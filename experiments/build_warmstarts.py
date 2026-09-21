"""
experiments/build_warmstarts.py

For each (demand_seed, distance_seed) pair, this:
  1. reseeds the world (OD demand + door distances + schedules)
  2. builds the world implicitly via run_greedy()
  3. runs Greedy   (algorithms.greedy.run_greedy)
  4. saves the resulting state to data/warmstarts/gs_d{dem}_x{dist}.json
  5. writes a sidecar data/warmstarts/gs_d{dem}_x{dist}.meta.json
     with the Greedy objective values (Z_greedy, P_greedy).

The full set written by default:
  demand_seeds   = [2026, 2027, 2028, 2029, 2030]
  distance_seeds = [124, 125, 126, 127, 128]
-> 25 greedy state files (+ 25 .meta.json sidecars).

Usage
-----
  # Default 5 x 5 = 25 greedy states
  python -m experiments.build_warmstarts

  # Custom seed grids
  python -m experiments.build_warmstarts --demand-seeds 2026 --distance-seeds 124 125 126 127 128
  python -m experiments.build_warmstarts --demand-seeds 2027 --distance-seeds 125


  # Skip existing files (resume after a crash)
  python -m experiments.build_warmstarts --skip-existing

"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.world as _core_world

from core.world     import reseed_world, schedules, _dist_cache
from core.flow      import compute_flows, compute_segment_loads
from core.objective import compute_objective
from core.verify    import verify_demand_satisfied

from algorithms.greedy import run_greedy

DATA_DIR     = ROOT / "data"
WS_DIR       = DATA_DIR / "warmstarts"
WS_DIR.mkdir(parents=True, exist_ok=True)


class NpEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalar and array values."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_state(pie_door_factor, runs_added, W, save_path,
               demand_seed: int | None = None) -> None:
    """Save a greedy state using the runner-compatible JSON format."""
    fp = compute_flows(W, pie_door_factor)

    all_paths = []
    for u in W["grouped_paths"]:
        for j in W["grouped_paths"][u]:
            for v in W["grouped_paths"][u][j]:
                for tr in W["grouped_paths"][u][j][v]:
                    for p in W["grouped_paths"][u][j][v][tr]:
                        e = fp.get((u, j, tuple(p)))
                        all_paths.append({
                            "u": u, "j": j, "v": v,
                            "tr": list(tr) if tr else None,
                            "path": list(p),
                            "flow": e["flow"] if e else 0.0,
                        })

    data = {
        "schedules": {
            ln: [{"train": r["train"], "route": r["route"]}
                 for r in runs]
            for ln, runs in schedules.items()
        },
        "pie_door_factor": {str(k): v for k, v in pie_door_factor.items()},
        "_dist_seed": _core_world._dist_seed,
        "dist_cache": {str(k): v for k, v in _dist_cache.items()},
        "runs_added": runs_added or [],
        "all_paths": all_paths,
        "od_demand": {f"{o}|{d}": v
                       for (o, d), v in _core_world.OD.items()},
    }
    if demand_seed is not None:
        data["demand_seed"] = int(demand_seed)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NpEncoder)

    sl = compute_segment_loads(W, fp)
    _, Z, P = compute_objective(W, fp, sl)
    print(f"  Saved -> {save_path}  (Z={Z:.4f}  P={P:.2f}  "
          f"paths={len(all_paths)})")


def greedy_state_path(demand_seed: int, distance_seed: int) -> Path:
    return WS_DIR / f"gs_d{demand_seed}_x{distance_seed}.json"

def greedy_state_meta_path(demand_seed: int, distance_seed: int) -> Path:
    return WS_DIR / f"gs_d{demand_seed}_x{distance_seed}.meta.json"

def build_one(demand_seed: int, distance_seed: int) -> dict:
    """Reseed, run Greedy, save its state and metadata, and return a summary."""
    t_start = time.perf_counter()

    print(f"\n{'-'*70}")
    print(f"  building greedy state  demand={demand_seed}  distance={distance_seed}")
    print(f"{'-'*70}")

    # reset_schedules=True (default) is important: each instance must start
    # from one-run-per-line, otherwise extra runs added in a previous
    # instance leak across (instance N+1 inherits N's schedule).
    reseed_world(rng_seed=distance_seed, od_seed=demand_seed)

    # ── Greedy ─────────────────────────────────────────────────────────
    t_g = time.perf_counter()

    W, pdf, runs_added = run_greedy()
    t_greedy = time.perf_counter() - t_g

    fp = compute_flows(W, pdf)
    sl = compute_segment_loads(W, fp)
    _, Z_g, P_g = compute_objective(W, fp, sl)
    print(f"  greedy done   Z={Z_g:.4f}  P={P_g:.4f}  ({t_greedy:.1f}s)")


    # ── Save greedy state ──────────────────────────────────────────────
    g_ok, g_report = verify_demand_satisfied(W, fp)
    g_out_path = greedy_state_path(demand_seed, distance_seed)
    save_state(pdf, runs_added, W, str(g_out_path), demand_seed=demand_seed)
    g_meta_path = greedy_state_meta_path(demand_seed, distance_seed)
    with open(g_meta_path, "w") as f:
        json.dump({
            "demand_seed":   demand_seed,
            "distance_seed": distance_seed,
            "Z_greedy":      Z_g,
            "P_greedy":      P_g,
            "feasible":      bool(g_ok),
            "state_file":    g_out_path.name,
        }, f, indent=2, cls=NpEncoder)
    print(f"  greedy state -> {g_out_path.name}")

    dt_total = time.perf_counter() - t_start
    return {
        "demand_seed":   demand_seed,
        "distance_seed": distance_seed,
        "path":          str(g_out_path),
        "meta_path":     str(g_meta_path),
        "Z_greedy":      Z_g,
        "P_greedy":      P_g,
        "feasible":      g_ok,
        "n_unreachable": len(g_report["unreachable"]),
        "demand_total":  g_report["total_demand"],
        "demand_served": g_report["total_served"],
        "time_greedy":   t_greedy,
        "time_total":    dt_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand-seeds", nargs="+", type=int,
                    default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--distance-seeds", nargs="+", type=int,
                    default=[124, 125, 126, 127, 128])
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip combinations whose greedy state file AND sidecar "
                         "already exist (older states missing a sidecar "
                         "are regenerated)")
    args = ap.parse_args()

    print(f"\n{'='*70}")
    print(f"  GREEDY STATE GENERATION")
    print(f"{'='*70}")
    print(f"  demand   seeds:  {args.demand_seeds}")
    print(f"  distance seeds:  {args.distance_seeds}")
    print(f"  total combos:    {len(args.demand_seeds) * len(args.distance_seeds)}")
    print(f"  output dir:      {WS_DIR}")
    print(f"  skip existing:   {args.skip_existing}")

    summaries = []
    t_all = time.perf_counter()

    for dem in args.demand_seeds:
        for dist in args.distance_seeds:
            out = greedy_state_path(dem, dist)
            meta = greedy_state_meta_path(dem, dist)
            # Only skip when BOTH state file and sidecar exist, so older
            # greedy states without a .meta.json get regenerated.
            if args.skip_existing and out.exists() and meta.exists():
                print(f"\n  skip (exists): {out.name}")
                continue
            summaries.append(build_one(dem, dist))

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GREEDY STATE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'demand':>7} {'dist':>5} {'Z_greedy':>10} "
          f"{'P_greedy':>10} {'feas':>5} {'time(s)':>9}")
    print(f"  {'-'*54}")
    for s in summaries:
        feas = "OK" if s["feasible"] else "FAIL"
        print(f"  {s['demand_seed']:>7d} {s['distance_seed']:>5d} "
              f"{s['Z_greedy']:>10.4f} {s['P_greedy']:>10.4f} "
              f"{feas:>5} {s['time_total']:>9.1f}")

    print(f"\n  total elapsed: {time.perf_counter() - t_all:.1f}s")

    # ── Manifest (used by sweep_ts_* scripts to discover warm-starts) ──
    manifest_path = WS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "demand_seeds":   args.demand_seeds,
            "distance_seeds": args.distance_seeds,
            "entries":        summaries,
        }, f, indent=2, cls=NpEncoder)
    print(f"  manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
