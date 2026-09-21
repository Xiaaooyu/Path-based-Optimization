"""Generate raw JSONL data for OFAT experiments on TS, SA, and GA.

Implements the following OFAT experiment stages:
python -m experiments.ofat_tuning --demand-seed 2026 2027 2028 2029 2030 --algo-seeds 0 1 2 3 4 --distance-seed 124 --stages ts1 

"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from functools import lru_cache
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config    import max_pie_index
from core.world     import reseed_world, snapshot_schedules, restore_schedules
from core.flow      import compute_flows, compute_segment_loads, reset_eval_count, get_eval_count
from core.objective import compute_objective
from core.verify    import verify_demand_satisfied

from algorithms.tabu_search_unified  import run_tabu_search, TSConfig
from algorithms.simulated_annealing  import run_simulated_annealing_basic
from algorithms.genetic_algorithm    import run_genetic_algorithm
from algorithms._ts_common           import _snapshot_paths, _restore_paths

from experiments.runner import load_state

DATA_DIR  = ROOT / "data"
WS_DIR    = DATA_DIR / "warmstarts"
RESULTS   = ROOT / "results"
RAW_DIR   = RESULTS / "ofat" / "raw"

FEAS_TOL       = 1e-9
DEFAULT_LAMBDA_EVAL = 16.0     # fixed evaluation-side penalty weight (§0 penalized_Z).
                               # Independent of TSConfig.ts_lambda / sa_lambda.
BIG_MAX_ITER   = 100_000       # effectively unbounded — max_evaluations is the budget.


# ===========================================================================
# 0. Misc small helpers
# ===========================================================================

@lru_cache(maxsize=1)
def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "no-git"


def _snapshot_pdf(pdf):
    from collections import defaultdict
    snap = defaultdict(lambda: max_pie_index)
    snap.update(pdf)
    return snap


def append_jsonl(row: dict, path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


# ===========================================================================
# 1. Instance loading (load once, restore cheaply before every run)
# ===========================================================================

def load_instance(demand_seed: int, distance_seed: int) -> dict:
    reseed_world(rng_seed=distance_seed, od_seed=demand_seed)
    path = WS_DIR / f"gs_d{demand_seed}_x{distance_seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"greedy warm-start not found: {path}\n"
            f"  run: python -m experiments.build_warmstarts "
            f"--demand-seeds {demand_seed} --distance-seeds {distance_seed}"
        )
    pdf, W, _runs_added, _hist = load_state(str(path))
    fp = compute_flows(W, pdf)
    sl = compute_segment_loads(W, fp)
    _, Z_ws, P_ws = compute_objective(W, fp, sl)
    return {
        "W":             W,
        "pdf":           pdf,
        "pdf_snap":      _snapshot_pdf(pdf),
        "paths_snap":    _snapshot_paths(W),
        "sched_snap":    snapshot_schedules(),
        "demand_seed":   demand_seed,
        "distance_seed": distance_seed,
        "instance_id":   f"d{demand_seed}_x{distance_seed}",
        "Z_ws":          Z_ws,
        "P_ws":          P_ws,
    }


def restore_instance(inst: dict):
    """Reset W / pdf / schedules to the instance's warm-start snapshot."""
    reseed_world(rng_seed=inst["distance_seed"], od_seed=inst["demand_seed"],
                reset_schedules=False)
    restore_schedules(inst["sched_snap"])
    _restore_paths(inst["W"], inst["paths_snap"])
    inst["pdf"].clear()
    inst["pdf"].update(inst["pdf_snap"])
    return inst["W"], inst["pdf"]


# ===========================================================================
# 2. Per-algorithm single-run wrappers → §0 raw-row schema
# ===========================================================================

def _base_row(stage, algorithm, mode, swept_param, swept_value, config,
              inst, seed, max_evaluations, lambda_eval):
    return {
        "run_id":       uuid.uuid4().hex[:12],
        "timestamp":    datetime.now().astimezone().isoformat(),
        "git_commit":   _git_commit(),
        "stage":        stage,
        "algorithm":    algorithm,
        "mode":         mode,
        "swept_param":  swept_param if swept_param is not None else "baseline",
        "swept_value":  swept_value,
        "config":       config,
        "instance_id":  inst["instance_id"],
        "demand_seed":  inst["demand_seed"],
        "distance_seed": inst["distance_seed"],
        "seed":         seed,
        "max_evaluations": max_evaluations,
        "lambda_eval":  lambda_eval,
    }


def run_ts_once(inst, cfg_dict, stage, mode, swept_param, swept_value,
                seed, max_evaluations, lambda_eval) -> dict:
    W, pdf = restore_instance(inst)
    row = _base_row(stage, "TS", mode, swept_param, swept_value, cfg_dict,
                    inst, seed, max_evaluations, lambda_eval)
    cfg = TSConfig(**{**cfg_dict, "seed": seed,
                      "max_evaluations": max_evaluations,
                      "max_iter": BIG_MAX_ITER, "verbose": False,
                      "FEAS_TOL": FEAS_TOL})
    reset_eval_count()
    t0 = time.perf_counter()
    try:
        pdf_out, history, curve = run_tabu_search(W, pdf, cfg)
    except Exception as e:
        row.update(Z=None, P=None, feasible=False, penalized_Z=None,
                   evaluations=get_eval_count(), runtime=time.perf_counter() - t0,
                   iterations=None, termination_reason="error",
                   diag={}, error=repr(e), traceback=traceback.format_exc())
        return row
    dt = time.perf_counter() - t0
    # Snapshot the search budget before final validation.  compute_flows() may
    # increment the global counter, but validation work is not search effort.
    evaluations = get_eval_count()

    fp = compute_flows(W, pdf_out)
    sl = compute_segment_loads(W, fp)
    _, Z, P = compute_objective(W, fp, sl)
    ok, _rep = verify_demand_satisfied(W, fp)
    feasible = bool(P <= FEAS_TOL and ok)

    aspirations = sum(1 for h in history if h[5])
    infeas_in_hist = sum(1 for h in history if h[4] is not None and h[4] > FEAS_TOL)
    total_evaluated = getattr(cfg, "evaluations", None)   # set in-place by run_tabu_search

    if evaluations >= max_evaluations - 1:
        termination_reason = "budget"
    else:
        termination_reason = "no_improve"

    row.update(
        Z=float(Z), P=float(P), feasible=feasible,
        penalized_Z=float(Z + lambda_eval * P),
        evaluations=int(evaluations), runtime=dt,
        iterations=max(0, len(curve) - 1),
        termination_reason=termination_reason,
        diag={
            "aspirations":          aspirations,
            "infeas_accepted_hist": infeas_in_hist,
            "n_history_entries":    len(history),
            "total_evaluated_pool": total_evaluated,
            "cls_enabled":          cfg.use_cls,
            "penalty_enabled":      cfg.use_penalty,
        },
        error=None, traceback=None,
    )
    return row


def run_sa_once(inst, cfg_dict, stage, mode, swept_param, swept_value,
                seed, max_evaluations, lambda_eval) -> dict:
    W, pdf = restore_instance(inst)
    row = _base_row(stage, "SA", mode, swept_param, swept_value, cfg_dict,
                    inst, seed, max_evaluations, lambda_eval)
    kwargs = dict(cfg_dict)
    kwargs.update(max_iter=BIG_MAX_ITER, max_evaluations=max_evaluations,
                 seed=seed, verbose=False)
    reset_eval_count()
    t0 = time.perf_counter()
    try:
        pdf_out, history, Z_best, curve = run_simulated_annealing_basic(W, pdf, **kwargs)
    except Exception as e:
        row.update(Z=None, P=None, feasible=False, penalized_Z=None,
                   evaluations=get_eval_count(), runtime=time.perf_counter() - t0,
                   iterations=None, termination_reason="error",
                   diag={}, error=repr(e), traceback=traceback.format_exc())
        return row
    dt = time.perf_counter() - t0
    evaluations = get_eval_count()

    fp = compute_flows(W, pdf_out)
    sl = compute_segment_loads(W, fp)
    _, Z, P = compute_objective(W, fp, sl)
    ok, _rep = verify_demand_satisfied(W, fp)
    feasible = bool(P <= FEAS_TOL and ok)

    n_accept_worse = sum(1 for h in history if h[5] and h[2] > 1e-9)
    n_reject_worse = sum(1 for h in history if (not h[5]) and h[2] > 1e-9)
    denom = n_accept_worse + n_reject_worse
    worse_accept_rate = (n_accept_worse / denom) if denom else None

    T0 = kwargs.get("T0", 0.5)
    alpha = kwargs.get("alpha", 0.95)
    t_min = kwargs.get("t_min", 1e-4)
    moves = max(1, int(kwargs.get("moves_per_temperature", 20)))
    temp_idx_est = evaluations // moves
    T_final_est = max(T0 * (alpha ** temp_idx_est), t_min)

    if evaluations >= max_evaluations - 1:
        termination_reason = "budget"
    elif T_final_est <= t_min + 1e-12:
        termination_reason = "t_min"
    else:
        termination_reason = "max_iter"

    row.update(
        Z=float(Z), P=float(P), feasible=feasible,
        penalized_Z=float(Z + lambda_eval * P),
        evaluations=int(evaluations), runtime=dt,
        iterations=max(0, len(curve) - 1),
        termination_reason=termination_reason,
        diag={
            "worse_accept_rate":  worse_accept_rate,
            "n_accept_worse":     n_accept_worse,
            "n_reject_worse":     n_reject_worse,
            "n_history_entries":  len(history),
            "T_final_est":        T_final_est,
            "Z_best_returned":    None if Z_best == float("inf") else float(Z_best),
        },
        error=None, traceback=None,
    )
    return row


def run_ga_once(inst, cfg_dict, stage, mode, swept_param, swept_value,
                seed, max_evaluations, lambda_eval) -> dict:
    W, pdf = restore_instance(inst)
    row = _base_row(stage, "GA", mode, swept_param, swept_value, cfg_dict,
                    inst, seed, max_evaluations, lambda_eval)
    kwargs = dict(cfg_dict)
    kwargs.update(max_evaluations=max_evaluations, seed=seed, verbose=False)
    reset_eval_count()
    t0 = time.perf_counter()
    try:
        pdf_out, curve, stats = run_genetic_algorithm(W, pdf, **kwargs)
    except Exception as e:
        row.update(Z=None, P=None, feasible=False, penalized_Z=None,
                   evaluations=get_eval_count(), runtime=time.perf_counter() - t0,
                   iterations=None, termination_reason="error",
                   diag={}, error=repr(e), traceback=traceback.format_exc())
        return row
    dt = time.perf_counter() - t0

    fp = compute_flows(W, pdf_out)
    sl = compute_segment_loads(W, fp)
    _, Z, P = compute_objective(W, fp, sl)
    Z, P = float(Z), float(P)
    ok, _rep = verify_demand_satisfied(W, fp)
    feasible = bool(P <= FEAS_TOL and ok)
    evaluations = int(stats["evaluations"])

    pop_size = max(1, int(kwargs.get("population_size", 30)))
    generation_est = (evaluations + pop_size - 1) // pop_size
    termination_reason = "budget" if evaluations >= max_evaluations - 1 else "stagnation"

    row.update(
        Z=Z, P=P, feasible=feasible,
        penalized_Z=float(Z + lambda_eval * P),
        evaluations=evaluations, runtime=dt,
        iterations=generation_est,
        termination_reason=termination_reason,
        diag={
            "n_curve_points":  len(curve),
            "generation_est":  generation_est,
            "final_F":         float(stats["F"]),
            "reported_Z":      float(stats["Z"]),
            "reported_P":      float(stats["P"]),
            "switches":        int(stats["switches"]),
            "doors":           int(stats["doors"]),
        },
        error=None, traceback=None,
    )
    return row


RUNNERS = {"TS": run_ts_once, "SA": run_sa_once, "GA": run_ga_once}


# ===========================================================================
# 3. Generic OFAT stage engine
# ===========================================================================

def run_stage(stage_key, algorithm, mode, baseline_dict, param_grids,
             instances, algo_seeds, max_evaluations, lambda_eval, raw_path) -> int:
    """Write one baseline block and every non-baseline OFAT run to JSONL."""
    run_fn = RUNNERS[algorithm]
    n_written = 0

    print(f"\n{'=' * 84}\n  {algorithm} :: {stage_key} ({mode})\n{'=' * 84}")
    print(f"  baseline config: {baseline_dict}")
    print(f"  params swept:    {list(param_grids)}")

    # ── baseline block ──────────────────────────────────────────────────
    for inst in instances:
        for s in algo_seeds:
            r = run_fn(inst, baseline_dict, stage_key, mode, None, None, s,
                       max_evaluations, lambda_eval)
            append_jsonl(r, raw_path)
            n_written += 1
            _log_run("baseline", None, inst["instance_id"], s, r)

    # ── per-param, non-baseline grid ────────────────────────────────────
    for param, grid in param_grids.items():
        baseline_value = _baseline_level(baseline_dict, param)
        for v in grid:
            if v == baseline_value:
                continue
            cfg_dict = dict(baseline_dict)
            if param == "perturb_steps":         # covaries min == max
                cfg_dict["perturb_steps_min"] = v
                cfg_dict["perturb_steps_max"] = v
            else:
                cfg_dict[param] = v
            for inst in instances:
                for s in algo_seeds:
                    r = run_fn(inst, cfg_dict, stage_key, mode, param, v, s,
                               max_evaluations, lambda_eval)
                    append_jsonl(r, raw_path)
                    n_written += 1
                    _log_run(param, v, inst["instance_id"], s, r)

    return n_written


def _log_run(param, value, instance_id, seed, r):
    if r.get("error"):
        print(f"    [{param}={value} {instance_id} seed={seed}]  ERROR  {r['error'][:90]}")
        return
    tag = "P=0  " if r["feasible"] else "Over!"
    print(f"    [{param}={value} {instance_id} seed={seed}]  {tag}  "
          f"Z={r['Z']:.4f}  penZ={r['penalized_Z']:.4f}  "
          f"evals={r['evaluations']}  term={r['termination_reason']}  "
          f"{r['runtime']:.2f}s")


# ===========================================================================
# 4. Parameter levels (baseline value included — see module docstring)
# ===========================================================================

TS_STAGE1_DEFAULTS = dict(
    use_perturbation=True, use_cls=False, use_penalty=False,
    tenure=8, perturbation_trigger=15,
    perturb_steps_min=5, perturb_steps_max=5,
    perturb_heavy_bias=0.30, perturb_top_k=3,
    label="TS-OFAT",
)
TS_STAGE1_GRID = {
    "tenure":               [4, 8, 16],
    "perturbation_trigger": [8, 15, 25],
    "perturb_steps":        [3, 5, 8],       # sets perturb_steps_min == max
    "perturb_heavy_bias":   [0.0, 0.3, 0.6],
    "perturb_top_k":        [1, 3, 6],
}

# Update this explicitly after analyzing the ts1 JSONL.  Dependent TS stages
# use this declared config and never select parameters from results internally.
TS_STAGE1_FROZEN = dict(TS_STAGE1_DEFAULTS)

TS_CLS_EXTRA_DEFAULTS = dict(use_cls=True, top_cls=24, random_extra=6,
                             random_ratio=0.05, cls_pool_max=64)
TS_STAGE2_GRID = {
    "tenure":               [4, 8, 16],
    "perturbation_trigger": [8, 15, 25],
    "top_cls":              [12, 24, 48],
    "random_extra":         [0, 6, 12],
    "random_ratio":         [0.0, 0.05, 0.15],
    "cls_pool_max":         [32, 64, 96],
}

TS_PENALTY_EXTRA_DEFAULTS = dict(use_penalty=True, ts_lambda= 5.0
                                 )
TS_STAGE3_GRID = {
    "tenure":               [4, 8, 16],
    "perturbation_trigger": [8, 15, 25],
    "perturb_steps":        [3, 5, 8],
    "perturb_heavy_bias":   [0.0, 0.3, 0.6],
    "perturb_top_k":        [1, 3, 6],
    "ts_lambda":            [0.0, 5.0, 16.0],
}

# Final combined TS stage.  Its baseline inherits the currently frozen stage-1
# perturbation settings plus the declared CLS and penalty settings.  Re-sweeping
# the union of their parameters makes interactions visible under the full mode.
TS_STAGE4_GRID = {
    # Core / perturbation parameters
    "tenure":               [4, 8, 16],
    "perturbation_trigger": [8, 15, 25],
    "perturb_steps":        [3, 5, 8],
    "perturb_heavy_bias":   [0.0, 0.3, 0.6],
    "perturb_top_k":        [1, 3, 6],
    # CLS parameters
    "top_cls":              [12, 24, 48],
    "random_extra":         [0, 6, 12],
    "random_ratio":         [0.0, 0.05, 0.15],
    "cls_pool_max":         [32, 64, 96],
    # Penalty parameter
    "ts_lambda":            [1.0, 5.0, 16.0
                             ],
}

SA_STAGE1_DEFAULTS = dict(T0=0.5
                          , alpha=0.95, t_min=1e-4,moves_per_temperature=20, use_penalty=False,sa_lambda=16.0)
SA_STAGE1_GRID = {
    "T0":                    [0.1, 0.5, 2.0],
    "alpha":                 [0.90, 0.95, 0.99],
    "moves_per_temperature": [10, 20, 40],
}

# Update this explicitly after analyzing the sa1 JSONL.
SA_STAGE1_FROZEN = dict(SA_STAGE1_DEFAULTS)

SA_STAGE2_GRID = {
    "T0":                    [0.1, 0.5, 2.0],
    "alpha":                 [0.90, 0.95, 0.99],
    "moves_per_temperature": [10, 20, 40],
    "sa_lambda": [0,4.0, 16.0, 32.0],
}


GA_DEFAULTS = dict(population_size=30, tournament_size=2, p_c=0.8, p_m=0.03,elite_size=2)
GA_GRID = {
    "population_size": [15, 30, 60],
    "tournament_size": [1, 2, 4],
    "p_c":             [0.5, 0.8, 1.0],
    "p_m":             [0.01, 0.03, 0.10],
    "elite_size":      [1, 2, 4],
}


def _shrink_for_quick(grid: dict, baseline_dict: dict) -> dict:
    """Keep exactly one non-baseline level for each parameter."""
    return {
        param: [next(value for value in levels
                     if value != _baseline_level(baseline_dict, param))]
        for param, levels in grid.items()
    }


# ===========================================================================
# 5. Design validation
# ===========================================================================

def _baseline_level(config: dict, param: str):
    if param == "perturb_steps":
        low = config.get("perturb_steps_min")
        high = config.get("perturb_steps_max")
        if low != high:
            raise ValueError(
                "OFAT perturb_steps requires perturb_steps_min == "
                "perturb_steps_max in the baseline config."
            )
        return low
    return config.get(param)


def validate_stage_design(stage_key: str, baseline_dict: dict, grid: dict) -> None:
    """Reject duplicate levels and require one declared baseline level."""
    for param, levels in grid.items():
        if not levels:
            raise ValueError(f"{stage_key}: grid for {param} is empty.")
        if len(levels) != len(set(levels)):
            raise ValueError(f"{stage_key}: duplicate levels in {param}: {levels}")
        baseline = _baseline_level(baseline_dict, param)
        if baseline is None:
            raise ValueError(
                f"{stage_key}: baseline config has no value for {param}."
            )
        if baseline not in levels and len(levels) > 1:
            raise ValueError(
                f"{stage_key}: {param} levels must include baseline value "
                f"{baseline}: {levels}"
            )


# ===========================================================================
# 6. Stage orchestration
# ===========================================================================

def do_stage(stage_key, algorithm, mode, baseline_dict, grid, instances, algo_seeds,
             max_evaluations, lambda_eval, raw_dir, overwrite):
    validate_stage_design(stage_key, baseline_dict, grid)
    raw_path = raw_dir / f"{algorithm.lower()}_{stage_key}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {raw_path}. Use --overwrite to "
                "replace it, or choose another --output-dir."
            )
        raw_path.unlink()

    n_written = run_stage(
        stage_key,
        algorithm,
        mode,
        baseline_dict,
        grid,
        instances,
        algo_seeds,
        max_evaluations,
        lambda_eval,
        raw_path,
    )
    print(f"\n  [{algorithm}::{stage_key}] wrote {n_written} rows -> {raw_path}")
    return raw_path


# ===========================================================================
# 7. Main
# ===========================================================================

DEFAULT_DEMAND_SEEDS = [2026, 2027, 2028, 2029, 2030]
DEFAULT_DISTANCE_SEEDS = [124]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demand-seed", "--demand-seeds", dest="demand_seeds",
                    type=int, nargs="+", default=DEFAULT_DEMAND_SEEDS,
                    help="one or more demand seeds (default: %(default)s)")
    ap.add_argument("--distance-seed", "--distance-seeds", dest="distance_seeds",
                    type=int, nargs="+", default=DEFAULT_DISTANCE_SEEDS,
                    help="one or more distance seeds (default: %(default)s)")
    ap.add_argument("--algo-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-evaluations", type=int, default=30000)
    ap.add_argument("--lambda-eval", type=float, default=DEFAULT_LAMBDA_EVAL,
                    help="fixed evaluation-side penalty weight for penalized_Z "
                         "(independent of TSConfig.ts_lambda / SA sa_lambda)")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help="directory for raw JSONL files (default: %(default)s)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing stage JSONL instead of stopping",
    )
    ap.add_argument("--stages", nargs="+",
                    choices=["ts1", "ts2", "ts3", "ts4",
                             "sa1", "sa2", "ga", "all"],
                    default=["ts1"])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: first 2 seeds, 1 non-baseline level/param")
    return ap


def main():
    args = build_arg_parser().parse_args()

    if args.max_evaluations <= 0:
        raise SystemExit("--max-evaluations must be positive.")
    for name, values in (
        ("demand seeds", args.demand_seeds),
        ("distance seeds", args.distance_seeds),
        ("algorithm seeds", args.algo_seeds),
    ):
        if not values:
            raise SystemExit(f"At least one {name} value is required.")
        if len(values) != len(set(values)):
            raise SystemExit(f"Duplicate {name} are not allowed: {values}")

    stages = (["ts1", "ts2", "ts3", "ts4", "sa1", "sa2", "ga"]
             if "all" in args.stages else args.stages)
    if len(stages) != len(set(stages)):
        raise SystemExit(f"Duplicate stages are not allowed: {stages}")

    demand_seeds = args.demand_seeds[:1] if args.quick else args.demand_seeds
    distance_seeds = args.distance_seeds[:1] if args.quick else args.distance_seeds
    algo_seeds = args.algo_seeds[:2] if args.quick else args.algo_seeds
    raw_dir = args.output_dir.resolve()

    print(f"{'=' * 84}\n  OFAT RAW JSONL GENERATOR — TS / SA / GA\n{'=' * 84}")
    print(f"  demand seeds:    {demand_seeds}")
    print(f"  distance seeds:  {distance_seeds}")
    print(f"  warm starts:     {len(demand_seeds) * len(distance_seeds)}")
    print(f"  algo seeds:      {algo_seeds}")
    print(f"  max_evaluations: {args.max_evaluations}")
    print(f"  lambda_eval:     {args.lambda_eval}")
    print(f"  stages:          {stages}")
    print(f"  output dir:      {raw_dir}")
    print(f"  git commit:      {_git_commit()}")

    instances = [load_instance(demand_seed, distance_seed)
                 for demand_seed, distance_seed
                 in product(demand_seeds, distance_seeds)]
    print("  warm-starts loaded: "
          f"{len(instances)} (Z range {min(i['Z_ws'] for i in instances):.4f}.."
          f"{max(i['Z_ws'] for i in instances):.4f})")

    t_all = time.perf_counter()

    if "ts1" in stages:
        grid = (_shrink_for_quick(TS_STAGE1_GRID, TS_STAGE1_DEFAULTS)
                if args.quick else TS_STAGE1_GRID)
        do_stage("ts1_perturbation", "TS", "perturbation",
                 TS_STAGE1_DEFAULTS, grid, instances, algo_seeds,
                 args.max_evaluations, args.lambda_eval, raw_dir, args.overwrite)

    if "ts2" in stages:
        base2 = {**TS_STAGE1_FROZEN, **TS_CLS_EXTRA_DEFAULTS}
        grid = (_shrink_for_quick(TS_STAGE2_GRID, base2)
                if args.quick else TS_STAGE2_GRID)
        do_stage("ts2_perturbation_cls", "TS", "perturbation_cls",
                 base2, grid, instances, algo_seeds,
                 args.max_evaluations, args.lambda_eval, raw_dir, args.overwrite)

    if "ts3" in stages:
        base3 = {**TS_STAGE1_FROZEN, **TS_PENALTY_EXTRA_DEFAULTS}
        grid = (_shrink_for_quick(TS_STAGE3_GRID, base3)
                if args.quick else TS_STAGE3_GRID)
        do_stage("ts3_perturbation_penalty", "TS", "perturbation_penalty",
                 base3, grid, instances, algo_seeds,
                 args.max_evaluations, args.lambda_eval, raw_dir, args.overwrite)

    if "ts4" in stages:
        base4 = {
            **TS_STAGE1_FROZEN,
            **TS_CLS_EXTRA_DEFAULTS,
            **TS_PENALTY_EXTRA_DEFAULTS,
            "label": "TS-Full-OFAT",
        }
        grid = (_shrink_for_quick(TS_STAGE4_GRID, base4)
                if args.quick else TS_STAGE4_GRID)
        do_stage("ts4_full", "TS", "full",
                 base4, grid, instances, algo_seeds,
                 args.max_evaluations, args.lambda_eval, raw_dir, args.overwrite)

    if "sa1" in stages:
        grid = (_shrink_for_quick(SA_STAGE1_GRID, SA_STAGE1_DEFAULTS)
                if args.quick else SA_STAGE1_GRID)
        do_stage("sa1_base_hard", "SA", "base",
                 SA_STAGE1_DEFAULTS, grid, instances, algo_seeds,
                 args.max_evaluations, args.lambda_eval, raw_dir, args.overwrite)

    if "sa2" in stages:
        base_p = {**SA_STAGE1_FROZEN, "use_penalty": True}
        grid = (_shrink_for_quick(SA_STAGE2_GRID, base_p)
                if args.quick else SA_STAGE2_GRID)
        do_stage("sa2_penalty", "SA", "penalty",
                 base_p, grid, instances, algo_seeds,
                 args.max_evaluations, args.lambda_eval, raw_dir, args.overwrite)

    if "ga" in stages:
        grid = (_shrink_for_quick(GA_GRID, GA_DEFAULTS)
                if args.quick else GA_GRID)
        do_stage("ga_default", "GA", "ga",
                 GA_DEFAULTS, grid, instances, algo_seeds,
                 args.max_evaluations, args.lambda_eval, raw_dir, args.overwrite)

    print(f"\n  total elapsed: {time.perf_counter() - t_all:.1f}s")
    print(f"  raw JSONL:     {raw_dir}")


if __name__ == "__main__":
    main()
