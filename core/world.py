"""
core/world.py — Schedule management & world builder.
"""

import copy
import hashlib
import random
from collections import defaultdict
from itertools import permutations

import numpy as np

from core.config import (
    LINE_DEFS, TRAVEL_TIME, doors_per_run, MAX_WAIT, MAX_XFER_WAIT,
    max_capacity, pie, max_pie_index,
    beta0, beta1, beta2, beta3,
    alpha0, alpha1, alpha2, penalty_weight,
)

# ── Reproducible RNG (default seed) ───────

DEFAULT_RNG_SEED   = 124
OD_SEED            = 2026


rng    = np.random.default_rng(DEFAULT_RNG_SEED)
od_rng = random.Random(OD_SEED)    # Dedicated RNG for OD demand only

# ── Time slot registry ────────────────────────────────────────────────────────
time_order: dict[str, int] = {}
time_rev:   dict[int, str] = {}

def ensure_time(n: int) -> None:
    if n not in time_rev:
        t = f"t{n}"
        time_order[t] = n
        time_rev[n]   = t

for _i in range(1, 200):
    ensure_time(_i)

# ── OD demand ─────────────────────────────────────────────────────────────────
# OD demand uses its own dedicated RNG (od_rng) so that it stays fixed

stations = sorted({s for stops in LINE_DEFS.values() for s in stops})
od_pairs = list(permutations(stations, 2))
D_od:  dict   = {f"{o}_{d}": od_rng.randint(0, 45) for o, d in od_pairs}
OD:    dict   = {(o, d): D_od[f"{o}_{d}"] for o, d in od_pairs}
DDSet: set    = set(k[1] for k in OD)

# ── Door helpers ──────────────────────────────────────────────────────────────
def door_id(station, time, train, door) -> str:
    return f"{station}_{time}_{train}_d{door}"

def parse_door(did: str):
    p = did.split("_")
    return p[0], p[1], p[2], int(p[3][1:])

# ── Door position (order-independent, reproducible across processes) ──────────
_dist_cache: dict[tuple[str, int], float] = {}
_dist_seed: int = DEFAULT_RNG_SEED


def _door_dist_rng(did: str, seed: int) -> np.random.Generator:
    """Deterministic per-door RNG derived from (seed, door_id)."""
    h = hashlib.sha256(f"{seed}:{did}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def get_dist(did: str) -> float:
    key = (did, _dist_seed)
    if key not in _dist_cache:
        _dist_cache[key] = float(_door_dist_rng(did, _dist_seed).integers(2, 26))
    return _dist_cache[key]

# ── Schedule management ───────────────────────────────────────────────────────
def make_run(line_num: str, dep: str) -> dict:
    stops    = LINE_DEFS[line_num]
    dep_idx  = time_order[dep]
    route    = []
    for hop, st in enumerate(stops):
        ti_idx = dep_idx + hop * TRAVEL_TIME
        ensure_time(ti_idx)
        route.append((st, time_rev[ti_idx]))
    return {"train": line_num, "route": route}


schedules: dict[str, list] = {ln: [make_run(ln, "t1")] for ln in LINE_DEFS}

def next_departure(ln: str) -> str:
    last = max(time_order[r["route"][0][1]] for r in schedules[ln])
    ensure_time(last + 1)
    return time_rev[last + 1]


def add_run_to_schedule(ln: str, verbose: bool = True) -> dict:
    dep = next_departure(ln)
    nr  = make_run(ln, dep)
    schedules[ln].append(nr)
    if verbose:
        print(f"    [+] Added run  line={ln}  dep={dep}  route={nr['route']}")
    return nr


def all_runs() -> list:
    return [r for runs in schedules.values() for r in runs]


# ── Schedule snapshot / restore ───────────────────────────────────────────────
def snapshot_schedules() -> dict:
    return copy.deepcopy(schedules)


def restore_schedules(snap: dict) -> None:
    schedules.clear()
    schedules.update(copy.deepcopy(snap))


# ── build_world ───────────────────────────────────────────────────────────────
def build_world(verbose: bool = True) -> dict:
    flat     = all_runs()
    ev_rows  = []
    for run in flat:
        for st, ti in run["route"]:
            for d in range(1, doors_per_run + 1):
                ev_rows.append((st, ti, run["train"], d))

    events_list = list(dict.fromkeys(ev_rows))
    event_set   = set(events_list)
    adj         = {ev: [] for ev in events_list}

    # Move arcs
    for run in flat:
        tr, route = run["train"], run["route"]
        for d in range(1, doors_per_run + 1):
            for i in range(len(route) - 1):
                nf = (route[i][0],   route[i][1],   tr, d)
                nt = (route[i+1][0], route[i+1][1], tr, d)
                if nf in event_set and nt in event_set:
                    adj[nf].append((nt, "move"))

    # Transfer arcs (at s2, different train, 0 <= bo - ao <= MAX_XFER_WAIT)
    wait_time_xfer: dict = {}
    transfer_set:   set  = set()
    s2_evs = [ev for ev in events_list if ev[0] == "s2"]
    for a in s2_evs:
        for b in s2_evs:
            if a == b or a[2] == b[2]:
                continue
            ao, bo = time_order[a[1]], time_order[b[1]]
            if 0 <= bo - ao <= MAX_XFER_WAIT:
                adj[a].append((b, "transfer"))
                da, db = door_id(*a), door_id(*b)
                wait_time_xfer[(da, db)] = bo - ao
                transfer_set.add((da, db))

    # Wait arcs (same station+line, w <= MAX_WAIT)
    wait_arc_set: dict = {}
    by_st_tr = defaultdict(list)
    for ev in events_list:
        by_st_tr[(ev[0], ev[2])].append(ev)

    for (station, train_num), evs in by_st_tr.items():
        evs_s = sorted(evs, key=lambda e: time_order[e[1]])
        for i, ev_e in enumerate(evs_s):
            for ev_l in evs_s[i + 1:]:
                if ev_e[3] != ev_l[3]:
                    continue
                w = time_order[ev_l[1]] - time_order[ev_e[1]]
                if w > MAX_WAIT:
                    break
                adj[ev_e].append((ev_l, "wait"))
                de, dl = door_id(*ev_e), door_id(*ev_l)
                wait_arc_set[(de, dl)] = w

    # Path finding
    def find_paths(origin, dest, max_steps=30):
        paths  = []
        earliest_by_line: dict = {}     # line_num -> earliest time index
        for ev in events_list:
            if ev[0] != origin:
                continue
            ln, ti_idx = ev[2], time_order[ev[1]]
            if ln not in earliest_by_line or ti_idx < earliest_by_line[ln]:
                earliest_by_line[ln] = ti_idx

        # start from all events at the origin whose departure time is the earliest for their line
        starts = [
            ev for ev in events_list
            if ev[0] == origin and time_order[ev[1]] == earliest_by_line.get(ev[2], -1)
        ]

        stack  = [(ev, [ev], {ev}, "start") for ev in starts]
        while stack:
            node, path, visited, last = stack.pop()
            if node[0] == dest:
                paths.append(path)
                continue
            if len(path) >= max_steps:
                continue
            for nb, etype in adj.get(node, []):
                if nb in visited:
                    continue
                if etype == "transfer" and last != "move":
                    continue
                if etype == "wait" and last != "start":
                    continue
                stack.append((nb, path + [nb], visited | {nb}, etype))
        return paths

    feasible_paths_per_od = defaultdict(lambda: defaultdict(list))
    for o, d in od_pairs:
        for p in find_paths(o, d):
            path_doors = [door_id(*n) for n in p]
            feasible_paths_per_od[o][d].append(path_doors)
            for i in range(len(p) - 1):
                if p[i][2] != p[i+1][2]:
                    transfer_set.add((door_id(*p[i]), door_id(*p[i+1])))

    # Path helpers (closures over local data)
    def get_wait_units(path):
        k = (path[0], path[1]) if len(path) >= 2 else None
        return wait_arc_set[k] if k and k in wait_arc_set else 0

    def boarding_door_id(path):
        if len(path) >= 2 and (path[0], path[1]) in wait_arc_set:
            return path[1]
        return path[0]

    def get_transfer_arc(path):
        return next(((a, b) for a, b in zip(path, path[1:]) if (a, b) in transfer_set), None)

    def n_transfers(path):
        return sum(1 for a, b in zip(path, path[1:]) if (a, b) in transfer_set)

    # grouped_paths[o][d][boarding_door][transfer_arc] = [path, ...]
    grouped_paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for o in feasible_paths_per_od:
        for d in feasible_paths_per_od[o]:
            for path in feasible_paths_per_od[o][d]:
                if len(path) < 2:
                    continue
                grouped_paths[o][d][boarding_door_id(path)][get_transfer_arc(path)].append(path)

        # ── Run metadata + fast lookup ───────────────────────────────
    run_info = {}

    # (station,time,train) -> run_id
    event_to_run = {}

    # (station,time,train) -> (run_id, seg_idx)
    event_to_segment = {}

    for run in flat:

        rid = (
            run["train"],
            run["route"][0][1],   # departure time
        )

        route = run["route"]

        run_info[rid] = {
            "route": route,
            "seg_count": len(route) - 1,
        }

        for i, (st, ti) in enumerate(route):

            key = (
                st,
                ti,
                run["train"],
            )

            event_to_run[key] = rid

            # last stop has no outgoing segment
            if i < len(route) - 1:

                event_to_segment[key] = (
                    rid,
                    i,
                )

    n_p = sum(
        len(feasible_paths_per_od[o][d])
        for o in feasible_paths_per_od
        for d in feasible_paths_per_od[o]
    )
    if verbose:
        print(f"  [world] runs={len(flat)}  events={len(events_list)}  "
              f"paths={n_p}  wait_arcs={len(wait_arc_set)}")

    return dict(
        events_list           = events_list,
        wait_arc_set          = wait_arc_set,
        transfer_set          = transfer_set,
        wait_time_xfer        = wait_time_xfer,
        feasible_paths_per_od = feasible_paths_per_od,
        grouped_paths         = grouped_paths,

        run_info              = run_info,
        event_to_run          = event_to_run,
        event_to_segment      = event_to_segment,

        get_wait_units        = get_wait_units,
        boarding_door_id      = boarding_door_id,
        get_transfer_arc      = get_transfer_arc,
        n_transfers           = n_transfers,
    )
# ── reseed_world ────────────────
def reseed_world(rng_seed: int | None = None,
                 od_seed:  int | None = None,
                 reset_schedules: bool = True) -> None:
    """
    Reseed RNGs and recompute demand-related state.
    Call this before build_world() to get a different problem instance.
    """
    global rng, od_rng, D_od, OD, _dist_seed

    if rng_seed is not None:
        _dist_seed = rng_seed
        # Kept for any other module that imports core.world.rng directly.
        # Door distances no longer depend on this object (see get_dist).
        rng = np.random.default_rng(rng_seed)

    if od_seed is not None:
        od_rng = random.Random(od_seed)
        new_demand = {f"{o}_{d}": od_rng.randint(0, 45) for o, d in od_pairs}
        D_od.clear();  D_od.update(new_demand)
        OD.clear();    OD.update({(o, d): D_od[f"{o}_{d}"] for o, d in od_pairs})

    if reset_schedules:
        schedules.clear()
        schedules.update({ln: [make_run(ln, "t1")] for ln in LINE_DEFS})


def enumerate_wait_extended_paths(W: dict, max_added_waits: int = 2) -> int:
    """Add wait-extended paths to W['grouped_paths'] in place.

    For each base path already in grouped_paths, try inserting wait arcs at
    every internal node (origin and transfer points), with wait duration
    w in {1..MAX_WAIT}, and add the resulting extended path if it is
    feasible (all required wait arcs exist in wait_arc_set, and the
    resulting tail is consistent with the train's actual schedule).

    Parameters
    ----------
    W : world dict from build_world()
    max_added_waits : how many wait insertions to chain per base path.
        1 = single wait insertion (covers what greedy's A2/A3 do per action)
        2 = up to two waits in the same path (rare in practice but
            covers compound discoveries from successive A2/A3 calls)
        Higher values explode quickly — keep at 2 unless instance is small.

    Returns
    -------
    n_added : number of new paths inserted into grouped_paths.
    """
    gp        = W["grouped_paths"]
    was       = W["wait_arc_set"]
    ts        = W["transfer_set"]
    ri        = W["run_info"]
    gbd       = W["boarding_door_id"]
    gtr       = W["get_transfer_arc"]

    # Snapshot current path set so we don't re-enumerate over our own additions
    base_paths = []
    existing   = set()
    for u in gp:
        for j in gp[u]:
            for v in gp[u][j]:
                for tr in gp[u][j][v]:
                    for p in gp[u][j][v][tr]:
                        base_paths.append((u, j, tuple(p)))
                        existing.add(tuple(p))

    n_added = 0

    def _add_path(u, j, new_path):
        """Insert new_path into grouped_paths under (u, j, boarding, xfer)."""
        nonlocal n_added
        if tuple(new_path) in existing:
            return False
        nv  = gbd(new_path)
        ntr = gtr(new_path)
        gp.setdefault(u, {}) \
          .setdefault(j, {}) \
          .setdefault(nv, {}) \
          .setdefault(ntr, []).append(list(new_path))
        existing.add(tuple(new_path))
        n_added += 1
        return True

    def _try_origin_wait(u, j, base_path):
        """Open wait at origin: equivalent to greedy A2."""
        starts_with_wait = (len(base_path) >= 2
                            and (base_path[0], base_path[1]) in was)
        bd_idx = 1 if starts_with_wait else 0
        bd     = base_path[bd_idx]
        p      = bd.split("_")
        origin_st, origin_ti, train_num = p[0], p[1], p[2]
        dn = p[3][1:]

        for w in range(1, MAX_WAIT + 1):
            ti_idx = time_order[origin_ti] + w
            if ti_idx not in time_rev:
                continue
            new_ti = time_rev[ti_idx]
            d_e = f"{origin_st}_{origin_ti}_{train_num}_d{dn}"
            d_l = f"{origin_st}_{new_ti}_{train_num}_d{dn}"
            if (d_e, d_l) not in was:
                continue

            # Rebuild the tail from the actual train_num run stopping at
            # (origin_st, new_ti), through destination j.
            new_tail = None
            for rid2, ri2 in ri.items():
                if rid2[0] != train_num:
                    continue
                for si2, (rst, rti) in enumerate(ri2["route"]):
                    if rst == origin_st and rti == new_ti:
                        raw = [f"{rs}_{rt}_{train_num}_d{dn}"
                               for rs, rt in ri2["route"][si2:]]
                        if any(t.split("_")[0] == j for t in raw):
                            cut = next(
                                i + 1 for i, t in enumerate(raw)
                                if t.split("_")[0] == j
                            )
                            new_tail = raw[:cut]
                        break
                if new_tail:
                    break
            if not new_tail:
                continue

            # new_tail starts at d_l; prepend only d_e.
            new_path = [d_e] + new_tail
            _add_path(u, j, new_path)

    def _try_xfer_wait(u, j, base_path):
        """Open wait at every transfer point: equivalent to greedy A3."""
        for k in range(len(base_path) - 1):
            da, db = base_path[k], base_path[k + 1]
            if (da, db) not in ts:
                continue
            p = db.split("_")
            xfer_st, ti_from, train_num = p[0], p[1], p[2]
            dn = p[3][1:]
            arrive_ti = da.split("_")[1]

            for w in range(1, MAX_WAIT + 1):
                ti_idx = time_order[ti_from] + w
                if ti_idx not in time_rev:
                    continue
                new_ti = time_rev[ti_idx]
                if time_order[new_ti] < time_order[arrive_ti]:
                    continue
                # Find a tail that uses train_num starting at (xfer_st, new_ti)
                # and eventually hits the destination j.
                new_tail = None
                for rid2, ri2 in ri.items():
                    if rid2[0] != train_num:
                        continue
                    for si2, (rst, rti) in enumerate(ri2["route"]):
                        if rst == xfer_st and rti == new_ti:
                            raw = [f"{rs}_{rt}_{train_num}_d{dn}"
                                   for rs, rt in ri2["route"][si2:]]
                            if any(t.split("_")[0] == j for t in raw):
                                cut = next(
                                    i + 1 for i, t in enumerate(raw)
                                    if t.split("_")[0] == j
                                )
                                new_tail = raw[:cut]
                            break
                    if new_tail:
                        break
                if not new_tail:
                    continue
                new_path = list(base_path[:k + 1]) + new_tail
                _add_path(u, j, new_path)

    # Round 1: extend every base path with one wait insertion
    for (u, j, bp) in base_paths:
        _try_origin_wait(u, j, bp)
        _try_xfer_wait(u, j, bp)

    if max_added_waits >= 2:
        # Round 2: extend round-1 results with one more wait insertion.
        # We snapshot the post-round-1 set so round 2 only chains onto
        # round-1 discoveries, not its own outputs.
        round2_base = []
        for u in gp:
            for j in gp[u]:
                for v in gp[u][j]:
                    for tr in gp[u][j][v]:
                        for p in gp[u][j][v][tr]:
                            tp = tuple(p)
                            # Only chain onto paths NOT already in our
                            # original base — i.e. the round-1 additions.
                            if tp not in {bp for (_, _, bp) in base_paths}:
                                round2_base.append((u, j, tp))
        for (u, j, bp) in round2_base:
            _try_origin_wait(u, j, bp)
            _try_xfer_wait(u, j, bp)

    print(f"  [world] enumerate_wait_extended_paths: "
          f"added {n_added} paths "
          f"(base={len(base_paths)} → total={len(base_paths) + n_added})")
    return n_added
