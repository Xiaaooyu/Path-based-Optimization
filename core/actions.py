"""
core/actions.py — Apply / Undo logic and candidate generators for
                  Algorithm 1 actions (A1–A4) and Algorithm 2/3 actions (B1–B6).
"""

from collections import defaultdict, Counter

from core.config import MAX_WAIT, max_pie_index, pie, max_capacity
from core.world  import time_order, time_rev
from core.objective import get_overloaded
from core.flow import compute_flows, compute_segment_loads

def rebuild_tail(ri: dict, station: str, ti: str,
                 train_num: str, dn: str, dest: str) -> list | None:
    """
    Find a run of train_num stopping at (station, ti), and return its door
    nodes from that stop through the first arrival at dest. Return None if
    no such run reaches dest. The first node is
    f"{station}_{ti}_{train_num}_d{dn}".
    """
    for rid2, ri2 in ri.items():
        if rid2[0] != train_num:
            continue
        for si2, (rst, rti) in enumerate(ri2["route"]):
            if rst != station or rti != ti:
                continue
            raw = [f"{rs}_{rt}_{train_num}_d{dn}"
                   for rs, rt in ri2["route"][si2:]]
            cut = next((i + 1 for i, t in enumerate(raw)
                        if t.split("_")[0] == dest), None)
            if cut is not None:
                return raw[:cut]
            break          # This run stops here but misses dest; try the next run.
    return None
 
 
def path_is_time_monotone(path: list) -> bool:
    """Check t_n <= t_{n+1}; useful as a development assertion."""
    idx = [time_order[p.split("_")[1]] for p in path]
    return all(a <= b for a, b in zip(idx, idx[1:]))
 
 
def build_origin_wait_path(ri: dict, was: dict, orig: list, gbd,
                           origin_st: str, origin_ti: str,
                           train_num: str, new_ti: str, dest: str):
    """
    Build a path that waits at the origin until new_ti before boarding.
    Return None when the path is invalid. A2/B3 application and candidate
    checks share this constructor so a generated candidate adds a path.
    """
    bd = gbd(orig)
    p  = bd.split("_")
    if p[0] != origin_st or p[1] != origin_ti or p[2] != train_num:
        return None
 
    dn  = p[3][1:]
    d_e = f"{origin_st}_{origin_ti}_{train_num}_d{dn}"
    d_l = f"{origin_st}_{new_ti}_{train_num}_d{dn}"
    if (d_e, d_l) not in was:
        return None
 
    # Rebuild the tail from the later run's actual schedule.
    new_tail = rebuild_tail(ri, origin_st, new_ti, train_num, dn, dest)
    if not new_tail:
        return None
 
    new_path = [d_e] + new_tail          # new_tail[0] == d_l
    assert path_is_time_monotone(new_path), f"non-monotone path: {new_path}"
    return new_path
# =============================================================================
# Low-level path openers (shared by A2/A3 and B3/B5)
# =============================================================================
def build_xfer_wait_path(ri: dict, ts: set, orig: list, k: int,
                         xfer_st: str, ti_from: str, train_num: str,
                         new_ti: str, dest: str):
    """At transfer arc k, build a path that boards at new_ti after waiting.
    Return None when invalid. A3/B5 application and candidate checks share
    this constructor, as A2/B3 share build_origin_wait_path.
    """
    da, db = orig[k], orig[k + 1]
    if (da, db) not in ts:
        return None
    p = db.split("_")
    if p[0] != xfer_st or p[1] != ti_from or p[2] != train_num:
        return None
    dn = p[3][1:]

    new_tail = rebuild_tail(ri, xfer_st, new_ti, train_num, dn, dest)
    if not new_tail:
        return None
    if time_order[new_ti] < time_order[da.split("_")[1]]:
        return None
    # The rebuilt transfer arc must itself exist in transfer_set, otherwise
    # get_transfer_arc() returns None for the new path and the transfer wait
    # penalty silently disappears from U().
    if (da, new_tail[0]) not in ts:
        return None

    new_path = orig[:k + 1] + new_tail
    if not path_is_time_monotone(new_path):
        return None
    return new_path

def _open_origin_wait_paths(W: dict, origin_st, origin_ti, train_num, w) -> list:
    """Open wait-w paths from origin. Does NOT touch discounts.
    Returns new_paths_log for undo."""
    was  = W["wait_arc_set"]
    fpod = W["feasible_paths_per_od"]
    gp   = W["grouped_paths"]
    gbd  = W["boarding_door_id"]
    gtr  = W["get_transfer_arc"]
    ri   = W["run_info"]
 
    ti_idx = time_order[origin_ti] + w
    if ti_idx not in time_rev:
        return []
    new_ti = time_rev[ti_idx]
    log    = []
 
    for u in list(fpod):
        for j in list(fpod[u]):
            for orig in fpod[u][j]:
                np_ = build_origin_wait_path(
                    ri, was, orig, gbd, origin_st, origin_ti,
                    train_num, new_ti, j,
                )
                if np_ is None:
                    continue
                nv   = gbd(np_)
                ntr_ = gtr(np_)
                if any(tuple(q) == tuple(np_) for q in gp[u][j][nv][ntr_]):
                    continue
                gp[u][j][nv][ntr_].append(np_)
                log.append((u, j, nv, ntr_, np_))
    return log


def _open_xfer_wait_paths(W: dict, xfer_st, train_num, ti_from, w) -> list:
    """Open transfer-wait-w paths. Does NOT touch discounts.
    Returns new_paths_log for undo."""
    ts   = W["transfer_set"]
    fpod = W["feasible_paths_per_od"]
    gp   = W["grouped_paths"]
    gbd  = W["boarding_door_id"]
    gtr  = W["get_transfer_arc"]
    ri   = W["run_info"]

    ti_idx = time_order[ti_from] + w
    if ti_idx not in time_rev:
        return []
    new_ti = time_rev[ti_idx]
    log    = []

    for u in list(fpod):
        for j in list(fpod[u]):
            for orig in fpod[u][j]:
                for k in range(len(orig) - 1):
                    np_ = build_xfer_wait_path(
                        ri, ts, orig, k, xfer_st, ti_from,
                        train_num, new_ti, j)
                    if np_ is None:
                        continue
                    nv   = gbd(np_)
                    ntr_ = gtr(np_)
                    if any(tuple(q) == tuple(np_) for q in gp[u][j][nv][ntr_]):
                        continue
                    gp[u][j][nv][ntr_].append(np_)
                    log.append((u, j, nv, ntr_, np_))
                    break          # Process only the first matching transfer arc per path.
    return log


# =============================================================================
# Algorithm 1 — Apply / Undo  (A1–A3; A4 handled in greedy.py)
# =============================================================================

def apply_action(W: dict, pie_door_factor: dict, action: tuple):
    """Apply an Algorithm-1 action in-place. Returns undo_info."""
    atype = action[0]

    if atype == "A1":
        d = action[1]
        pie_door_factor[d] -= 1
        return ("A1", d, +1)

    elif atype == "A2":
        origin_st, origin_ti, train_num, w = action[1]
        log = _open_origin_wait_paths(W, origin_st, origin_ti, train_num, w)
        return ("A23", log)

    elif atype == "A3":
        xfer_st, train_num, ti_from, w = action[1]
        log = _open_xfer_wait_paths(W, xfer_st, train_num, ti_from, w)
        return ("A23", log)

    elif atype == "A4":
        raise ValueError("A4 must be handled outside apply/undo cycle")

    raise ValueError(f"Unknown action type: {atype}")


def undo_action(W: dict, pie_door_factor: dict, undo_info: tuple) -> None:
    """Undo a previously applied action."""
    if undo_info[0] == "A1":
        _, d, delta = undo_info
        pie_door_factor[d] += delta

    elif undo_info[0] == "A23":
        _, log = undo_info
        gp     = W["grouped_paths"]
        for (u, j, nv, ntr_, np_) in log:
            key = tuple(np_)
            gp[u][j][nv][ntr_] = [
                q for q in gp[u][j][nv][ntr_] if tuple(q) != key
            ]


# =============================================================================
# Algorithm 1 — Candidate generators
# =============================================================================

def gen_A1_candidates(W: dict, pie_door_factor: dict) -> list:
    """A1: raise discount on any existing boarding door (not yet at max)."""
    gp      = W["grouped_paths"]
    seen    = set()
    actions = []
    for u in gp:
        for j in gp[u]:
            for v in gp[u][j]:
                if v in seen:
                    continue
                seen.add(v)
                if pie_door_factor[v] > 0:
                    actions.append(("A1", v))
    return actions


def gen_A2_A3_candidates(W: dict, pie_door_factor: dict, seg_load: dict) -> list:
    """A2/A3: open wait paths for overloaded segments only."""
    was  = W["wait_arc_set"]
    ts   = W["transfer_set"]
    fpod = W["feasible_paths_per_od"]
    gp   = W["grouped_paths"]
    gbd  = W["boarding_door_id"]
    gtr  = W["get_transfer_arc"]
    ri   = W["run_info"]

    actions = []
    seen    = set()

    for rid, si, _ in get_overloaded(seg_load):
        route      = ri[rid]["route"]
        train_num  = rid[0]
        origin_st  = route[0][0]
        origin_ti  = route[0][1]
        st_from    = route[si][0]
        ti_from    = route[si][1]

        # A2: origin-wait
        for w in range(1, MAX_WAIT + 1):
            key = ("A2", origin_st, origin_ti, train_num, w)
            if key in seen:
                continue
            ti_idx = time_order[origin_ti] + w
            if ti_idx not in time_rev:
                continue
            new_ti       = time_rev[ti_idx]
            has_potential = False
            for u in fpod:
                if has_potential:
                    break
                for j in fpod[u]:
                    for orig in fpod[u][j]:
                        bd = gbd(orig)
                        p  = bd.split("_")
                        if p[0] != origin_st or p[1] != origin_ti or p[2] != train_num:
                            continue
                        dn  = p[3][1:]
                        d_e = f"{origin_st}_{origin_ti}_{train_num}_d{dn}"
                        d_l = f"{origin_st}_{new_ti}_{train_num}_d{dn}"
                        if (d_e, d_l) not in was:
                            continue
                        bd_idx = orig.index(bd)
                        np_    = [d_e, d_l] + orig[bd_idx + 1:]
                        nv     = gbd(np_)
                        ntr_   = gtr(np_)
                        if not any(tuple(q) == tuple(np_) for q in gp[u][j][nv][ntr_]):
                            has_potential = True
                            break
                    if has_potential:
                        break
            if has_potential:
                actions.append(("A2", (origin_st, origin_ti, train_num, w)))
                seen.add(key)

        # A3: transfer-wait (only for si > 0)
        if si > 0:
            for w in range(1, MAX_WAIT + 1):
                key = ("A3", st_from, train_num, ti_from, w)
                if key in seen:
                    continue
                ti_idx = time_order[ti_from] + w
                if ti_idx not in time_rev:
                    continue
                has_potential = False
                for u in fpod:
                    if has_potential:
                        break
                    for j in fpod[u]:
                        for orig in fpod[u][j]:
                            for k in range(len(orig) - 1):
                                da, db = orig[k], orig[k + 1]
                                if (da, db) not in ts:
                                    continue
                                p = db.split("_")
                                if p[0] == st_from and p[1] == ti_from and p[2] == train_num:
                                    has_potential = True
                                    break
                            if has_potential:
                                break
                        if has_potential:
                            break
                if has_potential:
                    actions.append(("A3", (st_from, train_num, ti_from, w)))
                    seen.add(key)

    return actions


def gen_A4_candidates(seg_load: dict) -> list:
    """A4: add a run to each line that has at least one overloaded segment."""
    lines_over = {rid[0] for (rid, si, _) in get_overloaded(seg_load)}
    return [("A4", ln) for ln in lines_over]


# =============================================================================
# Algorithm 2/3 — Shared cascade helper
# =============================================================================

def _cascade_remove(gp: dict, gbd, gtr, seed_v: str) -> list:
    """
    Remove paths from grouped_paths when their boarding door is seed_v or
    their transfer arc ends at seed_v. Return the removed paths for undo.
    The direct waiting paths have already been removed by the caller; this
    function handles their cascading effects.
    """
    removed = []
    for u in list(gp):
        for j in list(gp[u]):
            for v in list(gp[u][j]):
                for tr in list(gp[u][j][v]):
                    keep = []
                    for path in gp[u][j][v][tr]:
                        hit = False
                        # Condition 1: the boarding door matches.
                        if gbd(path) == seed_v:
                            hit = True
                        # Condition 2: the transfer destination matches.
                        if not hit:
                            xfer = gtr(path)
                            if xfer is not None:
                                _, db = xfer
                                if db == seed_v:
                                    hit = True
                        if hit:
                            removed.append((u, j, v, tr, path))
                        else:
                            keep.append(path)
                    gp[u][j][v][tr] = keep
    return removed


def _restore_removed(gp: dict, removed: list) -> None:
    """Restore removed paths to their original groups in gp."""
    for (u, j, v, tr, path) in removed:
        if not any(tuple(q) == tuple(path) for q in gp[u][j][v][tr]):
            gp[u][j][v][tr].append(path)


def _try_remove_with_capacity_guard(W: dict, pie_door_factor: dict,
                                    seed_vs: set,
                                    baseline_unreachable: set | None = None,
                                    enforce_capacity: bool = True) -> list | None:
    """
    Remove paths with boarding door or transfer destination in seed_vs.
    Optionally reject the removal if a segment exceeds max_capacity.
    Always reject it if an OD pair that was previously reachable loses all
    paths; baseline_unreachable lists pairs already unreachable at startup.

    Disabling the capacity guard allows infeasible intermediate states during
    strategic oscillation while preserving OD reachability.

    Return the removed paths on success, or None after restoring gp on failure.
    """
    from core.world import OD

    gp  = W["grouped_paths"]
    gbd = W["boarding_door_id"]
    gtr = W["get_transfer_arc"]

    removed = []
    for seed_v in seed_vs:
        removed.extend(_cascade_remove(gp, gbd, gtr, seed_v))

    if not removed:
        return []   # Nothing was removed; this is a no-op.

    # Guard 1: capacity (optional).
    if enforce_capacity:
        fpp  = compute_flows(W, pie_door_factor)
        load = compute_segment_loads(W, fpp)
        if any(ld > max_capacity for ld in load.values()):
            _restore_removed(gp, removed)
            return None

    # Guard 2: OD reachability (required).
    baseline = baseline_unreachable or set()
    for (o, d), dem in OD.items():
        if dem <= 0 or (o, d) in baseline:
            continue
        has = False
        if o in gp and d in gp[o]:
            for v in gp[o][d]:
                for tr in gp[o][d][v]:
                    if gp[o][d][v][tr]:
                        has = True
                        break
                if has:
                    break
        if not has:
            _restore_removed(gp, removed)
            return None

    return removed


# =============================================================================
# Algorithm 2/3 — Apply / Undo  (B1–B6)
# =============================================================================

def apply_hc_action(W: dict, pie_door_factor: dict, action: tuple,
                    baseline_unreachable: set | None = None,
                    enforce_capacity: bool = True):
    """Apply a Hill Climbing / Tabu action. Returns undo_info.

    baseline_unreachable: Optional OD pairs already unreachable at startup;
                          B4/B6 ignore these in the reachability guard.
    enforce_capacity:     True by default. When False, B4/B6 may cause
                          overloaded segments during strategic oscillation.
                          The reachability guard remains active.
    """
    atype = action[0]

    if atype == "B1":                          # raise discount (= A1)
        return apply_action(W, pie_door_factor, ("A1", action[1]))

    elif atype == "B2":                        # lower discount
        d = action[1]
        pie_door_factor[d] += 1
        return ("B2", d, -1)

    elif atype == "B3":                        # open origin-wait (= A2)
        return apply_action(W, pie_door_factor, ("A2", action[1]))

    elif atype == "B4":                        # close origin-wait paths + cascade
        origin_st, origin_ti, train_num, w = action[1]
        ti_idx = time_order[origin_ti] + w
        if ti_idx not in time_rev:
            return ("B4", [])
        new_ti = time_rev[ti_idx]

        # seed_v includes every {origin_st}_{new_ti}_{train_num}_d* door.
        seed_vs = set()
        for d in pie_door_factor:
            parts = d.split("_")
            if (parts[0] == origin_st
                    and parts[1] == new_ti
                    and parts[2] == train_num):
                seed_vs.add(d)

        removed = _try_remove_with_capacity_guard(W, pie_door_factor, seed_vs, baseline_unreachable, enforce_capacity)
        if removed is None:
            # A failed guard makes this action a no-op.
            return ("B4", [])
        return ("B4", removed)

    elif atype == "B5":                        # open xfer-wait (= A3)
        return apply_action(W, pie_door_factor, ("A3", action[1]))

    elif atype == "B6":                        # close xfer-wait paths + cascade
        st_from, train_num, ti_from_xfer, w = action[1]

        # seed_v includes every {st_from}_{ti_from_xfer}_{train_num}_d* door.
        seed_vs = set()
        for d in pie_door_factor:
            parts = d.split("_")
            if (parts[0] == st_from
                    and parts[1] == ti_from_xfer
                    and parts[2] == train_num):
                seed_vs.add(d)

        removed = _try_remove_with_capacity_guard(W, pie_door_factor, seed_vs, baseline_unreachable, enforce_capacity)
        if removed is None:
            return ("B6", [])
        return ("B6", removed)

    raise ValueError(f"Unknown HC action type: {atype}")


def undo_hc_action(W: dict, pie_door_factor: dict, undo_info: tuple) -> None:
    """Undo a Hill Climbing / Tabu action."""
    utype = undo_info[0]

    if utype in ("A1", "A23"):                 # B1 / B3 / B5 reuse greedy undo
        undo_action(W, pie_door_factor, undo_info)

    elif utype == "B2":
        _, d, delta = undo_info
        pie_door_factor[d] += delta            # delta = -1

    elif utype in ("B4", "B6"):
        # Restore all directly and transitively removed paths together.
        _, removed = undo_info
        gp = W["grouped_paths"]
        for (u, j, v, tr, path) in removed:
            if not any(tuple(q) == tuple(path) for q in gp[u][j][v][tr]):
                gp[u][j][v][tr].append(path)


# =============================================================================
# Algorithm 2/3 — Candidate generators (B1–B6)
# =============================================================================

def gen_B1_candidates(W: dict, pie_door_factor: dict) -> list:
    """B1: raise discount (same logic as A1, but returns B1-typed actions)."""
    gp      = W["grouped_paths"]
    seen    = set()
    actions = []
    for u in gp:
        for j in gp[u]:
            for v in gp[u][j]:
                if v in seen:
                    continue
                seen.add(v)
                if pie_door_factor[v] > 0:
                    actions.append(("B1", v))   # ← B1, not A1
    return actions


def gen_B2_candidates(W: dict, pie_door_factor: dict) -> list:
    """B2: lower discount on doors that have been raised at least once."""
    gp      = W["grouped_paths"]
    seen    = set()
    actions = []
    for u in gp:
        for j in gp[u]:
            for v in gp[u][j]:
                if v in seen:
                    continue
                seen.add(v)
                if pie_door_factor[v] < max_pie_index:
                    actions.append(("B2", v))
    return actions


def gen_B3_candidates(W: dict) -> list:
    """B3: open origin-wait-w paths for ALL runs (not just overloaded)."""
    was     = W["wait_arc_set"]
    fpod    = W["feasible_paths_per_od"]
    gp      = W["grouped_paths"]
    gbd     = W["boarding_door_id"]
    gtr     = W["get_transfer_arc"]
    ri      = W["run_info"]
    actions = []
    seen    = set()

    for rid, rinfo in ri.items():
        train_num = rid[0]
        origin_st = rinfo["route"][0][0]
        origin_ti = rinfo["route"][0][1]
        for w in range(1, MAX_WAIT + 1):
            key = ("B3", origin_st, origin_ti, train_num, w)
            if key in seen:
                continue
            ti_idx = time_order[origin_ti] + w
            if ti_idx not in time_rev:
                continue
            new_ti   = time_rev[ti_idx]
            has_new  = False
            for u in fpod:
                if has_new:
                    break
                for j in fpod[u]:
                    for orig in fpod[u][j]:
                        bd = gbd(orig)
                        p  = bd.split("_")
                        if p[0] != origin_st or p[1] != origin_ti or p[2] != train_num:
                            continue
                        dn  = p[3][1:]
                        d_e = f"{origin_st}_{origin_ti}_{train_num}_d{dn}"
                        d_l = f"{origin_st}_{new_ti}_{train_num}_d{dn}"
                        if (d_e, d_l) not in was:
                            continue
                        bd_idx = orig.index(bd)
                        np_    = [d_e, d_l] + orig[bd_idx + 1:]
                        nv     = gbd(np_)
                        ntr_   = gtr(np_)
                        if not any(tuple(q) == tuple(np_) for q in gp[u][j][nv][ntr_]):
                            has_new = True
                            break
                    if has_new:
                        break
            if has_new:
                actions.append(("B3", (origin_st, origin_ti, train_num, w)))
                seen.add(key)
    return actions


def gen_B4_candidates(W: dict) -> list:
    """B4: close currently open origin-wait paths."""
    was     = W["wait_arc_set"]
    gp      = W["grouped_paths"]
    actions = []
    seen    = set()

    for u in gp:
        for j in gp[u]:
            for v in gp[u][j]:
                for tr in gp[u][j][v]:
                    for path in gp[u][j][v][tr]:
                        if len(path) < 2 or (path[0], path[1]) not in was:
                            continue
                        p0        = path[0].split("_")
                        origin_st = p0[0]; origin_ti = p0[1]; train_num = p0[2]
                        w         = was[(path[0], path[1])]
                        key       = ("B4", origin_st, origin_ti, train_num, w)
                        if key not in seen:
                            actions.append(("B4", (origin_st, origin_ti, train_num, w)))
                            seen.add(key)
    return actions


def gen_B5_candidates(W: dict) -> list:
    """B5: open transfer-wait-w paths.
    Candidate checks and application share build_xfer_wait_path, so each
    generated candidate has an effect."""
    ts      = W["transfer_set"]
    fpod    = W["feasible_paths_per_od"]
    gp      = W["grouped_paths"]
    gbd     = W["boarding_door_id"]
    gtr     = W["get_transfer_arc"]
    ri      = W["run_info"]
    actions = []
    seen    = set()

    for rid, rinfo in ri.items():
        train_num = rid[0]
        for si, (st_from, ti_from) in enumerate(rinfo["route"]):
            if si == 0:
                continue
            for w in range(1, MAX_WAIT + 1):
                key = ("B5", st_from, train_num, ti_from, w)
                if key in seen:
                    continue
                ti_idx = time_order[ti_from] + w
                if ti_idx not in time_rev:
                    continue
                new_ti  = time_rev[ti_idx]
                has_new = False
                for u in fpod:
                    if has_new:
                        break
                    for j in fpod[u]:
                        if has_new:
                            break
                        for orig in fpod[u][j]:
                            for k in range(len(orig) - 1):
                                np_ = build_xfer_wait_path(
                                    ri, ts, orig, k, st_from, ti_from,
                                    train_num, new_ti, j)
                                if np_ is None:
                                    continue
                                nv, ntr_ = gbd(np_), gtr(np_)
                                if not any(tuple(q) == tuple(np_)
                                           for q in gp[u][j][nv][ntr_]):
                                    has_new = True
                                break      # Match _open_xfer_wait_paths: use the first match.
                            if has_new:
                                break
                if has_new:
                    actions.append(("B5", (st_from, train_num, ti_from, w)))
                    seen.add(key)
    return actions


def gen_B6_candidates(W: dict) -> list:
    """B6: close currently open transfer-wait paths."""
    was     = W["wait_arc_set"]
    gp      = W["grouped_paths"]
    gtr     = W["get_transfer_arc"]
    actions = []
    seen    = set()

    for u in gp:
        for j in gp[u]:
            for v in gp[u][j]:
                for tr in gp[u][j][v]:
                    for path in gp[u][j][v][tr]:
                        if len(path) < 2 or (path[0], path[1]) in was:
                            continue
                        xfer = gtr(path)
                        if xfer is None:
                            continue
                        da, db = xfer
                        p      = db.split("_")
                        st_from      = p[0]
                        ti_from_xfer = p[1]
                        train_num    = p[2]
                        arrive_ti    = da.split("_")[1]
                        if time_order[ti_from_xfer] > time_order[arrive_ti]:
                            w   = time_order[ti_from_xfer] - time_order[arrive_ti]
                            key = ("B6", st_from, train_num, ti_from_xfer, w)
                            if key not in seen:
                                actions.append(("B6", (st_from, train_num, ti_from_xfer, w)))
                                seen.add(key)
    return actions
