"""
algorithms/_ts_common.py — All common tools for TS and SA variants.


Exports
-------
    _snapshot_paths(W)            → dict              
    _restore_paths(W, snap)       → None               
    _make_tabu_key(action)        → hashable tuple     
    _get_path_flow(W, fp, action) → float              
    _cls_score(action, W, fp, sl, rng) → float              
"""

import copy
import random

from core.world import time_rev, time_order

def _snapshot_paths(W: dict) -> dict:
    """Return a fully detached deep copy of W['grouped_paths']."""
    return copy.deepcopy(W["grouped_paths"])


def _restore_paths(W: dict, snap: dict) -> None:
    """Replace W['grouped_paths'] with a fresh deep copy of the snapshot."""
    W["grouped_paths"] = copy.deepcopy(snap)


# ── Tabu key ──────────────────────────────────────────────────────────────────

def _make_tabu_key(action):
    atype = action[0]
    if atype in ("B1", "B2"):
        return ("discount", action[1])          
    if atype in ("B3", "B4"):
        p = action[1]
        return ("schedule", p[2], p[3])          
    if atype in ("B5", "B6"):
        p = action[1]
        return ("transfer", p[1], p[3])          
    return tuple(action)


# ── B4/B6 Path Flow ───────────────────────────────────────────────────────────

def _get_path_flow(W: dict, flow_per_path: dict, action: tuple) -> float:
    atype = action[0]
    was, gtr = W["wait_arc_set"], W["get_transfer_arc"]
    total = 0.0
    for (_, _, pt), entry in flow_per_path.items():
        path = list(pt)
        if atype == "B4":
            origin_st, origin_ti, train_num, w = action[1]
            new_ti = time_rev[time_order[origin_ti] + w]
            if (len(path) >= 2
                    and (path[0], path[1]) in was
                    and path[0].split("_")[0] == origin_st
                    and path[0].split("_")[1] == origin_ti
                    and path[0].split("_")[2] == train_num
                    and path[1].split("_")[1] == new_ti):
                total += entry["flow"]
        elif atype == "B6":
            st_from, train_num, ti_from_xfer, _w = action[1]
            xfer = gtr(path)
            if xfer is not None:
                db = xfer[1]
                p = db.split("_")
                if p[0] == st_from and p[1] == ti_from_xfer and p[2] == train_num:
                    total += entry["flow"]
    return total


# ── CLS ──────────────────────────────────────

def _cls_score(action, W, fp, sl, rng, high_flow_first=None) -> float:
    """
    The higher the score, the more likely the action is to be selected by CLS.

    B1/B2  → Discount factor-lowering/raising for a specific door, which may increase/decrease the flow through that door
    B3/B5  → The maximum load of the train
    B4/B6  → close the path with large flow first (20% chance), or close the path with small flow first (80% chance)
    """
    atype = action[0]

    if atype in ("B1", "B2"):
        door = action[1]
        return sum(e["flow"] for (_, _, pt), e in fp.items() if pt and pt[0] == door)

    if atype in ("B3", "B5"):
        train_num = action[1][2] if atype == "B3" else action[1][1]
        best = 0.0
        for (rid, _si), load in sl.items():
            if str(rid[0]) == str(train_num):
                if load > best:
                    best = load
        return best

    if atype in ("B4", "B6"):
        flow = _get_path_flow(W, fp, action)

        if high_flow_first is None:
            high_flow_first = rng.random() < 0.2

        return flow if high_flow_first else -flow

    return 0.0