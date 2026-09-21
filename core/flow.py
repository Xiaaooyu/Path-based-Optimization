"""
core/flow.py — Passenger flow computation (logit model).
"""

import numpy as np
from collections import defaultdict

from core.config import pie, max_pie_index, beta0, beta1, beta2, beta3
from core.world  import OD, get_dist


_EVAL_COUNT = {"n": 0}

def reset_eval_count() -> None:
    _EVAL_COUNT["n"] = 0

def get_eval_count() -> int:
    return _EVAL_COUNT["n"]


def compute_flows(W: dict, pie_door_factor: dict) -> dict:
    _EVAL_COUNT["n"] += 1   
    """
    U(path) = beta0 + beta1*(1 - pie[pie_door_factor[boarding_door]])
            + beta2*dist(boarding_door)
            - beta3*(origin_wait + s2_transfer_wait + n_transfers)

    Returns flow_per_path: {(o, d, path_tuple): {"flow": float, "path": list}}
    """
    gwu = W["get_wait_units"]
    gbd = W["boarding_door_id"]
    gtr = W["get_transfer_arc"]
    ntr = W["n_transfers"]
    wtx = W["wait_time_xfer"]
    gp  = W["grouped_paths"]

    def U(path):
        bd   = gbd(path)
        disc = pie[min(pie_door_factor[bd], max_pie_index)]
        tr   = gtr(path)
        return (
            beta0
            + beta1 * (1 - disc)
            + beta2 * get_dist(bd)
            + beta3 * (gwu(path) + (wtx.get(tr, 0) if tr else 0) + ntr(path))
        )

    flow_per_path = {}
    for (u, j), demand in OD.items():
        if u not in gp or j not in gp[u]:
            continue
        all_p = [
            p
            for v  in gp[u][j]
            for tr in gp[u][j][v]
            for p  in gp[u][j][v][tr]
        ]
        if not all_p:
            continue
        exps = [np.exp(U(p)) for p in all_p]
        S    = sum(exps)
        if S == 0:
            continue
        for p, e in zip(all_p, exps):
            flow_per_path[(u, j, tuple(p))] = {"flow": demand * e / S, "path": p}

    return flow_per_path


def compute_segment_loads(W: dict, flow_per_path: dict) -> dict:
    """
    Returns seg_load: {(run_id, seg_idx): float}
    Wait and transfer arcs do NOT contribute to on-train load.
    """
    was      = W["wait_arc_set"]
    ts       = W["transfer_set"]
    ri       = W["run_info"]
    seg_load = defaultdict(float)

    for (u, j, pt), entry in flow_per_path.items():
        flow = entry["flow"]
        if flow < 1e-9:
            continue
        path = list(pt)
        for k in range(len(path) - 1):
            da, db = path[k], path[k + 1]
            if (da, db) in was or (da, db) in ts:
                continue
            parts = da.split("_")
            st_f, ti_f, tr_n = parts[0], parts[1], parts[2]
            for rid, rinfo in ri.items():
                if str(rid[0]) != tr_n:
                    continue
                for si, (rst, rti) in enumerate(rinfo["route"][:-1]):
                    if rst == st_f and rti == ti_f:
                        seg_load[(rid, si)] += flow
                        break

    return seg_load