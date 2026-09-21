"""
core/objective.py — Objective function and overload helpers.
"""

from collections import defaultdict
from core.config import alpha0, alpha1, alpha2, max_capacity, penalty_weight
from core.world  import door_id


def compute_objective(W: dict, flow_per_path: dict, seg_load: dict) -> tuple[float, float, float]:
    """
    F = Z + lambda * P
    Returns (F, Z, P).
    """
    gbd = W["boarding_door_id"]
    gtr = W["get_transfer_arc"]
 
    # ── Per-door flow attribution ────────────────────────────────────────────
    B_v: dict = defaultdict(float)   # boardings per door
    A_v: dict = defaultdict(float)   # alightings per door
 
    for (_, _, pt), entry in flow_per_path.items():
        flow = entry["flow"]
        if flow < 1e-12:
            continue
        path = list(pt)
        if len(path) < 2:
            continue
 
        B_v[gbd(path)] += flow
 
        A_v[path[-1]] += flow
        # Mid-trip transfer at the hub: alight at the old train's door,
        tr = gtr(path)
        if tr is not None:
            A_v[tr[0]] += flow
            B_v[tr[1]] += flow
 
    # ── Group doors by stop ──────────────────────────────────────────────────
    # Each stop has a small list of door_ids (typically doors_per_run = 2).
    stop_to_doors: dict = defaultdict(list)
    for ev in W["events_list"]:
        st, ti, tr, _d = ev
        stop_to_doors[(st, ti, tr)].append(door_id(*ev))
 
    # ── Z: max-door dwell, summed over every stop ────────────────────────────
    Z = 0.0
    for _stop, doors in stop_to_doors.items():
        phis = []
        for d in doors:
            b = B_v.get(d, 0.0)
            a = A_v.get(d, 0.0)
            phis.append(alpha1 * b + alpha2 * a + 0.01 * b * a)
        # alpha0 once per stop
        Z += alpha0 + (max(phis) if phis else 0.0)
 
    # ── P: capacity overflow ─────────────────────────────────────────────────
    P = sum(max(0.0, load - max_capacity) for load in seg_load.values())
 
    return Z + penalty_weight * P, Z, P

def get_overloaded(seg_load: dict) -> list:
    """
    Returns overloaded segments sorted upstream-first (seg_idx ASC),
    then by load DESC.
    """
    return sorted(
        [(rid, si, ld) for (rid, si), ld in seg_load.items() if ld > max_capacity],
        key=lambda x: (x[0], x[1], -x[2]), # sort by route, then segment index, then load descending
    )
def compute_Z_fixed(W: dict) -> float:
    """Return the fixed part of Z: one alpha0 per timetable stop.

    The timetable, including GS A4 additions, determines this term; B1-B6
    cannot change it. Subtract it when comparing percentage improvements
    across instances.
    """
    stops = {(st, ti, tr) for (st, ti, tr, _d) in W["events_list"]}
    return alpha0 * len(stops)
