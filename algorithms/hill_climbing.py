"""
algorithms/hill_climbing.py — Algorithm 2: Hill Climbing.

Starts from Algorithm 1's feasible solution (P = 0) and improves Z
while keeping P = 0 at all times.

Neighbourhood N(s)
------------------
B1  Raise discount on existing boarding door (+1 level)
B2  Lower discount on existing boarding door (-1 level)
B3  Open origin-wait-w paths
B4  Close origin-wait-w paths
B5  Open transfer-wait-w paths
B6  Close transfer-wait-w paths
"""

from collections import defaultdict
import time

from core.config  import max_pie_index
from core.flow    import compute_flows, compute_segment_loads, get_eval_count
from core.objective import compute_objective
from core.actions import (
    apply_hc_action, undo_hc_action,
    gen_B1_candidates, gen_B2_candidates,
    gen_B3_candidates, gen_B4_candidates,
    gen_B5_candidates, gen_B6_candidates,
)

HC_MAX_ITER      = 300
NO_IMPROVE_LIMIT = 10
MIN_MEANINGFUL_CHANGE = 1e-4 
LAST_STATS = {}

def run_hill_climbing(W: dict, pie_door_factor: dict,
                      max_evaluations: int | None = None,
                      max_runtime: float | None = None,
                      max_iter: int = HC_MAX_ITER) -> tuple[dict, list]:
    """
    Run Algorithm 2 (Hill Climbing) starting from Algorithm 1's solution.

    Parameters
    ----------
    W               : world dict from build_world()
    pie_door_factor : discount-level dict from Algorithm 1

    Returns
    -------
    pie_door_factor : updated (in-place)
    hc_history      : list of (iter, action_type, dZ, Z, P)
    """
    start = time.perf_counter()
    termination_reason = None

    # Warm-start check
    fp0             = compute_flows(W, pie_door_factor)
    sl0             = compute_segment_loads(W, fp0)
    _, Z_ws, P_ws   = compute_objective(W, fp0, sl0)
    print(f"Warm-start from Algorithm 1:  F={_:.2f}  Z={Z_ws:.2f}  P={P_ws:.2f}")
    if P_ws > 0:
        print("  WARNING: warm-start not feasible (P > 0). Run greedy first.")

    hc_history       = []
    hc_iter          = 0
    no_improve_count = 0
    
    best_Z_sofar = Z_ws
    curve = [{"evaluation": get_eval_count(), "elapsed_time": 0.0,
              "best_Z": Z_ws, "best_P": P_ws}]
    print(f"\nStarting Hill Climbing (B1–B6)...")
    print(f"{'='*70}")

    while hc_iter < max_iter:
        if max_evaluations is not None and get_eval_count() >= max_evaluations:
            termination_reason = "max_evaluations"
            break
        if (max_runtime is not None
                and time.perf_counter() - start >= max_runtime):
            termination_reason = "max_runtime"
            break
        hc_iter += 1

        fp          = compute_flows(W, pie_door_factor)
        sl          = compute_segment_loads(W, fp)
        F0, Z0, P0  = compute_objective(W, fp, sl)

        print(f"\n HC iter {hc_iter:3d}  Z={Z0:.2f}  P={P0:.2f}  "
              f"no_improve={no_improve_count}/{NO_IMPROVE_LIMIT}")

        cands  = (
            gen_B1_candidates(W, pie_door_factor)
            + gen_B2_candidates(W, pie_door_factor)
            + gen_B3_candidates(W)
            + gen_B4_candidates(W)
            + gen_B5_candidates(W)
            + gen_B6_candidates(W)
        )
        print(f"  N(s): "
              + "  ".join(f"{t}={sum(1 for c in cands if c[0]==t)}"
                          for t in ("B1","B2","B3","B4","B5","B6")))

        best_action = None
        best_dZ     = 0.0
        best_Z      = Z0
        

        limit_reached = None
        for action in cands:
            if (max_evaluations is not None
                    and get_eval_count() >= max_evaluations):
                limit_reached = "max_evaluations"
                break
            if (max_runtime is not None
                    and time.perf_counter() - start >= max_runtime):
                limit_reached = "max_runtime"
                break
            undo_info        = apply_hc_action(W, pie_door_factor, action)
            if (max_evaluations is not None
                    and get_eval_count() >= max_evaluations):
                undo_hc_action(W, pie_door_factor, undo_info)
                limit_reached = "max_evaluations"
                break
            if (max_runtime is not None
                    and time.perf_counter() - start >= max_runtime):
                undo_hc_action(W, pie_door_factor, undo_info)
                limit_reached = "max_runtime"
                break
            fp2              = compute_flows(W, pie_door_factor)
            sl2              = compute_segment_loads(W, fp2)
            F2, Z2, P2       = compute_objective(W, fp2, sl2)
            dZ               = Z2 - Z0
            undo_hc_action(W, pie_door_factor, undo_info)

            if P2 == 0 and dZ < -MIN_MEANINGFUL_CHANGE and dZ < best_dZ:
                best_dZ     = dZ
                best_Z      = Z2
                best_action = action

        if limit_reached is not None:
            termination_reason = limit_reached
            print(f"  Stopped: {limit_reached} reached.")
            curve.append({"evaluation": get_eval_count(),
                          "elapsed_time": time.perf_counter() - start,
                          "best_Z": best_Z_sofar, "best_P": 0.0})
            break

        if best_action is not None:
            apply_hc_action(W, pie_door_factor, best_action)
            print(f"  Committed {best_action[0]}: {best_action[1]}  "
                  f"dZ={best_dZ:.2f}  Z: {Z0:.2f} → {best_Z:.2f}")
            hc_history.append((hc_iter, best_action[0], best_dZ, best_Z, 0))
            no_improve_count = 0
            if best_Z < best_Z_sofar:
                best_Z_sofar = best_Z
        else:
            no_improve_count += 1
            print(f"  No improvement  ({no_improve_count}/{NO_IMPROVE_LIMIT})")


        curve.append({"evaluation": get_eval_count(),
                      "elapsed_time": time.perf_counter() - start,
                      "best_Z": best_Z_sofar, "best_P": 0.0})

        if no_improve_count >= NO_IMPROVE_LIMIT:
            termination_reason = "own_stagnation"
            print(f"  Stopped: no improvement for {NO_IMPROVE_LIMIT} consecutive iters.")
            break

    if termination_reason is None:
        termination_reason = "max_iter"

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    # The committed candidate was already evaluated in the search. Reuse its
    # values so final reporting does not consume an unbudgeted flow evaluation.
    Z_fin = best_Z_sofar
    P_fin = 0.0 if P_ws <= 1e-9 else P_ws
    print(f"Hill Climbing complete after {hc_iter} iterations.")
    print(f"  Warm-start:    Z={Z_ws:.2f}  P={P_ws:.2f}")
    print(f"  Final:         Z={Z_fin:.2f}  P={P_fin:.2f}")
    print(f"  Z improvement: {Z_ws - Z_fin:.2f}  "
          f"({(Z_ws - Z_fin) / Z_ws * 100:.1f}%)")
    print(f"\nActions committed:")
    for it, atype, dZ, Z, P in hc_history:
        print(f"  iter={it:3d}  {atype}  dZ={dZ:+.2f}  Z={Z:.2f}")

    global LAST_STATS
    LAST_STATS = {
        "evaluations": get_eval_count(),
        "runtime": time.perf_counter() - start,
        "termination_reason": termination_reason,
        "iterations": hc_iter,
    }
    return pie_door_factor, hc_history, curve
