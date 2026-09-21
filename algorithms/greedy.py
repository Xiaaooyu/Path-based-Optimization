"""
algorithms/greedy.py — Algorithm 1: Greedy feasibility search.

Minimises F = Z + λ·P by greedily applying the best single action per
iteration until P = 0 (no capacity violations) or MAX_ITER is reached.

Action types
------------
A1  Raise discount on an existing boarding door by 1 level
A2  Open origin-wait-w paths (+ expose new boarding door)
A3  Open transfer-wait-w paths (+ expose new boarding door)
A4  Add a new run to a line  [irreversible fallback]
"""

from collections import defaultdict

from core.config    import max_pie_index, max_capacity, MAX_WAIT, pie
from core.world     import build_world, add_run_to_schedule, schedules
from core.flow      import compute_flows, compute_segment_loads, get_eval_count
from core.objective import compute_objective, get_overloaded
from core.actions   import (
    apply_action, undo_action,
    gen_A1_candidates, gen_A2_A3_candidates,
)

MAX_ITER        = 1000
DF_TOL          = 1e-4   # an action must improve F by more than this to be taken
MAX_STAGNANT_A4 = 3      # consecutive A4s that fail to reduce P before we stop


# ── A4 helpers ───────────────────────────────────────────────────────────────
def max_useful_runs() -> int:
    """
    How many runs of one line a passenger at that line's origin can reach.

    build_world().find_paths starts only from each line's earliest event at
    the origin station, and only allows a wait arc as the first step, bounded
    by MAX_WAIT.  So departures t1 .. t1+MAX_WAIT are reachable and everything
    later is dead weight: zero boarding flow, +alpha0 per stop on Z.
    """
    return 1 + MAX_WAIT


def eval_A4(pie_door_factor: dict, ln: str):
    """
    Measure the true effect of adding one run to line `ln`.

    Adds the run, rebuilds the world, evaluates, then rolls the schedule back
    so the caller decides whether to commit.

    Returns (F, P), or None if the line has no reachable departure slot left.
    """
    if len(schedules[ln]) >= max_useful_runs():
        return None

    add_run_to_schedule(ln, verbose=False)
    try:
        W2 = build_world(verbose=False)
        fp = compute_flows(W2, pie_door_factor)
        sl = compute_segment_loads(W2, fp)
        F, _, P = compute_objective(W2, fp, sl)
    finally:
        schedules[ln].pop()          # roll back
    return F, P


# ── Main loop ────────────────────────────────────────────────────────────────
def run_greedy(max_evaluations: int | None = None,
               max_iter: int = MAX_ITER) -> tuple[dict, dict, list]:
    """
    Run Algorithm 1 from scratch.

    Returns
    -------
    W               : world dict
    pie_door_factor : defaultdict of discount levels
    runs_added      : list of (iteration, line_num, dep_time)
    """
    W               = build_world()
    pie_door_factor = defaultdict(lambda: max_pie_index)
    runs_added      = []
    iteration       = 0
    stagnant_A4     = 0
    stop_reason     = "max_iter reached"

    while (iteration < max_iter
           and (max_evaluations is None or get_eval_count() < max_evaluations)):
        iteration += 1

        fp         = compute_flows(W, pie_door_factor)
        sl         = compute_segment_loads(W, fp)
        F0, Z0, P0 = compute_objective(W, fp, sl)
        over       = get_overloaded(sl)

        print(f"\n{'='*70}")
        print(f" Iter {iteration:3d}  F={F0:.2f}  Z={Z0:.2f}  P={P0:.2f}  "
              f"overloaded_segs={len(over)}")
        print(f" Runs: " +
              ", ".join(f"line{ln}×{len(schedules[ln])}" for ln in sorted(schedules)))
        print(f"{'='*70}")

        if P0 == 0:
            print("  No capacity violations. Done.")
            stop_reason = "P = 0 (feasible)"
            break

        cands_A1  = gen_A1_candidates(W, pie_door_factor)
        cands_A23 = gen_A2_A3_candidates(W, pie_door_factor, sl)
        cands     = cands_A1 + cands_A23
        print(f"  Candidates: A1={len(cands_A1)}  A2/A3={len(cands_A23)}")

        best_action = None
        best_dF     = -DF_TOL

        budget_exhausted = False
        for action in cands:
            if (max_evaluations is not None
                    and get_eval_count() >= max_evaluations):
                budget_exhausted = True
                break
            undo_info = apply_action(W, pie_door_factor, action)
            fp2       = compute_flows(W, pie_door_factor)
            sl2       = compute_segment_loads(W, fp2)
            F2, _, _  = compute_objective(W, fp2, sl2)
            dF        = F2 - F0
            undo_action(W, pie_door_factor, undo_info)

            if dF < best_dF:
                best_dF     = dF
                best_action = action

        if budget_exhausted:
            print(f"  Stopped: evaluation budget reached ({max_evaluations}).")
            stop_reason = f"evaluation budget ({max_evaluations}) reached"
            break

        # ── Commit best A1 / A2 / A3 ─────────────────────────────────────────
        if best_action is not None:
            apply_action(W, pie_door_factor, best_action)
            atype = best_action[0]
            if atype == "A1":
                d = best_action[1]
                print(f"  Committed A1: door={d}  "
                      f"discount={pie[pie_door_factor[d]]:.4f}  dF={best_dF:.4f}")
            elif atype == "A2":
                print(f"  Committed A2 (origin-wait): params={best_action[1]}  "
                      f"dF={best_dF:.4f}")
            elif atype == "A3":
                print(f"  Committed A3 (xfer-wait):   params={best_action[1]}  "
                      f"dF={best_dF:.4f}")
            continue

        # ── Fallback: A4, chosen by measured marginal effect ─────────────────
        scored = {}
        for ln in sorted(schedules):
            r = eval_A4(pie_door_factor, ln)
            if r is not None:
                scored[ln] = r          # ln -> (F_after, P_after)

        if scored:
            print("  A4 trial: " + ",  ".join(
                f"line{ln}: dF={F2 - F0:+.2f} P {P0:.2f}->{P2:.2f}"
                for ln, (F2, P2) in sorted(scored.items())))
        else:
            print(f"  A4 trial: every line has used all "
                  f"{max_useful_runs()} reachable departure slots.")

        # A4 always raises Z, so we cannot require dF < 0 here — we require
        # that it actually buys a reduction in P.
        helpful = {ln: v for ln, v in scored.items() if v[1] < P0 - 1e-9}

        if helpful:
            # Among the lines that really reduce P, take the cheapest one.
            worst_line  = min(helpful, key=lambda ln: helpful[ln][0])
            stagnant_A4 = 0
        elif scored and stagnant_A4 < MAX_STAGNANT_A4:
            # Plateau: no single run reduces P, but two might (an extra run on
            # one line can open transfer paths that only pay off once another
            # line also gets one).  Take the least-bad line and keep count.
            worst_line   = min(scored, key=lambda ln: (scored[ln][1], scored[ln][0]))
            stagnant_A4 += 1
            print(f"  (exploratory A4 {stagnant_A4}/{MAX_STAGNANT_A4}: "
                  f"no single run reduces P)")
        else:
            print("  No A1/A2/A3 improvement and no line whose extra run "
                  "reduces P — A4 is spent. Stopping.")
            print(f"  Residual overloaded segments "
                  f"(load / cap={max_capacity}):")
            for rid, si, ld in over:
                print(f"    line {rid[0]} run dep={rid[1]}  seg {si}  "
                      f"load={ld:.2f}  (+{ld - max_capacity:.2f})")
            stop_reason = f"infeasible under current action set (P={P0:.2f})"
            break

        F2, P2 = scored[worst_line]
        print(f"  No A1/A2/A3 improvement → A4: adding run to line "
              f"{worst_line}  (dF={F2 - F0:+.2f}, P {P0:.2f}->{P2:.2f})")

        add_run_to_schedule(worst_line)
        runs_added.append((iteration, worst_line,
                           schedules[worst_line][-1]["route"][0][1]))
        W = build_world()

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Total iterations : {iteration}")
    print(f"Stop reason      : {stop_reason}")
    fp = compute_flows(W, pie_door_factor)
    sl = compute_segment_loads(W, fp)
    F, Z, P = compute_objective(W, fp, sl)
    print(f"Final  F={F:.2f}  Z={Z:.2f}  P={P:.2f}")
    print(f"\nFinal schedule:")
    for ln in sorted(schedules):
        print(f"  Line {ln}: {len(schedules[ln])} run(s)"
              f"  (reachable cap = {max_useful_runs()})")
        for run in schedules[ln]:
            print(f"    {run['route']}")
    if runs_added:
        print(f"\nRuns added:")
        for it, ln, dep in runs_added:
            print(f"  iter={it:3d}  line {ln}  dep={dep}")
    print(f"\nDoors with raised discounts:")
    for d, level in sorted(pie_door_factor.items()):
        if level < max_pie_index:
            print(f"  {d:38s}: pie[{level}]={pie[level]:.4f}")

    return W, pie_door_factor, runs_added