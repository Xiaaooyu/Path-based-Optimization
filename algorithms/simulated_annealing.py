"""
algorithms/simulated_annealing_basic.py

Basic SA with two feasibility modes:

  * Hard-constraint mode (default, use_penalty=False)
        Infeasible moves (P>0) are rejected outright, exactly as before.

  * Penalty mode (use_penalty=True)
        Infeasible moves are allowed; the acceptance decision is driven by
        the *penalized* objective  pen = Z + sa_lambda * P  (ported from
        algorithms/simulated_annealing_unified.py). Capacity is NOT enforced
        at apply time in this mode (enforce_capacity=False) so the search can
        traverse infeasible regions, but only *feasible* (P==0) states are
        ever recorded as best-so-far.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from collections import defaultdict

from core.config    import max_pie_index
from core.flow      import compute_flows, compute_segment_loads, get_eval_count
from core.objective import compute_objective
from core.actions   import (
    apply_hc_action, undo_hc_action,
    gen_B1_candidates, gen_B2_candidates,
    gen_B3_candidates, gen_B4_candidates,
    gen_B5_candidates, gen_B6_candidates,
)
from algorithms._ts_common import _snapshot_paths, _restore_paths

from dataclasses import dataclass

LAST_STATS = {}


@dataclass
class SAConfig:
    max_iter: int = 3000
    T0: float = 0.1
    alpha: float = 0.95
    t_min: float = 1e-4
    moves_per_temperature: int = 20
    use_penalty: bool = True
    sa_lambda: float = 16.0
    seed: int | None = None
    label: str = "SA-Basic"
    verbose: bool = True 


PRESETS = {
    "base": SAConfig(use_penalty=False, label="SA-Basic-Hard"),
    "penalty": SAConfig(use_penalty=True, label="SA-Basic-Penalty"),
}


def run_simulated_annealing(W, pie_door_factor, cfg=None):
    """Compatibility wrapper exposing the new basic-SA implementation."""
    cfg = cfg or SAConfig()
    t0 = cfg.T0
    pdf, history, _ = run_simulated_annealing_basic(
        W, pie_door_factor, max_iter=cfg.max_iter, T0=t0,
        alpha=cfg.alpha, t_min=cfg.t_min,
        moves_per_temperature=cfg.moves_per_temperature,
        seed=cfg.seed, use_penalty=cfg.use_penalty,
        sa_lambda=cfg.sa_lambda)
    return pdf, history


_GENERATORS = [
    (gen_B1_candidates, True),
    (gen_B2_candidates, True),
    (gen_B3_candidates, False),
    (gen_B4_candidates, False),
    (gen_B5_candidates, False),
    (gen_B6_candidates, False),
]


def _pick_candidate(W, pdf, rng):
    """Pick a random candidate move using the supplied local RNG."""
    gen, needs_pdf = rng.choice(_GENERATORS)
    cands = gen(W, pdf) if needs_pdf else gen(W)
    return rng.choice(cands) if cands else None


def _snapshot_pdf(pdf):
    """Copy pie_door_factor while preserving defaultdict behavior."""
    snap = defaultdict(lambda: max_pie_index)
    snap.update(pdf)
    return snap





# ── main loop ────────────────────────────────────────────────────────────────

def run_simulated_annealing_basic(W, pie_door_factor,
                                  max_iter=5000,
                                  max_evaluations=None,
                                  max_runtime=None,
                                  T0=0.1,
                                  alpha=0.95,
                                  t_min=1e-4,
                                  moves_per_temperature=20,
                                  log_every=200,
                                  seed=None,
                                  verbose=True,
                                  use_penalty=False,
                                  sa_lambda=10.0):
    """
    Parameters
    ----------
    use_penalty : bool
        False → hard-constraint mode (infeasible moves rejected outright).
        True  → penalty mode: acceptance driven by Z + sa_lambda * P,
                infeasible intermediate states allowed.
    sa_lambda : float
        Penalty weight on P. Only used when use_penalty=True.

    Returns
    -------
    pie_door_factor : defaultdict
        Mutated in place to the best feasible solution found (or the last
        state if penalty mode never reached feasibility).
    history : list[tuple]
        (it, atype_str, dZ_true, Z_new, P_new, accepted, T)
        — compatible with plot_sa_convergence in run_experiment.py.
        dZ_true is the *true* objective delta (Z_new - Z_cur); the
        penalized delta is what actually drives acceptance in penalty mode.
    Z_best : float
        Best feasible Z found, or float("inf") if none was feasible.
    """
    start = time.perf_counter()
    termination_reason = None
    rng = random.Random(seed)
    enforce_cap = not use_penalty

    # ── initial state ─────────────────────────────────────────────────────
    fp = compute_flows(W, pie_door_factor)
    sl = compute_segment_loads(W, fp)
    _, Z_cur, P_cur = compute_objective(W, fp, sl)

    if P_cur > 0 and not use_penalty:
        raise ValueError(
            f"SA-Basic started from an infeasible warm-start "
            f"(P0={P_cur:.4f}, Z0={Z_cur:.4f}). The initial state is not "
            f"feasible — check that the warm-start was loaded and restored "
            f"faithfully before calling SA. (Pass use_penalty=True to allow "
            f"an infeasible start in penalty mode.)"
        )
    if P_cur > 0 and use_penalty and verbose:
        print(f"  SA-Basic[penalty]: warm-start is infeasible "
              f"(P0={P_cur:.4f}); penalty mode will tolerate it.")

    pen_cur = Z_cur + sa_lambda * P_cur if use_penalty else Z_cur

    # ── best-so-far snapshot (FEASIBLE states only) ───────────────────────
    # Even in penalty mode, "best" must be a genuinely feasible solution.
    if P_cur == 0:
        Z_best  = Z_cur
        it_best = 0
    else:
        Z_best  = float("inf")
        it_best = -1
    pdf_best   = _snapshot_pdf(pie_door_factor)
    paths_best = _snapshot_paths(W)

    # ── diagnostics counters ──────────────────────────────────────────────
    dZ_pos_samples   = []           # penalized dZ > 0 (drives acceptance)
    n_accept_improve = 0
    n_accept_worse   = 0
    n_reject_worse   = 0
    n_infeasible     = 0            # hard-mode reverts only
    n_infeas_accept  = 0            # penalty-mode accepted infeasible states
    n_no_action      = 0

    history = []
    
    curve = [{"evaluation": get_eval_count(), "elapsed_time": 0.0,
              "best_Z": (Z_cur if P_cur == 0 else None),
              "best_P": P_cur}]

    def _curve_point():
        curve.append({
            "evaluation": get_eval_count(),
            "elapsed_time": time.perf_counter() - start,
            "best_Z": None if it_best < 0 else Z_best,
            "best_P": 0.0,
        })
        
    T = T0
    moves_at_temperature = 0
    temperature_index = 0
    moves_per_temperature = max(1, int(moves_per_temperature))

    def finish_temperature_move():
        """Count one move and cool only after a complete temperature block."""
        nonlocal T, moves_at_temperature, temperature_index
        moves_at_temperature += 1
        if moves_at_temperature < moves_per_temperature:
            return False
        old_T = T
        T = max(T * alpha, t_min)
        temperature_index += 1
        moves_at_temperature = 0
        if verbose:
            print(f"  SA temperature={temperature_index:4d} complete "
                  f"moves={moves_per_temperature} "
                  f"T={old_T:.5f}->{T:.5f}")
        return T <= t_min

    if verbose:
        mode = (f"PENALTY (λ={sa_lambda:g})" if use_penalty
                else "HARD-CONSTRAINT")
        print(f"\n  SA-Basic start | mode={mode} | "
              f"Z0={Z_cur:.4f}  P0={P_cur:.2f}  "
              f"T0={T0}  alpha={alpha}  t_min={t_min}  "
        f"max_iter={max_iter}  max_evaluations={max_evaluations}  seed={seed}")
    
    for it in range(1, max_iter + 1):
        if (max_evaluations is not None
                and get_eval_count() >= max_evaluations):
            termination_reason = "max_evaluations"
            if verbose:
                print(f"  SA stopped: evaluation budget reached "
                      f"({max_evaluations}).")
            break
        if (max_runtime is not None
                and time.perf_counter() - start >= max_runtime):
            termination_reason = "max_runtime"
            if verbose:
                print(f"  SA stopped: runtime budget reached "
                      f"({max_runtime}s).")
            break
        temp_move = moves_at_temperature + 1
        temp_no = temperature_index + 1
        action = _pick_candidate(W, pie_door_factor, rng)
        if action is None:
            n_no_action += 1
            if verbose:
                print(f"  SA iter={it:4d} temperature={temp_no:4d} "
                      f"move={temp_move}/{moves_per_temperature} "
                      f"no action T={T:.5f}")
            _curve_point()
            if finish_temperature_move():
                termination_reason = "temperature_min"
                break
            continue

        ui = apply_hc_action(W, pie_door_factor, action,
                             enforce_capacity=enforce_cap)
        if (max_evaluations is not None
                and get_eval_count() >= max_evaluations):
            undo_hc_action(W, pie_door_factor, ui)
            termination_reason = "max_evaluations"
            _curve_point()
            break
        if (max_runtime is not None
                and time.perf_counter() - start >= max_runtime):
            undo_hc_action(W, pie_door_factor, ui)
            termination_reason = "max_runtime"
            _curve_point()
            break
        fp = compute_flows(W, pie_door_factor)
        sl = compute_segment_loads(W, fp)
        _, Z_new, P_new = compute_objective(W, fp, sl)

        # hard mode: infeasible → revert immediately
        if not use_penalty and P_new > 0:
            undo_hc_action(W, pie_door_factor, ui)
            n_infeasible += 1
            if verbose:
                print(f"  SA iter={it:4d} action={action[0]} "
                      f"temperature={temp_no:4d} "
                      f"move={temp_move}/{moves_per_temperature} "
                      f"rejected infeasible P={P_new:.4f} "
                      f"Z_cur={Z_cur:.4f} best={Z_best:.4f}")
            _curve_point()
            if finish_temperature_move():
                termination_reason = "temperature_min"
                break
            continue

        # penalized objective drives the Metropolis decision
        pen_new = Z_new + sa_lambda * P_new if use_penalty else Z_new
        dZ      = pen_new - pen_cur          # acceptance / diagnostics
        dZ_true = Z_new - Z_cur              # true objective delta (history)

        if dZ > 0:
            dZ_pos_samples.append(dZ)

        atype = action[0]
        accepted = False

        if dZ < 0:
            accepted = True
            n_accept_improve += 1
        elif rng.random() < math.exp(-dZ / max(T, 1e-12)):
            accepted = True
            n_accept_worse += 1
        else:
            undo_hc_action(W, pie_door_factor, ui)
            n_reject_worse += 1

        if accepted:
            Z_cur, P_cur, pen_cur = Z_new, P_new, pen_new
            if P_new > 0:
                n_infeas_accept += 1

            # best-so-far update — FEASIBLE improvements only
            if P_new == 0 and Z_cur < Z_best - 1e-4:
                Z_best     = Z_cur
                pdf_best   = _snapshot_pdf(pie_door_factor)
                paths_best = _snapshot_paths(W)
                it_best    = it

        history.append((it, str(atype), dZ_true, Z_new, P_new, accepted, T))
        _curve_point()

        if verbose:
            zb = "n/a" if it_best < 0 else f"{Z_best:.4f}"
            status = "accepted" if accepted else "rejected"
            print(f"  SA iter={it:4d} temperature={temp_no:4d} "
                  f"move={temp_move}/{moves_per_temperature} "
                  f"action={atype} {status} "
                  f"dZ={dZ_true:+.4f} dF={dZ:+.4f} T={T:.5f} "
                  f"Z_cur={Z_cur:.4f} best={zb}")

        if finish_temperature_move():
            termination_reason = "temperature_min"
            break

    if termination_reason is None:
        termination_reason = "max_iter"

    # ── restore best feasible snapshot before returning ───────────────────
    if it_best >= 0:
        _restore_paths(W, paths_best)
        pie_door_factor.clear()
        pie_door_factor.update(pdf_best)
    else:
        # penalty mode never reached a feasible state — keep last state.
        if verbose:
            print("\n  WARNING: SA-Basic never found a feasible solution; "
                  "returning the last visited state.")

    if verbose:
        _print_dZ_report(dZ_pos_samples, T0,
                         n_accept_improve, n_accept_worse,
                         n_reject_worse, n_infeasible, n_no_action,
                         max_iter, Z_best, it_best,
                         use_penalty, sa_lambda, n_infeas_accept)

    global LAST_STATS
    LAST_STATS = {
        "evaluations": get_eval_count(),
        "runtime": time.perf_counter() - start,
        "termination_reason": termination_reason,
        "moves_per_temperature": moves_per_temperature,
        "T0": T0,
        "alpha": alpha,
        "lambda": sa_lambda,
    }
    return pie_door_factor, history, Z_best, curve


# ── dZ diagnostics ───────────────────────────────────────────────────────────

def _print_dZ_report(dZ_pos, T0,
                     n_imp, n_w, n_rej, n_inf, n_no, max_iter,
                     Z_best, it_best,
                     use_penalty=False, sa_lambda=16.0, n_infeas_accept=0):
    print(f"\n  {'─' * 64}")
    print(f"  SA-Basic diagnostics"
          f"  [{'penalty λ=%g' % sa_lambda if use_penalty else 'hard-constraint'}]")
    print(f"  {'─' * 64}")
    print(f"  iterations         = {max_iter}")
    print(f"  no-action          = {n_no}")
    if use_penalty:
        print(f"  infeasible accepted= {n_infeas_accept}  "
              f"(penalty mode tolerates P>0)")
    else:
        print(f"  infeasible (P>0)   = {n_inf}  (rejected outright)")
    print(f"  accepted improve   = {n_imp}")
    print(f"  accepted worse     = {n_w}")
    print(f"  rejected worse     = {n_rej}")
    total_metro = n_w + n_rej
    if total_metro > 0:
        print(f"  worse-move acc.    = {n_w / total_metro:.3f}  "
              f"(empirical Metropolis acceptance)")

    if dZ_pos:
        s = sorted(dZ_pos)
        def q(p): return s[min(len(s) - 1, int(p * len(s)))]
        unit = "penalized dZ" if use_penalty else "dZ"
        print(f"\n  positive-{unit} stats  (n={len(s)})")
        print(f"    min     = {s[0]:.6f}")
        print(f"    p10     = {q(0.10):.6f}")
        print(f"    p25     = {q(0.25):.6f}")
        print(f"    median  = {q(0.50):.6f}")
        print(f"    mean    = {statistics.mean(s):.6f}")
        print(f"    p75     = {q(0.75):.6f}")
        print(f"    p95     = {q(0.95):.6f}")
        print(f"    max     = {s[-1]:.6f}")

        # T0 suggestion: exp(-dZ_med / T0) = p  ⇒  T0 = dZ_med / ln(1/p)
        med = q(0.50)
        T0_80 = med / math.log(1 / 0.80)
        T0_50 = med / math.log(1 / 0.50)
        T0_20 = med / math.log(1 / 0.20)
        print(f"\n  suggested T0 (based on median {unit} = {med:.4f}):")
        print(f"    acceptance ~= 80%  ->  T0 ~= {T0_80:.4f}")
        print(f"    acceptance ~= 50%  ->  T0 ~= {T0_50:.4f}")
        print(f"    acceptance ~= 20%  ->  T0 ~= {T0_20:.4f}")

        acc_mean = sum(math.exp(-dz / max(T0, 1e-12)) for dz in s) / len(s)
        print(f"\n  your T0={T0}  ->  theoretical mean acceptance of worse "
              f"moves at start ≈ {acc_mean:.3f}")
    else:
        print(f"\n  (no positive-dZ samples — every accepted move was improving)")
        print("  -> T0 has no practical effect for this run; SA degenerated to HC")

    if it_best < 0:
        print(f"\n  best Z = (none — no feasible solution found)")
    else:
        print(f"\n  best Z = {Z_best:.4f}  (found at iter {it_best})")
    print(f"  {'─' * 64}")
