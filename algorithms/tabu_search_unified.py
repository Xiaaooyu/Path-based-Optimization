"""
Tabu Search version with unified configuration.
--------
    base         = TSConfig()
    cls          = TSConfig(use_cls=True)
    penalty      = TSConfig(use_penalty=True, ts_lambda=16)
    cls_penalty  = TSConfig(use_cls=True, use_penalty=True, ts_lambda=16)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from collections import defaultdict, Counter
from core.config    import max_pie_index, pie
from core.flow      import compute_flows, compute_segment_loads, get_eval_count
from core.objective import compute_objective
from core.actions   import (
    apply_hc_action, undo_hc_action,
    gen_B1_candidates, gen_B2_candidates,
    gen_B3_candidates, gen_B4_candidates,
    gen_B5_candidates, gen_B6_candidates,
)
from algorithms._ts_common import (
    _snapshot_paths, _restore_paths,
    _make_tabu_key, _get_path_flow, _cls_score,
)

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TSConfig:
    # ── Core TS ──
    max_iter:              int   = 500
    tenure:                int   = 4
    perturbation_trigger: int = 15
    no_improve_ratio:     int = 5              # Number of perturbation intervals allowed.
    no_improve_limit:     int | None = None    # Derived automatically when None.
    def __post_init__(self):
            if self.no_improve_limit is None:
                self.no_improve_limit = (self.perturbation_trigger
                                         * self.no_improve_ratio)

    use_perturbation:      bool  = True
    perturb_steps:         int   = 5
    min_meaningful_change: float = 1e-4
    aspiration_threshold:  float = 0.05
    perturb_top_k:         int   = 3
    FEAS_TOL: float = 1e-9
    
    # ── Perturbation (multi-step kick) ──
    # Kept as compatibility aliases for old sweep scripts. New experiments
    # use the single Markdown-defined k_p value above.
    perturb_steps_min:     int   = 5
    perturb_steps_max:     int   = 5
    perturb_heavy_bias:    float = 0.30      # prob of biasing toward high-flow B4/B6
    perturb_accept:        str   = "always"

    # ── CLS (Candidate List Strategy) ──
    use_cls:               bool  = False
    top_cls:               int   = 24        # compatibility count
    cls_ratio:             float = 1.00      # legacy compatibility field
    random_extra:          int   = 6
    random_ratio:          float = 0.05
    cls_pool_max:          int   = 64        # Leave room for random candidates.
    

    # ── Dynamic / fixed penalty (P>0) ──
    use_penalty:           bool  = False
    ts_lambda:             float = 5.0
    seed:                  int | None = None
    label:                 str   = "TS"
    max_evaluations:       int | None = None
    max_runtime:           float | None = None
    verbose:               bool  = True


PRESETS: dict[str, TSConfig] = {
    # Four ablations required by the experimental framework:
    "perturbation": TSConfig(use_perturbation=True, label="TS"),
    "perturbation_cls": TSConfig(use_perturbation=True, use_cls=True,
                                  top_cls=24,
                                  label="TS+CLS"),
    "perturbation_penalty": TSConfig(use_perturbation=True, use_penalty=True,
                                      ts_lambda=16.0,
                                      label="TS+OverPenalty"),
    "full": TSConfig(use_perturbation=True, use_cls=True, top_cls=24,
                      use_penalty=True,
                      ts_lambda=16.0, label="TS+CLS+OverPenalty"),
    # Compatibility names used by the old runner and old sweep scripts.
    "base": TSConfig(use_perturbation=True, label="TS"),
    "cls": TSConfig(use_perturbation=True, use_cls=True, label="TS+CLS"),
    "penalty": TSConfig(use_perturbation=True, use_penalty=True,
                         ts_lambda=16.0, label="TS+OverPenalty"),
    "cls_penalty": TSConfig(use_perturbation=True, use_cls=True,
                             use_penalty=True, ts_lambda=16.0,
                             label="TS+CLS+OverPenalty"),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_tabu_search(W: dict,
                    pie_door_factor: dict,
                    cfg: TSConfig | None = None
                    ) -> tuple[dict, list]:
    """
    history : [(iter, atype, dZ, Z, P, is_asp, penZ), ...]
    """
    if cfg is None:
        cfg = TSConfig()

    start = time.perf_counter()
    cfg._start_time = start
    termination_reason = None

    log = print if cfg.verbose else (lambda *args, **kwargs: None)

    LAMBDA = cfg.ts_lambda
    enforce_cap = not cfg.use_penalty
    rng = random.Random(cfg.seed)

    # ── Tabu list ────────────────────────────────────────────────────────────
    tabu_list: dict[tuple, int] = {}

    def is_tabu(action, it):
        return tabu_list.get(_make_tabu_key(action), 0) > it

    def add_tabu(action, it):
        tabu_list[_make_tabu_key(action)] = it + cfg.tenure

    def clean_tabu(it):
        for k in [k for k, exp in tabu_list.items() if exp <= it]:
            del tabu_list[k]

    # ── Warm start ───────────────────────────────────────────────────────────
    fp0 = compute_flows(W, pie_door_factor)
    sl0 = compute_segment_loads(W, fp0)
    _, Z_start, P_start = compute_objective(W, fp0, sl0)
    pen_start = Z_start + LAMBDA * P_start

    log(f"[{cfg.label}] Warm start: Z={Z_start:.4f}  P={P_start:.2f}")
    if P_start > 0 and not cfg.use_penalty:
        log("  WARNING: warm start infeasible & penalty mode OFF -> P >0")

    # P=0 best-feasible tracking
    FEAS_TOL: float = 1e-9
    best_Z_global = Z_start if P_start <= FEAS_TOL else float("inf")
    best_Z_iter       = 0 if P_start <= FEAS_TOL else -1
    best_pdf_snapshot = dict(pie_door_factor)
    best_gp_snapshot  = _snapshot_paths(W)

    if P_start <= FEAS_TOL:
        warm_pdf_snap = dict(pie_door_factor)
        warm_gp_snap  = _snapshot_paths(W)
    else:
        warm_pdf_snap = None
        warm_gp_snap  = None

    # Aspiration (penalty mode)
    best_pen_ever = pen_start

    history             = []
    ts_iter             = 0
    no_improve_count    = 0      # cumulative stagnation counter (FIX 4)
    iters_since_perturb = 0      # for triggering perturbation
    perturbation_count  = 0      # report-only
    total_generated     = 0
    total_evaluated     = 0
    infeas_accepted     = 0
    
    curve = [{"evaluation": get_eval_count(), "elapsed_time": 0.0,
              "best_Z": Z_start, "best_P": P_start}]

    def _curve_point():
        curve.append({
            "evaluation": get_eval_count(),
            "elapsed_time": time.perf_counter() - start,
            "best_Z": None if best_Z_global == float("inf") else best_Z_global,
            "best_P": 0.0,
        })  
        
    log(f"\n{'='*60}")
    tag = (f"tenure={cfg.tenure} | "
           f"CLS={'on' if cfg.use_cls else 'off'} | "
           f"Penalty={'on(λ=%.1f)' % LAMBDA if cfg.use_penalty else 'off'}")
    log(f"[{cfg.label}] {tag}")
    log(f"{'='*60}")

    def _limit_reason():
        if (cfg.max_evaluations is not None
                and get_eval_count() >= cfg.max_evaluations):
            return "max_evaluations"
        if (cfg.max_runtime is not None
                and time.perf_counter() - start >= cfg.max_runtime):
            return "max_runtime"
        return None

    while ts_iter < cfg.max_iter:
        limit_reason = _limit_reason()
        if limit_reason is not None:
            termination_reason = limit_reason
            break
        ts_iter += 1
        clean_tabu(ts_iter)

        fp        = compute_flows(W, pie_door_factor)
        sl        = compute_segment_loads(W, fp)
        _, Z0, P0 = compute_objective(W, fp, sl)
        pen_Z0    = Z0 + LAMBDA * P0

        log(f"\n TS {ts_iter:3d}  Z={Z0:.4f}  P={P0:.2f}  "
              f"best={best_Z_global:.4f}  "
              f"no_imp={no_improve_count}/{cfg.no_improve_limit}  "
              f"|tabu|={len(tabu_list)}")

        if (cfg.use_perturbation
                and iters_since_perturb >= cfg.perturbation_trigger):
            # Restore from best feasible snapshot before kicking
            if best_Z_iter >= 0:
                pie_door_factor.clear()
                pie_door_factor.update(best_pdf_snapshot)
                _restore_paths(W, best_gp_snapshot)

                fp        = compute_flows(W, pie_door_factor)
                sl        = compute_segment_loads(W, fp)
                _, Z0, P0 = compute_objective(W, fp, sl)
                pen_Z0    = Z0 + LAMBDA * P0

            _do_perturbation(W, pie_door_factor, fp, pen_Z0,
                             cfg, add_tabu, ts_iter, LAMBDA, enforce_cap, rng)

            perturbation_count  += 1
            # Keep the cumulative stopping counter.  Only the distance from
            # the previous perturbation is restarted; no-improvement history
            # must still reach no_improve_limit and terminate the run.
            iters_since_perturb = 0
            _curve_point()      # Include perturbation flow evaluations on the x-axis.
            limit_reason = _limit_reason()
            if limit_reason is not None:
                termination_reason = limit_reason
                break
            continue

        # ── Generate all candidates ──────────────────────────────────────────
        all_cands = (
            gen_B1_candidates(W, pie_door_factor)
            + gen_B2_candidates(W, pie_door_factor)
            + gen_B3_candidates(W)
            + gen_B4_candidates(W)
            + gen_B5_candidates(W)
            + gen_B6_candidates(W)
        )
        total_generated += len(all_cands)

        # ── Apply CLS filter if enabled ──────────────────────────────────────
        if cfg.use_cls and len(all_cands) > cfg.top_cls:
            # Rank within action types because scores across types differ in scale.
            by_type: dict[str, list] = defaultdict(list)
            for a in all_cands:
                by_type[a[0]].append(a)

            quota = max(1, cfg.top_cls // max(1, len(by_type)))
            high_flow_first = rng.random() < 0.2
            top = []
            for atype in sorted(by_type):
                scored = sorted(by_type[atype],
                                key=lambda a: (_cls_score(a, W, fp, sl, rng,
                                                          high_flow_first),
                                               rng.random()),
                                reverse=True)
                top.extend(scored[:quota])

            top_ids = {id(a) for a in top}
            rest = [a for a in all_cands if id(a) not in top_ids]

            n_rand = max(cfg.random_extra,
                         int(cfg.random_ratio * len(all_cands)))
            n_rand = min(n_rand, len(rest),
                         cfg.cls_pool_max - len(top))
            n_rand = max(0, n_rand)

            rand_extra = rng.sample(rest, n_rand) if n_rand > 0 else []
            eval_pool  = top + rand_extra
            log(f"  N={len(all_cands)} -> CLS {len(eval_pool)} "
                  f"(top{len(top)}{dict(sorted(Counter(a[0] for a in top).items()))}"
                  f"+rand{len(rand_extra)})")
        else:
            eval_pool = all_cands
            log(f"  N={len(all_cands)} (full eval)")

        total_evaluated += len(eval_pool)

        # ── Evaluate pool ────────────────────────────────────────────────────
        best_action = None
        best_score  = float("inf")
        best_true_Z = float("inf")
        best_true_P = float("inf")
        best_pen_Z  = float("inf")
        best_is_asp = False

        limit_reached = None
        for action in eval_pool:
            limit_reached = _limit_reason()
            if limit_reached is not None:
                break

            try:
                ui = apply_hc_action(W, pie_door_factor, action,
                                     enforce_capacity=enforce_cap)
            except Exception:
                continue
            limit_reached = _limit_reason()
            if limit_reached is not None:
                undo_hc_action(W, pie_door_factor, ui)
                break
            fp2       = compute_flows(W, pie_door_factor)
            sl2       = compute_segment_loads(W, fp2)
            _, Z2, P2 = compute_objective(W, fp2, sl2)
            undo_hc_action(W, pie_door_factor, ui)

            if not cfg.use_penalty and P2 > FEAS_TOL:
                continue

            pen_Z2   = Z2 + LAMBDA * P2 if cfg.use_penalty else Z2
            pen_base = pen_Z0 if cfg.use_penalty else Z0
            dZ       = pen_Z2 - pen_base

            tabu_flag = is_tabu(action, ts_iter)
            if cfg.use_penalty:
                aspiration = pen_Z2 < best_pen_ever
            else:
                aspiration = Z2 < best_Z_global - cfg.aspiration_threshold

            if tabu_flag and not aspiration:
                continue

            score = pen_Z2

            if score < best_score:
                best_score  = score
                best_action = action
                best_true_Z = Z2
                best_true_P = P2
                best_pen_Z  = pen_Z2
                best_is_asp = aspiration and tabu_flag

        if limit_reached is not None:
            termination_reason = limit_reached
            log(f"  Stopped: {limit_reached} reached.")
            _curve_point()
            break

        # ── Submit best ──────────────────────────────────────────────────────
        if best_action is not None:
            apply_hc_action(W, pie_door_factor, best_action,
                            enforce_capacity=enforce_cap)
            add_tabu(best_action, ts_iter)

            pen_improved = (cfg.use_penalty
                            and best_pen_Z < best_pen_ever - cfg.min_meaningful_change)

            if cfg.use_penalty and best_pen_Z < best_pen_ever:
                best_pen_ever = best_pen_Z
            if best_true_P > FEAS_TOL:
                infeas_accepted += 1

            dZ_true  = best_true_Z - Z0
            feas_tag = "" if best_true_P <= FEAS_TOL else f" [P={best_true_P:.2f}]"
            asp_tag  = " [Asp]" if best_is_asp else ""
            pen_str  = f"  penZ={best_pen_Z:.4f}" if cfg.use_penalty else ""
            log(f"  OK {best_action[0]}  dZ={dZ_true:+.4f}  Z={best_true_Z:.4f}"
                  f"{pen_str}{feas_tag}{asp_tag}")

            # Best feasible solution update
            new_best_feasible = (
                best_true_P <= FEAS_TOL
                and best_true_Z < best_Z_global - cfg.min_meaningful_change
            )
            if new_best_feasible:
                best_Z_global       = best_true_Z
                best_Z_iter         = ts_iter
                best_pdf_snapshot   = dict(pie_door_factor)
                best_gp_snapshot    = _snapshot_paths(W)
                no_improve_count    = 0
                iters_since_perturb = 0
                log(f"  *** New best feasible Z={best_Z_global:.4f} ***")
            elif pen_improved:
                no_improve_count    = 0
                iters_since_perturb = 0
                log(f"  (penalised improvement penZ={best_pen_Z:.4f}"
                      f" — stagnation clock reset)")
            else:
                no_improve_count    += 1
                iters_since_perturb += 1

            history.append((ts_iter, best_action[0], dZ_true,
                            best_true_Z, best_true_P, best_is_asp,
                            best_pen_Z if cfg.use_penalty else None))
        else:
            no_improve_count    += 1
            iters_since_perturb += 1
            log(f"  No improvement ({no_improve_count}/{cfg.no_improve_limit})")
            
        _curve_point()
        
        if no_improve_count >= cfg.no_improve_limit:
            termination_reason = "own_stagnation"
            log(f"  Terminate: {cfg.no_improve_limit} cumulative "
                  f"no-improve iterations "
                  f"({perturbation_count} perturbations applied)")
            break

    if termination_reason is None:
        termination_reason = "max_iter"

    # ── Restore best feasible ────────────────────────────────────────────────
    # Best snapshots were verified when first encountered. Restore directly,
    # without post-budget flow evaluations that would distort fairness counts.
    if best_Z_iter >= 0:
        log(f"  Restore iter={best_Z_iter} snapshot (Z={best_Z_global:.4f})")
        pie_door_factor.clear()
        pie_door_factor.update(best_pdf_snapshot)
        _restore_paths(W, best_gp_snapshot)
        Z_fin, P_fin = best_Z_global, 0.0
    else:
        log("  WARNING: never found a feasible solution during this run; "
            "returning the last state.")
        Z_fin, P_fin = Z_start, P_start

    # ── Final report ─────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"[{cfg.label}] terminated  iter={ts_iter}")
    log(f"  Z: {Z_start:.4f} -> {Z_fin:.4f}  P={P_fin:.2f}")
    if Z_start > 0:
        log(f"  Improvement: {Z_start-Z_fin:.4f} "
              f"({(Z_start-Z_fin)/Z_start*100:.2f}%)")
    if cfg.use_cls:
        rate = total_evaluated / max(total_generated, 1) * 100
        log(f"  CLS eval rate: {total_evaluated}/{total_generated} "
              f"= {rate:.1f}% (saved {100-rate:.1f}% compute_flows)")
    log(f"  Perturbations: {perturbation_count}  "
          f"Aspirations: {sum(1 for h in history if h[5])}  "
          f"Infeas accepted: {infeas_accepted}")

    nd = {d: v for d, v in pie_door_factor.items() if v < max_pie_index}
    log(f"\nNon-default Discount Doors: {len(nd)}")
    for d, lv in sorted(nd.items()):
        log(f"  {d:40s}: pie[{lv}]={pie[lv]:.4f}")

    cfg.evaluations = get_eval_count()
    cfg.runtime = time.perf_counter() - start
    cfg.termination_reason = termination_reason
    return pie_door_factor, history, curve


# ── Perturbation helper ──────────────────────────────────────────────────────

def _do_perturbation_legacy(W, pie_door_factor, fp, pen_Z0,
                     cfg: TSConfig, add_tabu_fn, ts_iter,
                     LAMBDA, enforce_cap, rng):
    """
    Multi-step random kick across ALL B1-B6 neighborhoods.

    The perturbation is always accepted (true diversification).
    """
    # Snapshot for potential rollback
    pdf_snap = dict(pie_door_factor)
    gp_snap  = _snapshot_paths(W)

    K = max(1, int(cfg.perturb_steps))

    actions_taken = []
    kick_keys = set()
    for _ in range(K):
        fp_current = compute_flows(W, pie_door_factor)
        if (cfg.max_evaluations is not None
                and get_eval_count() >= cfg.max_evaluations):
            break
        if (cfg.max_runtime is not None
                and time.perf_counter() - cfg._start_time >= cfg.max_runtime):
            break
        pool = (
            gen_B1_candidates(W, pie_door_factor)
            + gen_B2_candidates(W, pie_door_factor)
            + gen_B3_candidates(W)
            + gen_B4_candidates(W)
            + gen_B5_candidates(W)
            + gen_B6_candidates(W)
        )
        # Avoid repeating a tabu family within one perturbation.
        pool = [
            a for a in pool
            if _make_tabu_key(a) not in kick_keys
        ]

        if not pool:
            break

        if rng.random() < cfg.perturb_heavy_bias:
            heavy = [a for a in pool if a[0] in ("B4", "B6")]
            if heavy:
                heavy_sorted = sorted(heavy,
                                      key=lambda a: _get_path_flow(W, fp_current, a),
                                      reverse=True)
                pick = rng.choice(heavy_sorted[:max(1, cfg.perturb_top_k)])
            else:
                pick = rng.choice(pool)
        else:
            pick = rng.choice(pool)

        try:
            apply_hc_action(
                W,
                pie_door_factor,
                pick,
                enforce_capacity=enforce_cap
            )
            actions_taken.append(pick)
            kick_keys.add(_make_tabu_key(pick))
        except Exception:
            continue


    if not actions_taken:
        if cfg.verbose:
            print(f"  [Perturbation cancelled]  No action could be applied")
        return

    # Evaluate post-kick state
    fp_t = compute_flows(W, pie_door_factor)
    sl_t = compute_segment_loads(W, fp_t)
    _, Zt, Pt = compute_objective(W, fp_t, sl_t)

    # Accepted — tabu all actions in the kick
    for a in actions_taken:
        add_tabu_fn(a, ts_iter)
    types = "+".join(a[0] for a in actions_taken)
    if cfg.verbose:
        print(f"  [Perturbation] {len(actions_taken)}-step kick [{types}]  "
              f"Z'={Zt:.4f} P'={Pt:.2f}")


# Active perturbation implementation: multi-step random kick over B1-B6.
def _do_perturbation(W, pie_door_factor, fp, pen_Z0,
                     cfg: TSConfig, add_tabu_fn, ts_iter,
                     LAMBDA, enforce_cap, rng):
    pdf_snap = dict(pie_door_factor)
    gp_snap = _snapshot_paths(W)
    k = rng.randint(cfg.perturb_steps_min, cfg.perturb_steps_max)
    actions_taken = []
    kick_keys = set()

    for _ in range(k):
        if (cfg.max_evaluations is not None
                and get_eval_count() >= cfg.max_evaluations):
            break
        if (cfg.max_runtime is not None
                and time.perf_counter() - cfg._start_time >= cfg.max_runtime):
            break
        fp_current = compute_flows(W, pie_door_factor)
        if (cfg.max_evaluations is not None
                and get_eval_count() >= cfg.max_evaluations):
            break
        if (cfg.max_runtime is not None
                and time.perf_counter() - cfg._start_time >= cfg.max_runtime):
            break
        pool = (
            gen_B1_candidates(W, pie_door_factor)
            + gen_B2_candidates(W, pie_door_factor)
            + gen_B3_candidates(W)
            + gen_B4_candidates(W)
            + gen_B5_candidates(W)
            + gen_B6_candidates(W)
        )
        pool = [a for a in pool if _make_tabu_key(a) not in kick_keys]
        if not pool:
            break

        if rng.random() < cfg.perturb_heavy_bias:
            heavy = [a for a in pool if a[0] in ("B4", "B6")]
            if heavy:
                scored = sorted(
                    heavy,
                    key=lambda a: _get_path_flow(W, fp_current, a),
                    reverse=True,
                )
                pick = rng.choice(scored[:max(1, cfg.perturb_top_k)])
            else:
                pick = rng.choice(pool)
        else:
            pick = rng.choice(pool)

        try:
            apply_hc_action(W, pie_door_factor, pick,
                            enforce_capacity=enforce_cap)
            actions_taken.append(pick)
            kick_keys.add(_make_tabu_key(pick))
        except Exception:
            continue

    if not actions_taken:
        if cfg.verbose:
            print("  [Perturbation cancelled] No action could be applied")
        return False

    if (cfg.max_evaluations is not None
            and get_eval_count() >= cfg.max_evaluations):
        pie_door_factor.clear()
        pie_door_factor.update(pdf_snap)
        _restore_paths(W, gp_snap)
        if cfg.verbose:
            print(f"  [Perturbation cancelled] evaluation budget reached "
                  f"({cfg.max_evaluations})")
        return False
    if (cfg.max_runtime is not None
            and time.perf_counter() - cfg._start_time >= cfg.max_runtime):
        pie_door_factor.clear()
        pie_door_factor.update(pdf_snap)
        _restore_paths(W, gp_snap)
        if cfg.verbose:
            print("  [Perturbation cancelled] runtime budget reached")
        return False

    fp_t = compute_flows(W, pie_door_factor)
    sl_t = compute_segment_loads(W, fp_t)
    _, z_t, p_t = compute_objective(W, fp_t, sl_t)
    for action in actions_taken:
        add_tabu_fn(action, ts_iter)
    types = "+".join(a[0] for a in actions_taken)
    if cfg.verbose:
        print(f"  [Perturbation] {len(actions_taken)}-step kick [{types}] "
              f"Z'={z_t:.4f} P'={p_t:.2f}")
    return True
