"""GA using the Markdown representation [discount genes | waiting-offer genes].

Before evaluation, the actions required to move from the GS warm start to those target
states are applied to a fresh copy of the GS state, so the available path set
is derived from G.
"""
from __future__ import annotations

import copy
import random
import time
from collections import defaultdict

from core.config import max_pie_index
from core.flow import compute_flows, compute_segment_loads, get_eval_count
from core.objective import compute_objective
from core.actions import (apply_hc_action, gen_B3_candidates,gen_B4_candidates, gen_B6_candidates,
                           gen_B5_candidates)
from algorithms._ts_common import _snapshot_paths, _restore_paths

STAGNATION_EVAL_LIMIT = 2000


def _copy_pdf(pdf):
    out = defaultdict(lambda: max_pie_index)
    out.update(pdf)
    return out


def _doors(W):
    result = []
    for u in W["grouped_paths"]:
        for d in W["grouped_paths"][u]:
            for v in W["grouped_paths"][u][d]:
                if v not in result:
                    result.append(v)
    return result


def _switch_universe(W):
    """Build the universe of waiting-offer switches.

    Each G gene stores the target open state for a (station, time, line, w)
    switch: 1 means open, 0 means closed. B3/B5 provide opening actions for
    closed switches; B4/B6 provide closing actions for open switches.
    Both may exist for a partially open switch.

    Return (kind, key, open_action, close_action, init_bit) tuples in a
    fixed order for deterministic decoding.
    """
    table = {}      # (kind, key) -> [open_action, close_action]

    for a in gen_B3_candidates(W):
        table.setdefault(("origin", a[1]), [None, None])[0] = a
    for a in gen_B4_candidates(W):
        table.setdefault(("origin", a[1]), [None, None])[1] = a
    for a in gen_B5_candidates(W):
        table.setdefault(("xfer", a[1]), [None, None])[0] = a
    for a in gen_B6_candidates(W):
        table.setdefault(("xfer", a[1]), [None, None])[1] = a

    switches = []
    for (kind, key) in sorted(table, key=repr):     # sort by string repr to get a fixed order
        open_a, close_a = table[(kind, key)]
        init = 1 if close_a is not None else 0      # currently open if a closing action exists, else closed
        switches.append((kind, key, open_a, close_a, init))
    return switches


def _evaluate(chrom, base_pdf, base_paths, W, doors, offers,
              max_evaluations=None, max_runtime=None, start=None):
    pdf = _copy_pdf(base_pdf)
    _restore_paths(W, base_paths)
    for d, level in zip(doors, chrom["D"]):
        pdf[d] = max(0, min(max_pie_index, int(level)))
    for bit, (_kind, _key, open_action, close_action, init) in zip(
            chrom["G"], offers):
        if (max_evaluations is not None
                and get_eval_count() >= max_evaluations):
            return None
        if (max_runtime is not None and start is not None
                and time.perf_counter() - start >= max_runtime):
            return None
        # G stores the target state, rather than an instruction to open an
        # offer.  Therefore a gene equal to its GS-state value needs no move;
        # otherwise decode it to the genuine B3/B4/B5/B6 action recorded for
        # this switch.  Passing the switch descriptor itself would make
        # apply_hc_action see "origin" or "xfer" as the action type.
        bit = int(bool(bit))
        if bit == init:
            continue

        action = open_action if bit else close_action
        if action is None:
            target = "open" if bit else "closed"
            raise RuntimeError(
                f"Cannot set waiting-offer switch {_kind!r}, {_key!r} "
                f"to {target}: its corresponding HC action is unavailable."
            )
        apply_hc_action(W, pdf, action, enforce_capacity=False)
        if (max_evaluations is not None
                and get_eval_count() >= max_evaluations):
            return None
        if (max_runtime is not None and start is not None
                and time.perf_counter() - start >= max_runtime):
            return None
    if (max_evaluations is not None
            and get_eval_count() >= max_evaluations):
        return None
    if (max_runtime is not None and start is not None
            and time.perf_counter() - start >= max_runtime):
        return None
    fp = compute_flows(W, pdf)
    sl = compute_segment_loads(W, fp)
    F, Z, P = compute_objective(W, fp, sl)
    return F, Z, P, pdf


def _clone(c):
    return {"D": list(c["D"]), "G": list(c["G"])}


def _fitness_key(row):
    """Sort feasible solutions first, then compare their penalized F.

    Rows have the form ``(F, Z, P, chromosome, pdf)``.  Keeping this rule in
    one place ensures elite preservation and tournament selection agree about
    feasibility.
    """
    return (row[2] > 1e-9, row[0])


def _mutate(c, rng, p_m):
    changed = 0
    for i in range(len(c["D"])):
        if rng.random() < p_m:
            c["D"][i] = max(0, min(max_pie_index,
                                      c["D"][i] + rng.choice([-1, 1])))
            changed += 1
    for i in range(len(c["G"])):
        if rng.random() < p_m:
            c["G"][i] = 1 - c["G"][i]
            changed += 1
    return changed


def _crossover(a, b, rng, p_c):
    x = _clone(a)
    if rng.random() > p_c:
        return x, False
    for i in range(len(x["D"])):
        x["D"][i] = a["D"][i] if rng.random() < 0.5 else b["D"][i]
    for i in range(len(x["G"])):
        x["G"][i] = a["G"][i] if rng.random() < 0.5 else b["G"][i]
    return x, True


def run_genetic_algorithm(W, pie_door_factor, max_evaluations=3000,
                          time_budget=None, seed=124, population_size=15,
                          tournament_size=4, p_c=0.8, p_m=0.01,
                          elite_size=4, verbose=False, max_runtime=None,
                          max_generations=100000):
    start = time.perf_counter()
    runtime_limit = max_runtime if max_runtime is not None else time_budget
    rng = random.Random(seed)
    base_pdf = _copy_pdf(pie_door_factor)
    base_paths = _snapshot_paths(W)
    doors = _doors(W)
    switches = _switch_universe(W)
    init_bits = [s[4] for s in switches]        # 30 * 1 + 40 * 0
    seed_chrom = {"D": [base_pdf[d] for d in doors],
                  "G": list(init_bits)}          # GS warm start

    def fresh():
        c = _clone(seed_chrom)
        for i in range(len(c["D"])):
            if rng.random() < 0.25:
                c["D"][i] = max(0, min(max_pie_index,
                                       c["D"][i] + rng.choice([-1, 1])))
        for i in range(len(c["G"])):
            if rng.random() < 0.10:
                c["G"][i] = 1 - c["G"][i]        # Flip the bit instead of setting it to 1.
        return c

    pop = [seed_chrom] + [fresh() for _ in range(max(1, population_size - 1))]
    best = None
    best_eval = None
    best_pdf = None
    best_paths = None
    curve = []
    evaluations = 0
    stagnant_evaluations = 0
    last_best_z = None
    generation = 0
    termination_reason = None

    while generation < max_generations:
        if max_evaluations is not None and evaluations >= max_evaluations:
            termination_reason = "max_evaluations"
            break
        if (runtime_limit is not None
                and time.perf_counter() - start >= runtime_limit):
            termination_reason = "max_runtime"
            break
        generation += 1
        scored = []
        for c in pop:
            if (max_evaluations is not None
                    and evaluations >= max_evaluations):
                break
            if (runtime_limit is not None
                    and time.perf_counter() - start >= runtime_limit):
                break
            evaluated = _evaluate(
                c, base_pdf, base_paths, W, doors, switches,
                max_evaluations=max_evaluations,
                max_runtime=runtime_limit, start=start)
            if evaluated is None:
                termination_reason = (
                    "max_evaluations"
                    if (max_evaluations is not None
                        and get_eval_count() >= max_evaluations)
                    else "max_runtime")
                break
            F, Z, P, pdf = evaluated
            evaluations += 1
            scored.append((F, Z, P, _clone(c), pdf))
            if P <= 1e-9 and (best_eval is None or Z < best_eval[1]):
                best = _clone(c)
                best_eval = (F, Z, P)
                best_pdf = _copy_pdf(pdf)
                best_paths = _snapshot_paths(W)
            # Use the global counter on the x-axis; GA's local count tracks
            # individuals and is not directly comparable with other algorithms.
            curve.append({"evaluation": get_eval_count(),
                          "elapsed_time": time.perf_counter() - start,
                          "best_Z": None if best_eval is None else best_eval[1],
                          "best_F": None if best_eval is None else best_eval[0],
                          "best_P": None if best_eval is None else best_eval[2]})
        if not scored:
            if (max_evaluations is not None
                    and evaluations >= max_evaluations):
                termination_reason = "max_evaluations"
            elif (runtime_limit is not None
                    and time.perf_counter() - start >= runtime_limit):
                termination_reason = "max_runtime"
            break
        scored.sort(key=_fitness_key)
        if termination_reason is not None:
            break
        current_feasible = next((x[1] for x in scored if x[2] <= 1e-9), None)
        generation_best = (scored[0][1] if scored[0][2] <= 1e-9
                           else None)
        if current_feasible is not None and (last_best_z is None or current_feasible < last_best_z - 1e-4):
            last_best_z = current_feasible
            stagnant_evaluations = 0
        else:
            stagnant_evaluations += len(scored)
        if stagnant_evaluations >= STAGNATION_EVAL_LIMIT:
            termination_reason = "own_stagnation"
            break
        next_pop = [row[3] for row in scored[:max(1, elite_size)]]
        crossover_count = 0
        mutation_count = 0
        while len(next_pop) < population_size:
            k = min(tournament_size, len(scored))
            pa = min(rng.sample(scored, k), key=_fitness_key)[3]
            pb = min(rng.sample(scored, k), key=_fitness_key)[3]
            child, did_crossover = _crossover(pa, pb, rng, p_c)
            crossover_count += int(did_crossover)
            mutation_count += _mutate(child, rng, p_m)
            next_pop.append(child)
        if verbose:
            best_text = "n/a" if best_eval is None else f"{best_eval[1]:.4f}"
            gen_text = "n/a" if generation_best is None else f"{generation_best:.4f}"
            feasible_count = sum(1 for row in scored if row[2] <= 1e-9)
            print(f"  GA generation={generation:4d} evaluations={evaluations:5d} "
                  f"feasible={feasible_count}/{len(scored)} "
                  f"gen_best={gen_text} global_best={best_text} "
                  f"elite={min(elite_size, len(scored))} "
                  f"crossovers={crossover_count} mutations={mutation_count} "
                  f"stagnant_evaluations={stagnant_evaluations}")
        pop = next_pop

    if termination_reason is None:
        termination_reason = "max_iter"

    # Restore the recorded best without another flow evaluation. This keeps
    # the reported search-evaluation count exact and within the common budget.
    if best_pdf is None:
        pdf = base_pdf
        _restore_paths(W, base_paths)
    else:
        pdf = best_pdf
        _restore_paths(W, best_paths)
    pie_door_factor.clear()
    pie_door_factor.update(pdf)
    if not curve or curve[-1]["evaluation"] != get_eval_count():
        curve.append({
            "evaluation": get_eval_count(),
            "elapsed_time": time.perf_counter() - start,
            "best_Z": None if best_eval is None else best_eval[1],
            "best_F": None if best_eval is None else best_eval[0],
            "best_P": None if best_eval is None else best_eval[2],
        })
    return pie_door_factor, curve, {"evaluations": get_eval_count(),
                                    "runtime": time.perf_counter() - start,
                                    "termination_reason": termination_reason,
                                    "generations": generation,
                                    "switches": len(switches),
                                    "doors": len(doors)}
