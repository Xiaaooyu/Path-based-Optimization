"""Single runner for GS, HC, TS, SA and GA.

Examples
--------
python -m experiments.runner --algorithm gs
python -m experiments.runner --algorithm hc
python -m experiments.runner --algorithm ts --ts-preset full
python -m experiments.runner --algorithm sa --sa-preset base
python -m experiments.runner --algorithm ga
python -m experiments.runner --algorithm all --condition evaluations
python -m experiments.runner --algorithm all --condition runtime --max-runtime 60
python -m experiments.runner --algorithm sa --condition own_stop
python -m experiments.runner --algorithm ga --condition runtime --max-runtime 60

"""
from __future__ import annotations

import argparse
import ast
import json
import time
from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from core.config import max_pie_index
from core.flow import (compute_flows, compute_segment_loads, get_eval_count,
                       reset_eval_count)
from core.objective import compute_Z_fixed, compute_objective
from core.verify import verify_demand_satisfied
import core.world as _core_world
from core.world import reseed_world, build_world, schedules, _dist_cache
from algorithms.greedy import run_greedy
import algorithms.hill_climbing as hc_module
from algorithms.hill_climbing import run_hill_climbing
from algorithms.tabu_search_unified import (PRESETS as TS_PRESETS,
                                            run_tabu_search)
import algorithms.simulated_annealing as sa_basic
from algorithms.simulated_annealing import run_simulated_annealing_basic
from algorithms.genetic_algorithm import run_genetic_algorithm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "runner"
RESULTS.mkdir(parents=True, exist_ok=True)
IMPROVEMENT_EPS = 1e-4

# Frozen confirmatory configurations from
# results/ofat/raw/metaheuristic_comparison/Z/final_recommendations.csv.
# Keeping them here makes comparison runs reproducible and independent of a
# mutable analysis-output file.
PRIMARY_TS_PRESETS = (
    "perturbation",
    "perturbation_cls",
    "perturbation_penalty",
    "full",
)
PRIMARY_SA_PRESETS = ("base", "penalty")

TS_PRESET_ALIASES = {
    "base": "perturbation",
    "cls": "perturbation_cls",
    "penalty": "perturbation_penalty",
    "cls_penalty": "full",
}

TUNED_TS_PARAMETERS = {
    "perturbation": {
        "tenure": 4,
        "perturbation_trigger": 15,
        "perturb_steps": 5,
        "perturb_heavy_bias": 0.30,
        "perturb_top_k": 3,
    },
    "perturbation_cls": {
        "tenure": 8,
        "perturbation_trigger": 15,
        "perturb_steps": 5,
        "perturb_heavy_bias": 0.30,
        "perturb_top_k": 3,
        "top_cls": 24,
        "random_extra": 12,
        "random_ratio": 0.15,
        "cls_pool_max": 64,
    },
    "perturbation_penalty": {
        "tenure": 8,
        "perturbation_trigger": 15,
        "perturb_steps": 5,
        "perturb_heavy_bias": 0.0,
        "perturb_top_k": 3,
        "ts_lambda": 5.0,
    },
    "full": {
        "tenure": 8,
        "perturbation_trigger": 25,
        "perturb_steps": 5,
        "perturb_heavy_bias": 0.3,
        "perturb_top_k": 3,
        "top_cls": 24,
        "random_extra": 6,
        "random_ratio": 0.05,
        "cls_pool_max": 64,
        "ts_lambda": 5.0,
    },
}

TUNED_SA_PARAMETERS = {
    "base": {
        "T0": 0.1,
        "alpha": 0.95,
        "moves_per_temperature": 20,
        "sa_lambda": 16.0,
    },
    "penalty": {
        "T0": 0.5,
        "alpha": 0.95,
        "moves_per_temperature": 20,
        "sa_lambda": 16.0,
    },
}

TUNED_GA_PARAMETERS = {
    "population_size": 15,
    "tournament_size": 4,
    "p_c": 0.8,
    "p_m": 0.01,
    "elite_size": 4,
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_state(pie_door_factor, runs_added, histories, W, save_path,
               demand_seed: int | None = None) -> None:
    """Save a warm-start state without depending on another experiment module."""
    fp = compute_flows(W, pie_door_factor)

    all_paths = []
    for u in W["grouped_paths"]:
        for j in W["grouped_paths"][u]:
            for v in W["grouped_paths"][u][j]:
                for tr in W["grouped_paths"][u][j][v]:
                    for p in W["grouped_paths"][u][j][v][tr]:
                        e = fp.get((u, j, tuple(p)))
                        all_paths.append({
                            "u": u, "j": j, "v": v,
                            "tr": list(tr) if tr else None,
                            "path": list(p),
                            "flow": e["flow"] if e else 0.0,
                        })

    data = {
        "schedules": {
            ln: [{"train": r["train"], "route": r["route"]}
                 for r in runs]
            for ln, runs in schedules.items()
        },
        "pie_door_factor": {str(k): v for k, v in pie_door_factor.items()},
        "_dist_seed": _core_world._dist_seed,
        "dist_cache": {str(k): v for k, v in _dist_cache.items()},
        "runs_added": runs_added or [],
        "histories": (histories if isinstance(histories, dict)
                      else {"hc": histories, "ts": [], "sa": []}),
        "all_paths": all_paths,
        "od_demand": {f"{o}|{d}": v
                       for (o, d), v in _core_world.OD.items()},
    }
    if demand_seed is not None:
        data["demand_seed"] = int(demand_seed)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NpEncoder)

    sl = compute_segment_loads(W, fp)
    _, Z, P = compute_objective(W, fp, sl)
    print(f"  Saved -> {save_path}  (Z={Z:.4f}  P={P:.2f}  paths={len(all_paths)})")


def load_state(path: str, expected_demand_seed: int | None = None,
               expected_dist_seed: int | None = None):
    """Load a warm-start state saved by this runner or legacy runners."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    stored_demand_seed = data.get("demand_seed")
    stored_dist_seed = data.get("_dist_seed", data.get("distance_seed"))
    if (expected_demand_seed is not None and stored_demand_seed is not None
            and int(stored_demand_seed) != int(expected_demand_seed)):
        raise ValueError(
            f"Warm start {path} has demand_seed={stored_demand_seed}, "
            f"expected {expected_demand_seed}.")
    if (expected_dist_seed is not None and stored_dist_seed is not None
            and int(stored_dist_seed) != int(expected_dist_seed)):
        raise ValueError(
            f"Warm start {path} has distance_seed={stored_dist_seed}, "
            f"expected {expected_dist_seed}.")

    if "_dist_seed" in data:
        _core_world._dist_seed = data["_dist_seed"]
    elif data.get("distance_seed") is not None:
        _core_world._dist_seed = data["distance_seed"]

    if "od_demand" in data:
        _core_world.OD.clear()
        for key, value in data["od_demand"].items():
            origin, destination = key.split("|")
            _core_world.OD[(origin, destination)] = value
    elif data.get("demand_seed") is not None:
        reseed_world(od_seed=int(data["demand_seed"]), reset_schedules=False)
    else:
        print(f"  WARNING: {path} has neither od_demand nor demand_seed; "
              "OD left unchanged (caller must have set it).")

    schedules.clear()
    schedules.update({
        ln: [
            {"train": r["train"],
             "route": [tuple(stop) for stop in r["route"]]}
            for r in runs
        ]
        for ln, runs in data["schedules"].items()
    })

    _dist_cache.clear()
    for key, value in data.get("dist_cache", {}).items():
        if isinstance(key, str) and key.startswith("("):
            key = ast.literal_eval(key)
        _dist_cache[key] = value

    runs_added = data.get("runs_added", [])
    raw_histories = data.get("histories")
    if isinstance(raw_histories, dict):
        histories = {"hc": [], "ts": [], "sa": [], **raw_histories}
    elif isinstance(raw_histories, list):
        histories = {"hc": raw_histories, "ts": [], "sa": []}
    else:
        histories = {"hc": data.get("hc_history", []), "ts": [], "sa": []}

    W = build_world()
    grouped_paths = W["grouped_paths"]
    if "all_paths" in data:
        for u in list(grouped_paths):
            for j in list(grouped_paths[u]):
                for v in list(grouped_paths[u][j]):
                    for tr in list(grouped_paths[u][j][v]):
                        grouped_paths[u][j][v][tr] = []
        for entry in data["all_paths"]:
            u, j, v = entry["u"], entry["j"], entry["v"]
            tr = tuple(entry["tr"]) if entry["tr"] else None
            grouped_paths[u][j][v][tr].append(entry["path"])
        print(f"  Loaded {path}  (paths={len(data['all_paths'])})")
    else:
        for entry in data.get("extra_paths", []):
            u, j, v = entry["u"], entry["j"], entry["v"]
            tr = tuple(entry["tr"]) if entry["tr"] else None
            path_value = entry["path"]
            if not any(tuple(q) == tuple(path_value)
                       for q in grouped_paths[u][j][v][tr]):
                grouped_paths[u][j][v][tr].append(path_value)
        print(f"  Loaded {path}  (extra_paths={len(data.get('extra_paths', []))})")

    pdf = defaultdict(lambda: max_pie_index)
    pdf.update(data["pie_door_factor"])
    return pdf, W, runs_added, histories


def _metrics(W, pdf):
    fp = compute_flows(W, pdf)
    sl = compute_segment_loads(W, fp)
    F, Z, P = compute_objective(W, fp, sl)
    ok, report = verify_demand_satisfied(W, fp)
    z_fixed = compute_Z_fixed(W)
    return {"Z": float(Z), "P": float(P), "F": float(F),
            "Z_fixed": float(z_fixed),
            "Z_var": float(Z - z_fixed),          # variable component of Z
            "feasible": bool(P <= 1e-9 and ok)}


def _tuned_ts_config(preset, algo_seed, max_iter,
                     evaluation_limit, runtime_limit):
    """Return one primary/compatibility TS preset with tuned parameters."""
    canonical = TS_PRESET_ALIASES.get(preset, preset)
    if canonical not in TUNED_TS_PARAMETERS:
        raise ValueError(f"No tuned TS parameters declared for {preset!r}")
    cfg = deepcopy(TS_PRESETS[preset])
    tuned = TUNED_TS_PARAMETERS[canonical]
    for key, value in tuned.items():
        if key == "perturb_steps":
            # The active implementation samples between min/max.  Keep the
            # compatibility scalar synchronized too.
            cfg.perturb_steps = int(value)
            cfg.perturb_steps_min = int(value)
            cfg.perturb_steps_max = int(value)
        else:
            setattr(cfg, key, value)

    # __post_init__ ran before the tuned trigger was applied.
    cfg.no_improve_limit = (
        cfg.perturbation_trigger * cfg.no_improve_ratio
    )
    cfg.seed = algo_seed
    cfg.max_iter = max_iter
    cfg.max_evaluations = evaluation_limit
    cfg.max_runtime = runtime_limit
    cfg.verbose = False
    return cfg, canonical


def _resolved_sa_config(preset, overrides):
    cfg = deepcopy(sa_basic.PRESETS[preset])
    for key, value in TUNED_SA_PARAMETERS[preset].items():
        setattr(cfg, key, value)
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _resolved_ga_config(overrides):
    cfg = dict(TUNED_GA_PARAMETERS)
    cfg.update({key: value for key, value in overrides.items()
                if value is not None})
    return cfg


def _run_one(algorithm, gs_path, demand_seed, dist_seed, algo_seed,
             condition, max_iter, max_evaluations, max_runtime, ts_preset,
             sa_preset, sa_overrides, ga_overrides):
    pdf, W, runs_added, _ = load_state(
        str(gs_path), expected_demand_seed=demand_seed,
        expected_dist_seed=dist_seed)
    baseline = _metrics(W, pdf)          # Reporting evaluation, not search.
    z_gs = baseline["Z"]
    z_var_gs = baseline["Z_var"]
    n_runs_added = len(runs_added or [])

    reset_eval_count()
    start = time.perf_counter()
    evaluation_limit = (max_evaluations
                        if condition == "evaluations" else None)
    runtime_limit = max_runtime if condition == "runtime" else None
    algorithm_evaluations = 0
    algorithm_runtime = 0.0
    termination_reason = "baseline"
    algorithm_stats = {}
    algorithm_parameters = {}
    if algorithm == "gs":
        curve = [{"evaluation": 0, "elapsed_time": 0.0, "best_Z": z_gs,
                  "best_F": baseline["F"], "best_P": baseline["P"]}]
        result = baseline
    elif algorithm == "hc":
        pdf, history, curve = run_hill_climbing(
            W, pdf, max_evaluations=evaluation_limit,
            max_runtime=runtime_limit, max_iter=max_iter)
        algorithm_evaluations = get_eval_count()
        algorithm_runtime = time.perf_counter() - start
        termination_reason = hc_module.LAST_STATS.get(
            "termination_reason", "unknown")
        result = _metrics(W, pdf)
    elif algorithm == "ts":
        cfg, canonical_preset = _tuned_ts_config(
            ts_preset, algo_seed, max_iter,
            evaluation_limit, runtime_limit)
        pdf, history, curve = run_tabu_search(W, pdf, cfg)
        algorithm_evaluations = get_eval_count()
        algorithm_runtime = time.perf_counter() - start
        termination_reason = getattr(cfg, "termination_reason", "unknown")
        result = _metrics(W, pdf)
        algorithm_parameters = {
            **TUNED_TS_PARAMETERS[canonical_preset],
            "use_perturbation": cfg.use_perturbation,
            "use_cls": cfg.use_cls,
            "use_penalty": cfg.use_penalty,
            "no_improve_limit": cfg.no_improve_limit,
        }
    elif algorithm == "sa":
        sa_cfg = _resolved_sa_config(sa_preset, sa_overrides)
        pdf, history, _z_best, curve = run_simulated_annealing_basic(
            W, pdf, max_iter=max_iter, max_evaluations=evaluation_limit,
            max_runtime=runtime_limit,
            T0=sa_cfg.T0, alpha=sa_cfg.alpha, t_min=sa_cfg.t_min,
            moves_per_temperature=sa_cfg.moves_per_temperature,
            seed=algo_seed, use_penalty=sa_cfg.use_penalty,
            sa_lambda=sa_cfg.sa_lambda,
            verbose=False)
        algorithm_evaluations = get_eval_count()
        algorithm_runtime = time.perf_counter() - start
        termination_reason = sa_basic.LAST_STATS.get(
            "termination_reason", "unknown")
        result = _metrics(W, pdf)
        result["T0"] = sa_cfg.T0
        algorithm_parameters = {
            "T0": sa_cfg.T0,
            "alpha": sa_cfg.alpha,
            "t_min": sa_cfg.t_min,
            "moves_per_temperature": sa_cfg.moves_per_temperature,
            "use_penalty": sa_cfg.use_penalty,
            "sa_lambda": sa_cfg.sa_lambda,
        }
    elif algorithm == "ga":
        ga_cfg = _resolved_ga_config(ga_overrides)
        pdf, curve, algorithm_stats = run_genetic_algorithm(
            W, pdf, max_evaluations=evaluation_limit,
            max_runtime=runtime_limit, max_generations=max_iter,
            seed=algo_seed, verbose=False, **ga_cfg)
        algorithm_evaluations = get_eval_count()
        algorithm_runtime = time.perf_counter() - start
        termination_reason = algorithm_stats.get(
            "termination_reason", "unknown")
        result = _metrics(W, pdf)
        algorithm_parameters = ga_cfg
    else:
        raise ValueError(algorithm)

    if (not curve or int(curve[0].get("evaluation", 0)) > 0
            or curve[0].get("best_Z") is None):
        curve.insert(0, {
            "evaluation": 0, "elapsed_time": 0.0,
            "best_Z": z_gs, "best_F": baseline["F"],
            "best_P": baseline["P"],
        })

    # Z_fixed is determined by the timetable, so derive each curve point's
    # variable component for the Z_var convergence series.
    for row in curve:
        row["best_Z_var"] = (
            None if row["best_Z"] is None
            else row["best_Z"] - baseline["Z_fixed"]
        )

    # These values were captured immediately after the algorithm returned;
    # the result-verification call above is therefore excluded.
    runtime = algorithm_runtime
    evaluations = int(algorithm_evaluations)
    eligible = bool(baseline["feasible"] and result["feasible"])
    improvement_pct = (
        (z_gs - result["Z"]) / z_gs * 100.0
        if eligible and abs(z_gs) > 1e-12 else None)
    improvement_pct_var = (
        (z_var_gs - result["Z_var"]) / z_var_gs * 100.0
        if eligible and abs(z_var_gs) > 1e-12 else None)

    result.update({
        "instance_id": f"d{demand_seed}_x{dist_seed}",
        "model_seed": demand_seed,
        "demand_seed": demand_seed,
        "distance_seed": dist_seed,
        "algorithm": (
            f"TS-{ts_preset.upper()}" if algorithm == "ts" else
            f"SA-{sa_preset.upper()}" if algorithm == "sa" else
            algorithm.upper()
        ),
        "algorithm_family": algorithm.upper(),
        "preset": ts_preset if algorithm == "ts" else (
            sa_preset if algorithm == "sa" else None
        ),
        "algo_seed": algo_seed,
        "condition": condition,
        "eligible_for_comparison": eligible,
        "n_runs_added": int(n_runs_added),
        "evaluations": int(evaluations),
        "runtime": float(runtime),
        "runtime_limit": runtime_limit,
        "evaluation_limit": evaluation_limit,
        "termination_reason": termination_reason,
        "algorithm_parameters": algorithm_parameters,
        "improvement_pct": improvement_pct,
        "improvement_pct_var": improvement_pct_var,
        "best_so_far_curve": curve,
        "gs_Z_baseline": z_gs,
        "gs_Z_var_baseline": z_var_gs,
    })
    for key in ("generations", "switches", "doors"):
        if key in algorithm_stats:
            result[key] = algorithm_stats[key]
    return result


def _summary_stats(values, maximize=False):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"best": None, "median": None, "mean": None, "std": None}
    return {
        "best": (max(values) if maximize else min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _aggregate_seeds(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["instance_id"], row["algorithm"],
                row["condition"])].append(row)

    summaries = []
    for (instance_id, algorithm, condition), group in sorted(groups.items()):
        eligible = [r for r in group if r["eligible_for_comparison"]]
        summary = {
            "instance_id": instance_id,
            "demand_seed": group[0]["demand_seed"],
            "distance_seed": group[0]["distance_seed"],
            "algorithm": algorithm,
            "condition": condition,
            "total_runs": len(group),
            "feasible_runs": sum(bool(r["feasible"]) for r in group),
            "eligible_runs": len(eligible),
            "feasibility_rate": (sum(bool(r["feasible"]) for r in group)
                                 / len(group)),
            "algo_seeds": [r["algo_seed"] for r in group
                           if r["algo_seed"] is not None],
            "termination_reasons": dict(Counter(
                r["termination_reason"] for r in group)),
        }
        summary["improve_rate"] = (
            sum(r["Z"] < r["gs_Z_baseline"] - IMPROVEMENT_EPS
                for r in eligible) / len(eligible)
            if eligible else None)
        for metric in ("Z", "Z_var", "improvement_pct",
                       "improvement_pct_var", "runtime", "evaluations"):
            stats = _summary_stats(
                [r.get(metric) for r in eligible],
                maximize=metric.startswith("improvement_pct"))
            summary.update({f"{metric}_{name}": value
                            for name, value in stats.items()})
        summaries.append(summary)
    return summaries


def _aggregate_instances(seed_summaries):
    groups = defaultdict(list)
    for row in seed_summaries:
        groups[(row["algorithm"], row["condition"])].append(row)

    summaries = []
    for (algorithm, condition), group in sorted(groups.items()):
        summary = {
            "algorithm": algorithm,
            "condition": condition,
            "instances": len(group),
            "total_runs": sum(r["total_runs"] for r in group),
            "feasible_runs": sum(r["feasible_runs"] for r in group),
        }
        summary["feasibility_rate"] = (
            summary["feasible_runs"] / summary["total_runs"]
            if summary["total_runs"] else None)
        # Each instance contributes one mean relative improvement, preventing
        # large raw objectives or extra stochastic repetitions from receiving
        # disproportionate weight in the second-level comparison.
        for metric in ("improvement_pct", "improvement_pct_var"):
            stats = _summary_stats(
                [r.get(f"{metric}_mean") for r in group], maximize=True)
            summary.update({f"within_instance_{metric}_{name}": value
                            for name, value in stats.items()})
        summaries.append(summary)
    return summaries


def _locf(curve, evaluation, key):
    value = None
    for point in curve:
        if int(point["evaluation"]) > evaluation:
            break
        if point.get(key) is not None:
            value = point[key]
    return value


def _aggregate_convergence(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["eligible_for_comparison"]:
            groups[(row["instance_id"], row["algorithm"],
                    row["condition"])].append(row)

    output = []
    for (instance_id, algorithm, condition), group in sorted(groups.items()):
        grid = sorted({int(point["evaluation"])
                       for row in group
                       for point in row["best_so_far_curve"]})
        for evaluation in grid:
            record = {"instance_id": instance_id, "algorithm": algorithm,
                      "condition": condition, "evaluation": evaluation,
                      "seed_count": len(group)}
            elapsed = [_locf(r["best_so_far_curve"], evaluation,
                             "elapsed_time") for r in group]
            elapsed = [x for x in elapsed if x is not None]
            record["elapsed_time_median"] = (
                float(np.median(elapsed)) if elapsed else None)
            for key, reference_key, output_key in (
                    ("best_Z", "gs_Z_baseline", "improvement_pct"),
                    ("best_Z_var", "gs_Z_var_baseline",
                     "improvement_pct_var")):
                values = []
                for row in group:
                    value = _locf(row["best_so_far_curve"], evaluation, key)
                    reference = row[reference_key]
                    if value is not None and abs(reference) > 1e-12:
                        values.append((reference - value) / reference * 100.0)
                record[f"{output_key}_median"] = (
                    float(np.median(values)) if values else None)
                record[f"{output_key}_q25"] = (
                    float(np.quantile(values, 0.25)) if values else None)
                record[f"{output_key}_q75"] = (
                    float(np.quantile(values, 0.75)) if values else None)
            output.append(record)
    return output


def _plot_convergence(convergence, stem):
    by_instance = defaultdict(list)
    for row in convergence:
        by_instance[row["instance_id"]].append(row)
    for instance_id, instance_rows in by_instance.items():
        fig, axes = plt.subplots(
            1, 2, figsize=(10.0, 4.6), sharey=False,
            gridspec_kw={"wspace": 0.06})
        fig.patch.set_facecolor("white")
        algorithms = sorted({r["algorithm"] for r in instance_rows})
        colors = {algorithm: plt.get_cmap("tab10")(i % 10)
                  for i, algorithm in enumerate(algorithms)}
        seed_count = max(
            (int(r.get("seed_count", 0)) for r in instance_rows),
            default=5)
        for ax, metric, title in (
                (axes[0], "improvement_pct", r"(a) Total objective $Z$"),
                (axes[1], "improvement_pct_var",
                 r"(b) Variable component $Z_{\mathrm{var}}$")):
            ax.set_facecolor("white")
            for algorithm in algorithms:
                points = sorted((r for r in instance_rows
                                 if r["algorithm"] == algorithm),
                                key=lambda r: r["evaluation"])
                points = [r for r in points
                          if r[f"{metric}_median"] is not None]
                if not points:
                    continue
                x = np.asarray([r["evaluation"] for r in points])
                median = np.asarray([r[f"{metric}_median"] for r in points])
                q25 = np.asarray([r[f"{metric}_q25"] for r in points])
                q75 = np.asarray([r[f"{metric}_q75"] for r in points])
                ax.step(x, median, where="post", label=algorithm,
                        color=colors[algorithm], linewidth=2.0)
                ax.fill_between(x, q25, q75, step="post",
                                color=colors[algorithm], alpha=0.12)
                ax.plot(x[-1], median[-1], marker="o", markersize=4.5,
                        color=colors[algorithm], zorder=4)
                ax.annotate(
                    f"{median[-1]:.2f}%",
                    xy=(x[-1], median[-1]),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", color=colors[algorithm],
                    fontsize=9, fontweight="bold")
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_xlabel("Flow evaluations")
            ax.grid(True, color="0.85", linewidth=0.6, alpha=0.55)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel("Median improvement vs. GS (%)")
        axes[0].tick_params(
            axis="y", left=True, labelleft=True,
            right=False, labelright=False)
        axes[1].tick_params(
            axis="y", left=True, labelleft=True,
            right=False, labelright=False)
        axes[0].spines["right"].set_visible(False)
        axes[1].spines["left"].set_visible(True)
        axes[0].set_ylabel(r"$\Delta Z / Z_{\mathrm{GS}}$ (%)")
        axes[1].set_ylabel(r"$\Delta Z_{\mathrm{var}} / Z_{\mathrm{GS,var}}$ (%)")

        if len(algorithms) == 1:
            legend_handles = [
                Line2D([0], [0], color=colors[algorithms[0]], linewidth=2.0,
                       label=f"Median ({seed_count} seeds)"),
                Patch(facecolor=colors[algorithms[0]], edgecolor=colors[algorithms[0]],
                      alpha=0.12, label="IQR (25%–75%)"),
            ]
        else:
            legend_handles = [
                Line2D([0], [0], color=colors[algorithm], linewidth=2.0,
                       label=algorithm)
                for algorithm in algorithms
            ]
            legend_handles.append(
                Patch(facecolor=colors[algorithms[0]],
                      edgecolor=colors[algorithms[0]], alpha=0.12,
                      label="IQR (25%–75%)")
            )

        figure_subject = algorithms[0] if len(algorithms) == 1 else "Algorithm"
        fig.suptitle(
            f"{figure_subject} convergence on instance {instance_id}",
            fontsize=12, y=0.99)
        fig.subplots_adjust(
            left=0.085, right=0.995, bottom=0.20, top=0.84,
            wspace=0.06)
        fig.legend(
            handles=legend_handles, loc="lower center",
            bbox_to_anchor=(0.5, 0.015), ncol=len(legend_handles),
            frameon=True, facecolor="white", edgecolor="0.75",
            fontsize=9, handlelength=2.8, columnspacing=1.5)
        fig.savefig(
            f"{stem}_{instance_id}_convergence.png",
            dpi=300, bbox_inches="tight", pad_inches=0.08,
            facecolor="white")
        plt.close(fig)


def _replot_saved_convergence():
    """Recreate every runner convergence PNG from its saved JSON data."""
    paths = sorted(RESULTS.glob("*_convergence.json"))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            convergence = json.load(f)
        stem_name = path.name.removesuffix("_convergence.json")
        _plot_convergence(convergence, path.with_name(stem_name))
        print(f"[runner] replotted {path.name}")
    print(f"[runner] replotted {len(paths)} convergence data file(s)")


def _write_outputs(rows, condition, run_id="comparison"):
    stem = RESULTS / f"{run_id}_{condition}"
    seed_summary = _aggregate_seeds(rows)
    instance_summary = _aggregate_instances(seed_summary)
    convergence = _aggregate_convergence(rows)

    outputs = (("", rows), ("_seed_summary", seed_summary),
               ("_cross_instance_summary", instance_summary),
               ("_convergence", convergence))
    for suffix, data in outputs:
        with open(f"{stem}{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=NpEncoder)

    try:
        import pandas as pd
        raw_table = []
        for row in rows:
            flat = {k: v for k, v in row.items()
                    if k != "best_so_far_curve"}
            flat["best_so_far_curve"] = json.dumps(
                row["best_so_far_curve"], cls=NpEncoder)
            flat["algorithm_parameters"] = json.dumps(
                row.get("algorithm_parameters", {}), cls=NpEncoder)
            raw_table.append(flat)
        with pd.ExcelWriter(f"{stem}.xlsx") as writer:
            pd.DataFrame(raw_table).to_excel(
                writer, sheet_name="raw_runs", index=False)
            pd.DataFrame(seed_summary).to_excel(
                writer, sheet_name="seed_summary", index=False)
            pd.DataFrame(instance_summary).to_excel(
                writer, sheet_name="cross_instance", index=False)
            pd.DataFrame(convergence).to_excel(
                writer, sheet_name="convergence", index=False)
    except Exception as exc:
        print(f"[runner] Excel output skipped: {exc}")

    _plot_convergence(convergence, stem)
    print(f"[runner] outputs written with prefix {stem}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quiet", action="store_true",
        help="suppress per-run progress and result JSON; keep final output message")
    ap.add_argument(
        "--replot-only", action="store_true",
        help="recreate all saved runner convergence PNGs and exit")
    ap.add_argument("--algorithm", choices=["gs", "hc", "ts", "sa", "ga", "all"], default="all")
    ap.add_argument("--condition",
                    choices=["own_stop", "runtime", "evaluations"],
                    default="evaluations",
                    help="formal comparison condition")
    ap.add_argument("--gs-state", default=None,
                    help="GS warm-start path; defaults to the demand/dist seed name")
    ap.add_argument("--build-gs", action="store_true",
                    help="rebuild GS warm start for the requested demand/dist seeds")
    ap.add_argument("--demand-seeds", "--demand-seed", dest="demand_seeds",
                    type=int, nargs="+", default=list(range(2026, 2031)),
                    help="demand RNG seeds (default: 2026..2030)")
    ap.add_argument("--dist-seeds", "--dist-seed", dest="dist_seeds",
                    type=int, nargs="+", default=list(range(124, 129)),
                    help="independent distance RNG seeds (default: 124..128)")
    ap.add_argument("--algo-seeds", "--algo-seed", dest="algo_seeds",
                    type=int, nargs="+", default=list(range(5)),
                    help="independent algorithm RNG seeds (default: 0..4)")
    ap.add_argument("--max-iter", type=int, default=100000,
                    help="iteration safety cap; not the fairness budget")
    ap.add_argument("--max-evaluations", type=int, default=3000,
                    help="common budget for condition=evaluations")
    ap.add_argument("--max-runtime", type=float, default=60.0,
                    help="common seconds budget for condition=runtime")
    ap.add_argument(
        "--ts-preset", choices=list(TS_PRESETS), default="full",
        help="TS preset for --algorithm ts; all primary presets run with all")
    ap.add_argument(
        "--sa-preset", choices=list(sa_basic.PRESETS), default="base",
        help="SA preset for --algorithm sa; both presets run with all")
    # None means: use the frozen final recommendation for that preset.
    ap.add_argument("--sa-t0", type=float, default=None)
    ap.add_argument("--sa-lambda", type=float, default=None)
    ap.add_argument("--sa-moves", type=int, default=None)
    ap.add_argument("--sa-alpha", type=float, default=None)
    ap.add_argument("--ga-population", type=int, default=None)
    ap.add_argument("--ga-tournament", type=int, default=None)
    ap.add_argument("--ga-p-c", dest="ga_p_c", type=float, default=None)
    ap.add_argument("--ga-p-m", dest="ga_p_m", type=float, default=None)
    ap.add_argument("--ga-elite", type=int, default=None)
    ap.add_argument("--output-id", default="comparison",
                    help="output filename prefix under results/runner")
    args = ap.parse_args()
    if args.replot_only:
        _replot_saved_convergence()
        return
    if (args.gs_state is not None
            and (len(set(args.demand_seeds)) != 1
                 or len(set(args.dist_seeds)) != 1)):
        ap.error("--gs-state can only be used with one demand/distance "
                 "instance; omit it to select warm starts automatically")
    if args.condition == "runtime" and args.max_runtime <= 0:
        ap.error("--max-runtime must be positive")
    if args.condition == "evaluations" and args.max_evaluations <= 0:
        ap.error("--max-evaluations must be positive")
    for name, value in (
            ("--sa-moves", args.sa_moves),
            ("--ga-population", args.ga_population),
            ("--ga-tournament", args.ga_tournament),
            ("--ga-elite", args.ga_elite)):
        if value is not None and value <= 0:
            ap.error(f"{name} must be positive")
    for name, value in (("--ga-p-c", args.ga_p_c),
                        ("--ga-p-m", args.ga_p_m)):
        if value is not None and not 0.0 <= value <= 1.0:
            ap.error(f"{name} must be between 0 and 1")
    if args.sa_t0 is not None and args.sa_t0 <= 0:
        ap.error("--sa-t0 must be positive")
    if args.sa_alpha is not None and not 0.0 < args.sa_alpha <= 1.0:
        ap.error("--sa-alpha must be in (0, 1]")
    if args.sa_lambda is not None and args.sa_lambda < 0:
        ap.error("--sa-lambda must be non-negative")

    sa_overrides = {
        "T0": args.sa_t0,
        "alpha": args.sa_alpha,
        "moves_per_temperature": args.sa_moves,
        "sa_lambda": args.sa_lambda,
    }
    ga_overrides = {
        "population_size": args.ga_population,
        "tournament_size": args.ga_tournament,
        "p_c": args.ga_p_c,
        "p_m": args.ga_p_m,
        "elite_size": args.ga_elite,
    }
    if args.algorithm == "all":
        # Final comparison: deterministic baselines, tuned GA, and every
        # primary tuned TS/SA preset.
        jobs = [
            ("gs", None, None),
            ("hc", None, None),
            ("ga", None, None),
        ]
        jobs.extend(("ts", preset, None)
                    for preset in PRIMARY_TS_PRESETS)
        jobs.extend(("sa", None, preset)
                    for preset in PRIMARY_SA_PRESETS)
    else:
        jobs = [(
            args.algorithm,
            args.ts_preset if args.algorithm == "ts" else None,
            args.sa_preset if args.algorithm == "sa" else None,
        )]
    rows = []
    for demand_seed in args.demand_seeds:
        for dist_seed in args.dist_seeds:
            gs_state = (args.gs_state if args.gs_state is not None else
                        f"data/warmstarts/gs_d{demand_seed}_x{dist_seed}.json")
            gs_path = Path(gs_state)
            if not gs_path.is_absolute():
                gs_path = ROOT / gs_path

            if args.build_gs:
                reseed_world(od_seed=demand_seed, rng_seed=dist_seed)
                W_gs, pdf_gs, runs_added = run_greedy(
                    max_evaluations=args.max_evaluations,
                    max_iter=args.max_iter)
                gs_path.parent.mkdir(parents=True, exist_ok=True)
                save_state(pdf_gs, runs_added,
                           {"hc": [], "ts": [], "sa": []}, W_gs,
                           str(gs_path), demand_seed=demand_seed)
            if not gs_path.exists():
                raise FileNotFoundError(gs_path)

            for algorithm, ts_preset, sa_preset in jobs:
                # GS and HC are deterministic and run once per instance.
                seeds = ([None] if algorithm in ("gs", "hc")
                         else args.algo_seeds)
                for algo_seed in seeds:
                    if not args.quiet:
                        print(f"\n[runner] instance=d{demand_seed}_x{dist_seed} "
                              f"algorithm={algorithm} algo_seed={algo_seed} "
                              f"condition={args.condition}")
                    row = _run_one(
                        algorithm, gs_path, demand_seed, dist_seed, algo_seed,
                        args.condition, args.max_iter, args.max_evaluations,
                        args.max_runtime, ts_preset, sa_preset,
                        sa_overrides, ga_overrides)
                    rows.append(row)
                    if not args.quiet:
                        print(json.dumps(
                            {k: v for k, v in row.items()
                             if k != "best_so_far_curve"},
                            ensure_ascii=False, cls=NpEncoder))

    run_id = args.output_id
    if args.algorithm != "all":
        run_id += f"_{args.algorithm}"
        if args.algorithm == "ts":
            run_id += f"_{args.ts_preset}"
        elif args.algorithm == "sa":
            run_id += f"_{args.sa_preset}"
    _write_outputs(rows, args.condition, run_id)


if __name__ == "__main__":
    main()
