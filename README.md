# Code Workflow

Master Thesis: 
Metaheuristic Optimization of a Path-Based Network Flow Problem via Discrete Door Incentives

This README describes the current Python code in `core/`, `algorithms/`, and `experiments/`. It follows the functions that the experiment scripts call.

## Workflow map

```text
core/config.py
    ↓
core/world.py → core/flow.py → core/objective.py + core/verify.py
    ↓
algorithms/greedy.py → GS state files
                         ├→ experiments/runner.py → comparisons and plots
                         └→ experiments/ofat_tuning.py → raw OFAT JSONL
```

The runner can also build GS states with `--build-gs`. It loads a fresh GS state before each algorithm run. OFAT does the same before each parameter run.

## 1. Build and evaluate a world

| File | Work |
| --- | --- |
| [`core/config.py`](core/config.py) | Defines four lines, two doors per run, wait limits, discount levels, model coefficients, segment capacity `40`, and capacity penalty weight `50`. |
| [`core/world.py`](core/world.py) | Generates OD demand from a demand seed and door distances from a distance seed. Each line starts with one run. `build_world()` creates door events, ride arcs, transfer arcs, wait arcs, feasible OD paths, grouped paths, and run lookup data. `reseed_world()` resets seeds and, by default, schedules. |
| [`core/flow.py`](core/flow.py) | Calculates path utility and splits each reachable OD demand across its paths with `exp(U) / Σexp(U)`. It adds ride flow to train segments. Each `compute_flows()` call increases the global evaluation counter. |
| [`core/objective.py`](core/objective.py) | Calculates stop time `Z`, capacity overflow `P`, and `F = Z + 50P`. `compute_Z_fixed()` gives the timetable part of `Z`: `alpha0` for each stop. |
| [`core/verify.py`](core/verify.py) | Checks whether path flows sum to OD demand. It reports unreachable and mismatched OD pairs. |

The implemented path utility is:

```text
U = beta0 + beta1 × (1 − boarding discount)
          + beta2 × boarding-door distance
          + beta3 × (origin wait + transfer wait + number of transfers)
```

For each stop, `Z` adds `alpha0` and the largest door value of `alpha1 × boardings + alpha2 × alightings + 0.01 × boardings × alightings`. `P` sums `max(0, segment load − 40)` over train segments.

## 2. Change a solution

[`core/actions.py`](core/actions.py) generates, applies, and undoes these actions:

| Action | Change |
| --- | --- |
| `A1` | Raise the discount at a boarding door. |
| `A2`, `A3` | Open origin wait paths or transfer wait paths. |
| `A4` | Add a run. Greedy handles this outside the apply/undo cycle. |
| `B1`, `B2` | Raise or lower a boarding-door discount. |
| `B3`, `B4` | Open or close origin wait paths. |
| `B5`, `B6` | Open or close transfer wait paths. |

Closing paths can remove related paths. `B4` and `B6` guard reachable OD pairs with positive demand. They also guard capacity when `enforce_capacity=True`.

## 3. Search for a solution

| Algorithm | Search | Stop and return |
| --- | --- | --- |
| [`Greedy`](algorithms/greedy.py) | Starts from the initial schedules and zero discounts. It tests `A1–A3` and takes the best action that lowers `F` by more than `1e-4`. If none qualifies, it tests adding one run to each line with a reachable departure slot. It prefers a run that lowers `P`, then the lowest resulting `F`. It can explore up to three consecutive `A4` moves that do not lower `P`. It rebuilds the world after a committed `A4`. | Stops at `P=0`, a limit, or no usable action. Returns the world, discounts, and added runs. |
| [`Hill Climbing`](algorithms/hill_climbing.py) | Starts from a supplied state. It tests and undoes each `B1–B6` candidate. It commits the best action only when `P=0` and `Z` falls by more than `1e-4`. | Stops after 10 rounds without improvement or at an iteration, evaluation, or time limit. Returns discounts, action history, and a curve. |
| [`Tabu Search`](algorithms/tabu_search_unified.py) | Tests `B1–B6` candidates. It can filter candidates by action type, use a tabu list with aspiration, and perturb the best feasible state. Penalty mode scores `Z + ts_lambda × P`; the other mode accepts only candidates with `P` near zero. | Stops at stagnation or a limit. Restores the best feasible state if one exists. Returns discounts, history, and a curve. [`_ts_common.py`](algorithms/_ts_common.py) holds shared snapshot and scoring helpers. |
| [`Simulated Annealing`](algorithms/simulated_annealing.py) | Randomly chooses an action type and then an action from `B1–B6`. It accepts better moves. It can accept worse moves with `exp(−Δ/T)`. Temperature falls after a set number of moves. Hard mode rejects `P>0`; penalty mode uses `Z + sa_lambda × P`. | Stops at the minimum temperature or a limit. Restores the best feasible state if one exists. The main function returns discounts, history, best feasible `Z`, and a curve. |
| [`Genetic Algorithm`](algorithms/genetic_algorithm.py) | Uses `D` genes for door discounts and `G` genes for wait-path switches. It decodes each individual from the starting state and calculates `F`, `Z`, and `P`. It ranks feasible individuals first, then ranks by `F`. It uses elites, tournament selection, crossover, and mutation. | Stops after 2,000 evaluations without improvement or at a generation, evaluation, or time limit. Restores the best feasible individual if one exists; otherwise restores the starting state. Returns discounts, a curve, and statistics. |

## 4. Generate GS states

Run [`experiments/build_warmstarts.py`](experiments/build_warmstarts.py) from the project root:

```bash
python -m experiments.build_warmstarts
```

For each demand-seed and distance-seed pair, it resets the world, runs Greedy, evaluates the result, checks OD demand, and saves:

```text
data/warmstarts/gs_d{demand_seed}_x{distance_seed}.json
data/warmstarts/gs_d{demand_seed}_x{distance_seed}.meta.json
data/warmstarts/manifest.json
```

The default demand seeds are `2026–2030`. The default distance seeds are `124–128`. `--skip-existing` skips a pair only when both its state file and metadata file exist. The manifest contains summaries for pairs processed in that run. This script runs GS only.

Use these seed values to build the runner's default instances:

```bash
python -m experiments.build_warmstarts --demand-seeds 2026 2027 2028 2029 2030 --distance-seeds 124 125 126 127 128
```

## 5. Run the comparison

[`experiments/runner.py`](experiments/runner.py) reads a GS state for each instance. Its loader restores demand, distances, schedules, discounts, and saved paths. `--build-gs` builds that state in the runner instead. The runner reloads the GS state for each algorithm run. It measures the GS baseline, resets the evaluation counter, runs the algorithm, and checks the final result.

```bash
python -m experiments.runner --algorithm all --condition evaluations
```

By default, the runner uses demand seeds `2026–2030`, distance seeds `124–128`, and algorithm seeds `0–4`. `--algorithm all` runs GS, HC, GA, four TS presets, and two SA presets. GS and HC run once per instance. GA, TS, and SA run for each algorithm seed.

`--condition` accepts `own_stop`, `runtime`, or `evaluations`. The runner uses its declared tuned settings for TS, SA, and GA. It does not read OFAT output files. It compares a run only when both the GS baseline and the final result pass its capacity and OD-demand checks.

The runner writes raw runs, per-instance seed summaries, cross-instance summaries, and convergence data as JSON under `results/runner/`. It also writes a convergence PNG for each instance. It writes an Excel workbook when the Excel dependencies are available. The convergence data include medians and quartiles. The runner reports both `Z` and `Z_var = Z − Z_fixed` improvements.

The three scripts do not use the same default seed pairs. Set matching builder seeds, provide `--gs-state` for one runner instance, or use runner `--build-gs` when needed. The builder records demand satisfaction in its `feasible` metadata field. The runner checks demand satisfaction and `P ≤ 1e-9`.

## 6. Run one-factor-at-a-time experiments

[`experiments/ofat_tuning.py`](experiments/ofat_tuning.py) reads GS states from `data/warmstarts/`. It restores the same state before each TS, SA, or GA run. Each stage writes its baseline once per instance and algorithm seed. It then changes one parameter at a time and writes one row per run to `results/ofat/raw/<algorithm>_<stage>.jsonl`.

```bash
python -m experiments.ofat_tuning
```

The default stage is `ts1`. The default demand seeds are `2026–2030`; the distance seed is `124`; the algorithm seeds are `0–4`; and the evaluation limit is `30000`. `--quick` uses the first instance, the first two algorithm seeds, and one non-baseline level per parameter. `--overwrite` replaces an existing stage file. The script writes raw JSONL only. It does not aggregate results or choose parameters. Its reported `penalized_Z` uses `lambda_eval`, which defaults to `16` and is separate from the TS and SA search penalties.

| Stage | Mode | Parameters listed in the current grid |
| --- | --- | --- |
| `ts1` | TS - BASE | `tenure`, `perturbation_trigger`, `perturb_steps`, `perturb_heavy_bias`, `perturb_top_k` |
| `ts2` | TS  + CLS | `tenure`, `perturbation_trigger`, `top_cls`, `random_extra`, `random_ratio`, `cls_pool_max` |
| `ts3` | TS + penalty | `tenure`, `perturbation_trigger`, `perturb_steps`, `perturb_heavy_bias`, `perturb_top_k`, `ts_lambda` |
| `ts4` | Full TS | The five `ts1` parameters, the four CLS parameters, and `ts_lambda` |
| `sa1` | SA hard mode | `T0`, `alpha`, `moves_per_temperature` |
| `sa2` | SA penalty mode | `T0`, `alpha`, `moves_per_temperature`, `sa_lambda` |
| `ga` | GA | `population_size`, `tournament_size`, `p_c`, `p_m`, `elite_size` |

The declared `TS_STAGE1_FROZEN` and `SA_STAGE1_FROZEN` values feed later stages. They start as copies of the stage-one defaults. The script does not update them from results.


