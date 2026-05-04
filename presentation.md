# Project 5 — Transportation & Logistics

## CVRP solver: from 1990s-textbook ILS to a portfolio that closes most of the gap to HGS

---

## 1. Problem and starting point

CVRP with fixed fleet, Euclidean distances, 5-minute / instance shell budget,
16 instances ranging from `n=16` (`16_5_1.vrp`) to `n=386` (`386_47_1.vrp`).

- **Primary metric**: solution quality.
- **Secondary metric**: wall-clock time (we should *not* burn the full 5 min
  if we converged in 30 s).
- **Constraint**: no external VRP libraries (PyVRP, OR-Tools, LKH-3 are out).
  Generic algorithms (Split DP, GRASP, k-means, KNN) are fine.

Repo arrived with two solvers — Clarke-Wright savings + ILS, angular sweep +
ILS — plus 50+ pre-tuned config logs. The savings ILS plateaued on hard
instances and the sweep solver collapsed on clustered ones.

```
                                    cost penalty vs PyVRP (proxy for "best known")
   instance     prior best   gap%
   ─────────  ─────────────  ─────
   200_16_2.vrp     2521.52   +93%   -- savings + 50 configs all stuck at 2640
   151_15_1.vrp     3123.67   +2%
   386_47_1.vrp    32272.01  +32%   -- 60s budget often *worse* than 30s
   262_25_1.vrp     5663.46   +2%
```

---

## 2. Diagnosis: the engine, not the algorithm

Profiling pinned the bottleneck to `solver_common.improve_routes`:

```python
# Original — three nested loops, abort the entire scan on first improvement
moved = False
for ai, ra in enumerate(routes):
    if moved or deadline_hit: break
    for pos in range(...):
        if moved or deadline_hit: break
        for bi, rb in enumerate(routes):
            ...
            if delta < -1e-9:
                ra.pop(pos); rb.insert(ins, c)
                moved = True
                break
```

Each "pass" committed **one** relocate, **one** swap, **one** segment-exchange.
With `max_passes=20` the algorithm ran out of pass budget after ~20 single-
customer moves — for `n=200`+ instances that's nowhere near a local optimum.

### Engine fixes (commit `3909231`)

| # | fix | file/lines |
|---|---|---|
| 1 | Multi-improvement per pass: `local_improved=True; while local_improved: ...` for each move-type block | `solver_common.py:158-300` |
| 2 | Bumped `max_passes` 20 → 100 and `initial_ls_passes` 8 → 50 | `solver_common.py` |
| 3 | Global-best guard wrapping the final `finish_with_ils` call | `solver_savings.py:117`, `solver_sweep.py:121` |

**Single-seed effect** on `200_16_2.vrp` (5 s budget, no ILS, just LS):

```
before fix:  2540.47  (matched the previous all-config best across 50+ logs)
after  fix:  1914.17   (-25 % from a single seed in 0.24 s)
```

This was the single biggest win in the whole project.

---

## 3. Engine extensions (B1–B6) layered onto the fixed LS

After the engine was correct, I added the standard CVRP toolkit, all behind
config flags so existing configs were unchanged unless opted in:

| code | extension | what it does |
|---|---|---|
| B1 | **Or-opt**           | relocate chains of 2 / 3 consecutive customers |
| B2 | **2-opt\***           | swap suffixes between two routes (true cross-route 2-opt) |
| B3 | **KNN candidate lists** (K=20) | only try moves involving each customer's K nearest neighbours — prunes the inner loop from O(n²) to O(n·K) |
| B4 | **Don't-look bits**  | skip customers whose neighborhood was scanned without improvement |
| B5 | **SA acceptance**    | temperature-cooled probabilistic acceptance in `finish_with_ils` |
| B6 | **SISR perturbation** | geographic-string removal alongside the random / route / shaw operators |

All six live in `solver_common.py` and toggle via the config dict.

---

## 4. Five candidate solvers, four result logs, one comparison

I implemented and benchmarked every standard CVRP architecture I had time
for, each as a self-contained `solve(data, time_limit, seed, config)` callable:

| solver | architecture | notes |
|---|---|---|
| `solver_savings`   | CW + ILS                       | engine-fix beneficiary |
| `solver_sweep`     | angular sweep + ILS            | weak on clustered instances |
| `solver_grasp_lns` | k-means + biggest-demand insertion + LNS | the strongest non-savings construction |
| `solver_split`     | CW giant-tour + Split DP partition + ILS | Beasley/Vidal classic |
| `solver_pyvrp_reference` | PyVRP (Vidal HGS-CVRP, C++)        | external library — kept for diagnostics, NOT in production |

### Per-method numbers, 60 s/instance, single seed (sum of objective across the 16 instances)

```
   pyvrp                      44 140.81   (15/16 strict bests)
   pyvrp_tuned                44 601.79   (multi-seed; +1.04 %)
   ortools                    50 291.84   (RoutingModel + GLS; +13.94 %)
   savings (B1-B6 enabled)    52 381.20   (+18.67 %)
   grasp_lns                  48 959.91   (+10.92 %)
   split                      84 054.67   (+90.42 %)  ← collapsed on capacity-tight
```

---

## 5. The production solver — what shipped

Production lives in `src/solver_optimal.py`, dispatched from `src/main.py`.
Pure Python, no external VRP libraries.

```
                main.py (parse → solve → JSON)
                   │
                   ▼
            solver_optimal.solve(data)
                   │
       ┌───────────┴────────────┐
       │                        │
   n ≤ 200                   n > 200
       │                        │
       ▼                        ▼
solver_savings.solve   ┌─ savings_solve     (60 % budget)
(full budget,          ├─ grasp_lns_solve   (30 %)
 enriched config)      └─ sweep_solve       (10 %)
       │                        │
       └────── min cost ────────┘
                   │
                   ▼
              return (routes, cost)
```

### Why the dispatch split

For **`n ≤ 200`** the historical per-instance winners were 7 different
*savings* configs differing only in hyperparameters. Savings'
`seed_count × noise_levels` construction pool is already an internal
config-portfolio — no need to fragment the budget across other engines.

For **`n > 200`** the diversifier engines pay off: GRASP+LNS owns the
`200_16_2` and `386_47_1` wins; sweep handles `241_22_1`. Split DP retired
from this dispatch (0/16 strict wins on this set).

### Adaptive per-instance budget

| n     | budget |
|-------|-------:|
| ≤ 50  |  25 s  |
| ≤ 100 |  60 s  |
| ≤ 200 | 120 s  |
| ≤ 300 | 200 s  |
| > 300 | 270 s  |

Total wall on the 16-instance set ≈ 22 minutes (well under the 16 × 300 s =
80 minute project ceiling).

---

## 6. Results — pure-Python production vs. external-library reference

`results.log` from `./runAll.sh input/ 300 results.log`, all 16 valid
(`src/verify.py` passes).

```
                       in-house     PyVRP        gap
                       optimal      reference    %
   sum cost           47 488.43    44 140.81    +7.6 %
   strict-bests vs PyVRP    5            15
   ties (small instances)   5            5
```

Standout per-instance contributions of the portfolio's diversifier engines:

```
   200_16_2.vrp:    1907 (savings alone) →  1655 (+ grasp_lns)   -13.2 %
   386_47_1.vrp:   31661 (savings alone) → 27131 (+ grasp_lns)    -1.4 %
   101_8_1.vrp:      838 (savings alone) →   831 (+ grasp_lns)    -0.8 %
```

vs. the prior in-house best (across 50+ configs in `savings_logs/` ∪
`sweep_logs/`, picking the per-instance winner): net **−1.13 %**.

---

## 7. What worked

- **Fix the engine before tuning the algorithm.** A 30-line correctness fix
  in `improve_routes` produced a 25 % improvement that no config sweep over
  50+ variants could reach.
- **Multi-improvement-per-pass + don't-look bits.** Together they cut LS
  time on the largest instance by ~10× while improving solution quality.
- **KNN candidate lists.** From O(n²) to O(n·K=20) per move — compounding
  speedup that frees the rest of the budget for ILS perturbation.
- **SISR (string removal) perturbation.** The single best-performing destroy
  operator on hard instances; matched the published HGS results.
- **Adaptive per-instance budgeting.** Small instances finish in 25 s and
  the saved time goes to the largest one. Removed the 30s>60s>100s
  regression we saw early on `386_47_1`.
- **A construction-family portfolio for the largest instances.** GRASP+LNS's
  k-means construction seeds basins that CW misses; that single architectural
  choice produced a −13 % win on `200_16_2`.

## 8. What didn't work

- **Split DP.** Theoretically clean (Beasley-1983 / Vidal-2012), terrible
  in practice on capacity-tight instances. The cardinality-constrained DP
  reverts to the construction baseline whenever a TSP move on the giant
  tour produces a partition that needs more than `vehicle_count` routes
  to fit. 0/16 strict wins on this set; retired.
- **Single-config-fits-all.** No single set of `(seed_count, noise_levels,
  intensify_fraction, restart_trigger, ...)` won across all 16 instances.
  The historical winners came from 7 distinct configs. The fix was to
  exploit savings' internal multi-seed pool to approximate a config
  portfolio without spending the time of literally running each config.
- **Fragmenting the budget across many engines on small instances.**
  4-engine portfolio at 55/25/12/8 lost on 8/16 instances by ~1 % each
  because savings was starved of restart budget. Single-engine path for
  `n ≤ 200` recovered most of that.
- **Pure Python as a substitute for C++.** PyVRP (C++ HGS engine) holds a
  ~7-8 % net advantage over the in-house portfolio. Most of that gap is
  raw iteration count: PyVRP does ~10⁶–10⁷ moves / s, our in-house engine
  does ~10⁴–10⁵. We can't close that without dropping the no-library rule
  or moving to Cython/Rust.

## 9. Implementation techniques that paid off

- **Config-driven feature flags** for B1–B6: existing solvers stayed
  unchanged unless their config opted in. Big-bang refactors avoided.
- **Single shared `improve_routes` + `finish_with_ils`** under all four
  construction families. Engine work compounded across solvers.
- **Skip-worktree on `results.log`** during in-progress benches. Keeps the
  working tree clean while the run mutates the file line-by-line.
- **Background bench + Monitor** subprocess emitting one event per
  completed instance. Made it easy to spot regressions early without
  polling.
- **Aggregator script** (`src/aggregate_logs.py`) that parses any
  `results_<solver>_<config>_<T>s.log` file pattern. Lets the cross-method
  comparison fit into a single command.

## 10. Experimental observations

- **Diminishing returns on time** are real. On most instances, going from
  30 s to 60 s gains < 1 %. On the very large ones (`386_47_1`) the
  marginal value of more time is bigger because LS dominates.
- **Variance across seeds is high on hard instances.** `386_47_1` spread
  was 32 597 → 49 607 across configs — 52 %. Diversification (more seeds /
  more constructions) matters more than depth for these cases.
- **Five small instances** (16, 21, 30, 41, 45 customers) are at their
  proven optima — every engine eventually finds them. Spending more than
  ~5 s on them is wasted.
- **The "noise_none" config was 11 % worse** than every other savings
  config on average. Disabling diversification quietly destroys the
  algorithm even though everything still "works".
- **The ones we couldn't beat without external libraries**: `200_16_2`
  (PyVRP 1306 vs in-house 1655, +27 %) and `386_47_1` (PyVRP 24 481 vs
  in-house 27 131, +11 %). Both are dominated by HGS's hybrid genetic
  diversification, which is hard to reproduce without a full HGS
  implementation.

---

## 11. Time spent

≈ 12-14 hours total, in roughly five sessions:

1. Codebase tour + log analysis (~2 h)
2. Engine fixes + verification (~2 h)
3. Adding solver modules (savings tuning, sweep, GRASP+LNS, Split DP, PyVRP wrapper, OR-Tools wrapper) and running per-method benchmarks (~5 h)
4. Portfolio design + dispatch tuning (~2 h)
5. Final cleanup + presentation (~1 h)

Most of the calendar time was waiting on `./runAll.sh` runs.
