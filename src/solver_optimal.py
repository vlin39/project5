"""
Pure-Python CVRP solver — no external VRP libraries.

The production solver for this project. Runs a 3-engine
construction-family portfolio for every instance:

  - Clarke-Wright savings + ILS    (solver_savings.solve)
  - GRASP + LNS                    (solver_grasp_lns.solve)
  - Angular sweep + ILS            (solver_sweep.solve)

Each engine builds an initial solution using its own construction strategy,
then drives it through the same enhanced ILS in `solver_common`. The
diversity of starting solutions is what wins on the hardest instances:
GRASP+LNS's k-means construction seeds basins that CW misses on
200_16_2.vrp / 386_47_1.vrp / 41_14_1.vrp; sweep's angular construction
helps on 241_22_1.vrp.

All engines share the same enhanced ILS, with every B-series enhancement
always on:

  B1. Or-opt              — relocate segments of length 2 / 3
  B2. 2-opt*              — true cross-route 2-opt edge exchange
  B3. KNN candidate lists — top-K nearest-neighbour pruning of cross-route
                            insertions (huge per-iteration speedup)
  B4. Don't-look bits     — skip customers whose neighborhood was scanned
                            without improvement, until a neighbor changes
  B5. SA acceptance       — temperature-cooled probabilistic acceptance
                            inside `finish_with_ils`
  B6. SISR perturbation   — geographic-string removal alongside the
                            classic random / route / shaw destroy operators

Pipeline at a glance:

  main.py
   └── solver_optimal.solve(data)
         ├── adaptive_budget(n)      — per-instance time budget
         ├── _config_for_size(n)     — size-tuned ILS knobs + B1-B6
         └── 3-engine portfolio @ savings/grasp_lns/sweep = 60/27/13
                  → take the lowest-cost result

Earlier we tried single-savings for n<=200 to recover the historical
config-tuned wins, but that lost 200_16_2 (-143 cost), 41_14_1 (-64),
101_8_1 (-19), and 51_5_1 (-13) where grasp_lns or sweep was the actual
winner. Net was +0.48% worse, so we're back to portfolio-for-all.
"""

from solver_common import ensure_data, DEFAULT_TIME_LIMIT
from solver_savings import solve as savings_solve
from solver_sweep import solve as sweep_solve
from solver_grasp_lns import solve as grasp_lns_solve


# Always-on engine extensions (B1-B6).
ENGINE_EXTENSIONS = {
    "or_opt": True,                 # B1
    "two_opt_star": True,           # B2
    "candidate_list_size": 20,      # B3 — top-20 nearest neighbors
    "dont_look": True,              # B4
    "sa_acceptance": True,          # B5
    "perturb_mode": ["random", "route", "shaw", "sisr"],  # B6 + the originals
}


def _config_for_size(n):
    """ILS knobs adapted to instance size.

    Small (n<=50)         — many noisy seeds; LS converges in milliseconds
                            so basin diversity is the binding constraint.
    Medium (50<n<=200)    — wider construction pool tuned to the union of
                            the historical per-instance winners
                            (restart_trigger=12, noise_low, intensify=0.10,
                            etc.).
    Large (n>200)         — fewer seeds with deeper LS per restart;
                            paired with the construction-family portfolio
                            so different starting points get equal access
                            to the shared ILS.
    """
    cfg = dict(ENGINE_EXTENSIONS)
    if n <= 50:
        cfg.update({
            "noise_levels": (0.0, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50),
            "seed_count": 12,
            "elite_size": 6,
            "construction_fraction": 0.50,
            "construction_passes": 5,
            "intensify_fraction": 0.30,
            "base_remove": 4,
            "max_remove": 8,
            "top_k": 8,
            "ls_passes": 6,
            "restart_trigger": 6,
            "accept_prob": 0.20,
        })
    elif n <= 200:
        # Enriched config — union of historical per-instance winners
        # (restart12, noise_low, balanced, intensify10, restarttrigger8,
        # diversified). Wider noise vector covers both the broad-noise
        # cases and the narrow-noise winner on 151_15_1; seed_count and
        # elite_size match the prior winners' values.
        cfg.update({
            "noise_levels": (0.0, 0.005, 0.01, 0.02, 0.03, 0.05,
                             0.07, 0.12, 0.20),
            "seed_count": 8,
            "elite_size": 6,
            "construction_fraction": 0.40,
            "construction_passes": 3,
            "intensify_fraction": 0.12,
            "base_remove": 6,
            "max_remove": 20,
            "top_k": 8,
            "ls_passes": 5,
            "restart_trigger": 10,
            "accept_prob": 0.20,
        })
    else:
        cfg.update({
            "noise_levels": (0.0, 0.01, 0.03, 0.07, 0.12),
            "seed_count": 3,
            "elite_size": 3,
            "construction_fraction": 0.20,
            "construction_passes": 3,
            "intensify_fraction": 0.10,
            "base_remove": 8,
            "max_remove": 24,
            "top_k": 10,
            "ls_passes": 6,
            "restart_trigger": 10,
            "accept_prob": 0.20,
        })
    return cfg


def adaptive_budget(n):
    """Per-instance time budget in seconds, keyed on customer count.

    Sized so that small instances finish in a fraction of the project's
    300s/instance shell ceiling and large instances run their full ILS
    convergence cycle. Budgets are split across the 3-engine portfolio
    by the weights below — savings still ends up with the majority share
    on every instance.
    """
    if n <= 50:
        return 25.0
    if n <= 100:
        return 60.0
    if n <= 200:
        return 120.0
    if n <= 300:
        return 200.0
    return 270.0


# Portfolio composition. Weights chosen to give savings the lion's share
# (it's the winner on most instances) while still giving grasp_lns and
# sweep enough budget to find their basins on 200_16_2 / 386_47_1 /
# 41_14_1 / 51_5_1 / 101_8_1 / 241_22_1 — instances where they
# strict-beat savings in prior runs.
_PORTFOLIO = [
    ("savings",   savings_solve,   60),
    ("grasp_lns", grasp_lns_solve, 27),
    ("sweep",     sweep_solve,     13),
]


def solve(data, time_limit=None, seed=0, config=None):
    """
    Solve a CVRP instance with a pure-Python construction-family portfolio.

    Parameters mirror the other `solver_*.solve(...)` entry points so this
    module is a drop-in for `src/main.py`. `time_limit`, when supplied, is
    treated as an upper bound (the runAll.sh shell timeout); the adaptive
    budget is clamped under it.
    """
    data = ensure_data(data)
    n = int(data.n)

    total = adaptive_budget(n)
    if time_limit is not None:
        # Leave a margin so we always return before the shell timeout fires.
        total = min(total, max(1.0, float(time_limit) - 5.0))

    base_cfg = _config_for_size(n)
    if config:
        base_cfg.update(config)

    weight_sum = sum(w for _, _, w in _PORTFOLIO)
    best_routes, best_cost = None, float("inf")
    for i, (label, fn, weight) in enumerate(_PORTFOLIO):
        budget = max(0.5, total * weight / weight_sum)
        try:
            routes, cost = fn(
                data, time_limit=budget, seed=seed + i, config=base_cfg
            )
        except Exception:
            # Any single engine failure must not take down the portfolio.
            continue
        if cost < best_cost:
            best_routes, best_cost = routes, cost

    return best_routes, best_cost
