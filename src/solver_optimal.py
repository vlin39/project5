"""
Pure-Python portfolio CVRP solver — no external VRP libraries.

The production solver for this project. Combines two construction families
(Clarke-Wright savings and angular sweep) feeding a single ILS engine that
has every B-series enhancement enabled:

  B1. Or-opt   — relocate segments of length 2 and 3
  B2. 2-opt*   — true cross-route 2-opt edge exchange
  B3. KNN candidate lists — restrict cross-route insertions to positions
                             adjacent to one of c's K nearest neighbors
  B4. Don't-look bits     — skip customers whose neighborhood was scanned
                             without improvement until a neighbor changes
  B5. SA acceptance       — temperature-cooled acceptance in finish_with_ils
  B6. SISR (string removal) — geographic-string destroy operator alongside
                              random / route / shaw

All of the above live in `solver_common.py` and are toggled via the config
dict. This module wires the toggles, picks an adaptive per-instance time
budget (smaller instances converge in seconds), splits that budget between
the two construction engines, and returns the best result found.

The split is 80/20 in favour of savings, reflecting the empirical edge of
CW + ILS over sweep + ILS on this project's input/ instance set; sweep is
a cheap diversifier for the rare cases where its angular construction
seeds a basin that CW misses (historically ~1/16 of instances).

`solver_pyvrp_reference.py` lives in the tree as a reference baseline
only and is NOT imported here.
"""

from solver_common import ensure_data, DEFAULT_TIME_LIMIT
from solver_savings import solve as savings_solve
from solver_sweep import solve as sweep_solve


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

    Small instances (n<=50): the LS converges in milliseconds, so the
    binding constraint is *basin diversity* — push more seeds with wider
    noise levels and a heavier construction phase. (Without this, savings
    plateaus at the wrong local optimum on instances like 16_5_1, 21_4_1.)

    Medium (50<n<=200): the prior log winners — moderate noise, balanced
    construction/intensify split.

    Large (n>200): construction is expensive per seed, so spawn fewer
    seeds and put the time into deep ILS perturbation cycles.
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
        cfg.update({
            "noise_levels": (0.0, 0.01, 0.03, 0.07, 0.12, 0.20),
            "seed_count": 6,
            "elite_size": 4,
            "construction_fraction": 0.35,
            "construction_passes": 3,
            "intensify_fraction": 0.15,
            "base_remove": 6,
            "max_remove": 20,
            "top_k": 8,
            "ls_passes": 5,
            "restart_trigger": 8,
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

    Sized so that small instances finish within a fraction of the
    project's 300s/instance shell ceiling and large instances run their
    full ILS convergence cycle. With the small-instance config above,
    n<=50 needs ~20s to reliably escape the wrong local optimum.
    """
    if n <= 50:
        return 25.0
    if n <= 100:
        return 50.0
    if n <= 200:
        return 100.0
    if n <= 300:
        return 200.0
    return 270.0


def solve(data, time_limit=None, seed=0, config=None):
    """
    Solve a CVRP instance using the savings + sweep portfolio.

    Parameters mirror the other `solver_*.solve(...)` entry points so this
    module is a drop-in for `src/main.py`. `time_limit`, when supplied, is
    treated as an upper bound (the runAll.sh shell timeout); the adaptive
    budget is clamped under it.
    """
    data = ensure_data(data)

    total = adaptive_budget(int(data.n))
    if time_limit is not None:
        # Leave a margin so we always return before the shell timeout fires.
        total = min(total, max(1.0, float(time_limit) - 5.0))

    # Pick size-adapted config (engine extensions are always on); allow
    # external override via `config`.
    cfg = _config_for_size(int(data.n))
    if config:
        cfg.update(config)

    # 80/20 split — savings is the dominant engine on this instance set,
    # sweep's diversification is cheap insurance for clustered layouts.
    budget_savings = max(0.5, total * 0.80)
    budget_sweep = max(0.5, total * 0.20)

    best_routes, best_cost = None, float("inf")

    routes, cost = savings_solve(
        data, time_limit=budget_savings, seed=seed, config=cfg
    )
    if cost < best_cost:
        best_routes, best_cost = routes, cost

    routes, cost = sweep_solve(
        data, time_limit=budget_sweep, seed=seed + 1, config=cfg
    )
    if cost < best_cost:
        best_routes, best_cost = routes, cost

    return best_routes, best_cost
