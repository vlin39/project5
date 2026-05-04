"""
Pure-Python portfolio CVRP solver — no external VRP libraries.

The production solver for this project. The "external library" prohibition
rules out PyVRP and OR-Tools; classic algorithms (Split DP, GRASP+LNS,
KNN candidate lists, etc.) are fair game and are used heavily.

Construction-family portfolio
-----------------------------
  1. Clarke-Wright savings + ILS  (solver_savings.solve)
  2. GRASP construction + LNS     (solver_grasp_lns.solve)
       — k-means clusters, biggest-demand-first insertion, then LNS
  3. Angular sweep + ILS          (solver_sweep.solve)
  4. CW giant-tour + Split DP     (solver_split.solve)
       — only included on instances with loose capacity headroom; on
       capacity-tight instances the cardinality-constrained DP collapses
       to the construction baseline so it is excluded by default.

All four engines run the same enhanced ILS through `solver_common`, which
has every B-series enhancement enabled here:

  B1. Or-opt   — relocate segments of length 2 and 3
  B2. 2-opt*   — true cross-route 2-opt edge exchange
  B3. KNN candidate lists — top-K nearest-neighbor pruning of cross-route
                             insertions
  B4. Don't-look bits     — skip customers whose neighborhood was scanned
                             without improvement until a neighbor changes
  B5. SA acceptance       — temperature-cooled acceptance in finish_with_ils
  B6. SISR (string removal) — geographic-string destroy operator alongside
                              random / route / shaw

This module wires the B1-B6 toggles, picks a per-instance time budget,
allocates that budget across the engines (heaviest weight on savings — the
empirical winner on this project's instance set, with smaller weights on
the diversifier engines), runs them in sequence, and returns the best.

`solver_pyvrp_reference.py` is a separate file that wraps PyVRP and is
explicitly NOT imported here; it is kept in the tree only as a baseline
for diagnostic comparisons.
"""

from solver_common import ensure_data, DEFAULT_TIME_LIMIT
from solver_savings import solve as savings_solve
from solver_sweep import solve as sweep_solve
from solver_grasp_lns import solve as grasp_lns_solve
from solver_split import solve as split_solve


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


# Capacity headroom threshold below which Split DP is included in the
# portfolio. Above this (i.e. instances where total demand uses ≤ this
# fraction of fleet capacity) Split DP can move tail segments freely; below
# this it tends to revert to the construction seed and waste budget.
_SPLIT_HEADROOM_THRESHOLD = 0.85


def _engines_for(data):
    """Return [(label, solve_fn, weight), ...] for this instance.

    Split DP is gated on capacity utilisation so we don't waste budget on
    instances where it can't improve. Savings, sweep, and GRASP+LNS are
    always in the portfolio.
    """
    total_demand = sum(int(d) for d in data.demand[1:])
    fleet_cap = int(data.capacity) * int(data.vehicle_count)
    util = total_demand / fleet_cap if fleet_cap > 0 else 1.0

    engines = [
        ("savings",   savings_solve,   55),
        ("grasp_lns", grasp_lns_solve, 25),
        ("sweep",     sweep_solve,     12),
    ]
    if util <= _SPLIT_HEADROOM_THRESHOLD:
        engines.append(("split", split_solve, 8))
    return engines


def solve(data, time_limit=None, seed=0, config=None):
    """
    Solve a CVRP instance with a pure-Python construction-family portfolio.

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
    base_cfg = _config_for_size(int(data.n))
    if config:
        base_cfg.update(config)

    engines = _engines_for(data)
    weight_sum = sum(w for _, _, w in engines)

    best_routes, best_cost = None, float("inf")

    for i, (label, fn, weight) in enumerate(engines):
        budget = max(0.5, total * weight / weight_sum)
        try:
            routes, cost = fn(data, time_limit=budget, seed=seed + i, config=base_cfg)
        except Exception:
            # Any single engine failure must not take down the portfolio;
            # the others still produce valid solutions.
            continue
        if cost < best_cost:
            best_routes, best_cost = routes, cost

    return best_routes, best_cost
