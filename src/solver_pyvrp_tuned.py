"""
PyVRP backend with multi-seed pooling and lightly tuned solver parameters.

Wraps `solver_pyvrp.solve` and runs it with several seeds within the same
total budget, returning the best feasible solution observed. Also overrides a
handful of `pyvrp.SolveParams` knobs that the prior single-seed log suggests
have headroom on the bigger instances.

Like the other solver modules, exposes
    solve(data, time_limit=None, seed=0, config=None) -> (routes, cost)
"""

import time

from solver_common import DEFAULT_TIME_LIMIT, ensure_data, solution_cost
from solver_pyvrp import solve as base_solve


def _build_params(n):
    """Return a `pyvrp.SolveParams` lightly tuned for the instance size."""
    from pyvrp import IteratedLocalSearchParams
    from pyvrp.search import NeighbourhoodParams, PerturbationParams
    from pyvrp.solve import SolveParams

    # Larger instances benefit from more perturbations per cycle (more
    # diversity when the search settles); small instances don't need them.
    if n <= 50:
        max_pert = 15
        num_nb = 30
    elif n <= 150:
        max_pert = 25
        num_nb = 40
    elif n <= 250:
        max_pert = 35
        num_nb = 50
    else:
        max_pert = 50
        num_nb = 60

    return SolveParams(
        ils=IteratedLocalSearchParams(
            num_iters_no_improvement=150_000,
            history_length=300,
            exhaustive_on_best=True,
        ),
        neighbourhood=NeighbourhoodParams(num_neighbours=num_nb),
        perturbation=PerturbationParams(
            min_perturbations=1,
            max_perturbations=max_pert,
        ),
    )


def solve(data, time_limit=None, seed=0, config=None):
    data = ensure_data(data)
    tl = float(time_limit) if time_limit is not None else DEFAULT_TIME_LIMIT
    tl = max(0.3, tl)

    cfg = dict(config or {})
    seeds = list(cfg.pop("seeds", [seed, seed + 1, seed + 2]))
    if not seeds:
        seeds = [seed]

    # Build (and reuse) the tuned SolveParams once.
    params = cfg.get("params") or _build_params(data.n)
    cfg["params"] = params

    # Each seed gets a roughly equal share of the wall-clock. We track the
    # remaining budget and never let any seed exceed it.
    deadline = time.time() + tl
    best_routes, best_cost = None, float("inf")
    for s in seeds:
        remaining = deadline - time.time()
        if remaining <= 0.05:
            break
        # Give the current seed at most 1/N of the remaining budget but
        # always a small floor so it has a fair shot.
        share = max(0.5, remaining / max(1, len(seeds) - seeds.index(s)))
        share = min(share, remaining)
        try:
            routes, cost = base_solve(data, time_limit=share, seed=int(s), config=cfg)
        except Exception:
            continue
        if cost < best_cost:
            best_routes, best_cost = routes, cost

    if best_routes is None:
        # Should be unreachable since solver_pyvrp falls back to savings, but
        # guard against any unexpected shape.
        from solver_savings import solve as savings_solve
        return savings_solve(data, time_limit=tl, seed=seed)

    return best_routes, solution_cost(data, best_routes)
