"""
PyVRP backend — wraps Vidal's HGS-CVRP-derived Hybrid ILS engine.

Exposes a `solve(data, time_limit=None, seed=0, config=None)` entry point that
matches `solver_savings.solve` / `solver_sweep.solve`, so it can be plugged
into `src/main.py` or `src/main1.py` the same way.

Notes
-----
- PyVRP works in integer arithmetic. Coordinates, demands, and capacities are
  scaled by SCALE so capacity-tight instances (e.g. 16_5_1 at 94% utilisation)
  don't bottom out the integer penalty mechanism. We descale the cost on the
  way out.
- PyVRP picks how many vehicles to use; instances ask for a fixed fleet. We
  pad the route list up to `vehicle_count` with empty routes ([0, 0]) so the
  output matches the project's required format.
- `data` may be either a `VRPData` (from `solver_common.read_vrp`) or a
  `VRPInstance` (the existing project parser); `ensure_data()` normalises.
"""

import math
import time
import warnings

from solver_common import DEFAULT_TIME_LIMIT, ensure_data, normalize_routes, solution_cost

# Suppress PyVRP's PenaltyBoundWarning (we handle infeasibility ourselves)
try:
    from pyvrp.exceptions import PenaltyBoundWarning
    warnings.simplefilter("ignore", PenaltyBoundWarning)
except Exception:
    pass

SCALE = 1000  # PyVRP-recommended "exact" scaling factor


def _build_model(data):
    """Construct a pyvrp.Model from our VRPData representation."""
    from pyvrp import Model

    model = Model()
    depot = model.add_depot(
        x=int(round(data.x[0] * SCALE)),
        y=int(round(data.y[0] * SCALE)),
    )
    clients = []
    for i in range(1, data.n):
        clients.append(model.add_client(
            x=int(round(data.x[i] * SCALE)),
            y=int(round(data.y[i] * SCALE)),
            delivery=int(data.demand[i]) * SCALE,
        ))
    model.add_vehicle_type(
        num_available=int(data.vehicle_count),
        capacity=int(data.capacity) * SCALE,
        start_depot=depot,
        end_depot=depot,
    )

    # PyVRP's Model does not auto-derive a distance matrix from coordinates.
    # Without explicit edges, every distance defaults to MAX_VALUE (2^44),
    # which the solver flags as infeasible. Compute scaled Euclidean
    # distances and register an edge for every directed pair.
    locations = [depot] + clients
    for i, frm in enumerate(locations):
        for j, to in enumerate(locations):
            if i == j:
                continue
            d = math.hypot(frm.x - to.x, frm.y - to.y)
            model.add_edge(frm, to, distance=int(round(d)))
    return model


def _result_to_routes(data, result):
    """Translate a PyVRP Result into the project's [[0, ..., 0], ...] format."""
    routes = []
    for route in result.best.routes():
        custs = [int(c) for c in route.visits()]
        if custs:
            routes.append([0] + custs + [0])
    # Pad with empty [0, 0] routes to match the required fleet size.
    return normalize_routes(data, routes)


def solve(data, time_limit=None, seed=0, config=None):
    """
    Solve a CVRP instance using PyVRP's iterated local search engine.

    Parameters
    ----------
    data
        VRPData or VRPInstance (auto-detected via ensure_data).
    time_limit
        Wall-clock budget in seconds. Falls back to DEFAULT_TIME_LIMIT.
    seed
        RNG seed forwarded to PyVRP.
    config
        Optional dict; only `time_limit` and `seed` are read here, plus
        `display` (bool) for verbose output. Other keys are ignored — PyVRP
        has its own (rich) parameter surface; if you need it, pass a
        pyvrp.SolveParams via config['params'].

    Returns
    -------
    routes, cost : list[list[int]], float
    """
    from pyvrp.stop import MaxRuntime

    data = ensure_data(data)
    tl = float(time_limit) if time_limit is not None else DEFAULT_TIME_LIMIT
    tl = max(0.1, tl)

    display = bool(config.get("display", False)) if config else False
    params = (config or {}).get("params", None)

    model = _build_model(data)

    solve_kwargs = dict(stop=MaxRuntime(tl), seed=int(seed), display=display)
    if params is not None:
        solve_kwargs["params"] = params

    result = model.solve(**solve_kwargs)

    if not result.is_feasible():
        # Fall back to running the existing savings solver if PyVRP couldn't
        # find a feasible solution within the budget. This shouldn't happen
        # for well-posed instances at SCALE=1000, but we don't want to
        # silently emit garbage.
        from solver_savings import solve as savings_solve
        return savings_solve(data, time_limit=tl, seed=seed, config=None)

    routes = _result_to_routes(data, result)
    # Recompute cost in the project's (unscaled, float) metric.
    return routes, solution_cost(data, routes)
