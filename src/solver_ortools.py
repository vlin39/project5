"""
Google OR-Tools backend for CVRP using the routing-specific facade
(`pywrapcp.RoutingModel`).

Style mirrors the imperative classical-CP example referenced by the user
(github.com/sidprasad/prescriptive-analystics-employee-scheduling): build a
solver/model, declare variables and constraints, hand it a search strategy,
solve, then extract the solution. The routing facade collapses the variable
declaration and CVRP-specific constraints into one call set; the rest of the
flow is the same.

Exposes
    solve(data, time_limit=None, seed=0, config=None) -> (routes, cost)

Two implicit modes by instance size:
- n <= 50: TABU_SEARCH metaheuristic — converges to (often optimal-quality)
  solutions quickly on the small instances.
- n  > 50: GUIDED_LOCAL_SEARCH — the routing solver's strongest large-n LS.
The choice can be overridden via `config['metaheuristic']`.

Distances are integer Euclidean lengths scaled by SCALE so OR-Tools' integer
arc-cost interaction matches the project's float-distance objective when
cost is unscaled.
"""

import math

from solver_common import DEFAULT_TIME_LIMIT, ensure_data, normalize_routes, solution_cost

SCALE = 1000


def _build_distance_matrix(data):
    n = data.n
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = data.x[i], data.y[i]
        for j in range(i + 1, n):
            d = int(round(math.hypot(xi - data.x[j], yi - data.y[j]) * SCALE))
            matrix[i][j] = matrix[j][i] = d
    return matrix


def _meta_for(n, override=None):
    """Pick the local-search metaheuristic by instance size (or honor override)."""
    from ortools.constraint_solver.routing_enums_pb2 import LocalSearchMetaheuristic as M
    if override:
        return getattr(M, override.upper())
    # GLS is the routing solver's strongest general-purpose metaheuristic for
    # CVRP; it works well across all instance sizes in this folder.
    return M.GUIDED_LOCAL_SEARCH


def _first_solution_for(n, override=None):
    from ortools.constraint_solver.routing_enums_pb2 import FirstSolutionStrategy as F
    if override:
        return getattr(F, override.upper())
    return F.PATH_CHEAPEST_ARC


def solve(data, time_limit=None, seed=0, config=None):
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver.routing_enums_pb2 import LocalSearchMetaheuristic

    data = ensure_data(data)
    tl = float(time_limit) if time_limit is not None else DEFAULT_TIME_LIMIT
    tl = max(0.1, tl)

    cfg = config or {}
    meta = _meta_for(data.n, cfg.get("metaheuristic"))
    first_solution = _first_solution_for(data.n, cfg.get("first_solution"))

    n = data.n
    distance_matrix = _build_distance_matrix(data)

    # Routing manager: n nodes, vehicle_count vehicles, depot at index 0.
    manager = pywrapcp.RoutingIndexManager(n, int(data.vehicle_count), 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return distance_matrix[i][j]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Capacity dimension: each vehicle has the same capacity in this project.
    def demand_callback(from_index):
        return int(data.demand[manager.IndexToNode(from_index)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,                                    # null capacity slack
        [int(data.capacity)] * int(data.vehicle_count),
        True,                                 # start cumul to zero
        "Capacity",
    )

    # Search parameters: pick a starting heuristic + a local-search engine,
    # let the solver run until the time limit is hit.
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = first_solution
    search_params.local_search_metaheuristic = meta
    # `time_limit.FromMilliseconds` is the canonical way to set a sub-second
    # budget; the routing solver doesn't expose a public RNG-seed knob.
    search_params.time_limit.FromMilliseconds(max(100, int(round(tl * 1000))))

    assignment = routing.SolveWithParameters(search_params)
    if assignment is None:
        # Should be rare: we let it run to time, so worst case is a poor but
        # feasible solution. Fall back to the in-house savings solver.
        from solver_savings import solve as savings_solve
        return savings_solve(data, time_limit=tl, seed=seed)

    # Extract routes.
    routes = []
    for vehicle_id in range(int(data.vehicle_count)):
        index = routing.Start(vehicle_id)
        route = [0]
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(node)
            index = assignment.Value(routing.NextVar(index))
        route.append(0)
        if len(route) > 2:
            routes.append(route)

    routes = normalize_routes(data, routes)
    return routes, solution_cost(data, routes)
