"""
Clarke‑Wright giant tour + Split DP partition + ILS.

The CVRP search landscape is much smoother when capacity is handled by a
dynamic program instead of being woven into the operators. The classic
trick (Beasley 1983; Vidal 2012) is:

  1. Maintain a **giant tour** — a permutation of all customers.
  2. Run **Split**: an O(n^2) DP that recovers the optimal capacity-feasible
     partition of the giant tour into routes.
  3. Local-search the giant tour with TSP moves (2-opt, Or-opt) and re-Split
     after each accepted move.

Operators don't have to track per-route loads; the DP finds the cheapest
partition for any tour, so capacity is automatic.

Exposes the standard `solve(data, time_limit, seed, config)` interface.
"""

import random
import time

from solver_common import (
    DEFAULT_TIME_LIMIT,
    cfg,
    ensure_data,
    normalize_routes,
    solution_cost,
)
from solver_savings import clarke_wright


def _giant_tour_from_routes(routes):
    """Concatenate routes (depot-stripped) into one customer permutation."""
    tour = []
    for r in routes:
        for c in r:
            if c != 0:
                tour.append(c)
    return tour


def _split_table(data, tour, max_routes=None):
    """
    Cardinality-constrained Split DP.

    Returns (cost, partition_indices) where the partition uses at most
    `max_routes` segments. If `max_routes` is None, no cardinality limit
    is imposed (classical Beasley split).

    cost is the total objective; partition_indices is a list of split
    boundaries [j0=0, j1, j2, ..., jk=n] so that segment t is tour[jt:jt+1].
    """
    n = len(tour)
    if n == 0:
        return 0.0, [0]
    INF = float("inf")
    cap = data.capacity
    d = data.dist
    dem = data.demand
    V = max_routes if max_routes is not None else n  # large enough that limit is non-binding

    # f[k][i] = optimal cost partitioning tour[:i] into exactly k routes.
    # pred[k][i] = j such that the k-th route is tour[j:i].
    # Build only as needed; rolling along k is enough but we keep all for backtracking.
    f = [[INF] * (n + 1) for _ in range(V + 1)]
    pred = [[-1] * (n + 1) for _ in range(V + 1)]
    f[0][0] = 0.0
    for k in range(1, V + 1):
        for i in range(1, n + 1):
            load = 0
            cost = 0.0
            prev = 0
            best = INF
            best_j = -1
            for j in range(i, 0, -1):
                c = tour[j - 1]
                load += dem[c]
                if load > cap:
                    break
                cost += d[prev][c]
                prev = c
                # Cost of this k-th route serving tour[j-1:i] (depot -> ... -> depot).
                route_cost = cost + d[c][0]
                if f[k - 1][j - 1] + route_cost < best:
                    best = f[k - 1][j - 1] + route_cost
                    best_j = j - 1
            f[k][i] = best
            pred[k][i] = best_j

    # Find the best k <= V that minimizes f[k][n].
    best_k = 1
    best_cost = INF
    for k in range(1, V + 1):
        if f[k][n] < best_cost:
            best_cost = f[k][n]
            best_k = k

    # Backtrack the partition.
    boundaries = [n]
    k = best_k
    i = n
    while k > 0 and i > 0:
        j = pred[k][i]
        boundaries.append(j)
        i = j
        k -= 1
    boundaries.reverse()
    return best_cost, boundaries


def split_dp(data, tour, max_routes=None):
    """Cardinality-constrained partitioning (returns route lists)."""
    if max_routes is None:
        max_routes = data.vehicle_count
    _, boundaries = _split_table(data, tour, max_routes=max_routes)
    if len(tour) == 0:
        return []
    return [[0] + tour[boundaries[i]:boundaries[i + 1]] + [0]
            for i in range(len(boundaries) - 1)]


def split_cost(data, tour, max_routes=None):
    """Cardinality-constrained partition cost (used as the LS objective)."""
    if max_routes is None:
        max_routes = data.vehicle_count
    cost, _ = _split_table(data, tour, max_routes=max_routes)
    return cost


def two_opt_giant(data, tour, deadline=None):
    """Standard 2-opt sweep on the giant tour, accepting first improvement."""
    d = data.dist
    n = len(tour)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for i in range(n - 1):
            for k in range(i + 2, n):
                a = tour[i - 1] if i > 0 else 0
                b = tour[i]
                c = tour[k]
                e = tour[k + 1] if k + 1 < n else 0
                if d[a][c] + d[b][e] + 1e-9 < d[a][b] + d[c][e]:
                    tour[i:k + 1] = reversed(tour[i:k + 1])
                    improved = True
                    break
            if improved:
                break
    return tour


def or_opt_giant(data, tour, deadline=None):
    """Relocate chains of length 1, 2, 3 anywhere along the giant tour."""
    d = data.dist
    n = len(tour)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for L in (1, 2, 3):
            for i in range(0, n - L + 1):
                seg = tour[i:i + L]
                a = tour[i - 1] if i > 0 else 0
                b = tour[i + L] if i + L < n else 0
                remove_delta = d[a][b] - d[a][seg[0]] - d[seg[-1]][b]
                for j in range(0, n - L + 1):
                    if j == i or (i <= j <= i + L):
                        continue
                    p = tour[j - 1] if j > 0 else 0
                    q = tour[j] if j < n else 0
                    insert_delta = d[p][seg[0]] + d[seg[-1]][q] - d[p][q]
                    if remove_delta + insert_delta < -1e-9:
                        # Apply
                        del tour[i:i + L]
                        if j > i:
                            j -= L
                        for k, x in enumerate(seg):
                            tour.insert(j + k, x)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return tour


def _improve_tour(data, tour, deadline=None, passes=10):
    """Alternate 2-opt and Or-opt; only commit passes that improve cardinality-
    constrained split cost. Reverts on regression (which can happen because
    TSP-style moves don't see the per-route depot legs)."""
    best_cost = split_cost(data, tour)
    best = list(tour)
    for _ in range(passes):
        if deadline is not None and time.time() >= deadline:
            break
        # Try a 2-opt then or-opt sweep on a working copy.
        work = list(best)
        two_opt_giant(data, work, deadline)
        or_opt_giant(data, work, deadline)
        cost = split_cost(data, work)
        if cost + 1e-9 < best_cost:
            best = work
            best_cost = cost
        else:
            break
    tour[:] = best
    return tour


def _perturb_tour(data, tour, rng, k):
    """Remove K consecutive customers, reinsert them at random positions."""
    n = len(tour)
    if n < 2:
        return tour
    k = min(k, n - 1)
    start = rng.randint(0, n - k)
    seg = tour[start:start + k]
    del tour[start:start + k]
    for c in seg:
        pos = rng.randint(0, len(tour))
        tour.insert(pos, c)
    return tour


def solve(data, time_limit=None, seed=0, config=None):
    data = ensure_data(data)
    tl = float(time_limit) if time_limit is not None else DEFAULT_TIME_LIMIT
    tl = max(0.1, tl)
    deadline = time.time() + tl

    rng = random.Random(seed)

    # Construction: build a CW solution, concatenate into a giant tour.
    init_routes = clarke_wright(data, rng)
    tour = _giant_tour_from_routes(init_routes)
    if not tour:  # All singletons fallback
        tour = list(range(1, data.n))

    # Initial polish.
    _improve_tour(data, tour, deadline=deadline, passes=20)

    best_tour = list(tour)
    best_cost = split_cost(data, best_tour)
    cur_tour = list(best_tour)
    cur_cost = best_cost

    # ILS over the giant tour.
    n = data.n
    base_remove = max(3, n // 20)
    max_remove = max(8, n // 6)
    no_improve = 0
    accept_prob = float(cfg(config, "accept_prob", 0.20))
    while time.time() < deadline:
        k = min(max_remove, base_remove + no_improve // 2)
        cand = list(cur_tour)
        cand = _perturb_tour(data, cand, rng, k)
        _improve_tour(data, cand, deadline=deadline, passes=5)
        cc = split_cost(data, cand)
        if cc + 1e-9 < best_cost:
            best_tour = list(cand)
            best_cost = cc
            cur_tour = list(cand)
            cur_cost = cc
            no_improve = 0
        else:
            no_improve += 1
            if rng.random() < accept_prob:
                cur_tour = list(cand)
                cur_cost = cc
        if no_improve >= 30:
            cur_tour = list(best_tour)
            cur_cost = best_cost
            no_improve = 0

    routes = split_dp(data, best_tour)
    routes = normalize_routes(data, routes)
    return routes, solution_cost(data, routes)
