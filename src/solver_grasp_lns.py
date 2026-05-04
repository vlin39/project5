"""
GRASP+LNS solver — pure Python.

Inspired by `src/vrpinstance_neighbors.py` on origin/main, but rebuilt to
share the post-fix `improve_routes` engine and to drop the unused
`self.nearest` array. No Cython dependency: the existing project's
`compile.sh` does not compile `vrputils.pyx`, so we keep everything in
Python.

Pipeline:
  1. GRASP construction: k-means clusters with k = vehicle_count, then
     biggest-demand-first insertion into the cheapest feasible cluster.
     Repeated K times to seed a population, with the heuristic randomized
     by sampling the assignment from the top-r choices each step.
  2. Local-search polish: every member of the population is improved with
     `solver_common.improve_routes`.
  3. LNS phase: the best M members are perturbed (random destroy + greedy
     regret reinsertion via the project's existing operators) and re-polished
     in a budget-limited loop with simulated-annealing-ish acceptance.

Exposes the standard `solve(data, time_limit, seed, config)` interface.
"""

import math
import random
import time

from solver_common import (
    DEFAULT_TIME_LIMIT,
    cfg,
    ensure_data,
    finish_with_ils,
    greedy_reinsert,
    improve_routes,
    normalize_routes,
    solution_cost,
)


def _kmeans(data, k, rng, iters=20):
    """Lightweight Lloyd's k-means over the customer coordinates.

    Pure Python (no scipy / sklearn dependency); fine at the project's
    instance sizes (≤ 386 customers) and called once per construction.
    """
    customers = list(range(1, data.n))
    if k >= len(customers):
        return [[c] for c in customers]
    # k-means++ seeding for stability.
    centroids = [(data.x[rng.choice(customers)], data.y[rng.choice(customers)])]
    while len(centroids) < k:
        dists = []
        for c in customers:
            best = min((data.x[c] - cx) ** 2 + (data.y[c] - cy) ** 2 for cx, cy in centroids)
            dists.append(best)
        total = sum(dists)
        if total <= 0:
            centroids.append((data.x[rng.choice(customers)], data.y[rng.choice(customers)]))
            continue
        pick = rng.uniform(0, total)
        cum = 0
        for c, dist in zip(customers, dists):
            cum += dist
            if cum >= pick:
                centroids.append((data.x[c], data.y[c]))
                break
    # Lloyd's iterations
    assignment = [0] * len(customers)
    for _ in range(iters):
        changed = False
        for ix, c in enumerate(customers):
            best_k, best_d = 0, float("inf")
            for kk, (cx, cy) in enumerate(centroids):
                dd = (data.x[c] - cx) ** 2 + (data.y[c] - cy) ** 2
                if dd < best_d:
                    best_d = dd
                    best_k = kk
            if assignment[ix] != best_k:
                changed = True
                assignment[ix] = best_k
        if not changed:
            break
        # Recompute centroids
        new_centroids = []
        for kk in range(k):
            members = [customers[ix] for ix, a in enumerate(assignment) if a == kk]
            if members:
                cx = sum(data.x[m] for m in members) / len(members)
                cy = sum(data.y[m] for m in members) / len(members)
            else:
                cx = centroids[kk][0]
                cy = centroids[kk][1]
            new_centroids.append((cx, cy))
        centroids = new_centroids
    clusters = [[] for _ in range(k)]
    for ix, a in enumerate(assignment):
        clusters[a].append(customers[ix])
    return clusters


def _grasp_construction(data, rng, top_r=3):
    """Build one feasible solution: cluster, then insert biggest-demand
    customers into the most attractive feasible cluster (top-r randomized)."""
    k = int(data.vehicle_count)
    clusters = _kmeans(data, k, rng)
    centroids = []
    for cl in clusters:
        if cl:
            cx = sum(data.x[c] for c in cl) / len(cl)
            cy = sum(data.y[c] for c in cl) / len(cl)
        else:
            cx, cy = data.x[0], data.y[0]
        centroids.append((cx, cy))

    routes = [[] for _ in range(k)]
    loads = [0] * k
    customers = sorted(range(1, data.n), key=lambda c: -data.demand[c])
    for c in customers:
        ranked = sorted(
            range(k),
            key=lambda j: (loads[j] + data.demand[c] > data.capacity,
                           (data.x[c] - centroids[j][0]) ** 2 + (data.y[c] - centroids[j][1]) ** 2),
        )
        # randomize: pick from feasible top-r
        feasible = [j for j in ranked if loads[j] + data.demand[c] <= data.capacity]
        if not feasible:
            # All clusters full: pick the least-overloaded; LNS phase will fix.
            j = ranked[0]
        else:
            j = rng.choice(feasible[:max(1, top_r)])
        routes[j].append(c)
        loads[j] += data.demand[c]

    out = [[0] + r + [0] for r in routes]
    return normalize_routes(data, out)


def solve(data, time_limit=None, seed=0, config=None):
    data = ensure_data(data)
    tl = float(time_limit) if time_limit is not None else DEFAULT_TIME_LIMIT
    tl = max(0.1, tl)
    deadline = time.time() + tl

    rng = random.Random(seed)

    pop_size = int(cfg(config, "pop_size", 30))
    elite_size = int(cfg(config, "elite_size", 5))
    construction_passes = int(cfg(config, "construction_passes", 2))

    construct_deadline = time.time() + max(0.1, tl * 0.30)

    population = []
    while time.time() < construct_deadline and len(population) < pop_size:
        cand = _grasp_construction(data, rng)
        cand = improve_routes(data, cand, deadline=construct_deadline,
                              max_passes=construction_passes, config=config)
        cost = solution_cost(data, cand)
        population.append((cost, cand))

    if not population:
        # Pathologically tight time budget: at least produce one solution.
        cand = _grasp_construction(data, rng)
        population.append((solution_cost(data, cand), cand))

    population.sort(key=lambda x: x[0])
    elite = [p for p in population[: max(1, min(elite_size, len(population)))]]

    best, best_cost = elite[0][1], elite[0][0]

    # LNS phase: run finish_with_ils on each elite, share the remaining budget.
    elite_count = len(elite)
    per_elite_deadline_step = max(0.1, (deadline - time.time()) / max(1, elite_count + 1))

    for _, routes0 in elite:
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        share = min(remaining, per_elite_deadline_step)

        def restart_factory(_rng, _data=data):
            return _grasp_construction(data, _rng)

        cand, cc = finish_with_ils(
            data,
            [r[:] for r in routes0],
            time_limit=share,
            seed=rng.randrange(10 ** 9),
            restart_factory=restart_factory,
            config=config,
        )
        if cc + 1e-9 < best_cost:
            best, best_cost = cand, cc

    # Final pass on the best with whatever budget is left.
    remaining = max(0.05, deadline - time.time())
    if remaining > 0.5:
        def restart_factory(_rng, _data=data):
            return _grasp_construction(data, _rng)

        cand, cc = finish_with_ils(
            data, best, time_limit=remaining,
            seed=rng.randrange(10 ** 9),
            restart_factory=restart_factory,
            config=config,
        )
        if cc + 1e-9 < best_cost:
            best, best_cost = cand, cc

    best = normalize_routes(data, best)
    return best, solution_cost(data, best)
