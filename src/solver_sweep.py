import math
import random
import time
from solver_common import adaptive_time_limit, ensure_data, finish_with_ils, improve_routes, normalize_routes, solution_cost, cfg


def nearest_neighbor_order(data, customers):
    if not customers:
        return [0, 0]
    unvisited = set(customers)
    route = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: data.dist[cur][c])
        unvisited.remove(nxt)
        route.append(nxt)
        cur = nxt
    route.append(0)
    return route


def sweep_construct(data, offset=0.0, reverse=False, shuffle_ties=False, rng=None):
    cx, cy = data.x[0], data.y[0]
    keyed = []
    for i in range(1, data.n):
        angle = (math.atan2(data.y[i] - cy, data.x[i] - cx) + offset) % (2 * math.pi)
        radius = data.dist[0][i]
        tie = rng.random() if (shuffle_ties and rng is not None) else 0.0
        keyed.append((angle, radius, tie, i))
    keyed.sort()
    ordered = [i for _, _, _, i in keyed]
    if reverse:
        ordered.reverse()
    routes, cur, load = [], [], 0
    for c in ordered:
        dem = data.demand[c]
        if cur and load + dem > data.capacity:
            routes.append(nearest_neighbor_order(data, cur))
            cur, load = [], 0
        cur.append(c)
        load += dem
    if cur:
        routes.append(nearest_neighbor_order(data, cur))
    return normalize_routes(data, routes)


def default_offset_count(tl):
    if tl < 5: return 12
    if tl < 15: return 24
    if tl < 45: return 36
    if tl < 120: return 48
    return 72


def default_construction_fraction(tl):
    if tl < 10: return 0.30
    if tl < 30: return 0.35
    return 0.40


def solve(data, time_limit=None, seed=2, config=None):
    data = ensure_data(data)
    tl = adaptive_time_limit(data, time_limit)
    deadline = time.time() + tl
    offset_count = cfg(config, 'offset_count', default_offset_count(tl))
    construction_fraction = cfg(config, 'construction_fraction', default_construction_fraction(tl))
    construction_passes = cfg(config, 'construction_passes', 3 if tl < 30 else 4)
    elite_size = cfg(config, 'elite_size', 6)
    intensify_fraction = cfg(config, 'intensify_fraction', 0.20)
    use_reverse = cfg(config, 'use_reverse', True)
    use_shuffle_ties = cfg(config, 'use_shuffle_ties', True)
    construct_deadline = min(deadline, time.time() + max(0.01, tl * construction_fraction))
    master_rng = random.Random(seed)
    best, best_cost = None, float('inf')
    construction_pool = []
    offsets = [2 * math.pi * t / float(offset_count) for t in range(offset_count)]
    master_rng.shuffle(offsets)
    variants = []
    for off in offsets:
        variants.append((off, False, False))
        if use_reverse:
            variants.append((off, True, False))
        if use_shuffle_ties:
            variants.append((off, False, True))
    master_rng.shuffle(variants)
    for off, reverse, shuffle_ties in variants:
        if time.time() >= construct_deadline:
            break
        rng = random.Random(master_rng.randrange(10**9))
        routes = sweep_construct(data, off, reverse=reverse, shuffle_ties=shuffle_ties, rng=rng)
        routes = improve_routes(data, routes, deadline=construct_deadline, max_passes=construction_passes)
        cost = solution_cost(data, routes)
        construction_pool.append((cost, [r[:] for r in routes]))
        if cost < best_cost:
            best, best_cost = routes, cost
    if best is None:
        best = sweep_construct(data, 0.0)
        best_cost = solution_cost(data, best)
        construction_pool.append((best_cost, [r[:] for r in best]))
    construction_pool.sort(key=lambda x: x[0])
    elite = construction_pool[:max(1, min(elite_size, len(construction_pool)))]
    intensify_deadline = min(deadline, time.time() + max(0.01, intensify_fraction * tl))
    for idx, (_, routes0) in enumerate(elite):
        if time.time() >= intensify_deadline:
            break
        remaining = max(0.01, intensify_deadline - time.time())
        def restart_factory(rng):
            off2 = rng.choice(offsets)
            rev2 = use_reverse and (rng.random() < 0.5)
            shuf2 = use_shuffle_ties and (rng.random() < 0.5)
            return sweep_construct(data, off2, reverse=rev2, shuffle_ties=shuf2, rng=rng)
        cand, cc = finish_with_ils(data, [r[:] for r in routes0], time_limit=remaining, seed=seed + 1000 * (idx + 1), restart_factory=restart_factory, config=config)
        if cc < best_cost:
            best, best_cost = cand, cc
    remaining = max(0.01, deadline - time.time())
    def final_restart_factory(rng):
        off2 = rng.choice(offsets)
        rev2 = use_reverse and (rng.random() < 0.5)
        shuf2 = use_shuffle_ties and (rng.random() < 0.5)
        return sweep_construct(data, off2, reverse=rev2, shuffle_ties=shuf2, rng=rng)
    return finish_with_ils(data, best, time_limit=remaining, seed=seed + 777777, restart_factory=final_restart_factory, config=config)
