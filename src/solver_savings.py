import random
import time
from solver_common import adaptive_time_limit, ensure_data, finish_with_ils, improve_routes, normalize_routes, solution_cost, cfg


def clarke_wright(data, rng=None, noise=0.0):
    rng = rng or random.Random(0)
    routes = {i: [0, i, 0] for i in range(1, data.n)}
    route_of = {i: i for i in range(1, data.n)}
    loads = {i: data.demand[i] for i in range(1, data.n)}
    savings = []
    for i in range(1, data.n):
        for j in range(i + 1, data.n):
            s = data.dist[0][i] + data.dist[0][j] - data.dist[i][j]
            if noise:
                s *= 1.0 + rng.uniform(-noise, noise)
            savings.append((s, i, j))
    savings.sort(reverse=True)
    for _, i, j in savings:
        ri, rj = route_of.get(i), route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        a, b = routes[ri], routes[rj]
        if loads[ri] + loads[rj] > data.capacity:
            continue
        if a[1] == i:
            a = [0] + list(reversed(a[1:-1])) + [0]
        if b[-2] == j:
            b = [0] + list(reversed(b[1:-1])) + [0]
        if a[-2] != i or b[1] != j:
            continue
        merged = a[:-1] + b[1:]
        routes[ri] = merged
        loads[ri] += loads[rj]
        for c in b[1:-1]:
            route_of[c] = ri
        del routes[rj]
        del loads[rj]
    return normalize_routes(data, list(routes.values()))


def default_seed_count(tl):
    if tl < 5: return 2
    if tl < 15: return 4
    if tl < 45: return 7
    if tl < 120: return 10
    return 14


def default_noise_levels(tl):
    if tl < 8:
        return (0.0, 0.03, 0.07)
    if tl < 30:
        return (0.0, 0.01, 0.03, 0.07, 0.12, 0.20)
    return (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30)


def default_construction_fraction(tl):
    if tl < 10: return 0.35
    if tl < 30: return 0.40
    return 0.45


def solve(data, time_limit=None, seed=1, config=None):
    data = ensure_data(data)
    tl = adaptive_time_limit(data, time_limit)
    deadline = time.time() + tl
    construction_fraction = cfg(config, 'construction_fraction', default_construction_fraction(tl))
    seed_count = cfg(config, 'seed_count', default_seed_count(tl))
    noise_levels = tuple(cfg(config, 'noise_levels', default_noise_levels(tl)))
    construction_passes = cfg(config, 'construction_passes', 2)
    elite_size = cfg(config, 'elite_size', 6)
    intensify_fraction = cfg(config, 'intensify_fraction', 0.20)
    construct_deadline = min(deadline, time.time() + max(0.01, tl * construction_fraction))
    master_rng = random.Random(seed)
    best, best_cost = None, float('inf')
    construction_pool = []
    for _ in range(seed_count):
        if time.time() >= construct_deadline:
            break
        restart_seed = master_rng.randrange(10**9)
        rng = random.Random(restart_seed)
        local_noises = list(noise_levels)
        rng.shuffle(local_noises)
        for noise in local_noises:
            if time.time() >= construct_deadline:
                break
            init = clarke_wright(data, rng, noise=noise)
            init = improve_routes(data, init, deadline=construct_deadline, max_passes=construction_passes)
            c = solution_cost(data, init)
            construction_pool.append((c, [r[:] for r in init], restart_seed))
            if c < best_cost:
                best, best_cost = init, c
    if best is None:
        best = clarke_wright(data, random.Random(seed), noise=0.0)
        best_cost = solution_cost(data, best)
        construction_pool.append((best_cost, [r[:] for r in best], seed))
    construction_pool.sort(key=lambda x: x[0])
    elite = construction_pool[:max(1, min(elite_size, len(construction_pool)))]
    intensify_deadline = min(deadline, time.time() + max(0.01, intensify_fraction * tl))
    for idx, (_, routes0, restart_seed) in enumerate(elite):
        if time.time() >= intensify_deadline:
            break
        remaining = max(0.01, intensify_deadline - time.time())
        def restart_factory(rng):
            seed2 = rng.randrange(10**9)
            noise2 = rng.choice(noise_levels)
            return clarke_wright(data, random.Random(seed2), noise=noise2)
        cand, cc = finish_with_ils(data, [r[:] for r in routes0], time_limit=remaining, seed=restart_seed + 1000 * (idx + 1), restart_factory=restart_factory, config=config)
        if cc < best_cost:
            best, best_cost = cand, cc
    remaining = max(0.01, deadline - time.time())
    def final_restart_factory(rng):
        seed2 = rng.randrange(10**9)
        noise2 = rng.choice(noise_levels)
        return clarke_wright(data, random.Random(seed2), noise=noise2)
    final_routes, final_cost = finish_with_ils(data, best, time_limit=remaining, seed=seed + 424242, restart_factory=final_restart_factory, config=config)
    if final_cost + 1e-9 < best_cost:
        return final_routes, final_cost
    return best, best_cost
