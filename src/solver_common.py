import math
import random
import time
from dataclasses import dataclass

DEFAULT_TIME_LIMIT = 295.0

@dataclass
class VRPData:
    filename: str
    n: int
    vehicle_count: int
    capacity: int
    demand: list
    x: list
    y: list
    dist: list


def read_vrp(path: str) -> VRPData:
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n, v, cap = map(int, lines[0].split()[:3])
    demand, xs, ys = [], [], []
    for ln in lines[1:1+n]:
        parts = ln.split()
        demand.append(int(float(parts[0])))
        xs.append(float(parts[1]))
        ys.append(float(parts[2]))
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            dist[i][j] = dist[j][i] = d
    return VRPData(path.split('/')[-1], n, v, cap, demand, xs, ys, dist)


def from_vrp_instance(instance) -> VRPData:
    n = int(instance.numCustomers)
    demand = [int(x) for x in instance.demandOfCustomer]
    xs = [float(x) for x in instance.xCoordOfCustomer]
    ys = [float(x) for x in instance.yCoordOfCustomer]
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            dist[i][j] = dist[j][i] = d
    return VRPData('', n, int(instance.numVehicles), int(instance.vehicleCapacity), demand, xs, ys, dist)


def ensure_data(obj) -> VRPData:
    if isinstance(obj, VRPData):
        return obj
    return from_vrp_instance(obj)


def cfg(config, key, default):
    return default if config is None or key not in config else config[key]


def route_cost(data, route):
    return sum(data.dist[route[i]][route[i + 1]] for i in range(len(route) - 1))


def solution_cost(data, routes):
    return sum(route_cost(data, r) for r in routes)


def route_load(data, route):
    return sum(data.demand[i] for i in route if i != 0)


def format_solution(routes):
    return ' '.join(str(x) for r in routes for x in r)


def adaptive_time_limit(data, requested=None):
    if requested is not None:
        return requested
    return DEFAULT_TIME_LIMIT


def greedy_insert_customer(data, routes, loads, c):
    best = None
    for ri, r in enumerate(routes):
        if loads[ri] + data.demand[c] <= data.capacity:
            for pos in range(1, len(r)):
                delta = data.dist[r[pos - 1]][c] + data.dist[c][r[pos]] - data.dist[r[pos - 1]][r[pos]]
                if best is None or delta < best[0]:
                    best = (delta, ri, pos)
    if best is None:
        return False
    _, ri, pos = best
    routes[ri].insert(pos, c)
    loads[ri] += data.demand[c]
    return True


def greedy_pack_all_customers(data, customers=None):
    if customers is None:
        customers = list(range(1, data.n))
    cx, cy = data.x[0], data.y[0]
    customers = sorted(set(customers), key=lambda c: (-data.demand[c], math.atan2(data.y[c] - cy, data.x[c] - cx)))
    routes = [[0, 0] for _ in range(data.vehicle_count)]
    loads = [0] * data.vehicle_count
    for c in customers:
        if not greedy_insert_customer(data, routes, loads, c):
            raise ValueError(f'Could not pack customer {c}; instance may be infeasible')
    return routes


def normalize_routes(data, routes):
    seen, cleaned = set(), []
    for r in routes:
        custs = []
        for c in r:
            if c != 0 and 1 <= c < data.n and c not in seen:
                custs.append(c)
                seen.add(c)
        if custs:
            cleaned.append([0] + custs + [0])
    missing = [c for c in range(1, data.n) if c not in seen]
    for c in missing:
        cleaned.append([0, c, 0])
    if any(route_load(data, r) > data.capacity for r in cleaned):
        cleaned = greedy_pack_all_customers(data)
    if len(cleaned) > data.vehicle_count:
        cleaned = greedy_pack_all_customers(data)
    while len(cleaned) < data.vehicle_count:
        cleaned.append([0, 0])
    return cleaned


def validate_solution(data, routes):
    if len(routes) != data.vehicle_count:
        return False
    customers = [c for r in routes for c in r if c != 0]
    return sorted(customers) == list(range(1, data.n)) and all(route_load(data, r) <= data.capacity for r in routes)


def two_opt_route(data, route, deadline=None):
    if len(route) <= 4:
        return False
    d = data.dist
    n = len(route)
    for i in range(1, n - 2):
        if deadline is not None and time.time() >= deadline:
            return False
        a, b = route[i - 1], route[i]
        for k in range(i + 1, n - 1):
            c, e = route[k], route[k + 1]
            if d[a][c] + d[b][e] + 1e-9 < d[a][b] + d[c][e]:
                route[i:k + 1] = reversed(route[i:k + 1])
                return True
    return False


def improve_routes(data, routes, deadline=None, max_passes=20):
    routes = normalize_routes(data, routes)
    loads = [route_load(data, r) for r in routes]
    d = data.dist
    passes = 0
    while passes < max_passes and (deadline is None or time.time() < deadline):
        passes += 1
        improved = False
        for r in routes:
            while deadline is None or time.time() < deadline:
                if not two_opt_route(data, r, deadline):
                    break
                improved = True
        if deadline is not None and time.time() >= deadline:
            break
        moved = False
        for ai, ra in enumerate(routes):
            if moved or (deadline is not None and time.time() >= deadline): break
            for pos in range(1, len(ra) - 1):
                if moved or (deadline is not None and time.time() >= deadline): break
                c = ra[pos]
                remove_delta = d[ra[pos - 1]][ra[pos + 1]] - d[ra[pos - 1]][c] - d[c][ra[pos + 1]]
                for bi, rb in enumerate(routes):
                    if moved or (deadline is not None and time.time() >= deadline): break
                    if ai != bi and loads[bi] + data.demand[c] > data.capacity:
                        continue
                    for ins in range(1, len(rb)):
                        if ai == bi and (ins == pos or ins == pos + 1):
                            continue
                        before, after = rb[ins - 1], rb[ins]
                        delta = remove_delta + d[before][c] + d[c][after] - d[before][after]
                        if delta < -1e-9:
                            ra.pop(pos)
                            if ai == bi and ins > pos:
                                ins -= 1
                            rb.insert(ins, c)
                            if ai != bi:
                                loads[ai] -= data.demand[c]
                                loads[bi] += data.demand[c]
                            moved = improved = True
                            break
        if deadline is not None and time.time() >= deadline:
            break
        swapped = False
        for ai in range(len(routes)):
            if swapped or (deadline is not None and time.time() >= deadline): break
            ra = routes[ai]
            for bi in range(ai + 1, len(routes)):
                if swapped or (deadline is not None and time.time() >= deadline): break
                rb = routes[bi]
                for pa in range(1, len(ra) - 1):
                    if swapped or (deadline is not None and time.time() >= deadline): break
                    ca = ra[pa]
                    for pb in range(1, len(rb) - 1):
                        cb = rb[pb]
                        if loads[ai] - data.demand[ca] + data.demand[cb] > data.capacity: continue
                        if loads[bi] - data.demand[cb] + data.demand[ca] > data.capacity: continue
                        old = d[ra[pa - 1]][ca] + d[ca][ra[pa + 1]] + d[rb[pb - 1]][cb] + d[cb][rb[pb + 1]]
                        new = d[ra[pa - 1]][cb] + d[cb][ra[pa + 1]] + d[rb[pb - 1]][ca] + d[ca][rb[pb + 1]]
                        if new + 1e-9 < old:
                            ra[pa], rb[pb] = cb, ca
                            loads[ai] += data.demand[cb] - data.demand[ca]
                            loads[bi] += data.demand[ca] - data.demand[cb]
                            swapped = improved = True
                            break
        if deadline is not None and time.time() >= deadline:
            break
        exchanged = False
        for ai in range(len(routes)):
            if exchanged or (deadline is not None and time.time() >= deadline): break
            for bi in range(ai + 1, len(routes)):
                if exchanged or (deadline is not None and time.time() >= deadline): break
                ra, rb = routes[ai], routes[bi]
                for pa in range(1, len(ra) - 1):
                    if exchanged or (deadline is not None and time.time() >= deadline): break
                    tail_a = loads[ai] - sum(data.demand[x] for x in ra[1:pa])
                    for pb in range(1, len(rb) - 1):
                        head_b = sum(data.demand[x] for x in rb[1:pb])
                        tail_b = loads[bi] - head_b
                        new_la = loads[ai] - tail_a + tail_b
                        new_lb = loads[bi] - tail_b + tail_a
                        if new_la > data.capacity or new_lb > data.capacity: continue
                        old = d[ra[pa - 1]][ra[pa]] + d[rb[pb - 1]][rb[pb]]
                        new = d[ra[pa - 1]][rb[pb]] + d[rb[pb - 1]][ra[pa]]
                        if new + 1e-9 < old:
                            routes[ai] = ra[:pa] + rb[pb:-1] + [0]
                            routes[bi] = rb[:pb] + ra[pa:-1] + [0]
                            loads[ai], loads[bi] = new_la, new_lb
                            exchanged = improved = True
                            break
        if not improved:
            break
    return normalize_routes(data, routes)


def random_route_segment_removal(data, routes, rng, k):
    routes = [r[:] for r in routes]
    nonempty = [ri for ri, r in enumerate(routes) if len(r) > 3]
    if not nonempty:
        return routes, []
    ri = rng.choice(nonempty)
    r = routes[ri]
    seg_len = min(k, len(r) - 2)
    start = rng.randint(1, len(r) - seg_len - 1)
    removed = r[start:start + seg_len]
    routes[ri] = r[:start] + r[start + seg_len:]
    if len(routes[ri]) < 2:
        routes[ri] = [0, 0]
    return normalize_routes(data, routes), removed


def relatedness_key(data, anchor, c):
    return data.dist[anchor][c] + 0.10 * abs(data.demand[anchor] - data.demand[c])


def shaw_removal(data, routes, rng, k):
    routes = [r[:] for r in routes]
    customers = [c for r in routes for c in r if c != 0]
    if not customers:
        return routes, []
    anchor = rng.choice(customers)
    related = sorted(customers, key=lambda c: relatedness_key(data, anchor, c))
    removed = set(related[:min(k, len(related))])
    partial = [[0] + [c for c in r if c != 0 and c not in removed] + [0] for r in routes]
    return normalize_routes(data, partial), list(removed)


def greedy_reinsert(data, partial_routes, removed, rng=None, randomize=False, top_k=4, regret=False):
    routes = normalize_routes(data, partial_routes)
    loads = [route_load(data, r) for r in routes]
    remaining = list(removed)
    if randomize and rng is not None:
        rng.shuffle(remaining)
    else:
        remaining.sort(key=lambda z: -data.demand[z])
    while remaining:
        if regret:
            best_pick = None
            for c in remaining:
                candidates = []
                for ri, r in enumerate(routes):
                    if loads[ri] + data.demand[c] > data.capacity:
                        continue
                    for pos in range(1, len(r)):
                        delta = data.dist[r[pos - 1]][c] + data.dist[c][r[pos]] - data.dist[r[pos - 1]][r[pos]]
                        candidates.append((delta, ri, pos))
                if not candidates:
                    routes = greedy_pack_all_customers(data)
                    return normalize_routes(data, routes)
                candidates.sort(key=lambda x: x[0])
                regret_value = candidates[1][0] - candidates[0][0] if len(candidates) > 1 else candidates[0][0]
                score = (regret_value, -candidates[0][0], c)
                if best_pick is None or score > best_pick[0]:
                    best_pick = (score, c, candidates)
            _, c, candidates = best_pick
            if randomize and rng is not None:
                _, ri, pos = rng.choice(candidates[:min(top_k, len(candidates))])
            else:
                _, ri, pos = candidates[0]
            routes[ri].insert(pos, c)
            loads[ri] += data.demand[c]
            remaining.remove(c)
        else:
            c = remaining.pop()
            candidates = []
            for ri, r in enumerate(routes):
                if loads[ri] + data.demand[c] > data.capacity:
                    continue
                for pos in range(1, len(r)):
                    delta = data.dist[r[pos - 1]][c] + data.dist[c][r[pos]] - data.dist[r[pos - 1]][r[pos]]
                    candidates.append((delta, ri, pos))
            if not candidates:
                routes = greedy_pack_all_customers(data)
                break
            candidates.sort(key=lambda x: x[0])
            if randomize and rng is not None:
                _, ri, pos = rng.choice(candidates[:min(top_k, len(candidates))])
            else:
                _, ri, pos = candidates[0]
            routes[ri].insert(pos, c)
            loads[ri] += data.demand[c]
    return normalize_routes(data, routes)


def perturb(data, routes, rng, remove_count=5, mode=None, top_k=5):
    routes = [r[:] for r in routes]
    customers = [c for r in routes for c in r if c != 0]
    if not customers:
        return routes
    k = min(max(1, int(remove_count)), len(customers))
    if mode is None:
        mode = rng.choice(['random', 'route', 'shaw'])
    if mode == 'route':
        partial, removed = random_route_segment_removal(data, routes, rng, k)
    elif mode == 'shaw':
        partial, removed = shaw_removal(data, routes, rng, k)
    else:
        removed = set(rng.sample(customers, k))
        partial = [[0] + [c for c in r if c != 0 and c not in removed] + [0] for r in routes]
        partial = normalize_routes(data, partial)
        removed = list(removed)
    return greedy_reinsert(data, partial, removed, rng=rng, randomize=True, top_k=top_k, regret=True)


def ils_profile(data, time_limit, config=None):
    tl = max(0.01, float(time_limit))
    n = data.n
    if tl < 5:
        profile = {'base_remove': max(3, n // 30), 'max_remove': max(8, n // 10), 'top_k': 4, 'ls_passes': 2, 'restart_trigger': 10, 'accept_prob': 0.25}
    elif tl < 20:
        profile = {'base_remove': max(4, n // 25), 'max_remove': max(12, n // 8), 'top_k': 6, 'ls_passes': 3, 'restart_trigger': 12, 'accept_prob': 0.25}
    elif tl < 60:
        profile = {'base_remove': max(5, n // 20), 'max_remove': max(16, n // 6), 'top_k': 8, 'ls_passes': 4, 'restart_trigger': 14, 'accept_prob': 0.25}
    else:
        profile = {'base_remove': max(6, n // 18), 'max_remove': max(24, n // 5), 'top_k': 10, 'ls_passes': 5, 'restart_trigger': 16, 'accept_prob': 0.25}
    if config is not None:
        for key in ('base_remove', 'max_remove', 'top_k', 'ls_passes', 'restart_trigger', 'accept_prob'):
            if key in config:
                profile[key] = config[key]
    return profile


def finish_with_ils(data, initial_routes, time_limit=DEFAULT_TIME_LIMIT, seed=0, restart_factory=None, config=None):
    deadline = time.time() + max(0.01, float(time_limit))
    rng = random.Random(seed)
    profile = ils_profile(data, time_limit, config)
    current = improve_routes(data, initial_routes, deadline, max_passes=cfg(config, 'initial_ls_passes', max(8, profile['ls_passes'] + 2)))
    best = [r[:] for r in current]
    best_cost = solution_cost(data, best)
    no_improve = 0
    while time.time() < deadline:
        strength = min(profile['max_remove'], profile['base_remove'] + no_improve // 2)
        cand = perturb(data, best if no_improve > profile['restart_trigger'] else current, rng, remove_count=strength, mode=cfg(config, 'perturb_mode', None), top_k=profile['top_k'])
        cand = improve_routes(data, cand, deadline, max_passes=profile['ls_passes'])
        cc = solution_cost(data, cand)
        if cc + 1e-9 < best_cost:
            best = [r[:] for r in cand]
            best_cost = cc
            current = cand
            no_improve = 0
        else:
            no_improve += 1
            if rng.random() < profile['accept_prob']:
                current = cand
        if restart_factory is not None and no_improve >= 2 * profile['restart_trigger'] and time.time() < deadline:
            fresh = restart_factory(rng)
            fresh = improve_routes(data, fresh, deadline, max_passes=min(3, profile['ls_passes']))
            fc = solution_cost(data, fresh)
            current = fresh
            if fc + 1e-9 < best_cost:
                best = [r[:] for r in fresh]
                best_cost = fc
                no_improve = 0
            else:
                no_improve = max(0, no_improve // 2)
    best = normalize_routes(data, best)
    return best, solution_cost(data, best)
