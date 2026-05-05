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


def format_solution(routes, optimality_flag=0):
    """Flatten routes into the project's wire format.

    Output is "<flag> <r0_node0> ... <rN-1_nodeM>" where the leading flag
    is the proven-optimal indicator (0 = not proved, 1 = proved). We never
    prove optimality, so callers should leave the default of 0.
    """
    body = ' '.join(str(x) for r in routes for x in r)
    return f"{int(optimality_flag)} {body}"


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


def compute_neighbors(data, k):
    """Return per-customer K-nearest-neighbor lists, cached on `data`.

    `data._neighbors[c]` is a list of customer indices (excluding 0 and c)
    sorted by Euclidean distance from c, truncated to K.
    """
    cached = getattr(data, '_neighbors', None)
    cached_k = getattr(data, '_neighbors_k', None)
    if cached is not None and cached_k == k:
        return cached
    n = data.n
    d = data.dist
    nbrs = [None] * n
    others = list(range(1, n))
    for c in range(1, n):
        nbrs[c] = sorted((j for j in others if j != c), key=lambda j: d[c][j])[:k]
    data._neighbors = nbrs
    data._neighbors_k = k
    return nbrs


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


def improve_routes(data, routes, deadline=None, max_passes=100, config=None):
    """Apply 2-opt, relocate, swap, and segment exchange to local optimum.

    Optional moves controlled by `config`:
      - or_opt              (B1): relocate segments of length 2 / 3
      - two_opt_star        (B2): cross-route 2-opt edge exchange
      - candidate_list_size (B3): if > 0, restrict cross-route insertions to
                                  positions adjacent to one of c's K nearest
                                  neighbors
      - dont_look           (B4): skip customers whose neighborhood was
                                  scanned without improvement until a neighbor
                                  changes
    """
    routes = normalize_routes(data, routes)
    loads = [route_load(data, r) for r in routes]
    d = data.dist
    demand = data.demand
    capacity = data.capacity

    or_opt_enabled = bool(cfg(config, 'or_opt', False))
    two_opt_star_enabled = bool(cfg(config, 'two_opt_star', False))
    cl_size = int(cfg(config, 'candidate_list_size', 0) or 0)
    dont_look = bool(cfg(config, 'dont_look', False))

    neighbors = compute_neighbors(data, cl_size) if cl_size > 0 else None
    look = [True] * data.n if dont_look else None

    def _is_target(c, before, after):
        """Candidate-list filter: insertion is only considered if `before` or
        `after` is one of c's K nearest neighbors."""
        if neighbors is None:
            return True
        targets = neighbors[c]
        return before in targets or after in targets

    def _wake(c):
        """Don't-look-bits: mark c (and its neighbors) as worth re-scanning."""
        if look is None:
            return
        look[c] = True
        if neighbors is not None:
            for nb in neighbors[c]:
                look[nb] = True

    passes = 0
    while passes < max_passes and (deadline is None or time.time() < deadline):
        passes += 1
        improved = False

        # In-route 2-opt — drained for each route
        for r in routes:
            while deadline is None or time.time() < deadline:
                if not two_opt_route(data, r, deadline):
                    break
                improved = True
        if deadline is not None and time.time() >= deadline:
            break

        # Cross-route relocate — keep finding improving moves until none found this pass
        local_improved = True
        while local_improved and (deadline is None or time.time() < deadline):
            local_improved = False
            for ai in range(len(routes)):
                if deadline is not None and time.time() >= deadline:
                    break
                ra = routes[ai]
                pos = 1
                while pos < len(ra) - 1:
                    if deadline is not None and time.time() >= deadline:
                        break
                    c = ra[pos]
                    if look is not None and not look[c]:
                        pos += 1
                        continue
                    remove_delta = d[ra[pos - 1]][ra[pos + 1]] - d[ra[pos - 1]][c] - d[c][ra[pos + 1]]
                    moved_here = False
                    for bi in range(len(routes)):
                        rb = routes[bi]
                        if ai != bi and loads[bi] + demand[c] > capacity:
                            continue
                        for ins in range(1, len(rb)):
                            if ai == bi and (ins == pos or ins == pos + 1):
                                continue
                            before, after = rb[ins - 1], rb[ins]
                            if not _is_target(c, before, after):
                                continue
                            delta = remove_delta + d[before][c] + d[c][after] - d[before][after]
                            if delta < -1e-9:
                                ra.pop(pos)
                                if ai == bi and ins > pos:
                                    ins -= 1
                                rb.insert(ins, c)
                                if ai != bi:
                                    loads[ai] -= demand[c]
                                    loads[bi] += demand[c]
                                local_improved = improved = True
                                moved_here = True
                                _wake(c)
                                break
                        if moved_here:
                            break
                    if not moved_here:
                        if look is not None:
                            look[c] = False
                        pos += 1
                    # else: customer at pos changed; re-test same pos
        if deadline is not None and time.time() >= deadline:
            break

        # Cross-route swap — multi-improvement per pass
        local_improved = True
        while local_improved and (deadline is None or time.time() < deadline):
            local_improved = False
            for ai in range(len(routes)):
                if deadline is not None and time.time() >= deadline:
                    break
                ra = routes[ai]
                for bi in range(ai + 1, len(routes)):
                    if deadline is not None and time.time() >= deadline:
                        break
                    rb = routes[bi]
                    pa = 1
                    while pa < len(ra) - 1:
                        if deadline is not None and time.time() >= deadline:
                            break
                        ca = ra[pa]
                        swapped_here = False
                        for pb in range(1, len(rb) - 1):
                            cb = rb[pb]
                            if neighbors is not None and cb not in neighbors[ca] and ca not in neighbors[cb]:
                                continue
                            if loads[ai] - demand[ca] + demand[cb] > capacity:
                                continue
                            if loads[bi] - demand[cb] + demand[ca] > capacity:
                                continue
                            old = d[ra[pa - 1]][ca] + d[ca][ra[pa + 1]] + d[rb[pb - 1]][cb] + d[cb][rb[pb + 1]]
                            new = d[ra[pa - 1]][cb] + d[cb][ra[pa + 1]] + d[rb[pb - 1]][ca] + d[ca][rb[pb + 1]]
                            if new + 1e-9 < old:
                                ra[pa], rb[pb] = cb, ca
                                loads[ai] += demand[cb] - demand[ca]
                                loads[bi] += demand[ca] - demand[cb]
                                local_improved = improved = True
                                swapped_here = True
                                _wake(ca); _wake(cb)
                                break
                        if not swapped_here:
                            pa += 1
                        # else: re-test pa with newly swapped-in customer
        if deadline is not None and time.time() >= deadline:
            break

        # Cross-route segment exchange — multi-improvement per pass
        local_improved = True
        while local_improved and (deadline is None or time.time() < deadline):
            local_improved = False
            for ai in range(len(routes)):
                if deadline is not None and time.time() >= deadline:
                    break
                for bi in range(ai + 1, len(routes)):
                    if deadline is not None and time.time() >= deadline:
                        break
                    ra, rb = routes[ai], routes[bi]
                    pa = 1
                    while pa < len(ra) - 1:
                        if deadline is not None and time.time() >= deadline:
                            break
                        tail_a = loads[ai] - sum(demand[x] for x in ra[1:pa])
                        exchanged_here = False
                        for pb in range(1, len(rb) - 1):
                            head_b = sum(demand[x] for x in rb[1:pb])
                            tail_b = loads[bi] - head_b
                            new_la = loads[ai] - tail_a + tail_b
                            new_lb = loads[bi] - tail_b + tail_a
                            if new_la > capacity or new_lb > capacity:
                                continue
                            old = d[ra[pa - 1]][ra[pa]] + d[rb[pb - 1]][rb[pb]]
                            new = d[ra[pa - 1]][rb[pb]] + d[rb[pb - 1]][ra[pa]]
                            if new + 1e-9 < old:
                                routes[ai] = ra[:pa] + rb[pb:-1] + [0]
                                routes[bi] = rb[:pb] + ra[pa:-1] + [0]
                                ra, rb = routes[ai], routes[bi]
                                loads[ai], loads[bi] = new_la, new_lb
                                local_improved = improved = True
                                exchanged_here = True
                                break
                        if exchanged_here:
                            pa = 1  # routes mutated — restart the scan
                        else:
                            pa += 1
        if deadline is not None and time.time() >= deadline:
            break

        # B1: Or-opt — relocate chains of 2 or 3 consecutive customers.
        if or_opt_enabled:
            local_improved = True
            while local_improved and (deadline is None or time.time() < deadline):
                local_improved = False
                for ai in range(len(routes)):
                    if deadline is not None and time.time() >= deadline:
                        break
                    ra = routes[ai]
                    for L in (2, 3):
                        pos = 1
                        while pos + L <= len(ra) - 1:
                            if deadline is not None and time.time() >= deadline:
                                break
                            seg = ra[pos:pos + L]
                            seg_demand = sum(demand[c] for c in seg)
                            seg_cost = sum(d[seg[i]][seg[i + 1]] for i in range(L - 1))
                            remove_delta = d[ra[pos - 1]][ra[pos + L]] - d[ra[pos - 1]][seg[0]] - d[seg[-1]][ra[pos + L]]
                            moved_here = False
                            for bi in range(len(routes)):
                                rb = routes[bi]
                                if ai != bi and loads[bi] + seg_demand > capacity:
                                    continue
                                for ins in range(1, len(rb)):
                                    # Skip insertion sites overlapping the source segment.
                                    if ai == bi and pos <= ins <= pos + L:
                                        continue
                                    before, after = rb[ins - 1], rb[ins]
                                    if not _is_target(seg[0], before, after):
                                        continue
                                    delta = (remove_delta
                                             + d[before][seg[0]] + d[seg[-1]][after]
                                             - d[before][after])
                                    if delta < -1e-9:
                                        # Apply: remove seg from ra, insert into rb at ins.
                                        del ra[pos:pos + L]
                                        if ai == bi and ins > pos:
                                            ins -= L
                                        for k, x in enumerate(seg):
                                            rb.insert(ins + k, x)
                                        if ai != bi:
                                            loads[ai] -= seg_demand
                                            loads[bi] += seg_demand
                                        local_improved = improved = moved_here = True
                                        for x in seg:
                                            _wake(x)
                                        break
                                if moved_here:
                                    break
                            if not moved_here:
                                pos += 1
                            # else: stay at pos; ra has shifted
            if deadline is not None and time.time() >= deadline:
                break

        # B2: 2-opt* — cross-route edge exchange. For two routes A=...a-b... and
        # B=...c-d... try the reconnection (a,d)+(c,b) which exchanges the
        # tails after positions pa and pb.
        if two_opt_star_enabled:
            local_improved = True
            while local_improved and (deadline is None or time.time() < deadline):
                local_improved = False
                for ai in range(len(routes)):
                    if deadline is not None and time.time() >= deadline:
                        break
                    for bi in range(ai + 1, len(routes)):
                        if deadline is not None and time.time() >= deadline:
                            break
                        ra, rb = routes[ai], routes[bi]
                        pa = 0
                        while pa < len(ra) - 1:
                            if deadline is not None and time.time() >= deadline:
                                break
                            a, b = ra[pa], ra[pa + 1]
                            head_a = sum(demand[x] for x in ra[1:pa + 1])
                            tail_a_load = loads[ai] - head_a
                            exchanged_here = False
                            for pb in range(0, len(rb) - 1):
                                cnode, dnode = rb[pb], rb[pb + 1]
                                head_b = sum(demand[x] for x in rb[1:pb + 1])
                                tail_b_load = loads[bi] - head_b
                                new_la = head_a + tail_b_load
                                new_lb = head_b + tail_a_load
                                if new_la > capacity or new_lb > capacity:
                                    continue
                                old = d[a][b] + d[cnode][dnode]
                                new = d[a][dnode] + d[cnode][b]
                                if new + 1e-9 < old:
                                    routes[ai] = ra[:pa + 1] + rb[pb + 1:]
                                    routes[bi] = rb[:pb + 1] + ra[pa + 1:]
                                    ra, rb = routes[ai], routes[bi]
                                    loads[ai], loads[bi] = new_la, new_lb
                                    local_improved = improved = exchanged_here = True
                                    break
                            if exchanged_here:
                                pa = 0
                            else:
                                pa += 1
            if deadline is not None and time.time() >= deadline:
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


def sisr_removal(data, routes, rng, k):
    """B6: String-removal (Christiaens & Vanden Berghe 2020 SISR-style).

    Pick a random anchor customer, walk through its K nearest neighbors and
    remove a contiguous string of length L_i from each route the walk
    reaches. Continues until the cumulative removal count reaches k.
    """
    routes = [r[:] for r in routes]
    customers = [c for r in routes for c in r if c != 0]
    if not customers:
        return routes, []
    nbrs = compute_neighbors(data, max(20, k))
    anchor = rng.choice(customers)
    removed = set()
    visited_routes = set()
    candidates = [anchor] + list(nbrs[anchor])
    for c in candidates:
        if len(removed) >= k:
            break
        # Find which route c is in and at what position
        ri = None
        pos = None
        for i, r in enumerate(routes):
            if c in r:
                ri = i
                pos = r.index(c)
                break
        if ri is None or ri in visited_routes:
            continue
        visited_routes.add(ri)
        # Pick string length
        max_string = min(len(routes[ri]) - 2, max(2, k - len(removed)))
        L = rng.randint(1, max(1, max_string))
        # Random start within the route, containing pos
        start = max(1, pos - rng.randint(0, L - 1))
        end = min(len(routes[ri]) - 1, start + L)
        for j in range(start, end):
            removed.add(routes[ri][j])
    if not removed:
        # Fallback: at least one removal
        removed.add(anchor)
    partial = [[0] + [c for c in r if c != 0 and c not in removed] + [0] for r in routes]
    return normalize_routes(data, partial), list(removed)


def perturb(data, routes, rng, remove_count=5, mode=None, top_k=5):
    routes = [r[:] for r in routes]
    customers = [c for r in routes for c in r if c != 0]
    if not customers:
        return routes
    k = min(max(1, int(remove_count)), len(customers))
    if mode is None:
        mode = rng.choice(['random', 'route', 'shaw'])
    elif isinstance(mode, (list, tuple)):
        mode = rng.choice(list(mode))
    if mode == 'route':
        partial, removed = random_route_segment_removal(data, routes, rng, k)
    elif mode == 'shaw':
        partial, removed = shaw_removal(data, routes, rng, k)
    elif mode == 'sisr':
        partial, removed = sisr_removal(data, routes, rng, k)
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
    start = time.time()
    rng = random.Random(seed)
    profile = ils_profile(data, time_limit, config)
    sa_enabled = bool(cfg(config, 'sa_acceptance', False))
    current = improve_routes(data, initial_routes, deadline, max_passes=cfg(config, 'initial_ls_passes', max(50, profile['ls_passes'] + 2)), config=config)
    best = [r[:] for r in current]
    best_cost = solution_cost(data, best)
    cur_cost = best_cost
    # B5: SA temperature schedule. Start at ~5% of best_cost, cool to ~0.1%.
    T0 = 0.05 * best_cost
    T_end = 0.001 * best_cost
    no_improve = 0
    while time.time() < deadline:
        strength = min(profile['max_remove'], profile['base_remove'] + no_improve // 2)
        cand = perturb(data, best if no_improve > profile['restart_trigger'] else current, rng, remove_count=strength, mode=cfg(config, 'perturb_mode', None), top_k=profile['top_k'])
        cand = improve_routes(data, cand, deadline, max_passes=profile['ls_passes'], config=config)
        cc = solution_cost(data, cand)
        if cc + 1e-9 < best_cost:
            best = [r[:] for r in cand]
            best_cost = cc
            current = cand
            cur_cost = cc
            no_improve = 0
        else:
            no_improve += 1
            if sa_enabled:
                elapsed = (time.time() - start) / max(1e-9, float(time_limit))
                elapsed = min(1.0, max(0.0, elapsed))
                T = T0 * ((T_end / max(T0, 1e-9)) ** elapsed)
                if T > 1e-9 and rng.random() < math.exp(-(cc - cur_cost) / T):
                    current = cand
                    cur_cost = cc
            else:
                if rng.random() < profile['accept_prob']:
                    current = cand
                    cur_cost = cc
        if restart_factory is not None and no_improve >= 2 * profile['restart_trigger'] and time.time() < deadline:
            fresh = restart_factory(rng)
            fresh = improve_routes(data, fresh, deadline, max_passes=min(3, profile['ls_passes']), config=config)
            fc = solution_cost(data, fresh)
            current = fresh
            cur_cost = fc
            if fc + 1e-9 < best_cost:
                best = [r[:] for r in fresh]
                best_cost = fc
                no_improve = 0
            else:
                no_improve = max(0, no_improve // 2)
    best = normalize_routes(data, best)
    return best, solution_cost(data, best)
