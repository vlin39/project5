# distutils: language=c++
cimport cython
from libc.math cimport INFINITY, ceil
from libcpp.vector cimport vector
from libcpp.algorithm cimport sample, find
from libcpp.random cimport mt19937, random_device

cdef random_device rd
cdef mt19937 gen = mt19937(rd())

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double insertion_cost(int c, int pos, long usage, vector[int] route, double lamb,
                           double[:, :] dists, long[:] demandOfCustomer, int vehicleCapacity):
    # get distance cost
    cdef int prv = route[pos - 1]
    cdef int nxt = route[pos]
    cdef double old_dist = dists[prv,nxt]
    cdef double new_dist = dists[prv, c] + dists[c, nxt]
    cdef double cost = new_dist - old_dist

    # add constraint violation cost
    cdef long new_usage = usage + demandOfCustomer[c]
    cost += lamb * max(0, new_usage - vehicleCapacity)
    return cost

@cython.boundscheck(False)
@cython.wraparound(False)
cdef (vector[vector[int]], vector[int], double)\
repair(vector[int] unassigned, vector[vector[int]] sol, vector[int] usedCapacity, double total_cost,
       int numVehicles, int vehicleCapacity, double lamb, double[:, :] dists, long[:] demandOfCustomer):
    cdef double best_cost # TODO: some insertion costs don't change, so save them
    cdef int best_c, best_r, best_pos

    # find best places to re-insert each customer
    while not unassigned.empty():
        best_cost = INFINITY
        best_c = best_r = best_pos = 0
        for r in range(numVehicles):
            route = sol[r]
            for pos in range(1, route.size()):
                for c in unassigned:
                    cost = insertion_cost(c, pos, usedCapacity[r], route, lamb, dists, demandOfCustomer, vehicleCapacity)
                    if cost < best_cost:
                        best_cost = cost
                        best_c = c
                        best_r = r
                        best_pos = pos

        # TODO: try adding randomness to selection

        # apply move
        sol[best_r].insert(sol[best_r].begin() + best_pos, best_c)
        usedCapacity[best_r] += demandOfCustomer[best_c]
        total_cost += best_cost
        unassigned.erase(find(unassigned.begin(), unassigned.end(), best_c))

    return sol, usedCapacity, total_cost

@cython.boundscheck(False)
@cython.wraparound(False)
cdef (vector[int], vector[vector[int]], vector[int], double)\
destroy(vector[vector[int]] sol, vector[int] usedCapacity, double total_cost,
        int numCustomers, int numVehicles, int vehicleCapacity,
        double lamb, double[:,:] dists, long[:] demandOfCustomer):
    # randomly remove customers from routes TODO: remove highest cost customers first?
    cdef vector[int] options
    options.reserve(numCustomers - 1)
    for c in range(1, numCustomers):
        options.push_back(c)

    cdef int k = min(10, max(1, <int>ceil(0.2 * <double>numCustomers)))
    cdef vector[int] unassigned
    unassigned.resize(k)
    sample(options.begin(), options.end(), unassigned.begin(), k, gen)

    cdef int r, p, old_usage, new_usage
    cdef double penalty_before, penalty_after
    for r in range(numVehicles):
        for p in range(sol[r].size() - 2, 0, -1):
            c = sol[r][p]
            if find(unassigned.begin(), unassigned.end(), c) != unassigned.end():
                total_cost -= dists[sol[r][p-1], c] + dists[c, sol[r][p+1]]
                total_cost += dists[sol[r][p-1], sol[r][p+1]]
                old_usage = usedCapacity[r]
                new_usage = old_usage - demandOfCustomer[c]
                penalty_before = lamb * max(0, old_usage - vehicleCapacity)
                penalty_after = lamb * max(0, new_usage - vehicleCapacity)
                total_cost += penalty_after - penalty_before
                usedCapacity[r] = new_usage
                sol[r].erase(sol[r].begin() + p)

    return unassigned, sol, usedCapacity, total_cost

cdef bint is_feasible(vector[int] usedCapacity, int vehicleCapacity, int totalDemand):
    cdef int d, capacitySum = 0
    for d in usedCapacity:
        if d > vehicleCapacity:
            return False
        capacitySum += d
    return (capacitySum == totalDemand)

@cython.boundscheck(False)
@cython.wraparound(False)
def local_search(vector[vector[int]] sol, vector[int] usedCapacity, double total_cost,
                 int numCustomers, int numVehicles, int vehicleCapacity,
                 double lamb, double[:,:] dists, long[:] demandOfCustomer, long max_iter):
    cdef double best_feasible_cost = total_cost
    cdef vector[int] best_feasible_usedCapacity = usedCapacity
    cdef vector[vector[int]] best_feasible_sol = sol

    cdef long total_demand = 0
    for c in range(numCustomers):
        total_demand += demandOfCustomer[c]

    for count in range(max_iter):
        # print(f"\rcount: {count}", end='')
        unassigned, new_sol, newCapacities, new_cost = destroy(sol, usedCapacity, total_cost,
                                                               numCustomers, numVehicles, vehicleCapacity,
                                                               lamb, dists, demandOfCustomer)
        new_sol, newCapacities, new_cost = repair(unassigned, new_sol, newCapacities, new_cost,
                                                  numVehicles, vehicleCapacity, lamb,
                                                  dists, demandOfCustomer)

        # TODO: randomly reject new solution sometimes?
        if new_cost < total_cost:
            sol = new_sol
            usedCapacity = newCapacities
            total_cost = new_cost

            if is_feasible(usedCapacity, vehicleCapacity, total_demand) and total_cost < best_feasible_cost:
                best_feasible_cost = total_cost
                best_feasible_usedCapacity = usedCapacity
                best_feasible_sol = sol

    return best_feasible_sol, best_feasible_usedCapacity, best_feasible_cost
