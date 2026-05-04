import numpy as np
from pathlib import Path
from scipy.spatial import distance
from sklearn.cluster import KMeans

import vrputils

rng = np.random.default_rng()

class VRPInstance:
    filename: str
    numCustomers: int  # the number of customers
    numVehicles: int  # the number of vehicles
    vehicleCapacity: int  # the capacity of the vehicles
    demandOfCustomer: np.ndarray  # the demand of each customer
    xCoordOfCustomer: np.ndarray  # the x coordinate of each customer
    yCoordOfCustomer: np.ndarray  # the y coordinate of each customer
    xyCoordsOfCustomer: np.ndarray # (x, y) coordinate of each customer
    dists: np.ndarray # matrix of distances between customers

    def __init__(self, filename: str):
        self.filename = filename
        self.load_from_file(filename)
        self.solution = None
        self.objective_value = 0

    def construct_solution(self) -> list:
        # semi-greedy solution construcition
        # K-means clustering of points
        cluster_dists = KMeans(n_clusters=self.numVehicles).fit_transform(self.xyCoordsOfCustomer)

        # assign each customer to a cluster
        sol = [[0] for _ in range(self.numVehicles)]
        usedCapacity = np.zeros(self.numVehicles, dtype=int)
        total_cost = 0
        customers = np.argsort(self.demandOfCustomer)[1:][::-1]
        for c in customers:
            choices = np.argsort(cluster_dists[c])
            for cluster in choices:
                if usedCapacity[cluster] + self.demandOfCustomer[c] <= self.vehicleCapacity:
                    lc = sol[cluster][-1]
                    usedCapacity[cluster] += self.demandOfCustomer[c]
                    total_cost += self.dists[lc, c]
                    sol[cluster].append(int(c))
                    break

        # must return to depot
        for i in range(self.numVehicles):
            lc = sol[i][-1]
            total_cost += self.dists[lc, 0]
            sol[i].append(0)

        return sol, usedCapacity, total_cost

    def is_feasible(self, usedCapacity: np.ndarray):
        return all(cap <= self.vehicleCapacity for cap in usedCapacity) and sum(usedCapacity) == self.demandOfCustomer.sum()

    def LNS(self):
        # parameters
        NUM_CANDIDATE_SOLUTIONS = 1000
        NUM_FILTERED_CANDIDATE_SOLUTIONS = 200
        NUM_ITERATIONS_1 = 100
        NUM_REMOVALS = 175
        NUM_ITERATIONS_2 = 1000

        assert NUM_FILTERED_CANDIDATE_SOLUTIONS <= NUM_CANDIDATE_SOLUTIONS
        assert NUM_REMOVALS < NUM_FILTERED_CANDIDATE_SOLUTIONS

        # produce initial solution
        sol, usedCapacity, total_cost = self.construct_solution()
        while not self.is_feasible(usedCapacity):
            sol, usedCapacity, total_cost = self.construct_solution()

        # record best feasible cost
        best_feasible_cost = total_cost
        best_feasible_usedCapacity = usedCapacity
        best_feasible_sol = sol

        # generate a series of candidate solutions
        sol_population = []
        usedCapacity_population = []
        total_cost_population = []
        for _ in range(NUM_CANDIDATE_SOLUTIONS):
            sol, usedCapacity, total_cost = self.construct_solution()
            sol_population.append(sol)
            usedCapacity_population.append(usedCapacity)
            total_cost_population.append(total_cost)
            if self.is_feasible(usedCapacity) and total_cost < best_feasible_cost:
                best_feasible_cost = total_cost
                best_feasible_usedCapacity = usedCapacity
                best_feasible_sol = sol

        # filter out the worst ones
        keep_idxs = np.argsort(total_cost_population)[:NUM_FILTERED_CANDIDATE_SOLUTIONS]
        sol_population = [sol_population[idx] for idx in keep_idxs]
        usedCapacity_population = [usedCapacity_population[idx] for idx in keep_idxs]
        total_cost_population = [total_cost_population[idx] for idx in keep_idxs]

        # run N iterations of local search on every candidate solution
        def compute_N_iterations(n: int):
            nonlocal best_feasible_cost, best_feasible_usedCapacity, best_feasible_sol
            for idx in range(len(sol_population)):
                # perform local search
                new_sol, newCapacities, new_cost = vrputils.local_search(
                    sol_population[idx], usedCapacity_population[idx], total_cost_population[idx],
                    self.numCustomers, self.numVehicles, self.vehicleCapacity,
                    self.lamb, self.dists, self.demandOfCustomer, n
                )

                # save solution
                sol_population[idx] = new_sol
                usedCapacity_population[idx] = newCapacities
                total_cost_population[idx] = new_cost

                # update best found solution so far
                if self.is_feasible(usedCapacity) and new_cost < best_feasible_cost:
                    best_feasible_cost = new_cost
                    best_feasible_usedCapacity = newCapacities
                    best_feasible_sol = new_sol

        # remove the N solutions with the highest cost
        def remove_N_solutions(n: int):
            worst = np.argsort(total_cost_population)[-n:]
            for idx in np.sort(worst)[::-1]:
                sol_population.pop(idx)
                usedCapacity_population.pop(idx)
                total_cost_population.pop(idx)

        # run filtered candidates solutions through local search
        compute_N_iterations(NUM_ITERATIONS_1)

        # remove the worst candidate solutions from consideration
        remove_N_solutions(NUM_REMOVALS)

        # run the best candidates through to completion
        compute_N_iterations(NUM_ITERATIONS_2)

        # return best result
        return best_feasible_sol, best_feasible_cost

    def solve(self, save_solution: bool = False):
        # compute distance matrix
        xy_coords = np.column_stack([self.xCoordOfCustomer, self.yCoordOfCustomer])
        self.dists = distance.cdist(xy_coords, xy_coords)
        self.xyCoordsOfCustomer = xy_coords

        # compute nearest neighbors (excluding itself and depot)
        self.nearest = self.dists[:, 1:].argsort(axis=1)[:, 1:] + 1

        # estimate parameter lambda
        avg_edge_length = self.dists.sum() / np.count_nonzero(self.dists)
        self.lamb = 10 * avg_edge_length

        # run GRASP
        sol, cost = self.LNS()
        if sol is None:
            return None, None

        # save solution
        if save_solution:
            self.write_solution(sol, cost, False)

        # flatten
        flat = [0] # solution is not provably optimal
        for route in sol:
            flat.extend(route)

        # convert to string
        sol_str = ' '.join(str(x) for x in flat)

        return sol_str, cost

    def write_solution(self, sol: list, cost: float, optimal=False):
        out_dir = Path('./solutions').absolute()
        out_dir.mkdir(exist_ok=True)
        out_filename = Path(self.filename).with_suffix('.vrp.sol').name
        out_file = out_dir / out_filename
        with open(out_file, 'w') as f:
            f.write(f"{round(cost, 1)} {int(optimal)}\n")
            for route in sol:
                f.write(' '.join(str(x) for x in route) + '\n')

    def load_from_file(self, filename: str):
        # Note from Taj: print feel free to remove print statements
        try:
            with open(filename, 'r') as f:
                # Read all numbers from the file
                content = f.read().split()
                iterator = iter(content)

                self.numCustomers = int(next(iterator))
                self.numVehicles = int(next(iterator))
                self.vehicleCapacity = int(next(iterator))

                print(f"Number of customers: {self.numCustomers}")
                print(f"Number of vehicles: {self.numVehicles}")
                print(f"Vehicle capacity: {self.vehicleCapacity}")

                self.demandOfCustomer = np.zeros(self.numCustomers, dtype=int)
                self.xCoordOfCustomer = np.zeros(self.numCustomers)
                self.yCoordOfCustomer = np.zeros(self.numCustomers)

                for i in range(self.numCustomers):
                    self.demandOfCustomer[i] = int(next(iterator))
                    self.xCoordOfCustomer[i] = float(next(iterator))
                    self.yCoordOfCustomer[i] = float(next(iterator))

                for i in range(self.numCustomers):
                    print(f"{self.demandOfCustomer[i]} {self.xCoordOfCustomer[i]} {self.yCoordOfCustomer[i]}")
        except Exception as e:
            print(f"Error reading instance file: {e}")
            exit(1)

    def __str__(self):
        out = f"Number of customers: {self.numCustomers}\n"
        out += f"Number of vehicles: {self.numVehicles}\n"
        out += f"Vehicle capacity: {self.vehicleCapacity}\n"
        for i in range(self.numCustomers):
            out += f"{self.demandOfCustomer[i]} {self.xCoordOfCustomer[i]} {self.yCoordOfCustomer[i]}\n"
        return out
