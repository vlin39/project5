import numpy as np


class VRPInstance:
    numCustomers: int  # the number of customers
    numVehicles: int  # the number of vehicles
    vehicleCapacity: int  # the capacity of the vehicles
    demandOfCustomer: np.ndarray  # the demand of each customer
    xCoordOfCustomer: np.ndarray  # the x coordinate of each customer
    yCoordOfCustomer: np.ndarray  # the y coordinate of each customer

    def __init__(self, filename:str, keep_print:bool = False):
        self.load_from_file(filename, keep_print)
        self.solution = None
        self.objective_value = 0

    def solve(self):
        self.solution = None
        self.objective_value = 0
        return self.solution, self.objective_value

    def load_from_file(self, filename:str, keep_print:bool = False):
        # Note from Taj: print feel free to remove print statements
        # Answer to Taj: the print statements are well-liked enough that I kept them 
        #                and just made them conditional :D
        try:
            with open(filename, 'r') as f:
                # Read all numbers from the file
                content = f.read().split()
                iterator = iter(content)

                self.numCustomers = int(next(iterator))
                self.numVehicles = int(next(iterator))
                self.vehicleCapacity = int(next(iterator))

                if keep_print: 
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

                if keep_print: 
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