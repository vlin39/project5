#!/usr/bin/env python3
import json, sys
from pathlib import Path
from typing import Any

"""
Mainly to call write_sol_file in main.py after getting output dict
Can also be called in terminal:
    python3 src/sol_file_writer.py '{"Instance":"5_4_10.vrp","Time":"1.23","Result":80.6,"Solution":"0 0 1 2 3 0 0 4 0 0 0 0 0"}'
        Output: 5_4_10.vrp.sol in folder named solutions
    python3 src/sol_file_writer.py <results>.log
        Output: lots of .vrp.sol files in folder named <results>_solutions (folder name corresponds to file name)

The Solution string in the JSON record carries the optimality flag as
its leading token: "<flag> <r0_node0> ... <rN-1_nodeM>". `sol_file_writer`
is a passthrough — it consumes the flag from the wire format and copies
it onto the first line of the .sol file. Don't try to override it here;
the flag is set upstream by `solver_common.format_solution`.

"""

def write_sol_file(solution: dict, num_vehicles=None, output_dir="solutions"):
    """
    output dictionary -> .vrp.sol file

    Example
    record: {"Instance": "5_4_10.vrp",
            "Time": "1.23",
            "Result": 80.6,
            "Solution": "0 0 1 2 3 0 0 4 0 0 0 0 0"}
    Output: 5_4_10.vrp.sol
        80.6 0
        0 1 2 3 0
        0 4 0
        0 0
        0 0
    """
    instance = solution.get("Instance")
    objective_value = solution.get("Result")
    solution = solution.get("Solution")

    if not solution:
        raise ValueError("Missing instance")
    if objective_value in (None, "--"):
        raise ValueError(f"{instance}: no result")
    if solution in (None, "--"):
        raise ValueError(f"{instance}: no solution")

    if num_vehicles is None:
        print("Inferring number of vehicles from file name...")
        num_vehicles = infer_num_vehicles(instance)

    flag, routes = parse_solution_string(solution, num_vehicles)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sol_path = output_path / f"{Path(instance).name}.sol"

    with open(sol_path, "w") as f:
        f.write(f"{objective_value} {flag}\n")
        for route in routes:
            f.write(" ".join(map(str, route)) + "\n")

    return sol_path

def infer_num_vehicles(instance_name:str)->int:
    """
    Gets vehicle count from file name. 

    The instance naming convention seems to be: 
        <numCustomers>_<numVehicles>_<capacity>.vrp

    Example: 5_4_10.vrp
        Number of customers: 5
        Number of vehicles: 4
        Vehicle capacity: 10
    """
    try:
        num_vehicles = Path(instance_name).name.split("_")[1]
        print("Number of vehicles: " + num_vehicles)
        return int(num_vehicles)
    except Exception as exc:
        raise ValueError(f"Could not infer number of vehicles from instance {instance_name}") from exc


def parse_solution_string(solution: str, num_vehicles: int) -> tuple[int, list[list[int]]]:
    """
    Convert wire-format Solution string into (optimality_flag, routes).

    Example: "0 0 1 2 3 0 0 4 0 0 0 0 0"
              ^ flag
                ^^^^^^^^^^^^^^^^^^^^^^^^ flattened routes
        -> (0, [[0, 1, 2, 3, 0], [0, 4, 0], [0, 0], [0, 0]])
    """
    tokens = [int(x) for x in str(solution).split()]
    if not tokens:
        raise ValueError("Empty solution string")
    flag = tokens[0]
    rest = tokens[1:]

    routes = []
    current = []
    for node in rest:
        current.append(node)
        if node == 0 and len(current) > 1:
            routes.append(current)
            current = []
            if len(routes) == num_vehicles: break

    while len(routes) < num_vehicles:
        routes.append([0, 0])

    return flag, routes[:num_vehicles]


def write_from_log(log_file):
    """results.log -> new-ish folder with .vrp.sol files"""
    log_path = Path(log_file)
    output_dir = Path(f"{log_path.stem}_solutions")
    sols = []
    with open(log_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line: continue
            try: 
                sols.append(write_sol_file(json.loads(line), output_dir=output_dir))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_num} of {log_file} is not valid JSON") from exc
    return sols

def write_from_string(json: str) -> Path:
    """ JSON string -> single .vrp.sol file in solutions folder
    # this is mainly to prevent overriding the provided 5_4_10.vrp.sol from the stencil 
    """
    sol : dict = json.loads(json)
    return write_sol_file(sol)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 src/sol_file_writer.py '<json-string>'")
        print("   or: python3 src/sol_file_writer.py results.log")
        raise SystemExit(65)

    arg = sys.argv[1]
    path = Path(arg)

    if path.exists() and path.is_file():
        written_files = write_from_log(path)
        for sol_path in written_files:
            print(f"Wrote {sol_path}")
        print(f"Done: wrote {len(written_files)} file(s)")
    else:
        sol_path = write_from_string(arg)
        print(f"Wrote {sol_path}")
