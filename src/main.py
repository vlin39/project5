"""
Project entry point — the optimal-solver branch.

Single solver: `solver_optimal` (adaptive-budget PyVRP). Following the
shape of `main_stencil.py` (parser → instance → solve → JSON output) and
the previous `main.py` (uses VRPInstance + format_solution and prints a
single-line JSON record on stdout, as required by the project handout).
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from vrpinstance_modified import VRPInstance
from timer import Timer
from solver_common import format_solution
from solver_optimal import solve


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    input_file = args.input_file
    filename = Path(input_file).name

    instance = VRPInstance(str(input_file), keep_print=False)

    timer = Timer()
    timer.start()
    solution, objective_value = solve(instance, seed=0)
    timer.stop()

    if objective_value:
        objective_value = round(objective_value, 2)
    else:
        objective_value = ""
    solution = format_solution(solution) if solution else "--"

    output_dict = {
        "Instance": filename,
        "Time": f"{timer.getTime():.2f}",
        "Result": objective_value,
        "Solution": solution,
    }
    print(json.dumps(output_dict))
