import json
from argparse import ArgumentParser
from pathlib import Path
from vrpinstance_modified import VRPInstance
from timer import Timer
from solver_common import format_solution
from sol_file_writer import write_sol_file 
# -----------------------------------------------------------------------------
# Different implementations go here!!
# Use one implementation at a time -- comment out the import/assignment lines that are not being used
# Not doing it probably won't break anything, but... :/
# To add a new solver, make sure it's solve(data, time_limit, seed)
# (But technically, only the data is needed.)
# Then import it and name it!
#
# from solver_savings import solve
# SOLVER = "savings_cw_ils" #"savings_clarke_wright_ils"
#
# from solver_sweep import solve
# SOLVER = "sweep_cluster_tsp_ils"
#
# from solver_pyvrp import solve
# SOLVER = "pyvrp"  # PyVRP / HGS engine — see src/solver_pyvrp.py

# results_savings_fastscreen_10s.log
SOLVER = "savings"
TIME_LIMIT = 10.0
SEED = 0
CONFIG = {
    "seed_count": 4,
    "noise_levels": (0.0, 0.03, 0.07, 0.12),
    "construction_fraction": 0.40,
    "construction_passes": 1,
    "elite_size": 3,
    "intensify_fraction": 0.10,
    "base_remove": 4,
    "max_remove": 12,
    "top_k": 6,
    "ls_passes": 3,
    "restart_trigger": 10,
    "accept_prob": 0.25,
}

if SOLVER == "savings":
    from solver_savings import solve
elif SOLVER == "sweep":
    from solver_sweep import solve
elif SOLVER == "pyvrp":
    from solver_pyvrp import solve

"""
    to make it easier to run things:

    ./compile.sh
    ./run_vrp.sh 5_4_10.vrp
    ./run.sh 5_4_10.vrp
    ./run.sh input/16_5_1.vrp
    python3 src/sol_file_writer.py results.log

    ./runAll.sh input/ 300 results.log

    python3 src/verify.py input/ results.log
    https://cs.brown.edu/courses/csci2951-o/p5vis.html
    https://csci2951o-hw5-leaderboard.vercel.app/

"""
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    input_file = args.input_file
    path = Path(input_file)
    filename = path.name

    # print(f"Solver: {SOLVER}")

    timer = Timer()
    timer.start()

    instance = VRPInstance(str(input_file), keep_print = False)
    # solution, objective_value = solve(instance)
    # solution, objective_value = solve(instance, time_limit = 30.0, seed = 0)
    solution, objective_value = solve(
        instance,
        time_limit=TIME_LIMIT,
        seed=SEED,
        config=CONFIG,
    )


    timer.stop()

    # This was the original
    # I changed it for ease of writing to a .vrp.sol file 
    # output_dict = {"Instance": filename,
    #                "Time": f"{timer.getTime():.2f}",
    #                "Result": round(objective_value, 2) if objective_value else "--",
    #                "Solution": format_solution(solution) if solution else "--"}

    if objective_value: objective_value = round(objective_value, 2)
    else: objective_value = ""
    if solution: solution = format_solution(solution) 
    else: solution = "--"

    output_dict = {
                #    "Solver": SOLVER,
                   "Instance": filename,
                   "Time": f"{timer.getTime():.2f}",
                   "Result": objective_value,
                   "Solution": solution}
    
    # write_sol_file(output_dict, num_vehicles = instance.numVehicles, output_dir = "")

    print(json.dumps(output_dict))
