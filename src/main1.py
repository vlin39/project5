import json
import os
from argparse import ArgumentParser
from pathlib import Path

from vrpinstance_modified import VRPInstance
from timer import Timer
from solver_common import format_solution

SWEEP_CONFIGS = {
    "default": None,
    "tuned": {  # data-driven defaults: balanced + lspass6 + restarttrigger8
        "offset_count": 36,
        "construction_fraction": 0.35,
        "construction_passes": 4,
        "elite_size": 6,
        "intensify_fraction": 0.20,
        "use_reverse": True,
        "use_shuffle_ties": True,
        "base_remove": 5,
        "max_remove": 16,
        "top_k": 8,
        "ls_passes": 6,
        "restart_trigger": 8,
        "accept_prob": 0.20,
    },
    "balanced": {
        "offset_count": 36,
        "construction_fraction": 0.35,
        "construction_passes": 4,
        "elite_size": 6,
        "intensify_fraction": 0.20,
        "use_reverse": True,
        "use_shuffle_ties": True,
        "base_remove": 6,
        "max_remove": 16,
        "top_k": 8,
        "ls_passes": 4,
        "restart_trigger": 14,
        "accept_prob": 0.25,
    },
    "intensified": {
        "offset_count": 24,
        "construction_fraction": 0.30,
        "construction_passes": 5,
        "elite_size": 8,
        "intensify_fraction": 0.30,
        "use_reverse": True,
        "use_shuffle_ties": True,
        "base_remove": 5,
        "max_remove": 16,
        "top_k": 6,
        "ls_passes": 6,
        "restart_trigger": 16,
        "accept_prob": 0.20,
    },
    "lspass6": {"ls_passes": 6},
    "constructionpass4": {"construction_passes": 4},
    "restarttrigger8": {"restart_trigger": 8},
}

SAVINGS_CONFIGS = {
    "default": None,
    "tuned": {  # data-driven defaults from log winners
        "noise_levels": (0.0, 0.01, 0.03, 0.07, 0.12, 0.20),
        "construction_fraction": 0.40,
        "construction_passes": 2,
        "elite_size": 4,
        "intensify_fraction": 0.10,
        "base_remove": 6,
        "max_remove": 20,
        "top_k": 8,
        "ls_passes": 4,
        "restart_trigger": 8,
        "accept_prob": 0.25,
    },
    "intensify10": {"intensify_fraction": 0.10},
    "restarttrigger8": {"restart_trigger": 8},
    "destroy_aggressive": {"base_remove": 8, "max_remove": 24},
    "noise_low": {"noise_levels": (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)},
    "fewseeds": {"seed_count": 3},  # bias toward fewer/stronger restarts (per log signal)
}

CONFIG_REGISTRY = {
    "savings": SAVINGS_CONFIGS,
    "sweep": SWEEP_CONFIGS,
}

SOLVER = os.getenv("SOLVER", "sweep")
TIME_LIMIT = float(os.getenv("TIME_LIMIT", "30"))
SEED = int(os.getenv("SEED", "0"))
CONFIG_NAME = os.getenv("CONFIG_NAME", "default")

if SOLVER not in CONFIG_REGISTRY:
    raise ValueError(
        f"Unknown SOLVER {SOLVER!r}. Expected one of: {', '.join(sorted(CONFIG_REGISTRY))}"
    )

if SOLVER == "savings":
    from solver_savings import solve as solve_vrp
else:
    from solver_sweep import solve as solve_vrp

CONFIGS = CONFIG_REGISTRY[SOLVER]
if CONFIG_NAME not in CONFIGS:
    raise ValueError(
        f"Unknown {SOLVER} config: {CONFIG_NAME}. "
        f"Available: {', '.join(sorted(CONFIGS.keys()))}"
    )
CONFIG = CONFIGS[CONFIG_NAME]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    input_file = args.input_file
    filename = Path(input_file).name

    timer = Timer()
    timer.start()

    instance = VRPInstance(str(input_file))
    solution, objective_value = solve_vrp(
        instance,
        time_limit=TIME_LIMIT,
        seed=SEED,
        config=CONFIG,
    )

    timer.stop()

    if objective_value:
        objective_value = round(objective_value, 2)
    else:
        objective_value = "--"
    solution_str = format_solution(solution) if solution else "--"

    output_dict = {
        "Solver": SOLVER,
        "Config": CONFIG_NAME,
        "Instance": filename,
        "Time": f"{timer.getTime():.2f}",
        "Result": objective_value,
        "Solution": solution_str,
    }

    print(json.dumps(output_dict))
