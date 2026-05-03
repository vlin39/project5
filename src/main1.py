import json
import os
from argparse import ArgumentParser
from pathlib import Path

from vrpinstance_modified import VRPInstance
from timer import Timer

SWEEP_CONFIGS = {
    "default": None,
    "diversified": {
        "offset_count": 48,
        "construction_fraction": 0.40,
        "construction_passes": 4,
        "elite_size": 6,
        "intensify_fraction": 0.20,
        "use_reverse": True,
        "use_shuffle_ties": True,
        "base_remove": 6,
        "max_remove": 24,
        "top_k": 10,
        "ls_passes": 4,
        "restart_trigger": 12,
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
    "offset24": {"offset_count": 24},
    "offset48": {"offset_count": 48},
    "offset72": {"offset_count": 72},
    "constructfrac25": {"construction_fraction": 0.25},
    "constructfrac35": {"construction_fraction": 0.35},
    "constructfrac45": {"construction_fraction": 0.45},
    "constructionpass1": {"construction_passes": 1},
    "constructionpass3": {"construction_passes": 3},
    "constructionpass5": {"construction_passes": 5},
    "noreverse": {"use_reverse": False, "use_shuffle_ties": True},
    "notieshuffle": {"use_reverse": True, "use_shuffle_ties": False},
    "forwardonly": {"use_reverse": False, "use_shuffle_ties": False},
    "elite3": {"elite_size": 3},
    "elite6": {"elite_size": 6},
    "elite8": {"elite_size": 8},
    "intensify10": {"intensify_fraction": 0.10},
    "intensify20": {"intensify_fraction": 0.20},
    "intensify30": {"intensify_fraction": 0.30},
    "destroy_conservative": {"base_remove": 4, "max_remove": 12},
    "destroy_medium": {"base_remove": 6, "max_remove": 16},
    "destroy_aggressive": {"base_remove": 8, "max_remove": 24},
    "topk4": {"top_k": 4},
    "topk8": {"top_k": 8},
    "topk12": {"top_k": 12},
    "lspass2": {"ls_passes": 2},
    "lspass4": {"ls_passes": 4},
    "lspass6": {"ls_passes": 6},
    "restarttrigger8": {"restart_trigger": 8},
    "restarttrigger12": {"restart_trigger": 12},
    "restarttrigger16": {"restart_trigger": 16},
    "restarttrigger20": {"restart_trigger": 20},
    "accept10": {"accept_prob": 0.10},
    "accept20": {"accept_prob": 0.20},
    "accept30": {"accept_prob": 0.30},
    "perturb_random": {"perturb_mode": "random"},
    "perturb_route": {"perturb_mode": "route"},
    "perturb_shaw": {"perturb_mode": "shaw"},
    "fastscreen": {
        "offset_count": 24,
        "construction_fraction": 0.35,
        "construction_passes": 2,
        "elite_size": 3,
        "intensify_fraction": 0.10,
        "use_reverse": True,
        "use_shuffle_ties": True,
        "base_remove": 4,
        "max_remove": 12,
        "top_k": 6,
        "ls_passes": 3,
        "restart_trigger": 10,
        "accept_prob": 0.25,
    },
}

SOLVER = os.getenv("SOLVER", "sweep")
TIME_LIMIT = float(os.getenv("TIME_LIMIT", "30"))
SEED = int(os.getenv("SEED", "0"))
CONFIG_NAME = os.getenv("CONFIG_NAME", "default")

if SOLVER != "sweep":
    raise ValueError(f"This main1.py is configured for sweep experiments only. Got SOLVER={SOLVER!r}")

from solver_sweep import solve as solve_vrp

if CONFIG_NAME not in SWEEP_CONFIGS:
    raise ValueError(
        f"Unknown sweep config: {CONFIG_NAME}. "
        f"Available configs: {', '.join(sorted(SWEEP_CONFIGS.keys()))}"
    )
CONFIG = SWEEP_CONFIGS[CONFIG_NAME]

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

    output_dict = {
        "Solver": SOLVER,
        "Config": CONFIG_NAME,
        "Instance": filename,
        "Time": f"{timer.getTime():.2f}",
        "Result": round(objective_value, 2) if objective_value is not None else "--",
        "Solution": solution if solution else "--",
    }

    print(json.dumps(output_dict))
