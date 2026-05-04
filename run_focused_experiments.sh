#!/bin/bash
set -euo pipefail

# Focused experiment grid for both savings and sweep solvers.
#
# Usage:
#   bash run_focused_experiments.sh [input_dir] [time_limit_seconds]
#
# Notes:
#   - Reads from src/main1.py via runAll_sweep.sh (env-var driven)
#   - Each (solver, config, time) batch produces one log file
#   - Default grid: 6 sweep configs + 6 savings configs x 2 time budgets x 16 instances
#     = ~12 batches per time budget. With a 30s/instance run that is roughly
#     2 hours per time budget on a single machine, vs ~22 hours for the original
#     run_sweep_experiments.sh grid.
#   - Sequential by default; for parallelism, run two shells with different
#     LOG_DIR / SOLVER subsets, or wrap the loop with `&` and `wait`.

INPUT_DIR="${1:-input}"
DEFAULT_TIME="${2:-30}"
LOG_DIR="${LOG_DIR:-focused_logs}"
mkdir -p "$LOG_DIR"

# Time budgets to evaluate. Two is enough to see scaling behavior; add more as
# needed. The handout's competition budget is 5 min so 60 is the more realistic
# value, while 30 gives faster iteration.
TIMES=(${DEFAULT_TIME} 60)

# Configs picked by the prior log analysis: each was a winner on ≥1 dimension.
# 'default' is the unmodified solver; 'tuned' bundles the data-driven picks.
SAVINGS_CONFIGS=(
  default
  tuned
  intensify10
  restarttrigger8
  destroy_aggressive
  noise_low
)

SWEEP_CONFIGS=(
  default
  tuned
  balanced
  intensified
  lspass6
  constructionpass4
)

run_batch() {
  local solver="$1"
  local cfg="$2"
  local t="$3"
  local log_file="${LOG_DIR}/results_${solver}_${cfg}_${t}s.log"
  local shell_timeout=$((t + 20))
  if [ -f "$log_file" ]; then
    echo "  skip (already exists): ${log_file}"
    return 0
  fi
  echo "  ${solver}/${cfg} @ ${t}s -> ${log_file}"
  SOLVER="$solver" TIME_LIMIT="$t" SEED="0" CONFIG_NAME="$cfg" \
    ./runAll_sweep.sh "$INPUT_DIR" "$shell_timeout" "$log_file"
}

for T in "${TIMES[@]}"; do
  echo "=== savings @ ${T}s ==="
  for CFG in "${SAVINGS_CONFIGS[@]}"; do
    run_batch savings "$CFG" "$T"
  done
  echo "=== sweep @ ${T}s ==="
  for CFG in "${SWEEP_CONFIGS[@]}"; do
    run_batch sweep "$CFG" "$T"
  done
done

echo
echo "Done. Logs in ${LOG_DIR}/"
echo "Aggregate with:  python3 src/aggregate_logs.py ${LOG_DIR}"
