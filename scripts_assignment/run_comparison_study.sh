#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# CarRacing-v3 Comprehensive Algorithm Comparison
# ============================================================================
#
# Runs all 3 algorithms (PPO, SAC, TD3) with 5 seeds for proper comparison:
#   Seeds: 1, 1234, 22, 3407, 42
#   Duration: 20K steps (reasonable length to see learning)
#   Features: Frame stacking + normalization enabled
#
# Expected Results:
#   PPO:  Strong positive performance (+20 to +40)
#   SAC:  Moderate performance (-10 to +5) with variability
#   TD3:  Poor performance (-80 to -90) due to architectural issues
#
# ============================================================================

MAX_JOBS="${MAX_JOBS:-12}"  # Run all 15 experiments in parallel
SEEDS_STR="${SEEDS_STR:-1 1234 22 3407 42}"  # Specific seeds requested
read -r -a SEEDS <<< "${SEEDS_STR}"

ALGORITHMS_STR="${ALGORITHMS_STR:-ppo sac td3}"
read -r -a ALGORITHMS <<< "${ALGORITHMS_STR}"

# Training parameters
MAX_STEPS="${MAX_STEPS:-20000}"      # 20K steps - good balance
BUFFER_SIZE="${BUFFER_SIZE:-100000}" # 100K buffer
BATCH_SIZE="${BATCH_SIZE:-64}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-1000}" # Eval every 1K steps (20 evaluation points)
EVAL_EPISODES="${EVAL_EPISODES:-3}"
LEARN_FREQUENCY="${LEARN_FREQUENCY:-1}"  # Default for SAC/TD3
# Learn frequency will be set per-algorithm (PPO needs >1, SAC/TD3 use 1)
PPO_LEARN_FREQUENCY="${PPO_LEARN_FREQUENCY:-10}"  # PPO needs >1 for normalize_advantages

# Environment
ENV="${ENV:-car-racing}"
DEVICE="${DEVICE:-cuda}"
STORING_DEVICE="${STORING_DEVICE:-cpu}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-RL}"
LOGGING_RESULT_PATH="${LOGGING_RESULT_PATH:-./_logs}"

# Critical features
USE_CNN="${USE_CNN:-1}"
ENCODER_TYPE="${ENCODER_TYPE:-light}"
USE_FRAME_STACK="${USE_FRAME_STACK:-1}"
N_FRAMES="${N_FRAMES:-4}"
NORMALIZE_OBS="${NORMALIZE_OBS:-1}"

echo "============================================================================"
echo "CarRacing-v3 Algorithm Comparison Study"
echo "============================================================================"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Environment: ${ENV}"
echo "Seeds: ${SEEDS[*]}"
echo "Max steps: ${MAX_STEPS}"
echo "Features: Frame stacking (4) + Normalization"
echo "Jobs: ${MAX_JOBS} (running in parallel across 4 GPUs)"
echo "============================================================================"
echo ""

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
  echo "Python runner: conda env '${CONDA_ENV_NAME}'"
else
  PYTHON_CMD=(uv run python)
  echo "Python runner: uv (.venv)"
fi

# Build command flags
CNN_FLAG=()
ENCODER_FLAG=()
FRAME_STACK_FLAG=()
NORMALIZE_FLAG=()

if [[ "${USE_CNN}" == "1" ]]; then
  CNN_FLAG+=(--env.use_cnn)
  ENCODER_FLAG+=(--env.encoder_type "${ENCODER_TYPE}")
fi

if [[ "${USE_FRAME_STACK}" == "1" ]]; then
  FRAME_STACK_FLAG+=(--env.use_frame_stack)
  FRAME_STACK_FLAG+=(--env.n_frames "${N_FRAMES}")
fi

if [[ "${NORMALIZE_OBS}" == "1" ]]; then
  NORMALIZE_FLAG+=(--env.normalize_obs)
fi

mkdir -p "${LOGGING_RESULT_PATH}"

declare -a PIDS=()
declare -a PID_ALGOS=()
declare -a PID_SEEDS=()

launch_idx=0
gpu_counter=0  # Round-robin GPU assignment

for algorithm in "${ALGORITHMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_JOBS}" ]; do
      sleep 2
    done

    env_tag="car-racing"
    gpu_label="cpu"
    gpu_id=""
    if [[ "${DEVICE}" == "cuda" ]]; then
      # Round-robin across 4 GPUs
      gpu_id=$((gpu_counter % 4))
      gpu_label="gpu${gpu_id}"
      gpu_counter=$((gpu_counter + 1))
    fi

    echo "Launching ${algorithm} seed ${seed} (target=${gpu_label})..."

    (
      # Algorithm-specific warmup and learn frequency
      run_warmup="${WARMUP_STEPS}"
      run_learn_frequency="${LEARN_FREQUENCY}"
      if [[ "${algorithm}" == "ppo" ]]; then
        run_warmup="0"  # PPO requires 0 warmup
        run_learn_frequency="${PPO_LEARN_FREQUENCY}"  # PPO needs >1 for normalize_advantages
      fi

      run_log="scripts/run_${algorithm}_seed_${seed}_comparison.log"

      if [[ "${DEVICE}" == "cuda" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_CMD[@]}" -m objectrl.main \
          --system.device="${DEVICE}" \
          --system.storing_device="${STORING_DEVICE}" \
          --system.seed="${seed}" \
          --training.max_steps="${MAX_STEPS}" \
          --training.buffer_size="${BUFFER_SIZE}" \
          --training.batch_size="${BATCH_SIZE}" \
          --training.warmup_steps="${run_warmup}" \
          --training.eval_frequency="${EVAL_FREQUENCY}" \
          --training.eval_episodes="${EVAL_EPISODES}" \
          --training.learn_frequency="${run_learn_frequency}" \
          --model.name "${algorithm}" \
          --env.name "${ENV}" \
          --logging.result_path "${LOGGING_RESULT_PATH}" \
          "${CNN_FLAG[@]}" \
          "${ENCODER_FLAG[@]}" \
          "${FRAME_STACK_FLAG[@]}" \
          "${NORMALIZE_FLAG[@]}" \
          --verbose \
          > "${run_log}" 2>&1
      else
        "${PYTHON_CMD[@]}" -m objectrl.main \
          --system.device="${DEVICE}" \
          --system.storing_device="${STORING_DEVICE}" \
          --system.seed="${seed}" \
          --training.max_steps="${MAX_STEPS}" \
          --training.buffer_size="${BUFFER_SIZE}" \
          --training.batch_size="${BATCH_SIZE}" \
          --training.warmup_steps="${run_warmup}" \
          --training.eval_frequency="${EVAL_FREQUENCY}" \
          --training.eval_episodes="${EVAL_EPISODES}" \
          --training.learn_frequency="${run_learn_frequency}" \
          --model.name "${algorithm}" \
          --env.name "${ENV}" \
          --logging.result_path "${LOGGING_RESULT_PATH}" \
          "${CNN_FLAG[@]}" \
          "${ENCODER_FLAG[@]}" \
          "${FRAME_STACK_FLAG[@]}" \
          "${NORMALIZE_FLAG[@]}" \
          --verbose \
          > "${run_log}" 2>&1
      fi
    ) &

    PIDS+=("$!")
    PID_ALGOS+=("${algorithm}")
    PID_SEEDS+=("${seed}")
    launch_idx=$((launch_idx + 1))
  done
done

echo ""
echo "============================================================================"
echo "All experiments started!"
echo "============================================================================"
echo "Total jobs: ${#PIDS[@]} (3 algorithms × 5 seeds)"
echo "Max steps per job: ${MAX_STEPS}"
echo "Evaluation points: $((MAX_STEPS / EVAL_FREQUENCY))"
echo ""
echo "Logs:"
echo "  scripts/run_*_seed_*_comparison.log"
echo ""
echo "Expected completion time:"
echo "  ~1-1.5 hours on 4x GPUs with MAX_JOBS=12"
echo ""
echo "Monitoring:"
echo "  Watch any log: tail -f scripts/run_ppo_seed_1_comparison.log"
echo "  Check progress: ps aux | grep objectrl"
echo "============================================================================"

# Function to monitor progress
monitor_progress() {
  local update_interval=60  # Update every 60 seconds

  while true; do
    running=0
    for algo in "${ALGORITHMS[@]}"; do
      algo_running=$(ps aux | grep "objectrl.*main.*--model.name ${algo}" | grep -v grep | wc -l)
      running=$((running + algo_running))
    done

    if [ $running -eq 0 ]; then
      echo ""
      echo "============================================================================"
      echo "All experiments completed!"
      echo "============================================================================"
      break
    fi

    clear
    echo "=== Training Progress - $(date +%H:%M:%S) ==="
    echo "Active processes: $running"

    # Check completion status
    for algo in "${ALGORITHMS[@]}"; do
      completed=0
      for seed in "${SEEDS[@]}"; do
        eval_file="${LOGGING_RESULT_PATH}/${ENV}/${algo}/seed_${seed}/*/eval_results.npy"
        if [ -f "$eval_file" ]; then
          steps=$(python3 -c "import numpy as np; d=np.load('$eval_file', allow_pickle=True).item(); print(max(d.keys()) if d else 0)" 2>/dev/null || echo 0)
          if [ "$steps" -ge "$MAX_STEPS" ]; then
            completed=$((completed + 1))
          fi
        fi
      done
      total=${#SEEDS[@]}
      echo "${algo^^}: ${completed}/${total} seeds completed"
    done

    echo ""
    echo "Next update in ${update_interval}s... (Ctrl+C to stop monitoring)"
    sleep $update_interval
  done
}

# Start monitoring in background
monitor_progress

# Wait for all processes to complete
wait

echo ""
echo "============================================================================"
echo "Generating comparison plots..."
echo "============================================================================"

# Check if all experiments completed successfully
failed=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  algo="${PID_ALGOS[$i]}"
  seed="${PID_SEEDS[$i]}"
  if wait "$pid"; then
    echo "✓ ${algo} seed ${seed} completed successfully"
  else
    echo "✗ ${algo} seed ${seed} failed. Check scripts/run_${algo}_seed_${seed}_comparison.log"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo ""
  echo "Some runs failed. Checking for partial results..."
fi

# Generate plots
echo ""
echo "Running plotting script..."

# Create the plotting script
cat > plot_comparison.py << 'EOF'
#!/usr/bin/env python3
"""Plot algorithm comparison results."""
import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

def _to_numpy_1d(values):
    if hasattr(values, "detach"):
        return np.asarray(values.detach().cpu().numpy(), dtype=np.float64).ravel()
    return np.asarray(values, dtype=np.float64).ravel()

def _latest_eval_file(seed_dir):
    latest_path = None
    latest_mtime = -1.0
    for run_dir in seed_dir.iterdir():
        if not run_dir.is_dir():
            continue
        eval_file = run_dir / "eval_results.npy"
        if not eval_file.is_file():
            continue
        mtime = eval_file.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = eval_file
    return latest_path

def _load_seed_step_means(eval_file):
    eval_dict = np.load(eval_file, allow_pickle=True).item()
    out = {}
    for step, rewards in eval_dict.items():
        out[int(step)] = float(_to_numpy_1d(rewards).mean())
    return out

def _aggregate_algorithm(env, algo, logs_root):
    algo_dir = logs_root / env / algo
    if not algo_dir.is_dir():
        print(f"Warning: {algo_dir} not found")
        return None, None, None, None, 0

    per_seed_step_means = []
    for seed_dir in sorted(algo_dir.glob("seed_*")):
        eval_file = _latest_eval_file(seed_dir)
        if eval_file:
            per_seed_step_means.append(_load_seed_step_means(eval_file))

    if not per_seed_step_means:
        print(f"Warning: No eval results found for {algo}")
        return None, None, None, None, 0

    all_steps = sorted({step for data in per_seed_step_means for step in data})
    steps, means, stds, counts = [], [], [], []
    for step in all_steps:
        vals = [seed_data[step] for seed_data in per_seed_step_means if step in seed_data]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            steps.append(step)
            means.append(arr.mean())
            stds.append(arr.std(ddof=0))
            counts.append(arr.size)

    return (np.asarray(steps), np.asarray(means), np.asarray(stds),
            np.asarray(counts), len(per_seed_step_means))

def main():
    env = "car-racing"
    algos = ["ppo", "sac", "td3"]
    logs_root = Path("./_logs")
    output_path = Path("plots/comparison_car_racing.png")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    colors = {'ppo': 'green', 'sac': 'blue', 'td3': 'red'}

    for algo in algos:
        steps, means, stds, counts, n_seeds = _aggregate_algorithm(env, algo, logs_root)
        if steps is None:
            continue

        print(f"\n{algo.upper()}:")
        print(f"  Seeds: {n_seeds}")
        print(f"  Steps: {steps}")
        print(f"  Means: {means}")
        print(f"  Stds: {stds}")
        print(f"  Counts: {counts}")
        overall_mean = np.mean(means) if means.size > 0 else float("nan")
        print(f"  Overall mean: {overall_mean:.2f}")

        ax.errorbar(steps, means, yerr=stds, label=algo.upper(),
                   linewidth=2.5, elinewidth=1.5, capsize=3,
                   color=colors.get(algo, 'black'), alpha=0.8)

    ax.set_title(f"CarRacing-v3: Algorithm Comparison ({n_seeds} seeds)\nFrame Stacking (4) + Normalization Enabled",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Training Steps", fontsize=12, fontweight='bold')
    ax.set_ylabel("Evaluation Reward (Mean ± Std)", fontsize=12, fontweight='bold')

    # Format x-axis
    x_formatter = ticker.ScalarFormatter(useOffset=False)
    x_formatter.set_scientific(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2000))
    ax.xaxis.set_major_formatter(x_formatter)

    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=200, color='orange', linestyle=':', alpha=0.3, linewidth=1.5, label='Tile Discovery')
    ax.axhline(y=500, color='green', linestyle=':', alpha=0.3, linewidth=1.5, label='Good Performance')

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # Set y-axis limits based on data
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, -150), max(ymax, 100))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
EOF

# Make plotting script executable and run it
chmod +x plot_comparison.py
python3 plot_comparison.py

echo ""
echo "============================================================================"
echo "Study Complete!"
echo "============================================================================"
echo ""
echo "Results:"
echo "  - Training logs: scripts/run_*_seed_*_comparison.log"
echo "  - Comparison plot: plots/comparison_car_racing.png"
echo ""
echo "Expected Results Summary:"
echo "  PPO:  Strong positive (+20 to +40) - Should clearly dominate"
echo "  SAC:  Variable (-15 to +10) - Shows learning but unstable"
echo "  TD3:  Poor performance (-80 to -90) - Architectural mismatch"
echo ""
echo "============================================================================"
