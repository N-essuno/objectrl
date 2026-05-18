#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# CarRacing-v3 V8 Study - OLD CONFIGURATION (Fast, No Features)
# ============================================================================
#
# Using the old working configuration that's fast:
# - NO frame stacking (causes 4x slowdown)
# - NO normalization (causes overhead)
# - Just CNN encoder (fast convergence)
# - 1M steps for overnight run
#
# ============================================================================

MAX_JOBS="${MAX_JOBS:-4}"  # ONE PER GPU
SEEDS_STR="${SEEDS_STR:-1 1234 22 3407 42}"
read -r -a SEEDS <<< "${SEEDS_STR}"

ALGORITHMS_STR="${ALGORITHMS_STR:-ppo sac td3}"
read -r -a ALGORITHMS <<< "${ALGORITHMS_STR}"

# Training hyperparameters
MAX_STEPS="${MAX_STEPS:-1000000}"  # 1M steps for overnight
BUFFER_SIZE="${BUFFER_SIZE:-100000}"   # 100K buffer (old config, no frame stack)
BATCH_SIZE="${BATCH_SIZE:-64}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
LEARN_FREQUENCY="${LEARN_FREQUENCY:-8}"

PPO_LEARN_FREQUENCY="${PPO_LEARN_FREQUENCY:-1024}"
PPO_N_EPOCHS="${PPO_N_EPOCHS:-10}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-64}"

DEVICE="${DEVICE:-cuda}"
STORING_DEVICE="${STORING_DEVICE:-cpu}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-RL}"
LOGGING_RESULT_PATH="${LOGGING_RESULT_PATH:-./_logs/V8_old_config}"

# OLD CONFIG: CNN ONLY (no frame stack, no normalize)
USE_CNN="${USE_CNN:-1}"
ENCODER_TYPE="${ENCODER_TYPE:-light}"
USE_FRAME_STACK="${USE_FRAME_STACK:-0}"  # DISABLED - causes 4x slowdown
NORMALIZE_OBS="${NORMALIZE_OBS:-0}"      # DISABLED - causes overhead

GPU_IDS_STR="${GPU_IDS_STR:-0 1 2 3}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
else
  PYTHON_CMD=(uv run python)
fi

warmup_for_algorithm() {
  local algo="$1"
  if [[ "${algo}" == "ppo" ]]; then
    echo "0"
  else
    echo "${WARMUP_STEPS}"
  fi
}

echo "============================================================================"
echo "CarRacing-v3 V8 Study - Old Configuration (FAST)"
echo "============================================================================"
echo "Configuration:"
echo "  Frame stacking: DISABLED (causes 4x slowdown)"
echo "  Normalization: DISABLED (causes overhead)"
echo "  CNN encoder: ENABLED (light version)"
echo "  Buffer size: ${BUFFER_SIZE}"
echo "  Max steps: ${MAX_STEPS}"
echo ""
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "MAX_JOBS: ${MAX_JOBS} (one per GPU)"
echo "============================================================================"
echo ""

mkdir -p "${LOGGING_RESULT_PATH}"

declare -a PIDS
launch_idx=0

for algorithm in "${ALGORITHMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # Wait for available job slot
    while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_JOBS}" ]; do
      sleep 2
    done

    # Round-robin GPU assignment
    gpu_id="${GPU_IDS[$((launch_idx % ${#GPU_IDS[@]}))]}"

    echo "[$((launch_idx + 1))/15] Launching ${algorithm} seed ${seed} on GPU ${gpu_id}..."

    run_warmup="$(warmup_for_algorithm "${algorithm}")"
    run_learn_frequency="${LEARN_FREQUENCY}"
    run_n_epochs="0"
    run_batch_size="${BATCH_SIZE}"

    if [[ "${algorithm}" == "ppo" ]]; then
      run_learn_frequency="${PPO_LEARN_FREQUENCY}"
      run_n_epochs="${PPO_N_EPOCHS}"
      run_batch_size="${PPO_BATCH_SIZE}"
    fi

    run_log="scripts/run_${algorithm}_car-racing_seed_${seed}_V8.log"

    # Build flags (CNN only, no frame stack/normalize)
    cnn_flag=()
    encoder_flag=()

    if [[ "${USE_CNN}" == "1" ]]; then
      cnn_flag=(--env.use_cnn)
      encoder_flag=(--env.encoder_type "${ENCODER_TYPE}")
    fi

    # NO frame_stack_flag
    # NO normalize_flag

    # Launch with proper GPU assignment
    if [[ "${DEVICE}" == "cuda" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_CMD[@]}" -m objectrl.main \
        --system.device="${DEVICE}" \
        --system.storing_device="${STORING_DEVICE}" \
        --system.seed="${seed}" \
        --training.max_steps="${MAX_STEPS}" \
        --training.buffer_size="${BUFFER_SIZE}" \
        --training.batch_size="${run_batch_size}" \
        --training.warmup_steps="${run_warmup}" \
        --training.eval_frequency="${EVAL_FREQUENCY}" \
        --training.eval_episodes="${EVAL_EPISODES}" \
        --training.learn_frequency="${run_learn_frequency}" \
        --training.n_epochs="${run_n_epochs}" \
        --model.name "${algorithm}" \
        --env.name car-racing \
        --logging.result_path "${LOGGING_RESULT_PATH}" \
        "${cnn_flag[@]}" \
        "${encoder_flag[@]}" \
        --verbose \
        > "${run_log}" 2>&1 &
    else
      "${PYTHON_CMD[@]}" -m objectrl.main \
        --system.device="${DEVICE}" \
        --system.storing_device="${STORING_DEVICE}" \
        --system.seed="${seed}" \
        --training.max_steps="${MAX_STEPS}" \
        --training.buffer_size="${BUFFER_SIZE}" \
        --training.batch_size="${run_batch_size}" \
        --training.warmup_steps="${run_warmup}" \
        --training.eval_frequency="${EVAL_FREQUENCY}" \
        --training.eval_episodes="${EVAL_EPISODES}" \
        --training.learn_frequency="${run_learn_frequency}" \
        --training.n_epochs="${run_n_epochs}" \
        --model.name "${algorithm}" \
        --env.name car-racing \
        --logging.result_path "${LOGGING_RESULT_PATH}" \
        "${cnn_flag[@]}" \
        "${encoder_flag[@]}" \
        --verbose \
        > "${run_log}" 2>&1 &
    fi

    PIDS+=("$!")
    launch_idx=$((launch_idx + 1))
  done
done

echo ""
echo "============================================================================"
echo "All 15 experiments launched!"
echo "============================================================================"
echo "Running with OLD CONFIGURATION (fast, no features)"
echo ""
echo "Expected timeline:"
echo "  ~1 hour: First checkpoint (10K steps)"
echo "  ~5 hours: Baseline comparison (50K steps)"
echo "  ~10 hours: Literature comparison (100K steps)"
echo "  ~20+ hours: Full convergence (1M steps)"
echo ""
echo "Monitoring:"
echo "  tail -f scripts/run_ppo_car-racing_seed_1_V8.log"
echo "  nvidia-smi (should show ~4 processes, one per GPU)"
echo "============================================================================"
echo ""

# Wait for all jobs
failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "✓ Job $((i+1))/15 completed"
  else
    echo "✗ Job $((i+1))/15 failed"
    failed=1
  fi
done

if [[ $failed -eq 0 ]]; then
  echo ""
  echo "============================================================================"
  echo "SUCCESS! All experiments completed!"
  echo "============================================================================"
else
  echo ""
  echo "Some jobs failed. Check logs."
  exit 1
fi
