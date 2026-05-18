#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# Quick Test: OLD WORKING CONFIGURATION (No frame stacking, no normalization)
# ============================================================================
#
# This tests if the frame stacking + normalization features are causing the hang
# Uses the same configuration that worked in previous studies
#
# ============================================================================

MAX_JOBS="${MAX_JOBS:-4}"
SEEDS_STR="${SEEDS_STR:-42}"  # Just one seed for quick test
read -r -a SEEDS <<< "${SEEDS_STR}"

ALGORITHMS_STR="${ALGORITHMS_STR:-ppo sac}"  # Just 2 algos for quick test
read -r -a ALGORITHMS <<< "${ALGORITHMS_STR}"

# OLD WORKING CONFIGURATION
MAX_STEPS="${MAX_STEPS:-50000}"      # Short test
BUFFER_SIZE="${BUFFER_SIZE:-50000}"  # Same as before
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
LOGGING_RESULT_PATH="${LOGGING_RESULT_PATH:-./_logs/test_old_config}"

# OLD CONFIG: NO FEATURES
USE_CNN="${USE_CNN:-1}"
ENCODER_TYPE="${ENCODER_TYPE:-light}"
USE_FRAME_STACK="${USE_FRAME_STACK:-0}"  # DISABLED
NORMALIZE_OBS="${NORMALIZE_OBS:-0}"      # DISABLED

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
echo "QUICK TEST: Old Configuration (No Features)"
echo "============================================================================"
echo "Purpose: Test if frame stacking + normalization cause the hang"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Max steps: ${MAX_STEPS}"
echo ""
echo "CONFIGURATION:"
echo "  Frame stacking: DISABLED"
echo "  Normalization: DISABLED"
echo "  Buffer size: ${BUFFER_SIZE}"
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

    echo "[$((launch_idx + 1))/2] Launching ${algorithm} seed ${seed} on GPU ${gpu_id}..."

    run_warmup="$(warmup_for_algorithm "${algorithm}")"
    run_learn_frequency="${LEARN_FREQUENCY}"
    run_n_epochs="0"
    run_batch_size="${BATCH_SIZE}"

    if [[ "${algorithm}" == "ppo" ]]; then
      run_learn_frequency="${PPO_LEARN_FREQUENCY}"
      run_n_epochs="${PPO_N_EPOCHS}"
      run_batch_size="${PPO_BATCH_SIZE}"
    fi

    run_log="scripts/test_${algorithm}_seed_${seed}_old_config.log"

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
echo "Test launched! 2 experiments (PPO + SAC, seed 42)"
echo "============================================================================"
echo ""
echo "Expected behavior:"
echo "  - Both GPUs should show activity"
echo "  - Log files should be written immediately"
echo "  - First eval at ~10 minutes (10K steps)"
echo "  - Complete in ~2 hours (50K steps)"
echo ""
echo "Monitoring commands:"
echo "  nvidia-smi  # Check GPU utilization (should see 2 GPUs active)"
echo "  tail -f scripts/test_ppo_seed_42_old_config.log"
echo "  tail -f scripts/test_sac_seed_42_old_config.log"
echo "============================================================================"
echo ""

# Wait for all jobs
failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "✓ Test $((i+1))/2 completed"
  else
    echo "✗ Test $((i+1))/2 failed"
    failed=1
  fi
done

if [[ $failed -eq 0 ]]; then
  echo ""
  echo "============================================================================"
  echo "SUCCESS! Old configuration works!"
  echo "This means frame stacking/normalization are causing the hang."
  echo "============================================================================"
else
  echo ""
  echo "Tests failed. Check logs."
  exit 1
fi
