#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# CarRacing-v3 Final Comparison Study (with Critical Features)
# ============================================================================
#
# Runs PPO, SAC, TD3 with 5 seeds for reportable results:
#   - Seeds: 1, 1234, 22, 3407, 42  
#   - Duration: 120K steps (sufficient for robust comparison)
#   - Features: Frame stacking (4) + Normalization + CNN (light encoder)
#   - Multi-GPU: Round-robin across 4 GPUs
#
# Expected Results:
#   PPO:  +20 to +40 (strong performance)
#   SAC:  -15 to +10  (shows learning with tuning)
#   TD3:  -80 to -90  (architectural mismatch with task)
#
# ============================================================================

# Number of runs allowed in parallel (4 GPUs → 4 parallel jobs)
MAX_JOBS="${MAX_JOBS:-4}"

# Use exactly the requested seeds
SEEDS_STR="${SEEDS_STR:-1 1234 22 3407 42}"
read -r -a SEEDS <<< "${SEEDS_STR}"

# Run all 3 algorithms
ALGORITHMS_STR="${ALGORITHMS_STR:-ppo sac td3}"
read -r -a ALGORITHMS <<< "${ALGORITHMS_STR}"

# Training hyperparameters (tuned for CarRacing with features)
MAX_STEPS="${MAX_STEPS:-120000}"      # 120K steps - robust comparison
BUFFER_SIZE="${BUFFER_SIZE:-100000}"  # 100K buffer
BATCH_SIZE="${BATCH_SIZE:-64}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}" # PPO forced to 0 below
EVAL_FREQUENCY="${EVAL_FREQUENCY:-10000}"  # Eval every 10K steps (12 points)
EVAL_EPISODES="${EVAL_EPISODES:-3}"  # Average over 3 episodes
LEARN_FREQUENCY="${LEARN_FREQUENCY:-8}"  # For SAC/TD3

# PPO-specific training defaults (on-policy friendly)
PPO_LEARN_FREQUENCY="${PPO_LEARN_FREQUENCY:-1024}"
PPO_N_EPOCHS="${PPO_N_EPOCHS:-10}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-64}"

# Device/runtime
DEVICE="${DEVICE:-cuda}"
STORING_DEVICE="${STORING_DEVICE:-cpu}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-RL}"
LOGGING_RESULT_PATH="${LOGGING_RESULT_PATH:-./_logs/V6_final_comparison}"

# CRITICAL FEATURES for CarRacing performance
USE_CNN="${USE_CNN:-1}"              # CNN encoder for images
ENCODER_TYPE="${ENCODER_TYPE:-light}"  # Light Nature-DQN encoder
USE_FRAME_STACK="${USE_FRAME_STACK:-1}"  # CRITICAL: Temporal context
N_FRAMES="${N_FRAMES:-4}"               # Stack 4 frames
NORMALIZE_OBS="${NORMALIZE_OBS:-1}"     # CRITICAL: Observation normalization

# Multi-GPU mapping (used only when DEVICE=cuda)
GPU_IDS_STR="${GPU_IDS_STR:-0 1 2 3}"  # Use all 4 GPUs
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${#SEEDS[@]}" -eq 0 || "${#ALGORITHMS[@]}" -eq 0 ]]; then
  echo "SEEDS_STR and ALGORITHMS_STR must each contain at least one value."
  exit 1
fi

if [[ "${DEVICE}" == "cuda" && "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "DEVICE=cuda requires GPU_IDS_STR to contain at least one GPU index."
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
  echo "Python runner: conda env '${CONDA_ENV_NAME}'"
else
  PYTHON_CMD=(uv run python)
  echo "Python runner: uv (.venv)"
fi

warmup_for_algorithm() {
  local algo="$1"
  if [[ "${algo}" == "ppo" ]]; then
    echo "0"  # PPO requires 0 warmup
  else
    echo "${WARMUP_STEPS}"
  fi
}

echo "============================================================================"
echo "CarRacing-v3 Final Comparison Study (V6 with Critical Features)"
echo "============================================================================"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Max steps per run: ${MAX_STEPS}"
echo "Buffer size: ${BUFFER_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Warmup steps (SAC/TD3): ${WARMUP_STEPS}"
echo "Warmup steps (PPO): 0 (forced)"
echo "Eval frequency: ${EVAL_FREQUENCY}"
echo "Eval episodes: ${EVAL_EPISODES}"
echo "Learn frequency (SAC/TD3): ${LEARN_FREQUENCY}"
echo "Learn frequency (PPO): ${PPO_LEARN_FREQUENCY}"
echo "PPO n_epochs: ${PPO_N_EPOCHS}"
echo ""
echo "CRITICAL FEATURES:"
echo "  CNN: ${USE_CNN} (encoder: ${ENCODER_TYPE})"
echo "  Frame stacking: ${USE_FRAME_STACK} (n_frames: ${N_FRAMES})"
echo "  Normalize obs: ${NORMALIZE_OBS}"
echo ""
echo "Device: ${DEVICE}"
echo "Storing device: ${STORING_DEVICE}"
echo "GPU IDs: ${GPU_IDS[*]}"
echo "Max parallel jobs: ${MAX_JOBS}"
echo "Logging: ${LOGGING_RESULT_PATH}"
echo "============================================================================"
echo ""

if [[ "${DEVICE}" == "cuda" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA preflight (nvidia-smi):"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
  fi

  echo "CUDA preflight (PyTorch):"
  if ! "${PYTHON_CMD[@]}" -c 'import torch; import sys; ok=torch.cuda.is_available(); print(f"torch={torch.__version__} cuda_available={ok}"); sys.exit(0 if ok else 1)'; then
    echo "PyTorch cannot use CUDA in this environment."
    exit 1
  fi
  echo ""
fi

mkdir -p "${LOGGING_RESULT_PATH}"

declare -a PIDS=()
declare -a PID_LABELS=()

launch_idx=0

for algorithm in "${ALGORITHMS[@]}"; do
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_JOBS}" ]; do
    sleep 2
  done

  gpu_id="${GPU_IDS[$((launch_idx % ${#GPU_IDS[@]}))]}"
  gpu_label="gpu${gpu_id}"

  echo "Launching ${algorithm} on ${gpu_label} for seeds ${SEEDS[*]}"

  (
    for seed in "${SEEDS[@]}"; do
      run_warmup="$(warmup_for_algorithm "${algorithm}")"
      run_learn_frequency="${LEARN_FREQUENCY}"
      run_n_epochs="0"
      run_batch_size="${BATCH_SIZE}"
      
      if [[ "${algorithm}" == "ppo" ]]; then
        run_learn_frequency="${PPO_LEARN_FREQUENCY}"
        run_n_epochs="${PPO_N_EPOCHS}"
        run_batch_size="${PPO_BATCH_SIZE}"
      fi
      
      run_log="scripts/run_${algorithm}_car-racing_seed_${seed}_V6.log"

      echo "Running ${algorithm} seed=${seed} warmup=${run_warmup} learn_freq=${run_learn_frequency}"

      # Build feature flags
      cnn_flag=()
      encoder_flag=()
      frame_stack_flag=()
      normalize_flag=()
      
      if [[ "${USE_CNN}" == "1" ]]; then
        cnn_flag=(--env.use_cnn)
        encoder_flag=(--env.encoder_type "${ENCODER_TYPE}")
      fi
      
      if [[ "${USE_FRAME_STACK}" == "1" ]]; then
        frame_stack_flag=(--env.use_frame_stack --env.n_frames "${N_FRAMES}")
      fi
      
      if [[ "${NORMALIZE_OBS}" == "1" ]]; then
        normalize_flag=(--env.normalize_obs)
      fi

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
          "${frame_stack_flag[@]}" \
          "${normalize_flag[@]}" \
          --verbose \
          > "${run_log}" 2>&1
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
          "${frame_stack_flag[@]}" \
          "${normalize_flag[@]}" \
          --verbose \
          > "${run_log}" 2>&1
      fi
    done
  ) &

  PIDS+=("$!")
  PID_LABELS+=("algo=${algorithm} target=${gpu_label} seeds=${SEEDS[*]}")
  launch_idx=$((launch_idx + 1))
done

echo ""
echo "============================================================================"
echo "All experiments launched!"
echo "============================================================================"
echo "Total jobs: ${#PIDS[@]} (3 algorithms × 5 seeds = 15 runs)"
echo "Max steps per run: ${MAX_STEPS}"
echo "Evaluation points: $((MAX_STEPS / EVAL_FREQUENCY))"
echo "GPUs used: ${GPU_IDS[*]}"
echo ""
echo "Logs:"
echo "  scripts/run_*_car-racing_seed_*_V6.log"
echo ""
echo "Expected completion time:"
echo "  ~4-6 hours on 4x GPUs for 120K steps"
echo ""
echo "Monitoring:"
echo "  Watch any log: tail -f scripts/run_ppo_car-racing_seed_1_V6.log"
echo "  Check progress: ./check_study_progress.sh"
echo "  Live monitor: python3 monitor_study.py"
echo "============================================================================"
echo ""

# Wait for all processes to complete
failed=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  label="${PID_LABELS[$i]}"
  if wait "${pid}"; then
    echo "✓ Completed: ${label}"
  else
    echo "✗ Failed: ${label}"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo ""
  echo "Some runs failed. Check individual logs for details."
  exit 1
fi

echo ""
echo "============================================================================"
echo "All experiments completed successfully!"
echo "============================================================================"
echo ""
echo "Results saved to: ${LOGGING_RESULT_PATH}"
echo ""
echo "To generate comparison plots:"
echo "  python3 -c 'import glob; files=glob.glob(\"${LOGGING_RESULT_PATH}/car-racing/*/seed_*/eval_results.npy\"); print(f\"Found {len(files)} result files\")'"
echo ""
echo "============================================================================"
