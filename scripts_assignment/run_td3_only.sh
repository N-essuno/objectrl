#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# CarRacing-v3 TD3 Only - Old Configuration (Fast, No Features)
# ============================================================================
#
# Running only TD3 algorithm with 5 seeds
# Using the old working configuration that's fast:
# - NO frame stacking (causes 4x slowdown)
# - NO normalization (causes overhead)
# - Just CNN encoder (fast convergence)
# - 1M steps for overnight run
#
# ============================================================================

DEVICE="${DEVICE:-cuda}"
if [[ "${DEVICE}" == "cuda" ]]; then
  STORING_DEVICE="${STORING_DEVICE:-cuda}"
else
  STORING_DEVICE="${STORING_DEVICE:-cpu}"
fi
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"

# Scheduler defaults: GPU mode keeps old behavior, CPU mode auto-sizes jobs.
if [[ "${DEVICE}" == "cpu" ]]; then
  cpu_cores="$(nproc)"
  CPU_TOTAL_CORES="${CPU_TOTAL_CORES:-50}"

  # Do not schedule more cores than physically available.
  effective_cpu_cores="${cpu_cores}"
  if [[ "${CPU_TOTAL_CORES}" -lt "${cpu_cores}" ]]; then
    effective_cpu_cores="${CPU_TOTAL_CORES}"
  fi

  if [[ -z "${MAX_JOBS:-}" ]]; then
    MAX_JOBS="$((effective_cpu_cores / CPU_THREADS_PER_JOB))"
    if [[ "${MAX_JOBS}" -lt 1 ]]; then
      MAX_JOBS=1
    fi
  else
    MAX_JOBS="${MAX_JOBS}"
  fi

  # Limit per-process CPU threads to avoid oversubscription across parallel jobs.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS_PER_JOB}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS_PER_JOB}}"
else
  MAX_JOBS="${MAX_JOBS:-}"
fi

SEEDS_STR="${SEEDS_STR:-1 1234 22 3407 42}"
read -r -a SEEDS <<< "${SEEDS_STR}"
SEED_REPEATS="${SEED_REPEATS:-1}"

# Expand seeds when many parallel runs are needed (e.g., large GPU fleets).
# Replicated seeds are offset deterministically to remain unique.
if [[ "${SEED_REPEATS}" -gt 1 ]]; then
  expanded_seeds=()
  for ((rep=0; rep<SEED_REPEATS; rep++)); do
    for seed in "${SEEDS[@]}"; do
      expanded_seeds+=("$((seed + rep * 100000))")
    done
  done
  SEEDS=("${expanded_seeds[@]}")
fi

# Only TD3
ALGORITHMS_STR="${ALGORITHMS_STR:-td3}"
read -r -a ALGORITHMS <<< "${ALGORITHMS_STR}"
TOTAL_RUNS="$(( ${#SEEDS[@]} * ${#ALGORITHMS[@]} ))"

# Training hyperparameters
MAX_STEPS="${MAX_STEPS:-1000000}"  # 1M steps for overnight
BUFFER_SIZE="${BUFFER_SIZE:-100000}"   # 100K buffer (old config, no frame stack)
BATCH_SIZE="${BATCH_SIZE:-64}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
LEARN_FREQUENCY="${LEARN_FREQUENCY:-8}"

# CPU threads per process (env stepping + PyTorch inter-op work).
# 10 threads × 5 jobs = 50 cores total on a GPU run.
NUM_THREADS_PER_JOB="${NUM_THREADS_PER_JOB:-10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_THREADS_PER_JOB}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NUM_THREADS_PER_JOB}}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-RL}"
LOGGING_RESULT_PATH="${LOGGING_RESULT_PATH:-./_logs/TD3_only_GPU}"

# OLD CONFIG: CNN ONLY (no frame stack, no normalize)
USE_CNN="${USE_CNN:-1}"
ENCODER_TYPE="${ENCODER_TYPE:-light}"
USE_FRAME_STACK="${USE_FRAME_STACK:-0}"  # DISABLED - causes 4x slowdown
NORMALIZE_OBS="${NORMALIZE_OBS:-0}"      # DISABLED - causes overhead

GPU_IDS_STR="${GPU_IDS_STR:-}"
GPU_IDS=()

if [[ "${DEVICE}" == "cuda" && -z "${GPU_IDS_STR:-}" ]]; then
  # Prefer CUDA_VISIBLE_DEVICES when set — nvidia-smi ignores it and would
  # return all physical GPUs, causing jobs to land on unintended devices.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
    if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
      echo "No GPUs detected by nvidia-smi; set GPU_IDS_STR explicitly."
      exit 1
    fi
  else
    echo "nvidia-smi not found; set GPU_IDS_STR or CUDA_VISIBLE_DEVICES."
    exit 1
  fi
elif [[ "${DEVICE}" == "cuda" ]]; then
  read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
fi

if [[ "${DEVICE}" == "cuda" && -z "${MAX_JOBS:-}" ]]; then
  MAX_JOBS="${#GPU_IDS[@]}"
fi

# Never schedule more workers than total runs.
if [[ -n "${MAX_JOBS:-}" && "${MAX_JOBS}" -gt "${TOTAL_RUNS}" ]]; then
  MAX_JOBS="${TOTAL_RUNS}"
fi

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
else
  PYTHON_CMD=(uv run python)
fi

echo "============================================================================"
echo "CarRacing-v3 TD3 Only - Old Configuration (FAST)"
echo "============================================================================"
echo "Configuration:"
echo "  Frame stacking: DISABLED (causes 4x slowdown)"
echo "  Normalization: DISABLED (causes overhead)"
echo "  CNN encoder: ENABLED (light version)"
echo "  Buffer size: ${BUFFER_SIZE}"
echo "  Max steps: ${MAX_STEPS}"
echo ""
echo "Algorithm: TD3"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: ${#SEEDS[@]}"
if [[ "${DEVICE}" == "cuda" ]]; then
  echo "GPUs: ${GPU_IDS[*]}"
  echo "MAX_JOBS: ${MAX_JOBS} (one per GPU by default)"
  echo "Storage device: ${STORING_DEVICE}"
  echo "CPU threads/job (OMP/MKL): ${NUM_THREADS_PER_JOB}"
else
  echo "MAX_JOBS: ${MAX_JOBS} (CPU parallel workers)"
  echo "CPU core budget: ${effective_cpu_cores}"
  echo "CPU threads/job: ${CPU_THREADS_PER_JOB}"
  echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
  echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
fi
echo "============================================================================"
echo ""

mkdir -p "${LOGGING_RESULT_PATH}"

declare -a PIDS
launch_idx=0
total_runs="${TOTAL_RUNS}"

for algorithm in "${ALGORITHMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # Wait for available job slot
    while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_JOBS}" ]; do
      sleep 1
    done

    # Round-robin GPU assignment (GPU mode only)
    gpu_id=""
    if [[ "${DEVICE}" == "cuda" ]]; then
      gpu_id="${GPU_IDS[$((launch_idx % ${#GPU_IDS[@]}))]}"
    fi

    if [[ "${DEVICE}" == "cuda" ]]; then
      echo "[$((launch_idx + 1))/${total_runs}] Launching ${algorithm} seed ${seed} on GPU ${gpu_id}..."
    else
      echo "[$((launch_idx + 1))/${total_runs}] Launching ${algorithm} seed ${seed} on CPU..."
    fi

    run_warmup="${WARMUP_STEPS}"
    run_learn_frequency="${LEARN_FREQUENCY}"
    run_n_epochs="0"
    run_batch_size="${BATCH_SIZE}"

    run_log="scripts/run_${algorithm}_car-racing_seed_${seed}_TD3_only.log"

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
        --system.num_threads="${NUM_THREADS_PER_JOB}" \
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
        --system.num_threads="${NUM_THREADS_PER_JOB}" \
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
echo "All ${total_runs} TD3 experiments launched!"
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
echo "  tail -f scripts/run_td3_car-racing_seed_1_TD3_only.log"
echo "  nvidia-smi (should show ~4 processes, one per GPU)"
echo "============================================================================"
echo ""

# Wait for all jobs
failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "✓ Job $((i+1))/5 completed"
  else
    echo "✗ Job $((i+1))/5 failed"
    failed=1
  fi
done

if [[ $failed -eq 0 ]]; then
  echo ""
  echo "============================================================================"
  echo "SUCCESS! All TD3 experiments completed!"
  echo "============================================================================"
else
  echo ""
  echo "Some jobs failed. Check logs."
  exit 1
fi
