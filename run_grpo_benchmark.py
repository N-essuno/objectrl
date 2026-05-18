#!/usr/bin/env python3
"""
Comprehensive benchmarking script comparing GRPO vs PPO vs SAC vs TD3.

Tests on:
1. CarRacing-v3 (Box2D) - with frame stacking
2. cartpole-swingup-v0 (DMC)
3. acrobot-swingup-v0 (DMC)
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Algorithms to benchmark
ALGORITHMS = ["grpo", "ppo", "sac", "td3"]

# Environments to test
ENVIRONMENTS = {
    "car-racing": {
        "name": "car-racing",
        "use_frame_stack": True,
        "n_frames": 4,
        "use_cnn": True,
        "training_steps": 1_000_000,
        "eval_frequency": 50_000,
        "description": "Box2D Car Racing with frame stacking"
    },
    "cartpole-swingup": {
        "name": "cartpole-swingup-v0",
        "use_frame_stack": False,
        "use_cnn": False,
        "training_steps": 500_000,
        "eval_frequency": 25_000,
        "description": "DMC Cartpole Swingup"
    },
    "acrobot-swingup": {
        "name": "acrobot-swingup-v0",
        "use_frame_stack": False,
        "use_cnn": False,
        "training_steps": 500_000,
        "eval_frequency": 25_000,
        "description": "DMC Acrobot Swingup"
    }
}

# Training configuration
BASE_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 256,
    "gamma": 0.99,
    "buffer_size": 1_000_000,
    "learn_frequency": 1,
    "max_iter": 1,
    "n_epochs": 0,  # PPO/GRPO will use on-policy epochs
    "eval_episodes": 10,
    "save_frequency": 100_000,  # This goes in logging config
}

# Algorithm-specific overrides
ALGORITHM_CONFIGS = {
    "grpo": {
        "clip_rate": 0.2,
        "GAE_lambda": 0.95,
        "normalize_advantages": True,
        "entropy_coef": 0.0,
        "n_groups": 8,
        "group_size": 4,
        "n_epochs": 10,  # On-policy epochs for GRPO
    },
    "ppo": {
        "clip_rate": 0.2,
        "GAE_lambda": 0.95,
        "normalize_advantages": True,
        "entropy_coef": 0.0,
        "n_epochs": 10,  # On-policy epochs for PPO
    },
    "sac": {
        "warmup_steps": 10_000,  # SAC benefits from warmup
        "learn_frequency": 1,
        "max_iter": 1,
        "n_epochs": 0,
    },
    "td3": {
        "warmup_steps": 10_000,  # TD3 benefits from warmup
        "learn_frequency": 1,
        "max_iter": 1,
        "n_epochs": 0,
        "policy_delay": 2,  # Delayed policy updates for TD3
    }
}


def build_command(
    algorithm: str,
    env_config: Dict,
    result_path: Path,
    seed: int = 42
) -> List[str]:
    """Build command line arguments for training run."""

    cmd = [
        "python", "-m", "objectrl.main",
        "--env.name", env_config['name'],
        "--model.name", algorithm,
        "--logging.result-path", str(result_path),
        "--system.seed", str(seed),
        "--system.device", "cpu",  # Use CPU to avoid GPU memory issues
        "--system.storing_device", "cpu",
    ]

    # Add environment-specific settings
    if env_config.get("use_frame_stack"):
        cmd.extend(["--env.use-frame-stack", "true"])
        cmd.extend(["--env.n_frames", str(env_config.get('n_frames', 4))])
    if env_config.get("use_cnn"):
        cmd.extend(["--env.use-cnn", "true"])

    # Add training configuration
    cmd.extend([
        "--training.learning_rate", str(BASE_CONFIG['learning_rate']),
        "--training.batch_size", str(BASE_CONFIG['batch_size']),
        "--training.gamma", str(BASE_CONFIG['gamma']),
        "--training.buffer_size", str(BASE_CONFIG['buffer_size']),
        "--training.eval_episodes", str(BASE_CONFIG['eval_episodes']),
        "--logging.save_frequency", str(BASE_CONFIG['save_frequency']),
        "--training.max_steps", str(env_config['training_steps']),
        "--training.eval_frequency", str(env_config['eval_frequency']),
    ])

    # Add algorithm-specific settings
    algo_config = ALGORITHM_CONFIGS.get(algorithm, {})
    for key, value in algo_config.items():
        cmd.extend([f"--training.{key}", str(value)])

    # Add model-specific settings for GRPO
    if algorithm == "grpo":
        for key, value in algo_config.items():
            if key in ["n_groups", "group_size"]:
                cmd.extend([f"--model.{key}", str(value)])

    return cmd


def run_experiment(
    algorithm: str,
    env_name: str,
    env_config: Dict,
    base_result_path: Path,
    seed: int = 42
) -> bool:
    """Run a single experiment."""

    result_path = base_result_path / f"{algorithm}_{env_name}_seed{seed}"
    result_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Running: {algorithm.upper()} on {env_config['description']}")
    print(f"Result path: {result_path}")
    print(f"Training steps: {env_config['training_steps']:,}")
    print(f"{'='*80}\n")

    cmd = build_command(algorithm, env_config, result_path, seed)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per experiment
        )

        if result.returncode == 0:
            print(f"✓ {algorithm.upper()} on {env_name} completed successfully")
            return True
        else:
            print(f"✗ {algorithm.upper()} on {env_name} failed")
            print(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ {algorithm.upper()} on {env_name} timed out")
        return False
    except Exception as e:
        print(f"✗ {algorithm.upper()} on {env_name} failed with exception: {e}")
        return False


def main():
    """Run all benchmarking experiments."""

    print("="*80)
    print("GRPO vs PPO vs SAC vs TD3 Benchmarking Suite")
    print("="*80)

    # Setup result directory
    base_result_path = Path("./benchmarking_results")
    base_result_path.mkdir(exist_ok=True)

    # Track results
    results = {}
    for algo in ALGORITHMS:
        results[algo] = {}

    # Run experiments
    total_experiments = len(ALGORITHMS) * len(ENVIRONMENTS)
    completed = 0

    for env_name, env_config in ENVIRONMENTS.items():
        print(f"\n\n{'#'*80}")
        print(f"# Environment: {env_config['description']}")
        print(f"# Training steps: {env_config['training_steps']:,}")
        print(f"{'#'*80}")

        for algorithm in ALGORITHMS:
            success = run_experiment(
                algorithm=algorithm,
                env_name=env_name,
                env_config=env_config,
                base_result_path=base_result_path,
                seed=42
            )

            results[algorithm][env_name] = success
            completed += 1

            print(f"\nProgress: {completed}/{total_experiments} experiments completed")

    # Print summary
    print("\n\n" + "="*80)
    print("BENCHMARKING SUMMARY")
    print("="*80)

    for env_name, env_config in ENVIRONMENTS.items():
        print(f"\n{env_config['description']}:")
        for algo in ALGORITHMS:
            status = "✓ PASS" if results[algo][env_name] else "✗ FAIL"
            print(f"  {algo.upper():6s}: {status}")

    # Count successes
    total_success = sum(
        1 for algo in ALGORITHMS
        for env_name in ENVIRONMENTS
        if results[algo][env_name]
    )

    print(f"\nTotal: {total_success}/{total_experiments} experiments successful")

    if total_success == total_experiments:
        print("\n🎉 All benchmarking experiments completed successfully!")
        return 0
    else:
        print("\n⚠️  Some experiments failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
