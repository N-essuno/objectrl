#!/usr/bin/env python3
"""
Quick script to run a single algorithm on a single environment for testing.
"""

import sys
from pathlib import Path

def run_single_experiment(
    algorithm: str = "grpo",
    env_name: str = "cartpole-swingup-v0",
    training_steps: int = 100_000,
    seed: int = 42,
    result_path: str = "./test_results"
):
    """Run a single experiment."""

    import subprocess

    result_path = Path(result_path) / f"{algorithm}_{env_name}_seed{seed}"
    result_path.mkdir(parents=True, exist_ok=True)

    print(f"Running {algorithm.upper()} on {env_name}")
    print(f"Training steps: {training_steps:,}")
    print(f"Result path: {result_path}\n")

    cmd = [
        "python", "-m", "objectrl.main",
        "--env.name", env_name,
        "--model.name", algorithm,
        "--logging.result-path", str(result_path),
        "--system.seed", str(seed),
        "--training.max_steps", str(training_steps),
        "--training.eval_frequency", str(training_steps//10),
        "--logging.save_frequency", str(training_steps//2),
        "--training.warmup_steps", "0",
        "--system.device", "cpu",  # Use CPU to avoid GPU memory issues
        "--system.storing_device", "cpu",
    ]

    # Add frame stacking for car racing
    if env_name == "car-racing":
        cmd.extend([
            "--env.use-frame-stack", "true",
            "--env.n_frames", "4",
            "--env.use-cnn", "true",
        ])

    # Add algorithm-specific settings
    if algorithm == "grpo":
        cmd.extend([
            "--model.n_groups", "8",
            "--model.group_size", "4",
            "--training.n_epochs", "10",
        ])
    elif algorithm == "ppo":
        cmd.extend([
            "--training.n_epochs", "10",
        ])
    elif algorithm in ["sac", "td3"]:
        cmd.extend([
            "--training.warmup_steps", "10000",
        ])

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✓ Experiment completed successfully!")
        print(f"Results saved to: {result_path}")
        return 0
    else:
        print(f"\n✗ Experiment failed!")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single RL experiment")
    parser.add_argument("--algorithm", type=str, default="grpo",
                       choices=["grpo", "ppo", "sac", "td3"],
                       help="Algorithm to run")
    parser.add_argument("--env", type=str, default="cartpole-swingup-v0",
                       help="Environment name")
    parser.add_argument("--steps", type=int, default=100_000,
                       help="Number of training steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--result-path", type=str, default="./test_results",
                       help="Base path for results")

    args = parser.parse_args()

    sys.exit(run_single_experiment(
        algorithm=args.algorithm,
        env_name=args.env,
        training_steps=args.steps,
        seed=args.seed,
        result_path=args.result_path
    ))
