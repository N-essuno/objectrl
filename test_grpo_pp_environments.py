#!/usr/bin/env python3
"""
Quick test script to verify GRPO++ works on all three environments.
Tests for basic functionality without full training runs.
"""

import subprocess
import sys
from pathlib import Path


def test_environment(env_name: str, use_frame_stack: bool = False, use_cnn: bool = False):
    """Test GRPO++ on a single environment."""

    result_path = Path("./test_grpo_pp") / f"{env_name}"
    result_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Testing GRPO++ on {env_name}")
    print(f"{'='*60}")

    cmd = [
        "python", "-m", "objectrl.main",
        "--env.name", env_name,
        "--model.name", "grpo_pp",
        "--logging.result-path", str(result_path),
        "--system.seed", "42",
        "--training.max_steps", "1000",  # Short test run
        "--training.eval_frequency", "500",
        "--training.warmup_steps", "0",
        "--system.device", "cpu",
        "--system.storing_device", "cpu",
    ]

    # Add frame stacking if needed
    if use_frame_stack:
        cmd.extend([
            "--env.use-frame-stack", "true",
            "--env.n_frames", "4",
        ])

    if use_cnn:
        cmd.extend([
            "--env.use-cnn", "true",
        ])

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print(f"✓ {env_name}: PASSED")
            return True
        else:
            print(f"✗ {env_name}: FAILED")
            print(f"STDERR: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ {env_name}: TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {env_name}: ERROR - {e}")
        return False


def main():
    """Test GRPO++ on all three environments."""

    print("="*60)
    print("GRPO++ Environment Compatibility Test")
    print("="*60)

    environments = [
        ("car-racing", True, True),           # CarRacing with frame stacking and CNN
        ("cartpole-swingup-v0", False, False),  # CartPole swingup (default)
        ("acrobot-swingup-v0", False, False),   # Acrobot swingup (default)
    ]

    results = {}
    for env_name, use_frame_stack, use_cnn in environments:
        results[env_name] = test_environment(env_name, use_frame_stack, use_cnn)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for env_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{env_name:25s}: {status}")

    total_passed = sum(1 for passed in results.values() if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! GRPO++ is compatible with all environments.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())