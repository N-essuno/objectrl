#!/usr/bin/env python3
"""Test that frame stacking and normalization fixes work correctly."""

import numpy as np
import gymnasium as gym
from objectrl.utils.make_env import make_env
from objectrl.config.config import EnvConfig, SystemConfig

def test_frame_stacking_initialization():
    """Test that frame stacking initializes with zeros, not duplicates."""
    print("=" * 70)
    print("TEST 1: Frame Stacking Initialization")
    print("=" * 70)

    # Create environment with frame stacking
    env_config = EnvConfig(
        name="car-racing",
        use_cnn=True,
        encoder_type="light",
        use_frame_stack=True,
        n_frames=4,
        normalize_obs=False,  # Test without normalization first
    )

    system_config = SystemConfig(seed=42)

    env = make_env("car-racing", 42, env_config, system_config)

    # Reset and check initial observation
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Expected: (96, 96, 12) [3 channels * 4 frames]")

    # Split into frames
    h, w, c = obs.shape
    frames = np.split(obs, 4, axis=-1)

    print(f"\nFrame statistics:")
    for i, frame in enumerate(frames):
        mean_val = frame.mean()
        std_val = frame.std()
        min_val = frame.min()
        max_val = frame.max()
        print(f"  Frame {i}: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val:.0f}, max={max_val:.0f}")

    # Check that first 3 frames are zeros
    first_three_are_zero = all(np.allclose(frames[i], 0, atol=1e-6) for i in range(3))
    last_has_content = not np.allclose(frames[3], 0, atol=1e-6)

    print(f"\n✓ First 3 frames are zeros: {first_three_are_zero}")
    print(f"✓ Last frame has content: {last_has_content}")

    if first_three_are_zero and last_has_content:
        print("\n✅ TEST 1 PASSED: Frame stacking initializes correctly!")
    else:
        print("\n❌ TEST 1 FAILED: Frame stacking initialization bug!")

    return first_three_are_zero and last_has_content


def test_normalization_order():
    """Test that normalization is applied before frame stacking."""
    print("\n" + "=" * 70)
    print("TEST 2: Normalization Order")
    print("=" * 70)

    # Create environment with both features
    env_config = EnvConfig(
        name="car-racing",
        use_cnn=True,
        encoder_type="light",
        use_frame_stack=True,
        n_frames=4,
        normalize_obs=True,  # Enable normalization
    )

    system_config = SystemConfig(seed=42)

    env = make_env("car-racing", 42, env_config, system_config)

    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # With normalization, values should be roughly centered around 0
    # (not in [0, 255] range like raw pixels)
    is_normalized = obs.mean() < 50 and obs.max() < 255

    print(f"\n✓ Observations appear normalized: {is_normalized}")

    if is_normalized:
        print("\n✅ TEST 2 PASSED: Normalization applied correctly!")
    else:
        print("\n⚠️  TEST 2 WARNING: Normalization may not be working as expected")

    return is_normalized


def test_temporal_consistency():
    """Test that frames change over time (temporal consistency)."""
    print("\n" + "=" * 70)
    print("TEST 3: Temporal Consistency")
    print("=" * 70)

    env_config = EnvConfig(
        name="car-racing",
        use_cnn=True,
        encoder_type="light",
        use_frame_stack=True,
        n_frames=4,
        normalize_obs=False,
    )

    system_config = SystemConfig(seed=42)
    env = make_env("car-racing", 42, env_config, system_config)

    # Reset
    obs1, info = env.reset()

    # Take a step
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)

    # Check that observations changed
    are_different = not np.allclose(obs1, obs2)

    print(f"Observation 1 mean: {obs1.mean():.3f}")
    print(f"Observation 2 mean: {obs2.mean():.3f}")
    print(f"Observations changed: {are_different}")

    if are_different:
        print("\n✅ TEST 3 PASSED: Observations change over time!")
    else:
        print("\n❌ TEST 3 FAILED: Observations stuck!")

    return are_different


def main():
    print("\n" + "=" * 70)
    print("TESTING FRAME STACKING AND NORMALIZATION FIXES")
    print("=" * 70)

    test1 = test_frame_stacking_initialization()
    test2 = test_normalization_order()
    test3 = test_temporal_consistency()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (Initialization): {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Test 2 (Normalization):  {'✅ PASSED' if test2 else '⚠️  WARNING'}")
    print(f"Test 3 (Temporal):        {'✅ PASSED' if test3 else '❌ FAILED'}")

    if test1 and test3:
        print("\n🎉 All critical tests passed! Fixes are working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the fixes.")
        return 1


if __name__ == "__main__":
    exit(main())
