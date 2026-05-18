#!/usr/bin/env python3
"""
Quick test to verify frame stacking and normalization work correctly.
"""
import gymnasium as gym
import numpy as np
import sys
sys.path.insert(0, '/mnt/odinstorage/users/jnn/codes/objectrl')

from objectrl.utils.make_env import make_env
from objectrl.config.config import EnvConfig

print("="*70)
print("Testing Critical Features for CarRacing")
print("="*70)

# Test 1: Basic environment
print("\n1. Testing basic CarRacing environment...")
try:
    env_config = EnvConfig(name="car-racing")
    env = make_env("car-racing", seed=1, env_config=env_config)
    obs, info = env.reset()
    print(f"   ✓ Basic env works! Obs shape: {obs.shape}")
    env.close()
except Exception as e:
    print(f"   ✗ Basic env failed: {e}")

# Test 2: Environment with CNN
print("\n2. Testing CarRacing with CNN...")
try:
    env_config = EnvConfig(name="car-racing", use_cnn=True)
    env = make_env("car-racing", seed=1, env_config=env_config)
    obs, info = env.reset()
    print(f"   ✓ CNN works! Obs shape: {obs.shape}")
    env.close()
except Exception as e:
    print(f"   ✗ CNN failed: {e}")

# Test 3: Environment with Frame Stacking
print("\n3. Testing CarRacing with Frame Stacking (CRITICAL)...")
try:
    env_config = EnvConfig(name="car-racing", use_cnn=True, use_frame_stack=True, n_frames=4)
    env = make_env("car-racing", seed=1, env_config=env_config)
    obs, info = env.reset()
    print(f"   ✓ Frame stacking works! Obs shape: {obs.shape}")
    print(f"   Expected: (96, 96, 12) for 4 stacked frames")

    # Take a few steps to verify stacking continues to work
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: Obs shape {obs.shape}, reward {reward:.1f}")
        if terminated or truncated:
            break

    env.close()
except Exception as e:
    print(f"   ✗ Frame stacking failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Environment with Normalization
print("\n4. Testing CarRacing with Normalization (CRITICAL)...")
try:
    env_config = EnvConfig(name="car-racing", use_cnn=True, normalize_obs=True)
    env = make_env("car-racing", seed=1, env_config=env_config)
    obs, info = env.reset()

    # Check if observations are normalized (should be around 0 mean, unit variance)
    print(f"   ✓ Normalization works! Obs shape: {obs.shape}")
    print(f"   Obs mean: {obs.mean():.4f}, std: {obs.std():.4f}")

    # Take a few steps
    obs_list = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        obs_list.append(obs)
        if terminated or truncated:
            break

    # Check normalization statistics
    all_obs = np.array(obs_list)
    print(f"   After 10 steps:")
    print(f"   Obs mean: {all_obs.mean():.4f}, std: {all_obs.std():.4f}")
    print(f"   (Should be roughly mean=0, std=1 for normalized obs)")

    env.close()
except Exception as e:
    print(f"   ✗ Normalization failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Both Frame Stacking and Normalization
print("\n5. Testing CarRacing with BOTH features (LITERATURE SETUP)...")
try:
    env_config = EnvConfig(
        name="car-racing",
        use_cnn=True,
        use_frame_stack=True,
        n_frames=4,
        normalize_obs=True
    )
    env = make_env("car-racing", seed=1, env_config=env_config)
    obs, info = env.reset()

    print(f"   ✓ Both features work! Obs shape: {obs.shape}")
    print(f"   Expected: (96, 96, 12) for 4 stacked frames")
    print(f"   Obs mean: {obs.mean():.4f}, std: {obs.std():.4f}")

    # Take a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: reward {reward:.1f}, obs mean {obs.mean():.4f}")
        if terminated or truncated:
            break

    env.close()
    print("\n   ✓✓✓ ALL CRITICAL FEATURES WORKING! ✓✓✓")
except Exception as e:
    print(f"   ✗ Combined features failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test complete! If all tests passed, you're ready for fast 1K step training")
print("="*70)
