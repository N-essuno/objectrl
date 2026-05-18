#!/usr/bin/env python3
"""
Test script for vanilla GRPO implementation.

This script verifies that GRPO can be instantiated and run on simple environments.
"""

import torch
from objectrl.config.config import MainConfig
from objectrl.models.get_model import get_model
from objectrl.utils.make_env import make_env
from objectrl.utils.utils import totorch, tonumpy


def test_grpo_instantiation():
    """Test that GRPO can be instantiated with default config."""
    print("Testing GRPO instantiation...")

    try:
        # Create a minimal config for GRPO using from_config
        config_dict = {
            "model": {
                "name": "grpo",
            },
            "env": {
                "name": "cheetah",  # Continuous action environment
            },
            "training": {
                "max_steps": 100,
                "warmup_steps": 0,  # GRPO requires no warmup
            },
        }
        config = MainConfig.from_config(config_dict)

        # Create environment and inject into config
        env = make_env(
            env_name=config.env.name,
            seed=config.system.seed,
            env_config=config.env,
            eval_env=False,
        )
        config.env.env = env

        agent = get_model(config)
        print(f"✓ GRPO agent created successfully: {agent._agent_name}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to create GRPO agent: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grpo_on_cheetah():
    """Test GRPO on HalfCheetah environment (continuous control)."""
    print("\nTesting GRPO on HalfCheetah...")

    try:
        # Create config
        config_dict = {
            "model": {
                "name": "grpo",
            },
            "env": {
                "name": "cheetah",  # Continuous action environment
            },
            "training": {
                "max_steps": 500,
                "eval_frequency": 100,
                "warmup_steps": 0,  # GRPO requires no warmup
            },
        }
        config = MainConfig.from_config(config_dict)

        # Create environment
        env = make_env(
            env_name=config.env.name,
            seed=config.system.seed,
            env_config=config.env,
            eval_env=False,
        )

        # Inject environment into config (as Experiment does)
        config.env.env = env

        # Create agent
        agent = get_model(config)
        print(f"✓ Agent created: {agent._agent_name}")

        # Test a few steps
        state, info = env.reset()
        state = totorch(state, device=agent.device)
        print(f"✓ Environment reset. State shape: {state.shape}")

        for step in range(10):
            action_dict = agent.select_action(state, is_training=True)
            action = action_dict["action"]
            action_logprob = action_dict["action_logprob"]

            next_state, reward, terminated, truncated, info = env.step(tonumpy(action))
            next_state = totorch(next_state, device=agent.device)

            # Store transition
            transition = agent.generate_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                truncated=truncated,
                info=info,
                action_logprob=action_logprob,
                step=step + 1,  # Add step parameter
            )
            agent.store_transition(transition)

            state = next_state

            if terminated or truncated:
                break

        print(f"✓ Completed {step + 1} steps successfully")

        # Test learning step
        if len(agent.experience_memory) > 0:
            agent.learn(max_iter=1)
            print("✓ Learning step completed successfully")

        env.close()
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grpo_config_parameters():
    """Test that GRPO-specific parameters are accessible."""
    print("\nTesting GRPO configuration parameters...")

    try:
        config_dict = {
            "model": {
                "name": "grpo",
            },
            "env": {
                "name": "cheetah",  # Continuous action environment
            },
        }
        config = MainConfig.from_config(config_dict)

        # Check GRPO-specific parameters
        assert hasattr(config.model, 'n_groups'), "Missing n_groups parameter"
        assert hasattr(config.model, 'group_size'), "Missing group_size parameter"
        assert hasattr(config.model, 'clip_rate'), "Missing clip_rate parameter"
        assert hasattr(config.model, 'GAE_lambda'), "Missing GAE_lambda parameter"

        print(f"✓ n_groups: {config.model.n_groups}")
        print(f"✓ group_size: {config.model.group_size}")
        print(f"✓ clip_rate: {config.model.clip_rate}")
        print(f"✓ GAE_lambda: {config.model.GAE_lambda}")
        print(f"✓ normalize_advantages: {config.model.normalize_advantages}")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("GRPO Implementation Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Instantiation", test_grpo_instantiation()))
    results.append(("Configuration", test_grpo_config_parameters()))
    results.append(("HalfCheetah Environment", test_grpo_on_cheetah()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)
