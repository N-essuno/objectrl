# -----------------------------------------------------------------------------------
# ObjectRL: An Object-Oriented Reinforcement Learning Codebase
# Copyright (C) 2025 ADIN Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------------

import warnings
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation, RescaleAction

from objectrl.utils.environment.dmc_wrappers import DMCEnv
from objectrl.utils.environment.metaworld_wrappers import SparsifyRewardWrapper
from objectrl.utils.environment.noisy_wrappers import (
    NoisyActionWrapper,
    NoisyObservationWrapper,
)
from objectrl.utils.environment.reward_wrappers import PositionDelayWrapper


class RunningMeanStd:
    """Tracks running mean and standard deviation of observations."""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class NormalizeObservation(gym.Wrapper):
    """Normalizes observations using running mean and std."""

    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_rms = RunningMeanStd(epsilon=epsilon, shape=env.observation_space.shape)
        self.training = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.training:
            self.obs_rms.update(obs[np.newaxis, ...])
        return self._normalize(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.training:
            self.obs_rms.update(obs[np.newaxis, ...])
        return self._normalize(obs), info

    def _normalize(self, obs):
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class FrameStack(gym.Wrapper):
    """Stacks the last n frames for temporal context.

    Fixed implementation:
    - Initializes with zeros for proper temporal history
    - Only fills frames with actual observations as they arrive
    - Prevents duplicate-frame bug that hurt training
    """

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Update observation space
        orig_space = env.observation_space
        if isinstance(orig_space, gym.spaces.Box):
            # Shape: (H, W, C) -> (H, W, C * n_frames)
            h, w, c = orig_space.shape
            new_shape = (h, w, c * n_frames)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=new_shape, dtype=orig_space.dtype
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # FIXED: Clear frames and append only the initial obs once
        self.frames.clear()
        # Fill with zeros first to indicate "no history yet"
        zero_frame = np.zeros_like(obs)
        for _ in range(self.n_frames - 1):
            self.frames.append(zero_frame)
        # Only the most recent slot gets the actual observation
        self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Concatenate frames along channel dimension
        return np.concatenate(list(self.frames), axis=-1)

gymnasium_mujoco_mappings = {
    "ant": "Ant-v5",
    "cartpole": "CartPole-v1",
    "cheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "humanoid": "Humanoid-v5",
    "reacher": "Reacher-v5",
    "swimmer": "Swimmer-v5",
    "walker2d": "Walker2d-v5",
}

gymnasium_box2d_mappings = {
    "car-racing": "CarRacing-v3",
    "car_racing": "CarRacing-v3",
    "CarRacing-v3": "CarRacing-v3",
}

dmc_mappings = {
    "dmc-quadruped-run": "dmc-quadruped-run",
    "dmc-humanoid-run": "dmc-humanoid-run",
    "dmc-cheetah-run": "dmc-cheetah-run",
    "dmc-hopper-hop": "dmc-hopper-hop",
    "dmc-walker-run": "dmc-walker-run",
    "dmc-cartpole-swingup": "dmc-cartpole-swingup",
    "dmc-acrobot-swingup": "dmc-acrobot-swingup",
    "cartpole-swingup-v0": "dmc-cartpole-swingup",
    "acrobot-swingup-v0": "dmc-acrobot-swingup",
}

metaworld_mappings = {
    "metaworld-window-close": "window-close-v3",
    "metaworld-window-open": "window-open-v3",
    "metaworld-drawer-close": "drawer-close-v3",
    "metaworld-drawer-open": "drawer-open-v3",
    "metaworld-reach": "reach-v3",
    "metaworld-button-press-topdown": "button-press-topdown-v3",
    "metaworld-door-open": "door-open-v3",
}

env_mappings = {
    # Gymnasium environments
    **gymnasium_mujoco_mappings,
    **gymnasium_box2d_mappings,
    # DMC environments
    **dmc_mappings,
    # MetaWorld environments
    **metaworld_mappings,
}


def _flatten_if_needed(env: gym.Env, keep_image_obs: bool = False) -> gym.Env:
    """Flatten image- or tensor-shaped Box observations to 1D vectors."""
    if (
        isinstance(env.observation_space, gym.spaces.Box)
        and len(env.observation_space.shape) > 1
    ):
        if keep_image_obs and len(env.observation_space.shape) == 3:
            return env
        env = FlattenObservation(env)
    return env


def make_env(  # noqa: C901
    env_name: str, seed: int, env_config, eval_env: bool = False, num_envs: int = 1
) -> gym.Env | gym.vector.VectorEnv:
    """
    Create and configure a Gymnasium environment with optional wrappers for noise,
    reward shaping, and consistent seeding.

    This function supports:
    - Gymnasium MuJoCo tasks
    - DM Control tasks, automatically wrapped for Gymnasium compatibility
    - MetaWorld tasks, with optional sparse rewards
    - Action rescaling to [-1, 1]
    - Noisy action and/or observation wrappers
    - Delayed reward and control cost penalties via PositionDelayWrapper
    - Frame stacking for temporal context (use_frame_stack)
    - Observation normalization (normalize_obs)
    - Reproducibility via consistent seeding for Gym, NumPy, and PyTorch

    Args:
        env_name (str): Name of the environment. Must be present in ``env_mappings`` and can belong to Gymnasium MuJoCo, DM Control, or MetaWorld suites.
        seed (int): Base random seed for reproducibility.
        env_config: Configuration object with nested attributes:
            - env_config.noisy.noisy_act (float): Std of Gaussian noise for actions.
            - env_config.noisy.noisy_obs (float): Std of Gaussian noise for observations.
            - env_config.position_delay (int): Delay threshold for reward.
            - env_config.control_cost_weight (float): Weight for control cost in reward.
            - env_config.use_frame_stack (bool): Whether to stack the last 4 frames.
            - env_config.normalize_obs (bool): Whether to normalize observations.
        eval_env (bool, optional): If True, modifies seed to separate training/testing. Defaults to False.
        num_envs (int, optional): Number of environments which are parallelized if > 1. Defaults to 1.

    Returns:
        gym.Env: The fully constructed and wrapped Gymnasium environment instance.

    Raises:
        gym.error.Error: If `env_name` is not registered in Gym.
    """
    vectorized = num_envs > 1

    seed = seed + (100 if eval_env else 0)
    # Check if the env is in gym.
    env_name = env_mappings.get(env_name, env_name)
    keep_image_obs = bool(getattr(env_config, "use_cnn", False))

    # ruff: noqa: C901
    def _make_single_env():
        registry_keys = set(gym.envs.registry.keys())
        if env_name in gymnasium_mujoco_mappings.values():
            if env_name in registry_keys:
                env = gym.make(
                    env_name,
                    render_mode="human" if getattr(env_config, "render", False) else None,
                )
            else:
                raise gym.error.Error(
                    f"Environment '{env_name}' is not registered in Gym."
                )
        elif env_name in gymnasium_box2d_mappings.values():
            if env_name in registry_keys:
                env = gym.make(
                    env_name,
                    continuous=True,
                    render_mode="human" if getattr(env_config, "render", False) else None,
                )
            else:
                raise gym.error.Error(
                    f"Environment '{env_name}' is not registered in Gym."
                )
        elif env_name in dmc_mappings.values():
            try:
                _, domain, task = env_name.split("-", 2)
            except ValueError as err:
                raise ValueError(
                    f"DMC environment name '{env_name}' is not in the expected format 'dmc-<domain>-<task>'."
                ) from err
            env = DMCEnv(
                domain_name=domain, task_name=task, task_kwargs={"random": seed}
            )
            if getattr(env_config, "render", False):
                warnings.warn(
                    "Live rendering is not supported for DMCEnv in this codepath; continuing without a render window.",
                    stacklevel=2,
                )
            env = FlattenObservation(env)
        elif env_name in metaworld_mappings.values():
            env = gym.make(
                "Meta-World/MT1",
                env_name=env_name,
                seed=seed,
                render_mode="human" if getattr(env_config, "render", False) else None,
            )
            if env_config.sparse_rewards:
                env = SparsifyRewardWrapper(env)
        elif env_name in registry_keys:
            env = gym.make(
                env_name,
                render_mode="human" if getattr(env_config, "render", False) else None,
            )
        else:
            raise gym.error.Error(f"Environment '{env_name}' not found.")

        env = _flatten_if_needed(env, keep_image_obs=keep_image_obs)

        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = RescaleAction(env, np.float32(-1.0), np.float32(1.0))

        if env_config.noisy:
            if env_config.noisy.noisy_act > 0:
                env = NoisyActionWrapper(env, noise_act=env_config.noisy.noisy_act)
            if env_config.noisy.noisy_obs > 0:
                env = NoisyObservationWrapper(env, noise_obs=env_config.noisy.noisy_obs)

        if env_config.position_delay or env_config.control_cost_weight:
            env = PositionDelayWrapper(
                env,
                position_delay=env_config.position_delay,
                ctrl_w=env_config.control_cost_weight,
            )

        # FIXED: Apply normalization FIRST (before stacking)
        # CRITICAL: Add observation normalization for stability
        if getattr(env_config, "normalize_obs", False):
            env = NormalizeObservation(env)

        # CRITICAL: Add frame stacking for temporal context (required for CarRacing)
        # Applied AFTER normalization so we stack normalized frames
        if getattr(env_config, "use_frame_stack", False) and keep_image_obs:
            n_frames = getattr(env_config, "n_frames", 4)
            env = FrameStack(env, n_frames=n_frames)

        return env

    def _make_wrappers(env, env_config):
        env = _flatten_if_needed(env, keep_image_obs=keep_image_obs)

        if not isinstance(env.action_space, gym.spaces.Discrete):
            env = RescaleAction(env, np.float32(-1.0), np.float32(1.0))

        if env_config.noisy:
            if env_config.noisy.noisy_act > 0:
                env = NoisyActionWrapper(env, noise_act=env_config.noisy.noisy_act)
            if env_config.noisy.noisy_obs > 0:
                env = NoisyObservationWrapper(env, noise_obs=env_config.noisy.noisy_obs)

        if env_config.position_delay or env_config.control_cost_weight:
            env = PositionDelayWrapper(
                env,
                position_delay=env_config.position_delay,
                ctrl_w=env_config.control_cost_weight,
            )

        return env

    if vectorized:
        env = gym.make_vec(
            env_name,
            num_envs=num_envs,
            wrappers=[lambda env: _make_wrappers(env, env_config)],
            vectorization_mode=(
                "sync" if eval_env else "async"
            ),  # Just a recommendation, no performance evaluation so far (2025-07-15)
        )
    else:
        env = _make_single_env()

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return env
