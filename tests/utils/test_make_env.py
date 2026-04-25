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

from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

from objectrl.utils.make_env import make_env


class DummyEnv(gym.Env):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.reset_called_with_seed = None

    def reset(self, seed=None, **kwargs):
        self.reset_called_with_seed = seed
        return np.zeros(self.observation_space.shape), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def action_space_seed(self, seed):
        pass

    def observation_space_seed(self, seed):
        pass


@pytest.fixture
def env_config():
    class NoisyConfig:
        noisy_act = 0.0
        noisy_obs = 0.0

    class Config:
        noisy = NoisyConfig()
        position_delay = 0
        control_cost_weight = 0.0

    return Config()


@patch("gymnasium.envs.registry")
@patch("gymnasium.make")
def test_make_env_basic(mock_make, mock_registry, env_config):
    mock_registry.keys.return_value = ["Dummy-v0"]

    dummy_env = DummyEnv(
        gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    )
    mock_make.return_value = dummy_env

    env = make_env("Dummy-v0", seed=123, env_config=env_config)

    assert env is not None
    assert dummy_env.reset_called_with_seed == 123
    assert isinstance(env, gym.Env)


@patch("gymnasium.envs.registry")
@patch("gymnasium.make")
def test_make_env_car_racing_continuous(mock_make, mock_registry, env_config):
    mock_registry.keys.return_value = ["CarRacing-v3"]

    dummy_env = DummyEnv(
        gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    )
    mock_make.return_value = dummy_env

    env = make_env("car-racing", seed=7, env_config=env_config)

    assert env is not None
    assert dummy_env.reset_called_with_seed == 7
    mock_make.assert_called_with("CarRacing-v3", continuous=True, render_mode=None)


@patch("objectrl.utils.make_env.FlattenObservation", side_effect=lambda env: env)
@patch("objectrl.utils.make_env.DMCEnv")
def test_make_env_dmc_swingup_alias(mock_dmc_env, mock_flatten, env_config):
    dummy_env = DummyEnv(
        gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    )
    mock_dmc_env.return_value = dummy_env

    env = make_env("cartpole-swingup-v0", seed=11, env_config=env_config)

    assert env is not None
    assert dummy_env.reset_called_with_seed == 11
    mock_dmc_env.assert_called_with(
        domain_name="cartpole",
        task_name="swingup",
        task_kwargs={"random": 11},
    )
    mock_flatten.assert_called_once()


@patch("objectrl.utils.make_env.FlattenObservation", side_effect=lambda env: env)
@patch("gymnasium.envs.registry")
@patch("gymnasium.make")
def test_make_env_flattens_multidim_box_observation(
    mock_make, mock_registry, mock_flatten, env_config
):
    mock_registry.keys.return_value = ["CarRacing-v3"]

    dummy_env = DummyEnv(
        gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    )
    dummy_env.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(96, 96, 3), dtype=np.uint8
    )
    mock_make.return_value = dummy_env

    env = make_env("car-racing", seed=5, env_config=env_config)

    assert env is not None
    mock_flatten.assert_called_once_with(dummy_env)
