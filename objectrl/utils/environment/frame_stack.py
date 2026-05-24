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

import numpy as np
from gymnasium import Wrapper


class FrameStack(Wrapper):
    """
    Frame stacking wrapper for temporal understanding in visual environments.

    Stacks the last N frames along the channel dimension to provide temporal context.
    Automatically detects if the environment is visual (has spatial observations) and
    enables frame stacking for CarRacing and similar environments.

    Args:
        env (gym.Env): The environment to wrap.
        num_frames (int): Number of frames to stack (default: 4 for CarRacing).
        lazy (bool): If True, frames are concatenated on the channel dimension.
                     If False, returns a list of frames (not recommended for training).
    """

    def __init__(self, env, num_frames: int = 4, lazy: bool = True):
        super().__init__(env)
        self.num_frames = num_frames
        self.lazy = lazy

        # Check if environment has visual observations
        self.is_visual = len(env.observation_space.shape) == 3

        if not self.is_visual:
            # For non-visual environments, don't modify observation space
            self.frames = None
            return

        # Calculate new observation space shape
        old_shape = env.observation_space.shape
        # Stack along channel dimension (assumes H, W, C format)
        new_shape = (old_shape[0], old_shape[1], old_shape[2] * num_frames)

        # Update observation space
        from gymnasium.spaces import Box
        self.observation_space = Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

        # Initialize frame buffer
        self.frames = np.zeros(new_shape, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        """Reset the environment and initialize frame buffer."""
        obs, info = self.env.reset(**kwargs)

        if not self.is_visual:
            return obs, info

        # Fill frame buffer with initial observation
        for _ in range(self.num_frames):
            self._insert_frame(obs)

        return self.frames.copy(), info

    def step(self, action):
        """Step the environment and update frame buffer."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.is_visual:
            return obs, reward, terminated, truncated, info

        # Update frame buffer
        self._insert_frame(obs)

        return self.frames.copy(), reward, terminated, truncated, info

    def _insert_frame(self, frame):
        """Insert a new frame into the buffer, shifting old frames."""
        if not self.is_visual:
            return

        # Shift existing frames
        frame_height, frame_width, num_channels = self.env.observation_space.shape
        self.frames[:, :, :-num_channels] = self.frames[:, :, num_channels:]
        # Insert new frame
        self.frames[:, :, -num_channels:] = frame