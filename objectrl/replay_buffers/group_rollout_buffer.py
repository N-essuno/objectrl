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

"""
Group Rollout Buffer for GRPO++.

This buffer maintains trajectory data in a structured format [group_size, max_episode_length, feature_dim]
to support group-wise policy updates with proper masking for variable-length episodes.

Key differences from standard replay buffer:
- Maintains group structure throughout storage and retrieval
- Handles variable-length episodes with padding and masking
- Supports synchronized seeding for identical initial states across groups
- Computes time-aligned advantages using Monte Carlo returns-to-go
"""

import warnings
from typing import Optional

import torch
from tensordict import TensorDict


class GroupRolloutBuffer:
    """
    A rollout buffer designed for group-wise RL algorithms like GRPO++.

    Stores trajectories in structured groups with proper masking for variable-length episodes.
    Maintains data shape: [group_size, max_episode_length, feature_dim].

    Args:
        group_size (int): Number of parallel environments/trajectories per group.
        max_episode_length (int): Maximum trajectory length (for padding).
        state_dim (int | tuple): State observation dimension.
        action_dim (int): Action space dimension.
        device (torch.device): Device for storage.
        gamma (float): Discount factor for computing returns.
        use_identical_seeding (bool): Whether to use identical seeds across all environments.
            False = diverse seeds for better exploration (recommended for stochastic envs).
            True = identical seeds for reduced variance (recommended for deterministic envs).
    """

    def __init__(
        self,
        group_size: int,
        max_episode_length: int,
        state_dim: int | tuple,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        use_identical_seeding: bool = False,
    ):
        self.group_size = group_size
        self.max_episode_length = max_episode_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.use_identical_seeding = use_identical_seeding

        # Handle state dimensions
        if isinstance(state_dim, tuple):
            # Image observations: (H, W, C)
            self.state_shape = state_dim
        else:
            # Vector observations
            self.state_shape = (state_dim,)

        # Initialize storage tensors
        self._init_storage()

        # Track current episode lengths
        self.current_lengths = torch.zeros(group_size, dtype=torch.long, device=device)
        self.current_idx = 0  # Current timestep within episode

    def _init_storage(self):
        """Initialize tensors for storing episode data."""
        # State storage: [group_size, max_episode_length, *state_shape]
        self.states = torch.zeros(
            (self.group_size, self.max_episode_length, *self.state_shape),
            device=self.device,
        )

        # Action storage: [group_size, max_episode_length, action_dim]
        self.actions = torch.zeros(
            (self.group_size, self.max_episode_length, self.action_dim),
            device=self.device,
        )

        # Log probability storage: [group_size, max_episode_length]
        self.logprobs = torch.zeros(
            (self.group_size, self.max_episode_length),
            device=self.device,
        )

        # Reward storage: [group_size, max_episode_length]
        self.rewards = torch.zeros(
            (self.group_size, self.max_episode_length),
            device=self.device,
        )

        # Termination flags: [group_size, max_episode_length]
        self.terminated = torch.zeros(
            (self.group_size, self.max_episode_length),
            dtype=torch.bool,
            device=self.device,
        )

        # Valid mask: [group_size, max_episode_length]
        # False for padded timesteps, True for real data
        self.valid_mask = torch.zeros(
            (self.group_size, self.max_episode_length),
            dtype=torch.bool,
            device=self.device,
        )

    def reset(self, seed: Optional[int] = None):
        """
        Reset the buffer for a new group rollout.

        Args:
            seed (Optional[int]): Random seed for environment reset.
                If use_identical_seeding=True, all environments use this seed.
                If use_identical_seeding=False, environments use seed + i (diverse seeds).

        Returns:
            list[int] | int: List of seeds for each environment (if not identical), or single seed.
        """
        self.current_lengths.zero_()
        self.current_idx = 0
        self.valid_mask.zero_()

        # Reset storage tensors to zero
        self.states.zero_()
        self.actions.zero_()
        self.logprobs.zero_()
        self.rewards.zero_()
        self.terminated.zero_()

        # Generate seeds based on configuration
        if self.use_identical_seeding:
            # All environments use identical seed (reduced variance)
            return [seed] * self.group_size if seed is not None else None
        else:
            # Each environment uses different seed (better exploration)
            if seed is not None:
                return [seed + i for i in range(self.group_size)]
            return None

    def add_step(
        self,
        group_idx: int,
        state: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        terminated: bool,
    ):
        """
        Add a single timestep for a specific environment in the group.

        Args:
            group_idx (int): Index of the environment in the group [0, group_size).
            state (torch.Tensor): State observation.
            action (torch.Tensor): Action taken.
            logprob (torch.Tensor): Log probability of the action.
            reward (float): Reward received.
            terminated (bool): Whether episode terminated.
        """
        if self.current_idx >= self.max_episode_length:
            warnings.warn(
                f"Episode length exceeds max_episode_length ({self.max_episode_length}). "
                "Truncating data.",
                stacklevel=2,
            )
            return

        # Store data
        self.states[group_idx, self.current_idx] = state
        self.actions[group_idx, self.current_idx] = action
        self.logprobs[group_idx, self.current_idx] = logprob
        self.rewards[group_idx, self.current_idx] = reward
        self.terminated[group_idx, self.current_idx] = terminated
        self.valid_mask[group_idx, self.current_idx] = True

        # Update episode length if this environment is still active
        if not terminated:
            self.current_lengths[group_idx] = self.current_idx + 1

        # If this environment terminated, freeze its length
        if terminated and self.current_lengths[group_idx] == 0:
            self.current_lengths[group_idx] = self.current_idx + 1

    def increment_timestep(self):
        """Move to the next timestep across all environments."""
        self.current_idx += 1

    def finalize_episode(self):
        """
        Finalize the current group rollout episode.

        Returns:
            bool: True if at least one episode completed, False otherwise.
        """
        # Update episode lengths for any environments that haven't terminated
        for i in range(self.group_size):
            if self.current_lengths[i] == 0:
                # Episode didn't terminate naturally - use current length
                self.current_lengths[i] = self.current_idx

        # Check if we have valid data
        return self.current_lengths.max() > 0

    def compute_mc_returns(self) -> torch.Tensor:
        """
        Compute Monte Carlo returns-to-go for each timestep.

        Uses the valid mask to handle variable-length episodes properly.

        Returns:
            torch.Tensor: Returns-to-go with shape [group_size, max_episode_length].
        """
        returns = torch.zeros_like(self.rewards)

        # Compute returns backwards from terminal timestep
        for t in range(self.max_episode_length - 1, -1, -1):
            if t == self.max_episode_length - 1:
                # Last timestep: return = reward
                returns[:, t] = self.rewards[:, t]
            else:
                # R_t = r_t + gamma * R_{t+1}
                # Only accumulate for valid timesteps
                next_return = returns[:, t + 1]
                current_return = self.rewards[:, t] + self.gamma * next_return

                # Apply mask: zero out invalid timesteps
                valid = self.valid_mask[:, t]
                returns[:, t] = torch.where(valid, current_return, torch.zeros_like(current_return))

        return returns

    def compute_time_aligned_advantages(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages normalized within each group at each timestep.

        For each timestep t, normalize returns across the group dimension:
        A_{i,t} = (R_{i,t} - mean(R_{:,t})) / (std(R_{:,t}) + epsilon)

        Args:
            returns (torch.Tensor): Returns-to-go [group_size, max_episode_length].

        Returns:
            torch.Tensor: Normalized advantages [group_size, max_episode_length].
        """
        # Compute mean and std across group dimension at each timestep
        mean = returns.mean(dim=0, keepdim=True)
        std = returns.std(dim=0, keepdim=True)

        # Normalize advantages
        advantages = (returns - mean) / (std + 1e-8)

        # Zero out invalid timesteps using mask
        advantages = torch.where(self.valid_mask, advantages, torch.zeros_like(advantages))

        return advantages

    def get_batch(self) -> TensorDict:
        """
        Get the complete group rollout batch with computed advantages.

        Returns:
            TensorDict: Batch containing states, actions, logprobs, advantages, returns, rewards, and valid_mask.
        """
        # Compute returns and advantages
        returns = self.compute_mc_returns()
        advantages = self.compute_time_aligned_advantages(returns)

        # Create TensorDict batch
        batch = TensorDict(
            {
                "state": self.states,  # [group_size, max_episode_length, *state_dim]
                "action": self.actions,  # [group_size, max_episode_length, action_dim]
                "logprob": self.logprobs,  # [group_size, max_episode_length]
                "advantages": advantages,  # [group_size, max_episode_length]
                "returns": returns,  # [group_size, max_episode_length]
                "rewards": self.rewards,  # [group_size, max_episode_length] - for adaptive normalization
                "valid_mask": self.valid_mask,  # [group_size, max_episode_length]
            },
            batch_size=[self.group_size],
        )

        return batch

    def __len__(self) -> int:
        """Return the number of active environments in the group."""
        return self.group_size
