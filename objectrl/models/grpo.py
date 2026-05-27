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
Group Relative Policy Optimization (GRPO) — Vanilla Implementation.

Vanilla GRPO (DeepSeekMath) eliminates the learned value baseline entirely.
Instead of training a critic to predict V(s) and using it as a GAE baseline,
GRPO samples groups of trajectories and computes advantages relative to the
group mean at each timestep (time-aligned group normalization).

Algorithm:
1. Collect G episodes into a structured buffer [G, T, *state_dim]
2. Compute Monte Carlo returns-to-go: R_t = sum_{k=t}^{T} gamma^{k-t} r_k
3. Normalize across group at each timestep:
   A_{i,t} = (R_{i,t} - mean_j(R_{j,t})) / (std_j(R_{j,t}) + eps)
4. Update actor with symmetric PPO clipping (epsilon=0.2), mean aggregation
5. Update critic to predict MC returns (optional — critic not used for advantages)

Reference: DeepSeekMath: Pushing the Limits of Mathematical Reasoning
in Open Language Models (2024)
"""

import typing
from typing import Literal

import torch
from tensordict import TensorDict
from torch import nn as nn

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.actor import Actor
from objectrl.models.basic.critic import CriticEnsemble
from objectrl.replay_buffers.group_rollout_buffer import GroupRolloutBuffer
from objectrl.utils.net_utils import MLP, make_pixel_encoder

if typing.TYPE_CHECKING:
    from objectrl.config.config import MainConfig


class GRPOActorNetProbabilistic(nn.Module):
    """
    Probabilistic actor network for GRPO using a Gaussian policy.

    Args:
        dim_state (int | tuple[int,int,int]): Dimension of input state, or HWC image shape.
        dim_act (int): Dimension of action space.
        n_heads (int): Number of heads (only supports 1).
        depth (int): Depth of the MLP.
        width (int): Width of each MLP layer.
        act (Literal["crelu", "relu"]): Activation function.
        has_norm (bool): Whether to include normalization layers.
        upper_clamp (float): Maximum clamp value for log standard deviation.
        encoder_type (str): Pixel encoder variant when dim_state is a tuple: "light" or "vgg".
    """

    def __init__(
        self,
        dim_state: int | tuple[int, int, int],
        dim_act: int,
        n_heads: int = 1,
        depth: int = 3,
        width: int = 256,
        act: Literal["crelu", "relu"] = "relu",
        has_norm: bool = False,
        upper_clamp: float = -1.0,
        encoder_type: str = "light",
    ) -> None:
        super().__init__()
        assert n_heads == 1, "GRPOActorNetProbabilistic only supports n_heads=1"
        self.dim_act = dim_act
        self.n_heads = n_heads
        self.upper_clamp = upper_clamp

        self.encoder = None
        input_dim = dim_state
        if isinstance(dim_state, tuple):
            self.encoder = make_pixel_encoder(dim_state, feature_dim=width, encoder_type=encoder_type)
            input_dim = self.encoder.output_dim

        # Create the network architecture
        self.arch = MLP(input_dim, dim_act, depth, width, act, has_norm)
        self.action_logstd = nn.Parameter(torch.zeros(dim_act))

    def forward(self, x: torch.Tensor, is_training: bool = True) -> dict:
        """
        Forward pass to generate actions and log probabilities.

        Args:
            x (torch.Tensor): Input state tensor.
            is_training (bool): Whether to sample or return mode.

        Returns:
            dict: Dictionary containing action distribution, action, and log-prob.
        """
        action_mean = self.arch(x if self.encoder is None else self.encoder(x))
        action_logstd = self.action_logstd.clamp(max=self.upper_clamp).expand_as(
            action_mean
        )
        action_std = torch.exp(action_logstd)
        # Create a normal distribution with the mean and standard deviation
        dist = torch.distributions.Normal(loc=action_mean, scale=action_std)
        return_dict = {
            "dist": dist,
        }
        if is_training:
            y = dist.sample()
        else:
            y = dist.mode
        # Calculate the log probability of the sampled actions
        y_logprob = dist.log_prob(y).sum(dim=-1)
        return_dict["action"] = y
        return_dict["action_logprob"] = y_logprob
        return return_dict


class GRPOActor(Actor):
    """
    GRPO Actor with standard symmetric PPO clipping and mean aggregation.

    Vanilla GRPO uses the same clipping bound for positive and negative
    advantages (epsilon=0.2), and aggregates the loss by averaging uniformly
    over all timesteps and episodes.

    Args:
        config (MainConfig): Configuration object.
        dim_state (int): Dimension of state space.
        dim_act (int): Dimension of action space.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)

    def loss(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_logprob: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the GRPO clipped surrogate loss with symmetric clipping.

        Args:
            state (torch.Tensor): State input.
            actions (torch.Tensor): Actions taken.
            action_logprob (torch.Tensor): Old log-probs of actions.
            advantages (torch.Tensor): Group-normalized advantages (from MC returns).

        Returns:
            torch.Tensor: Computed loss scalar.
        """
        act_dict = self.act(state, is_training=True)
        new_logprob = act_dict["dist"].log_prob(actions).sum(dim=-1)
        log_ratio = new_logprob - action_logprob
        ratio = torch.exp(log_ratio)

        # Symmetric PPO clipping
        weighted_advantages = advantages * ratio
        weighted_clipped_advantages = advantages * torch.clamp(
            ratio,
            1 - self.config.model.clip_rate,
            1 + self.config.model.clip_rate,
        )
        # Mean aggregation over all timesteps
        loss = -torch.min(weighted_advantages, weighted_clipped_advantages).mean()

        # Add entropy bonus if configured
        if self.config.model.entropy_coef > 0:
            entropy_loss = act_dict["dist"].entropy().sum(-1).mean()
            loss += -self.config.model.entropy_coef * entropy_loss

        return loss

    def update(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_logprob: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        """
        Performs gradient update on the actor network.

        Args:
            state (torch.Tensor): Input state batch.
            actions (torch.Tensor): Sampled actions.
            action_logprob (torch.Tensor): Log-probs of sampled actions.
            advantages (torch.Tensor): Advantage estimates.
        """
        self.optim.zero_grad()
        loss = self.loss(state, actions, action_logprob, advantages)
        loss.backward()
        if self.config.model.actor.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.model.actor.max_grad_norm,
            )
        self.optim.step()
        self.iter += 1


class GRPOCritic(CriticEnsemble):
    """
    GRPO Critic — trains to predict MC returns (not used for advantage computation).

    In vanilla GRPO, advantages come from group statistics, not the critic.
    The critic is kept for value function estimation only.

    Args:
        config (MainConfig): Configuration object.
        dim_state (int): State dimension.
        dim_act (int): Action dimension.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int):
        super().__init__(config, dim_state, dim_act)

    @torch.no_grad()
    def get_bellman_target(self):
        """Placeholder for Bellman target computation."""
        pass

    def update(self, state: torch.Tensor, y: torch.Tensor) -> None:
        """
        Updates critic to predict MC returns.

        Args:
            state (torch.Tensor): State inputs.
            y (torch.Tensor): Target values (MC returns).
        """
        self.optim.zero_grad()
        loss = self.loss(self.Q(state).view_as(y), y)
        loss = loss.sum(0).mean() if self.n_members > 1 else loss.mean()
        loss.backward()
        if self.config.model.critic.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                list(self.parameters()),
                self.config.model.critic.max_grad_norm,
            )
        self.optim.step()
        self.iter += 1


class GroupRelativePolicyOptimization(ActorCritic):
    """
    Vanilla GRPO agent — no critic for advantages, group statistics as baseline.

    Collects G episodes into a structured GroupRolloutBuffer, computes MC
    returns-to-go, normalizes across the group at each timestep (time-aligned),
    then updates the actor with symmetric PPO clipping.

    The critic is trained to predict MC returns but is NOT used for advantage
    computation. This is the defining difference from GRPO++ V2, which adds
    a target critic + GAE.

    Algorithm:
    1. Collect G sequential episodes into GroupRolloutBuffer [G, T, ...]
    2. buf.get_batch() computes MC returns + time-aligned group advantages
    3. Update actor with symmetric clipping, mean aggregation
    4. Update critic to predict MC returns (optional)
    5. Reset buffer

    Reference: DeepSeekMath (2024)
    """

    _agent_name = "GRPO"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = GRPOCritic,
        actor_type: type = GRPOActor,
    ) -> None:
        """
        Initializes the vanilla GRPO agent.

        Args:
            config (MainConfig): Configuration dataclass instance.
            critic_type (type): Critic class type.
            actor_type (type): Actor class type.
        """
        assert config.training.warmup_steps == 0, "GRPO does not support warmup steps"
        super().__init__(config, critic_type, actor_type)

        self.group_size = config.model.group_size
        self.max_episode_length = config.model.max_episode_length

        # Replace the default flat ReplayBuffer with the structured group rollout buffer.
        self.experience_memory = GroupRolloutBuffer(
            group_size=self.group_size,
            max_episode_length=self.max_episode_length,
            state_dim=self.dim_state,
            action_dim=self.dim_act,
            device=torch.device(config.system.storing_device),
            gamma=config.training.gamma,
            use_identical_seeding=False,
        )
        self._episode_idx = 0
        self._step_in_episode = 0
        self._completed_episodes = 0

    def generate_transition(self, **kwargs):
        """
        Generates a transition dictionary for the group rollout buffer.

        No critic values are stored — vanilla GRPO does not use V(s) for advantages.

        Args:
            **kwargs: Keyword arguments containing step data.

        Returns:
            TensorDict: Transition with action log-probabilities.
        """
        transition = TensorDict(
            {
                "state": kwargs["state"],
                "action": kwargs["action"],
                "reward": kwargs["reward"],
                "next_state": kwargs["next_state"],
                "terminated": kwargs["terminated"],
                "truncated": kwargs["truncated"],
                "action_logprob": kwargs["action_logprob"],
            },
            batch_size=[],
        )
        return transition

    def store_transition(self, transition) -> None:
        """
        Writes a single transition into the group rollout buffer.

        Sequential episodes are mapped to consecutive group slots: episode k
        goes into group_idx = k % group_size at its own per-episode timestep.

        Once all group slots are filled, additional transitions are dropped
        until learn() is called and resets the buffer.
        """
        if self._completed_episodes >= self.group_size:
            return

        buf = self.experience_memory
        group_idx = self._episode_idx % self.group_size
        t = self._step_in_episode

        terminated = bool(transition["terminated"])
        truncated = bool(transition["truncated"])
        episode_end = terminated or truncated

        if t < self.max_episode_length:
            buf_device = buf.device
            buf.states[group_idx, t] = transition["state"].to(buf_device)
            buf.actions[group_idx, t] = transition["action"].to(buf_device)
            buf.logprobs[group_idx, t] = transition["action_logprob"].to(buf_device)
            buf.rewards[group_idx, t] = float(transition["reward"])
            buf.terminated[group_idx, t] = terminated
            buf.valid_mask[group_idx, t] = True
            buf.current_lengths[group_idx] = t + 1

        self._step_in_episode += 1

        if episode_end:
            self._completed_episodes += 1
            self._episode_idx += 1
            self._step_in_episode = 0

    def learn(self, max_iter: int = 1, n_epochs: int = 0) -> None:
        """
        Learns from the group rollout buffer using MC returns + group normalization.

        Gated on collecting at least ``group_size`` completed episodes so that
        every group slot holds a real trajectory before advantage normalization.

        Args:
            max_iter (int): Maximum number of update iterations (unused, kept for API compat).
            n_epochs (int): Number of passes over the collected group batch.
        """
        if self._completed_episodes < self.group_size:
            return None

        buf = self.experience_memory
        compute_device = torch.device(self.config.system.device)

        # Get batch with MC returns + time-aligned group advantages (computed in buffer)
        batch = buf.get_batch()

        # Flatten [G, T, ...] -> [G*T, ...]
        group_size, max_len = batch["state"].shape[:2]
        if isinstance(self.dim_state, tuple):
            states_flat = batch["state"].reshape(group_size * max_len, *self.dim_state)
        else:
            states_flat = batch["state"].reshape(group_size * max_len, *buf.state_shape)
        actions_flat = batch["action"].reshape(group_size * max_len, -1)
        logprobs_flat = batch["logprob"].reshape(group_size * max_len)
        advantages_flat = batch["advantages"].reshape(group_size * max_len)
        returns_flat = batch["returns"].reshape(group_size * max_len)

        states_flat = states_flat.to(compute_device)
        actions_flat = actions_flat.to(compute_device)
        logprobs_flat = logprobs_flat.to(compute_device)
        advantages_flat = advantages_flat.to(compute_device)
        returns_flat = returns_flat.to(compute_device)

        n_passes = max(1, n_epochs)
        for _ in range(n_passes):
            self.actor.update(states_flat, actions_flat, logprobs_flat, advantages_flat)
            self.critic.update(states_flat, returns_flat)
            self.n_iter += 1

        # Reset the buffer after learning (on-policy algorithm).
        buf.reset()
        self._episode_idx = 0
        self._step_in_episode = 0
        self._completed_episodes = 0

        return None
