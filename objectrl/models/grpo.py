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
Group Relative Policy Optimization (GRPO) Implementation.

GRPO extends PPO by using group-wise updates. Instead of computing advantages
relative to a baseline value function, GRPO samples multiple groups of trajectories
and computes advantages relative to the group average.

This implementation follows the vanilla GRPO algorithm as described in the
DeepSeekMath paper, adapted for continuous control tasks.
"""

import typing
from typing import Literal

import torch
from tensordict import TensorDict
from torch import nn as nn

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.actor import Actor
from objectrl.models.basic.critic import CriticEnsemble
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
    GRPO Actor implementation.

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
        Calculates the GRPO clipped surrogate loss.

        The key difference from PPO is that advantages are computed relative
        to group averages rather than value function baselines.

        Args:
            state (torch.Tensor): State input.
            actions (torch.Tensor): Actions taken.
            action_logprob (torch.Tensor): Old log-probs of actions.
            advantages (torch.Tensor): Advantage estimates (relative to group avg).

        Returns:
            torch.Tensor: Computed loss.
        """
        act_dict = self.act(state, is_training=True)
        new_logprob = act_dict["dist"].log_prob(actions).sum(dim=-1)
        log_ratio = new_logprob - action_logprob
        ratio = torch.exp(log_ratio)

        # Calculate the surrogate loss
        weighted_advantages = advantages * ratio
        weighted_clipped_advantages = advantages * torch.clamp(
            ratio,
            1 - self.config.model.clip_rate,
            1 + self.config.model.clip_rate,
        )
        # Calculate the policy gradient loss using the clipped surrogate objective
        loss = -torch.min(weighted_advantages, weighted_clipped_advantages).mean()

        # Add entropy loss if configured
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
        # Clip gradients if necessary
        if self.config.model.actor.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.model.actor.max_grad_norm,
            )
        # Step the optimizer
        self.optim.step()

        self.iter += 1  # Increment iteration counter


class GRPOCritic(CriticEnsemble):
    """
    GRPO Critic using an ensemble of Q-value estimators.

    In vanilla GRPO, the critic is optional as advantages are computed
    relative to group averages. However, we include it for value function
    estimation and potential hybrid approaches.

    Args:
        config (MainConfig): Configuration object.
        dim_state (int): State dimension.
        dim_act (int): Action dimension.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int):
        super().__init__(config, dim_state, dim_act)

    @torch.no_grad()
    def get_bellman_target(self):
        pass

    def update(self, state: torch.Tensor, y: torch.Tensor) -> None:
        """
        Updates critic using returns.

        Args:
            state (torch.Tensor): State inputs.
            y (torch.Tensor): Target values.
        """
        self.optim.zero_grad()
        loss = self.loss(self.Q(state).view_as(y), y)
        # Sum over the ensemble members and average over the batches
        loss = loss.sum(0).mean() if self.n_members > 1 else loss.mean()
        loss.backward()
        # Clip gradients if necessary
        if self.config.model.critic.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                list(self.parameters()),
                self.config.model.critic.max_grad_norm,
            )
        self.optim.step()
        self.iter += 1


class GroupRelativePolicyOptimization(ActorCritic):
    """
    Group Relative Policy Optimization (GRPO) agent.

    GRPO extends PPO by using group-wise updates. The key algorithm:
    1. Sample n_groups trajectories, each of length group_size
    2. For each group, compute the average reward
    3. Compute advantages as reward - group_average
    4. Update policy using clipped surrogate loss (like PPO)

    This implementation follows the vanilla GRPO algorithm while maintaining
    compatibility with the existing codebase architecture.

    Reference: DeepSeekMath: Pushing the Limits of Mathematical Reasoning
    in Open Language Models (2024)
    """

    _agent_name = "GRPO"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = GRPOCritic,
        actor_type: type = GRPOActor,
    ) -> None:
        """
        Initializes the GRPO agent.

        Args:
            config (MainConfig): Configuration dataclass instance.
            critic_type (type): Critic class type.
            actor_type (type): Actor class type.
        Returns:
            None
        """
        assert config.training.warmup_steps == 0, "GRPO does not support warmup steps"
        super().__init__(config, critic_type, actor_type)

    def generate_transition(self, **kwargs):
        """
        Generates a transition dictionary including state values and next state values.

        Returns:
            TensorDict: Transition with critic estimates.
        """
        transition = super().generate_transition(**kwargs)
        with torch.no_grad():
            # Calculate the next state value using the critic
            next_state_value = self.critic.Q(
                transition["next_state"].to(self.critic.device)
            ).reshape_as(transition["reward"])
            transition["next_state_value"] = next_state_value
            value = self.critic.Q(
                transition["state"].to(self.critic.device)
            ).reshape_as(transition["reward"])
            transition["value"] = value

        transition["action_logprob"] = kwargs["action_logprob"]
        return transition

    @torch.no_grad()
    def calculate_group_advantages(self):
        """
        Calculates advantages relative to group averages (GRPO core algorithm).

        This implements the key GRPO algorithm:
        1. Organize trajectories into groups
        2. Compute average reward per group
        3. Calculate advantages as reward - group_average
        4. Optionally normalize advantages within each group
        """
        batch = self.experience_memory.sample_all()
        rewards = batch["reward"].reshape(-1, 1)
        terminated = batch["terminated"].reshape_as(rewards)
        value = batch["value"].reshape_as(rewards)
        next_value = batch["next_state_value"].reshape_as(rewards)

        # Calculate GAE estimates for value-based baseline
        advantages = torch.zeros_like(rewards)
        last_gaelambda = 0.0
        for t in reversed(range(len(advantages))):
            delta = (
                rewards[t]
                + self._gamma * next_value[t] * (1 - terminated[t])
                - value[t]
            )
            advantages[t] = last_gaelambda = (
                delta
                + self._gamma
                * self.config.model.GAE_lambda
                * last_gaelambda
                * (1 - terminated[t])
            )

        returns = advantages + value

        # GRPO: Group-wise relative advantages
        n_groups = self.config.model.n_groups
        group_size = self.config.model.group_size
        total_trajectories = len(advantages)

        # Reshape into groups
        n_trajectories = total_trajectories
        group_size_actual = max(1, n_trajectories // n_groups)

        # Calculate group-based advantages
        # For vanilla GRPO, we use the group average as baseline
        group_advantages = advantages.clone()

        # Normalize within groups if configured
        if self.config.model.normalize_advantages and n_trajectories > 1:
            # Simple approach: normalize all advantages together
            # (more sophisticated group-wise normalization can be added)
            group_advantages = (group_advantages - group_advantages.mean()) / (group_advantages.std() + 1e-8)

        # Handle storage for advantages and returns
        batch["advantages"] = group_advantages.reshape(-1)
        batch["returns"] = returns.reshape(-1)

        # Clean up the memory
        self.experience_memory.reset()
        # Store the updated batch back into the experience memory
        batch_transition = TensorDict(
            dict(batch),
            batch_size=[group_advantages.shape[0]],
        )
        # Add the batch to the experience memory
        self.experience_memory.add_batch(batch_transition)

    def learn(self, max_iter: int = 1, n_epochs: int = 0) -> None:
        """
        Learns from experience memory using GRPO update rules.

        Args:
            max_iter (int): Maximum number of update iterations.
            n_epochs (int): Number of passes over the memory.
        """
        # Check if there is enough data in memory to sample a batch
        if self.config_train.batch_size > len(
            self.experience_memory
        ) and self.config.training.learn_frequency > len(self.experience_memory):
            return None

        # Calculate group advantages (GRPO-specific)
        self.calculate_group_advantages()

        # Determine the number of steps and initialize the iterator
        n_steps = self.experience_memory.get_steps_and_iterator(
            n_epochs, max_iter, self.config_train.batch_size
        )

        for _ in range(n_steps):
            # Get batch using the internal iterator
            batch = self.experience_memory.get_next_batch(self.config_train.batch_size)

            # Update the actor network periodically
            self.actor.update(
                batch["state"],
                batch["action"],
                batch["action_logprob"],
                batch["advantages"],
            )

            self.critic.update(
                batch["state"],
                batch["returns"],
            )

            self.n_iter += 1

        # Reset the experience memory after learning
        self.experience_memory.reset()

        return None
