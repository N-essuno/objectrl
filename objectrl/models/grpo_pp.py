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
Group Relative Policy Optimization Plus Plus (GRPO++) Implementation.

GRPO++ extends vanilla GRPO with several key improvements:

1. **DAPO Asymmetric Clipping**: Different epsilon bounds for positive vs negative advantages
   - Tighter bound (epsilon_high) for positive advantages prevents policy collapse
   - Looser bound (epsilon_low) for negative advantages allows rapid rejection of bad policies

2. **Dr. GRPO Aggregation**: Sum over time dimension, mean over group dimension
   - Fixes length bias where shorter successful episodes are penalized
   - Preserves episode length in gradient magnitude for sparse-reward tasks

3. **Zero KL Penalty**: Removes standard GRPO/PPO KL divergence tax
   - Accelerates convergence in deterministic physical environments
   - Relies on group-wise normalization for stability instead

4. **Adaptive Advantage Normalization**: Dynamic normalization based on reward sparsity
   - Adjusts normalization strength based on reward distribution
   - Prevents over-normalization in dense reward environments
   - Enhances signal in sparse reward environments

5. **Group Rollout Buffer**: Structured storage with [group_size, max_episode_length, feature_dim]
   - Proper masking for variable-length episodes
   - Time-aligned advantage computation using Monte Carlo returns
   - Synchronized seeding for identical initial states across groups

Reference: Adapted from DeepSeekMath and extended with DAPO and Dr. GRPO techniques.
"""

from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.actor import Actor
from objectrl.models.basic.critic import CriticEnsemble
from objectrl.replay_buffers.group_rollout_buffer import GroupRolloutBuffer
from objectrl.utils.net_utils import MLP, make_pixel_encoder

if __name__ == "__main__":
    from objectrl.config.config import MainConfig


class AdaptiveNormalizer:
    """
    Adaptive advantage normalization for sparse rewards.

    Dynamically adjusts normalization strength based on reward distribution:
    - Sparse rewards (low variance): stronger normalization for stability
    - Dense rewards (high variance): lighter normalization to preserve signal

    Args:
        epsilon (float): Small constant for numerical stability.
        momentum (float): Moving average momentum for statistics.
    """

    def __init__(self, epsilon: float = 1e-8, momentum: float = 0.99):
        self.epsilon = epsilon
        self.momentum = momentum
        self.registered = False

    def register_buffers(self, module: nn.Module):
        """Register running statistics as buffers."""
        module.register_buffer("running_mean", torch.zeros(1))
        module.register_buffer("running_var", torch.ones(1))
        module.register_buffer("running_count", torch.tensor(0.0))
        self.registered = True

    def compute_sparsity_metric(self, rewards: torch.Tensor, valid_mask: torch.Tensor) -> float:
        """
        Compute reward sparsity metric.

        Lower values indicate sparser rewards (more zeros/near-zeros).

        Args:
            rewards: Reward tensor [group_size, max_episode_length]
            valid_mask: Valid timestep mask [group_size, max_episode_length]

        Returns:
            float: Sparsity metric (0 = very sparse, 1 = dense)
        """
        # Only consider valid timesteps
        valid_rewards = rewards[valid_mask]

        if len(valid_rewards) == 0:
            return 0.0

        # Compute coefficient of variation (std/mean)
        mean_reward = valid_rewards.mean()
        std_reward = valid_rewards.std()

        if abs(mean_reward) < self.epsilon:
            return 0.0

        cv = std_reward / (abs(mean_reward) + self.epsilon)

        # Normalize to [0, 1] range
        # CV > 1.0 indicates very dense rewards, CV < 0.1 indicates very sparse
        sparsity = torch.clamp(cv / 2.0, 0.0, 1.0)

        return sparsity.item()


class GRPOPlusActorNetProbabilistic(nn.Module):
    """
    Probabilistic actor network for GRPO++ using a Gaussian policy.

    Identical to GRPO actor network but included for completeness and
    potential future extensions specific to GRPO++.

    Args:
        dim_state (int | tuple[int,int,int]): Dimension of input state, or HWC image shape.
        dim_act (int): Dimension of action space.
        n_heads (int): Number of heads (only supports 1).
        depth (int): Depth of the MLP.
        width (int): Width of each MLP layer.
        act (str): Activation function ("crelu" or "relu").
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
        act: str = "relu",
        has_norm: bool = False,
        upper_clamp: float = -1.0,
        encoder_type: str = "light",
    ) -> None:
        super().__init__()
        assert n_heads == 1, "GRPOPlusActorNetProbabilistic only supports n_heads=1"
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


class GRPOPlusActor(Actor):
    """
    GRPO++ Actor with DAPO asymmetric clipping and Dr. GRPO aggregation.

    Key improvements over vanilla GRPO:
    - DAPO asymmetric clipping: different bounds for positive/negative advantages
    - Dr. GRPO aggregation: sum over time, mean over group (not mean over both)
    - Zero KL penalty: removed for faster convergence in deterministic environments
    - Adaptive normalization: adjusts normalization strength based on reward sparsity

    Args:
        config (MainConfig): Configuration object.
        dim_state (int): Dimension of state space.
        dim_act (int): Dimension of action space.
    """

    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)

        # Initialize adaptive normalizer
        self.adaptive_normalizer = AdaptiveNormalizer()
        self.adaptive_normalizer.register_buffers(self)

    def loss(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_logprob: torch.Tensor,
        advantages: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the GRPO++ loss with DAPO asymmetric clipping and Dr. GRPO aggregation.

        DAPO Asymmetric Clipping:
        - For positive advantages (A > 0): tighter upper bound to prevent policy collapse
        - For negative advantages (A < 0): looser lower bound for rapid rejection

        Dr. GRPO Aggregation:
        - Sum over time dimension (preserves episode length in gradient)
        - Mean over group dimension (stable updates across groups)
        - Applies valid_mask to handle variable-length episodes

        Adaptive Normalization:
        - Computes reward sparsity metric
        - Adjusts advantage normalization strength dynamically
        - Stronger normalization for sparse rewards, lighter for dense rewards

        Mathematical Formulation:
        ```
        ratio = exp(log π(a|s) - log π_old(a|s))

        # Asymmetric clipping bounds
        clip_low = 1 - epsilon_low   # For negative advantages
        clip_high = 1 + epsilon_high  # For positive advantages

        # Adaptive normalization based on reward sparsity
        sparsity = compute_sparsity(rewards)
        alpha = adaptive_mixing_factor(sparsity)  # Higher for sparse rewards

        # Normalize advantages with adaptive strength
        A_normalized = alpha * normalize(A) + (1 - alpha) * A

        # Clipped surrogate with asymmetric bounds
        L_clip = min(
            ratio * A_normalized,
            clip(ratio, clip_low, clip_high) * A_normalized
        )

        # Dr. GRPO aggregation: sum over time, mean over group
        loss = -mean_group(sum_time(L_clip * mask))
        ```

        Args:
            state (torch.Tensor): State input with shape [batch_size, state_dim].
            actions (torch.Tensor): Actions taken with shape [batch_size, action_dim].
            action_logprob (torch.Tensor): Old log-probs with shape [batch_size].
            advantages (torch.Tensor): Advantage estimates with shape [batch_size].
            valid_mask (torch.Tensor, optional): Boolean mask for valid timesteps.
                Shape [batch_size]. Used to exclude padded timesteps from loss computation.
            rewards (torch.Tensor, optional): Rewards for adaptive normalization.

        Returns:
            torch.Tensor: Computed loss scalar.
        """
        # Get new policy distribution
        act_dict = self.act(state, is_training=True)
        new_logprob = act_dict["dist"].log_prob(actions).sum(dim=-1)

        # Compute probability ratio
        log_ratio = new_logprob - action_logprob
        ratio = torch.exp(log_ratio)

        # Adaptive Normalization: Adjust advantages based on reward sparsity
        adaptive_advantages = advantages.clone()
        if rewards is not None and valid_mask is not None:
            # Reshape rewards to match advantages if needed
            if rewards.dim() != advantages.dim():
                group_size = self.config.model.group_size
                if advantages.numel() % group_size == 0:
                    max_len = advantages.numel() // group_size
                    advantages_reshaped = advantages.view(group_size, max_len)
                    rewards_reshaped = rewards.view(group_size, max_len) if rewards.dim() == advantages.dim() else rewards
                    valid_mask_reshaped = valid_mask.view(group_size, max_len) if valid_mask.dim() == advantages.dim() else valid_mask
                else:
                    advantages_reshaped = advantages
                    rewards_reshaped = rewards
                    valid_mask_reshaped = valid_mask
            else:
                advantages_reshaped = advantages
                rewards_reshaped = rewards
                valid_mask_reshaped = valid_mask

            # Compute reward sparsity
            sparsity = self.adaptive_normalizer.compute_sparsity_metric(
                rewards_reshaped, valid_mask_reshaped
            )

            # Adaptive mixing factor: higher for sparse rewards (stronger normalization)
            # Sparsity < 0.3: sparse rewards -> use normalized advantages
            # Sparsity > 0.7: dense rewards -> use raw advantages
            alpha = 1.0 - torch.clamp(torch.tensor(sparsity), 0.3, 0.7)

            # Apply adaptive normalization
            if advantages_reshaped.dim() == 2:  # [group_size, max_len]
                # Normalize within each timestep across group dimension
                mean = advantages_reshaped.mean(dim=0, keepdim=True)
                std = advantages_reshaped.std(dim=0, keepdim=True)
                normalized_advantages = (advantages_reshaped - mean) / (std + 1e-8)

                # Apply adaptive mixing
                adaptive_advantages_reshaped = alpha * normalized_advantages + (1 - alpha) * advantages_reshaped

                # Flatten back
                if advantages.dim() == 1:
                    adaptive_advantages = adaptive_advantages_reshaped.view(-1)
                else:
                    adaptive_advantages = adaptive_advantages_reshaped

        # DAPO: Asymmetric clipping bounds
        epsilon_high = self.config.model.clip_rate_high  # Tighter bound for positive advantages
        epsilon_low = self.config.model.clip_rate_low    # Looser bound for negative advantages

        # Apply asymmetric clipping
        clipped_ratio = torch.where(
            adaptive_advantages >= 0,
            torch.clamp(ratio, 1 - epsilon_high, 1 + epsilon_high),  # Positive advantages: tight bound
            torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_low),    # Negative advantages: loose bound
        )

        # Surrogate objectives using adaptive advantages
        surrogate = ratio * adaptive_advantages
        clipped_surrogate = clipped_ratio * adaptive_advantages

        # PPO-style clipped objective (min of surrogate and clipped)
        policy_loss_per_step = -torch.min(surrogate, clipped_surrogate)

        # Dr. GRPO Aggregation: Handle variable-length episodes with masking
        if valid_mask is not None:
            # Apply mask: zero out invalid (padded) timesteps
            policy_loss_per_step = policy_loss_per_step * valid_mask

        # CRITICAL: Sum over time dimension, then mean over group dimension.
        # If the caller flattened to [group_size * max_episode_length], reshape
        # back so the sum/mean split actually preserves episode length in the
        # gradient magnitude.
        group_size = self.config.model.group_size
        if policy_loss_per_step.dim() == 1 and policy_loss_per_step.numel() % group_size == 0:
            max_len = policy_loss_per_step.numel() // group_size
            policy_loss_per_step = policy_loss_per_step.view(group_size, max_len)

        if policy_loss_per_step.dim() == 1:
            loss = policy_loss_per_step.mean()
        else:
            loss = policy_loss_per_step.sum(dim=-1).mean()

        # Add entropy bonus if configured (helps exploration)
        if self.config.model.entropy_coef > 0:
            entropy = act_dict["dist"].entropy().sum(-1)
            if valid_mask is not None:
                entropy = entropy * valid_mask
            # Mean aggregation for entropy (standard approach)
            entropy_loss = -entropy.mean()
            loss += self.config.model.entropy_coef * entropy_loss

        # NOTE: No KL penalty in GRPO++ (beta = 0.0)
        # We rely on group-wise normalization for stability instead

        return loss

    def update(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_logprob: torch.Tensor,
        advantages: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Performs gradient update on the actor network.

        Args:
            state (torch.Tensor): Input state batch.
            actions (torch.Tensor): Sampled actions.
            action_logprob (torch.Tensor): Log-probs of sampled actions.
            advantages (torch.Tensor): Advantage estimates.
            valid_mask (torch.Tensor, optional): Boolean mask for valid timesteps.
            rewards (torch.Tensor, optional): Rewards for adaptive normalization.
        """
        self.optim.zero_grad()
        loss = self.loss(state, actions, action_logprob, advantages, valid_mask, rewards)
        loss.backward()

        # Clip gradients if necessary
        if self.config.model.actor.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.model.actor.max_grad_norm,
            )

        # Step the optimizer
        self.optim.step()
        self.iter += 1


class GRPOPlusCritic(CriticEnsemble):
    """
    GRPO++ Critic using an ensemble of Q-value estimators.

    In GRPO++, the critic is optional as advantages are computed using
    Monte Carlo returns relative to group averages. However, we include
    it for value function estimation and potential hybrid approaches.

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

    def update(self, state: torch.Tensor, y: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> None:
        """
        Updates critic using returns with optional masking.

        Args:
            state (torch.Tensor): State inputs.
            y (torch.Tensor): Target values.
            valid_mask (torch.Tensor, optional): Boolean mask for valid timesteps.
        """
        self.optim.zero_grad()
        predictions = self.Q(state).view_as(y)
        loss = self.loss(predictions, y)

        # Apply mask if provided
        if valid_mask is not None:
            loss = loss * valid_mask
            loss = loss.sum() / valid_mask.sum()
        else:
            # Sum over ensemble members and average over batches
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


class GroupRelativePolicyOptimizationPlusPlus(ActorCritic):
    """
    Group Relative Policy Optimization Plus Plus (GRPO++) agent.

    GRPO++ extends vanilla GRPO with three key improvements:

    1. **DAPO Asymmetric Clipping**: Prevents policy collapse on positive advantages
       while allowing rapid rejection of failed strategies on negative advantages.

    2. **Dr. GRPO Aggregation**: Sums loss over time dimension instead of averaging,
       preserving episode length in gradient magnitude for sparse-reward tasks.

    3. **Zero KL Penalty**: Removes the KL divergence tax, relying on group-wise
       normalization for stability instead.

    Algorithm:
    1. Sample G parallel trajectories with synchronized seeds
    2. Store in structured buffer [group_size, max_episode_length, feature_dim]
    3. Compute Monte Carlo returns-to-go for each timestep
    4. Normalize returns at each timestep across group dimension (time-aligned advantages)
    5. Update policy using asymmetric clipped surrogate with Dr. GRPO aggregation
    6. Repeat

    This implementation is specifically designed for:
    - Sparse-reward environments (e.g., Acrobot swing-up with -1 per step)
    - Episodic tasks with variable-length episodes
    - Deterministic physical control environments

    Reference: Extended from DeepSeekMath GRPO with DAPO and Dr. GRPO techniques.
    """

    _agent_name = "GRPO++"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = GRPOPlusCritic,
        actor_type: type = GRPOPlusActor,
    ) -> None:
        """
        Initializes the GRPO++ agent.

        Args:
            config (MainConfig): Configuration dataclass instance.
            critic_type (type): Critic class type.
            actor_type (type): Actor class type.
        """
        assert config.training.warmup_steps == 0, "GRPO++ does not support warmup steps"
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

        Sequential episodes are mapped to consecutive group slots: episode k goes
        into group_idx = k % group_size at its own per-episode timestep.
        """
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
        Learns from the group rollout buffer using GRPO++ update rules.

        Gated on collecting at least ``group_size`` completed episodes so that
        every group slot holds a real trajectory before advantage normalization.

        Args:
            max_iter (int): Maximum number of update iterations.
            n_epochs (int): Number of passes over the collected group batch.
        """
        # Need at least group_size completed episodes before computing group-relative advantages.
        if self._completed_episodes < self.group_size:
            return None

        # Get the complete group rollout batch with computed advantages
        batch = self.experience_memory.get_batch()

        # Move batch to compute device for updates.
        compute_device = torch.device(self.config.system.device)
        states = batch["state"].to(compute_device)
        actions = batch["action"].to(compute_device)
        logprobs = batch["logprob"].to(compute_device)
        advantages = batch["advantages"].to(compute_device)
        returns = batch["returns"].to(compute_device)
        valid_mask = batch["valid_mask"].to(compute_device).float()

        # Get rewards for adaptive normalization (stored in buffer)
        rewards = batch.get("rewards", None)
        if rewards is not None:
            rewards = rewards.to(compute_device)

        group_size, max_len = states.shape[:2]
        if isinstance(self.dim_state, tuple):
            states_flat = states.view(group_size * max_len, *self.dim_state)
        else:
            states_flat = states.view(group_size * max_len, -1)
        actions_flat = actions.view(group_size * max_len, -1)
        logprobs_flat = logprobs.view(group_size * max_len)
        advantages_flat = advantages.view(group_size * max_len)
        valid_mask_flat = valid_mask.view(group_size * max_len)
        returns_flat = returns.view(group_size * max_len)
        rewards_flat = rewards.view(group_size * max_len) if rewards is not None else None

        n_passes = max(1, n_epochs)
        for _ in range(n_passes):
            self.actor.update(
                states_flat,
                actions_flat,
                logprobs_flat,
                advantages_flat,
                valid_mask_flat,
                rewards_flat,  # Pass rewards for adaptive normalization
            )
            self.critic.update(states_flat, returns_flat, valid_mask_flat)
            self.n_iter += 1

        # Reset the buffer after learning (on-policy algorithm).
        self.experience_memory.reset()
        self._episode_idx = 0
        self._step_in_episode = 0
        self._completed_episodes = 0

        return None
