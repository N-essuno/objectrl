from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from objectrl.models.basic.ac import ActorCritic
from objectrl.models.basic.actor import Actor
from objectrl.models.basic.critic import CriticEnsemble
from objectrl.replay_buffers.group_rollout_buffer import GroupRolloutBuffer

if __name__ == "__main__":
    from objectrl.config.config import MainConfig


class GRPOPlusV2Actor(Actor):
    def __init__(self, config: "MainConfig", dim_state: int, dim_act: int) -> None:
        super().__init__(config, dim_state, dim_act)

    def loss(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_logprob: torch.Tensor,
        advantages: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the GRPO++ V2 loss with DAPO asymmetric clipping and Dr. GRPO aggregation.

        Advantages are already GAE-computed and group-normalized before being
        passed to this method, so no normalization is done here.

        Args:
            state (torch.Tensor): State input with shape [batch_size, state_dim].
            actions (torch.Tensor): Actions taken with shape [batch_size, action_dim].
            action_logprob (torch.Tensor): Old log-probs with shape [batch_size].
            advantages (torch.Tensor): GAE-based group-normalized advantages [batch_size].
            valid_mask (torch.Tensor, optional): Boolean mask for valid timesteps.

        Returns:
            torch.Tensor: Computed loss scalar.
        """
        # Get new policy distribution
        act_dict = self.act(state, is_training=True)
        new_logprob = act_dict["dist"].log_prob(actions).sum(dim=-1)

        # Compute probability ratio
        log_ratio = new_logprob - action_logprob
        ratio = torch.exp(log_ratio)

        # DAPO: Asymmetric clipping bounds
        epsilon_high = self.config.model.clip_rate_high
        epsilon_low = self.config.model.clip_rate_low

        # Apply asymmetric clipping
        clipped_ratio = torch.where(
            advantages >= 0,
            torch.clamp(ratio, 1 - epsilon_high, 1 + epsilon_high),
            torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_low),
        )

        # Surrogate objectives
        surrogate = ratio * advantages
        clipped_surrogate = clipped_ratio * advantages

        # PPO-style clipped objective
        policy_loss_per_step = -torch.min(surrogate, clipped_surrogate)

        # Dr. GRPO Aggregation with masking
        if valid_mask is not None:
            policy_loss_per_step = policy_loss_per_step * valid_mask

        # Reshape to [group_size, max_len] for sum-over-time, mean-over-group
        group_size = self.config.model.group_size
        if policy_loss_per_step.dim() == 1 and policy_loss_per_step.numel() % group_size == 0:
            max_len = policy_loss_per_step.numel() // group_size
            policy_loss_per_step = policy_loss_per_step.view(group_size, max_len)

        if policy_loss_per_step.dim() == 1:
            loss = policy_loss_per_step.mean()
        else:
            loss = policy_loss_per_step.sum(dim=-1).mean()

        # Add entropy bonus if configured
        if self.config.model.entropy_coef > 0:
            entropy = act_dict["dist"].entropy().sum(-1)
            if valid_mask is not None:
                entropy = entropy * valid_mask
            entropy_loss = -entropy.mean()
            loss += self.config.model.entropy_coef * entropy_loss

        return loss

    def update(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_logprob: torch.Tensor,
        advantages: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Performs gradient update on the actor network.

        Args:
            state (torch.Tensor): Input state batch.
            actions (torch.Tensor): Sampled actions.
            action_logprob (torch.Tensor): Log-probs of sampled actions.
            advantages (torch.Tensor): Advantage estimates.
            valid_mask (torch.Tensor, optional): Boolean mask for valid timesteps.
        """
        self.optim.zero_grad()
        loss = self.loss(state, actions, action_logprob, advantages, valid_mask)
        loss.backward()

        if self.config.model.actor.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.model.actor.max_grad_norm,
            )

        self.optim.step()
        self.iter += 1


class GRPOPlusV2Critic(CriticEnsemble):
    """
    GRPO++ V2 Critic with target network for stable GAE estimation.

    Uses has_target=True so CriticEnsemble creates a Polyak-averaged target
    network. The target is used during GAE computation for V(s) estimates,
    and updated via soft averaging after each training step.

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
        Updates critic using MC returns with optional masking.

        Args:
            state (torch.Tensor): State inputs.
            y (torch.Tensor): Target values (MC returns).
            valid_mask (torch.Tensor, optional): Boolean mask for valid timesteps.
        """
        self.optim.zero_grad()
        predictions = self.Q(state).view_as(y)
        loss = self.loss(predictions, y)

        if valid_mask is not None:
            loss = loss * valid_mask
            loss = loss.sum() / valid_mask.sum()
        else:
            loss = loss.sum(0).mean() if self.n_members > 1 else loss.mean()

        loss.backward()

        if self.config.model.critic.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                list(self.parameters()),
                self.config.model.critic.max_grad_norm,
            )

        self.optim.step()
        self.iter += 1


class GRPOPlusV2(ActorCritic):
    """
    Group Relative Policy Optimization Plus Plus V2 (GRPO++ V2) agent.

    GRPO++ V2 combines group-relative advantages with a learned value function
    using GAE (multi-step TD), making it competitive with PPO/SAC/TD3.

    Algorithm:
    1. Sample G parallel trajectories with synchronized seeds
    2. Store in structured buffer [group_size, max_episode_length, feature_dim]
    3. Compute V(s) for all buffer states using target critic
    4. Compute TD residuals: delta = r + gamma * V(s') * (1-done) - V(s)
    5. Backward GAE accumulation: A_t = delta_t + gamma * lambda * A_{t+1} * (1-done)
    6. Normalize GAE across group dim at each timestep
    7. Update actor with DAPO clipping + Dr. GRPO aggregation
    8. Update critic with MC returns
    9. Soft-update target network
    """

    _agent_name = "GRPO++ V2"

    def __init__(
        self,
        config: "MainConfig",
        critic_type: type = GRPOPlusV2Critic,
        actor_type: type = GRPOPlusV2Actor,
    ) -> None:
        assert config.training.warmup_steps == 0, "GRPO++ V2 does not support warmup steps"
        super().__init__(config, critic_type, actor_type)

        self.group_size = config.model.group_size
        self.max_episode_length = config.model.max_episode_length
        self.gae_lambda = config.model.GAE_lambda

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

        Once all group slots are filled, additional transitions are dropped until
        learn() is called and resets the buffer.
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

    @torch.no_grad()
    def _compute_gae(self, states, rewards, terminated, valid_mask, compute_device):
        """
        Compute GAE advantages using the target critic.

        Args:
            states: [group_size, max_episode_length, *state_dim]
            rewards: [group_size, max_episode_length]
            terminated: [group_size, max_episode_length] bool
            valid_mask: [group_size, max_episode_length] bool
            compute_device: torch.device

        Returns:
            advantages: [group_size, max_episode_length] - group-normalized GAE
        """
        group_size, max_len = states.shape[:2]
        buf = self.experience_memory

        # Flatten states for target critic forward pass
        if isinstance(self.dim_state, tuple):
            states_flat = states.view(group_size * max_len, *self.dim_state)
        else:
            states_flat = states.view(group_size * max_len, -1)

        # V(s) from target critic: shape [n_members, batch, 1] -> [group_size, max_len]
        values_flat = self.critic.Q_t(states_flat)  # [n_members, group*max_len, 1]
        values = values_flat.squeeze(-1).view(group_size, max_len)
        if self.critic.n_members > 1:
            values = values.mean(dim=0)  # average over ensemble if n_members > 1
            # After mean over n_members, we need to reshape again
            values = values.view(group_size, max_len)

        gamma = self.config.training.gamma
        lam = self.gae_lambda

        # Compute TD residuals and GAE backwards
        deltas = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards)

        for t in reversed(range(max_len)):
            if t == max_len - 1:
                # No next state: V(s') = 0
                deltas[:, t] = rewards[:, t] - values[:, t]
            else:
                # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
                not_done = (~terminated[:, t]).float()
                deltas[:, t] = rewards[:, t] + gamma * values[:, t + 1] * not_done - values[:, t]

            # GAE accumulation: GAE_t = delta_t + gamma * lambda * GAE_{t+1} * (1 - done)
            if t == max_len - 1:
                gae[:, t] = deltas[:, t]
            else:
                not_done = (~terminated[:, t]).float()
                gae[:, t] = deltas[:, t] + gamma * lam * gae[:, t + 1] * not_done

        # Group-normalize GAE at each timestep (GRPO insight)
        mean = gae.mean(dim=0, keepdim=True)
        std = gae.std(dim=0, keepdim=True)
        advantages = (gae - mean) / (std + 1e-8)

        # Zero out invalid timesteps
        advantages = torch.where(valid_mask, advantages, torch.zeros_like(advantages))

        return advantages

    def learn(self, max_iter: int = 1, n_epochs: int = 0) -> None:
        """
        Learns from the group rollout buffer using GAE + group normalization.

        Gated on collecting at least ``group_size`` completed episodes so that
        every group slot holds a real trajectory before advantage normalization.

        Args:
            max_iter (int): Maximum number of update iterations.
            n_epochs (int): Number of passes over the collected group batch.
        """
        if self._completed_episodes < self.group_size:
            return None

        buf = self.experience_memory
        compute_device = torch.device(self.config.system.device)

        # Get raw data from buffer (NOT using buf.get_batch which computes MC advantages)
        # We will compute GAE advantages ourselves using the target critic for stable V(s) estimates.
        states = buf.states.to(compute_device)
        actions = buf.actions.to(compute_device)
        logprobs = buf.logprobs.to(compute_device)
        rewards = buf.rewards.to(compute_device)
        terminated = buf.terminated.to(compute_device)
        valid_mask = buf.valid_mask.to(compute_device).float()

        # Compute MC returns for critic training target
        returns = buf.compute_mc_returns().to(compute_device)

        # Compute GAE advantages using target critic
        advantages = self._compute_gae(
            states, rewards, terminated, valid_mask.bool(), compute_device
        )

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

        n_passes = max(1, n_epochs)
        for _ in range(n_passes):
            self.actor.update(
                states_flat,
                actions_flat,
                logprobs_flat,
                advantages_flat,
                valid_mask_flat,
            )
            self.critic.update(states_flat, returns_flat, valid_mask_flat)

            # Soft-update target network after each pass
            if self.critic.has_target:
                self.critic.update_target()

            self.n_iter += 1

        # Reset the buffer after learning (on-policy algorithm).
        buf.reset()
        self._episode_idx = 0
        self._step_in_episode = 0
        self._completed_episodes = 0

        return None
