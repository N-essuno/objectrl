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
GRPO++ Model Configuration.

This module defines the configuration classes for GRPO++, an enhanced version
of Group Relative Policy Optimization with asymmetric clipping, temporal
aggregation, and zero KL penalty.
"""

from dataclasses import dataclass, field

from objectrl.models.grpo_pp import (
    GRPOPlusActor,
    GRPOPlusActorNetProbabilistic,
    GRPOPlusCritic,
)
from objectrl.nets.critic_nets import ValueNet


# [start-config]
@dataclass
class GRPO_PPActorConfig:
    """
    Configuration for the GRPO++ actor network.

    Attributes:
        arch (type): The architecture class to be used for the actor network.
        actor_type (type): The class implementing the GRPO++ actor logic.
        has_target (bool): Whether to maintain a target network for the actor.
        width (int): Width of hidden layers in the actor network.
        max_grad_norm (float): Maximum norm for gradient clipping in the actor.
    """

    arch: type = GRPOPlusActorNetProbabilistic
    actor_type: type = GRPOPlusActor
    has_target: bool = False
    width: int = 64
    max_grad_norm: float = 0.5


@dataclass
class GRPO_PPCriticConfig:
    """
    Configuration for the GRPO++ critic network.

    Attributes:
        arch (type): The architecture class to be used for the critic network.
        critic_type (type): The class implementing the GRPO++ critic logic.
        has_target (bool): Whether to maintain a target network for the critic.
        n_members (int): Number of ensemble members in the critic network.
        width (int): Width of hidden layers in the critic network.
        max_grad_norm (float): Maximum norm for gradient clipping in the critic.
    """

    arch: type = ValueNet
    critic_type: type = GRPOPlusCritic
    has_target: bool = False
    n_members: int = 1
    width: int = 64
    max_grad_norm: float = 0.5


@dataclass
class GRPO_PPConfig:
    """
    Full configuration for a GRPO++ agent.

    GRPO++ introduces three key improvements over vanilla GRPO:
    1. DAPO asymmetric clipping (clip_rate_high and clip_rate_low)
    2. Dr. GRPO aggregation (sum over time, mean over group)
    3. Zero KL penalty (removed standard KL tax)

    Attributes:
        name (str): Identifier name for the GRPO++ configuration.
        loss (str): Name of the loss function to use (e.g., 'MSELoss').
        tau (float): Polyak averaging coefficient for target network updates.
        policy_delay (int): Delay interval between policy (actor) updates.
        max_grad_norm (float): Maximum norm for gradient clipping globally.
        clip_rate_high (float): Tighter clipping bound for positive advantages (DAPO).
        clip_rate_low (float): Looser clipping bound for negative advantages (DAPO).
        normalize_advantages (bool): Whether to normalize advantages during training.
        entropy_coef (float): Coefficient for entropy regularization.
        group_size (int): Number of parallel environments/trajectories per group.
        max_episode_length (int): Maximum trajectory length for buffer allocation.
        gamma (float): Discount factor for computing returns.
        actor (GRPO_PPActorConfig): Configuration object for the GRPO++ actor.
        critic (GRPO_PPCriticConfig): Configuration object for the GRPO++ critic.
    """

    name: str = "grpo_pp"
    loss: str = "MSELoss"

    tau: float = 0.0
    policy_delay: int = 1
    max_grad_norm: float = 0.5

    # DAPO: Asymmetric clipping bounds
    # Tighter bound for positive advantages (prevents policy collapse)
    clip_rate_high: float = 0.1
    # Looser bound for negative advantages (rapid rejection of bad policies)
    # Conservative default (0.3) for stability; increase to 0.5 if training is stable
    clip_rate_low: float = 0.3

    normalize_advantages: bool = True
    entropy_coef: float = 0.0  # Zero KL penalty

    # Group rollout configuration
    group_size: int = 8
    max_episode_length: int = 1000
    gamma: float = 0.99

    actor: GRPO_PPActorConfig = field(default_factory=GRPO_PPActorConfig)
    critic: GRPO_PPCriticConfig = field(default_factory=GRPO_PPCriticConfig)


# [end-config]
