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
GRPO++ V2 Model Configuration.

This module defines the configuration classes for GRPO++ V2, which combines
group-relative advantages with a learned value function using GAE (multi-step TD).
This produces lower-variance advantage estimates compared to pure Monte Carlo returns.
"""

from dataclasses import dataclass, field

from objectrl.models.grpo_pp import GRPOPlusActorNetProbabilistic
from objectrl.models.grpo_pp_v2 import GRPOPlusV2Actor, GRPOPlusV2Critic
from objectrl.nets.critic_nets import ValueNet


# [start-config]
@dataclass
class GRPO_PP_V2ActorConfig:
    """
    Configuration for the GRPO++ V2 actor network.

    Attributes:
        arch (type): The architecture class to be used for the actor network.
        actor_type (type): The class implementing the GRPO++ V2 actor logic.
        has_target (bool): Whether to maintain a target network for the actor.
        width (int): Width of hidden layers in the actor network.
        max_grad_norm (float): Maximum norm for gradient clipping in the actor.
    """

    arch: type = GRPOPlusActorNetProbabilistic
    actor_type: type = GRPOPlusV2Actor
    has_target: bool = False
    width: int = 64
    max_grad_norm: float = 0.5


@dataclass
class GRPO_PP_V2CriticConfig:
    """
    Configuration for the GRPO++ V2 critic network.

    The critic uses a target network (has_target=True) for stable V(s)
    estimates during GAE computation.

    Attributes:
        arch (type): The architecture class to be used for the critic network.
        critic_type (type): The class implementing the GRPO++ V2 critic logic.
        has_target (bool): Whether to maintain a target network for the critic.
        n_members (int): Number of ensemble members in the critic network.
        width (int): Width of hidden layers in the critic network.
        max_grad_norm (float): Maximum norm for gradient clipping in the critic.
    """

    arch: type = ValueNet
    critic_type: type = GRPOPlusV2Critic
    has_target: bool = True
    n_members: int = 1
    width: int = 64
    max_grad_norm: float = 0.5


@dataclass
class GRPO_PP_V2Config:
    """
    Full configuration for a GRPO++ V2 agent.

    GRPO++ V2 combines group-relative advantages with a learned value function
    using GAE (Generalized Advantage Estimation):

    1. GAE reduces variance via a learned baseline (target critic V(s))
    2. Group normalization preserves relative comparison (GRPO insight)
    3. DAPO asymmetric clipping prevents policy collapse
    4. Dr. GRPO aggregation (sum over time, mean over group)
    5. Entropy regularization for exploration

    Attributes:
        name (str): Identifier name for the GRPO++ V2 configuration.
        loss (str): Name of the loss function to use (e.g., 'MSELoss').
        tau (float): Polyak averaging coefficient for target network updates.
        policy_delay (int): Delay interval between policy (actor) updates.
        max_grad_norm (float): Maximum norm for gradient clipping globally.
        clip_rate_high (float): Clipping bound for positive advantages (DAPO).
        clip_rate_low (float): Clipping bound for negative advantages (DAPO).
        normalize_advantages (bool): Whether to normalize advantages during training.
        GAE_lambda (float): GAE lambda parameter for bias-variance tradeoff.
        entropy_coef (float): Coefficient for entropy regularization.
        group_size (int): Number of parallel environments/trajectories per group.
        max_episode_length (int): Maximum trajectory length for buffer allocation.
        gamma (float): Discount factor for computing returns.
        actor (GRPO_PP_V2ActorConfig): Configuration object for the GRPO++ V2 actor.
        critic (GRPO_PP_V2CriticConfig): Configuration object for the GRPO++ V2 critic.
    """

    name: str = "grpo_pp_v2"
    loss: str = "MSELoss"

    tau: float = 0.005
    policy_delay: int = 1
    max_grad_norm: float = 0.5

    # DAPO: Asymmetric clipping bounds
    clip_rate_high: float = 0.2
    clip_rate_low: float = 0.3

    normalize_advantages: bool = True
    GAE_lambda: float = 0.95
    entropy_coef: float = 0.01

    # Group rollout configuration
    group_size: int = 8
    max_episode_length: int = 1000
    gamma: float = 0.99

    actor: GRPO_PP_V2ActorConfig = field(default_factory=GRPO_PP_V2ActorConfig)
    critic: GRPO_PP_V2CriticConfig = field(default_factory=GRPO_PP_V2CriticConfig)


# [end-config]
