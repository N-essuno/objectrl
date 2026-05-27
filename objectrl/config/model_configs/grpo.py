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

from dataclasses import dataclass, field

from objectrl.models.grpo import GRPOActor, GRPOActorNetProbabilistic, GRPOCritic
from objectrl.nets.critic_nets import ValueNet


# [start-config]
@dataclass
class GRPOActorConfig:
    """
    Configuration for the GRPO actor network.

    Attributes:
        arch (type): The architecture class to be used for the actor network.
        actor_type (type): The class implementing the GRPO actor logic.
        has_target (bool): Whether to maintain a target network for the actor.
        width (int): Width of hidden layers in the actor network.
        max_grad_norm (float): Maximum norm for gradient clipping in the actor.
    """

    arch: type = GRPOActorNetProbabilistic
    actor_type: type = GRPOActor
    has_target: bool = False
    width: int = 64
    max_grad_norm: float = 0.5


@dataclass
class GRPOCriticConfig:
    """
    Configuration for the GRPO critic network.

    In vanilla GRPO, the critic is not used for advantage computation — advantages
    come from group statistics. The critic is trained to predict MC returns only.

    Attributes:
        arch (type): The architecture class to be used for the critic network.
        critic_type (type): The class implementing the GRPO critic logic.
        has_target (bool): Whether to maintain a target network for the critic.
        n_members (int): Number of ensemble members in the critic network.
        width (int): Width of hidden layers in the critic network.
        max_grad_norm (float): Maximum norm for gradient clipping in the critic.
    """

    arch: type = ValueNet
    critic_type: type = GRPOCritic
    has_target: bool = False
    n_members: int = 1
    width: int = 64
    max_grad_norm: float = 0.5


@dataclass
class GRPOConfig:
    """
    Full configuration for a vanilla GRPO agent.

    Vanilla GRPO eliminates the learned value baseline. Advantages are computed
    as MC returns normalized across the group at each timestep. The critic is
    optional and only trains to predict MC returns.

    Attributes:
        name (str): Identifier name for the GRPO configuration.
        loss (str): Name of the loss function to use (e.g., 'MSELoss').
        tau (float): Polyak averaging coefficient (unused in vanilla GRPO, kept for compat).
        policy_delay (int): Delay interval between policy (actor) updates.
        max_grad_norm (float): Maximum norm for gradient clipping globally.
        clip_rate (float): Symmetric clipping factor for the GRPO objective.
        GAE_lambda (float): Lambda for GAE (unused in vanilla GRPO, kept for compat).
        normalize_advantages (bool): Whether to normalize advantages during training.
        entropy_coef (float): Coefficient for entropy regularization.
        group_size (int): Number of trajectories per group rollout.
        max_episode_length (int): Maximum episode length for buffer sizing.
        gamma (float): Discount factor for MC returns.
        actor (GRPOActorConfig): Configuration object for the GRPO actor.
        critic (GRPOCriticConfig): Configuration object for the GRPO critic.
    """

    name: str = "grpo"
    loss: str = "MSELoss"

    tau: float = 0.0
    policy_delay: int = 1
    max_grad_norm: float = 0.5
    clip_rate: float = 0.2
    GAE_lambda: float = 0.95
    normalize_advantages: bool = True
    entropy_coef: float = 0.0
    group_size: int = 4
    max_episode_length: int = 1000
    gamma: float = 0.99

    actor: GRPOActorConfig = field(default_factory=GRPOActorConfig)
    critic: GRPOCriticConfig = field(default_factory=GRPOCriticConfig)


# [end-config]
