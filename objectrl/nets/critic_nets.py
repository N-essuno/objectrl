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

from typing import Literal

import numpy as np
import torch
from torch import nn as nn

from objectrl.utils.net_utils import MLP, BayesianMLP, FeatureExtractor, PixelEncoder, make_pixel_encoder


def _prepare_state_action_input(
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    encoder: PixelEncoder | None,
) -> torch.Tensor:
    if isinstance(x, tuple):
        state, action = x
        if encoder is not None:
            state = encoder(state)
        return torch.cat((state, action), dim=-1)
    else:
        # Plain state tensor (e.g. ValueNet / PPO critic with no action input)
        if encoder is not None:
            return encoder(x)
        return x


class CriticNet(nn.Module):
    """
    Deterministic Critic Network (Q-network).
    Estimates the expected return (Q-value) for a given state-action pair.

    Args:
        dim_state (int): Dimension of observation space.
        dim_act (int): Dimension of action space.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (str): Activation function to use.
        has_norm (bool): Whether to include normalization layers.
    """

    def __init__(
        self,
        dim_state: int | tuple[int, int, int],
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = False,
        encoder_type: str = "light",
    ) -> None:
        super().__init__()

        self.encoder: PixelEncoder | None = None
        input_dim = dim_state
        if isinstance(dim_state, tuple):
            self.encoder = make_pixel_encoder(dim_state, feature_dim=width, encoder_type=encoder_type)
            input_dim = self.encoder.output_dim

        self.arch = MLP(
            input_dim + dim_act, 1, depth, width, act=act, has_norm=has_norm
        )

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the critic network.

        Args:
            x (Tensor): Concatenated observation and action tensor.
        """
        return self.arch(_prepare_state_action_input(x, self.encoder))


class ValueNet(CriticNet):
    """
    Value network for estimating V(s) without action input.

    Inherits from CriticNet but ignores action dimensions by setting dim_act to 0.
    Suitable for use in value-based methods like PPO or baseline estimation.

    Args:
        dim_state (int): Dimension of the input state.
        dim_act (int): Unused, kept for compatibility (should be 0).
        depth (int): Number of hidden layers in the network.
        width (int): Width (number of units) in each hidden layer.
        act (str): Activation function to use ("relu" or "crelu").
        has_norm (bool): Whether to apply normalization (e.g., LayerNorm).
    """

    def __init__(
        self,
        dim_state: int,
        dim_act: int,  # kept for interface compatibility
        depth: int = 3,
        width: int = 256,
        act: str = "relu",
        has_norm: bool = False,
        encoder_type: str = "light",
    ) -> None:
        super().__init__(dim_state, 0, depth, width, act, has_norm, encoder_type=encoder_type)


class CriticNetProbabilistic(nn.Module):
    """
    Probabilistic Critic Network.

    Args:
        dim_state (int): Observation space dimension.
        dim_act (int): Action space dimension.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (str): Activation function to use.
        has_norm (bool): Whether to use normalization layers.
    """

    def __init__(
        self,
        dim_state: int | tuple[int, int, int],
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = False,
        encoder_type: str = "light",
    ) -> None:
        super().__init__()

        self.encoder: PixelEncoder | None = None
        input_dim = dim_state
        if isinstance(dim_state, tuple):
            self.encoder = make_pixel_encoder(dim_state, feature_dim=width, encoder_type=encoder_type)
            input_dim = self.encoder.output_dim

        self.arch = MLP(
            input_dim + dim_act, 2, depth, width, act=act, has_norm=has_norm
        )

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the probabilistic critic.

        Args:
           x  (Tensor): Concatenated observation and action tensor.
        """
        return self.arch(_prepare_state_action_input(x, self.encoder))


class BNNCriticNet(nn.Module):
    """
    A Bayesian Critic Network (Q-network).

    Args:
        dim_state (int): Observation space dimension.
        dim_act (int): Action space dimension.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (Literal["relu", "crelu"]): Activation function to use.
        has_norm (bool): Whether to include normalization layers.
    """

    def __init__(
        self,
        dim_state: int | tuple[int, int, int],
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = False,
        encoder_type: str = "light",
    ) -> None:
        super().__init__()

        self.encoder: PixelEncoder | None = None
        input_dim = dim_state
        if isinstance(dim_state, tuple):
            self.encoder = make_pixel_encoder(dim_state, feature_dim=width, encoder_type=encoder_type)
            input_dim = self.encoder.output_dim

        # A BNN with local-reparameterization layers
        self.arch = BayesianMLP(
            dim_in=input_dim + dim_act,
            dim_out=1,
            depth=depth,
            width=width,
            act=act,
            has_norm=has_norm,
            layer_type="lr",
        )

        self._map = False

    def map(self, on: bool = True) -> None:
        "Switch maximum a posteriori mode on/off"
        self._map = on
        for layer in self.arch:
            if hasattr(layer, "_map"):
                layer.map(on)

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        "Forward pass of the BNNCriticNet"

        return self.arch(_prepare_state_action_input(x, self.encoder))


class EMstyle(nn.Module):
    """
    Encoder network for EM-style models.

    Args:
        dim_state (int): Observation space dimension.
        dim_act (int): Action space dimension.
        depth (int): Number of hidden layers.
        width (int): Hidden layer width and output dimensionality.
        act (str): Activation function to use.
        has_norm (bool): Whether to use normalization layers.
    """

    def __init__(
        self,
        dim_state: int | tuple[int, int, int],
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = False,
        encoder_type: str = "light",
    ) -> None:
        super().__init__()
        self.encoder: PixelEncoder | None = None
        input_dim = dim_state
        if isinstance(dim_state, tuple):
            self.encoder = make_pixel_encoder(dim_state, feature_dim=width, encoder_type=encoder_type)
            input_dim = self.encoder.output_dim

        self.arch = MLP(
            input_dim + dim_act, width, depth, width, act=act, has_norm=has_norm
        )

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to produce latent feature encoding.

        Args:
            x (Tensor): Concatenated input tensor.
        """
        x = self.arch(_prepare_state_action_input(x, self.encoder))
        return x


class DQNNet(nn.Module):
    """
    Deterministic Critic Network (Q-network).

    Args:
        dim_state (int): Dimension of observation space.
        dim_act (int): Dimension of action space.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (str): Activation function to use.
        has_norm (bool): Whether to include normalization layers.
    """

    def __init__(
        self,
        dim_state: int | tuple[int, int, int],
        dim_act: int,
        depth: int = 3,
        width: int = 256,
        act: str = "relu",
        has_norm: bool = False,
        encoder_type: str = "light",
    ) -> None:
        super().__init__()

        self.encoder: PixelEncoder | None = None
        input_dim = dim_state
        if isinstance(dim_state, tuple):
            self.encoder = make_pixel_encoder(dim_state, feature_dim=width, encoder_type=encoder_type)
            input_dim = self.encoder.output_dim

        self.arch = MLP(input_dim, dim_act, depth, width, act=act, has_norm=has_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network.

        Args:
            x (Tensor): Concatenated observation and action tensor.
        """
        if self.encoder is not None:
            x = self.encoder(x)
        return self.arch(x)


class QuantileCriticNet(nn.Module):
    """
    Quantile Critic Network for Distributional RL.
    Estimates the quantile values for given state-action pairs.

    Args:
        dim_state (int): Dimension of observation space.
        dim_act (int): Dimension of action space.
        depth (int): Number of hidden layers.
        width (int): Width of each hidden layer.
        act (str): Activation function to use.
        has_norm (bool): Whether to include normalization layers.
    """

    def __init__(
        self,
        dim_state: int,
        dim_act: int,
        depth: int,
        width: int = 256,
        act: Literal["relu", "crelu"] = "relu",
        has_norm: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = 128

        self.base_arch = FeatureExtractor(
            dim_in=dim_state + dim_act,
            depth=1,
            width=width,
            act="relu",
            has_norm=has_norm,
        )
        self.tau_arch = FeatureExtractor(
            dim_in=self.embedding_dim,
            depth=1,
            width=width,
            act="sigmoid",
            has_norm=has_norm,
        )
        self.out_arch = MLP(
            dim_in=width, dim_out=1, depth=2, width=width, act="relu", has_norm=has_norm
        )
        for i, layer in enumerate(self.out_arch.model):
            if isinstance(layer, nn.LayerNorm):
                self.out_arch.model[i] = nn.LayerNorm(width)

        self.const_vec = torch.from_numpy(np.arange(1, self.embedding_dim + 1)).float()
        self.const_vec = nn.Parameter(self.const_vec, requires_grad=False)

    def forward(self, xtau: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the critic network.
        Args:
            xtau (tuple[Tensor, Tensor]): Tuple containing concatenated observation-action tensor and quantile fractions.
        Returns:
            Tensor: Estimated quantile values.
        """
        x, tau = xtau
        sa_embedding = self.base_arch(x)  # [batch_size x width]

        tau_embedding = torch.cos(
            tau.unsqueeze(-1) * self.const_vec * np.pi
        )  # [batch_size x n_samples x embedding_dim]
        tau_embedding = self.tau_arch(tau_embedding)  # [batch_size x n_samples x width]

        x = (
            sa_embedding.unsqueeze(1) * tau_embedding
        )  # [batch_size x n_samples x width]
        q_values = self.out_arch(x).squeeze(-1)
        return q_values
