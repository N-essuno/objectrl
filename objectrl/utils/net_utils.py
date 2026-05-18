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

import importlib
from collections.abc import Callable
from typing import Literal

import torch
import torch.optim as optim
from torch import nn as nn

from objectrl.nets.layers.bayesian_layers import (
    BBBLinear,
    CLTLinear,
    CLTLinearDet,
    LRLinear,
)
from objectrl.utils.custom_act import CReLU


def create_optimizer(config) -> Callable:
    """
    Creates a PyTorch optimizer based on the configuration.

    Args:
        config: Configuration object containing:
            - config.optimizer (str): Name of the optimizer (e.g., 'Adam', 'SGD').
            - config.learning_rate (float): Learning rate for the optimizer.
    Returns:
        Callable: A function that accepts model parameters and returns an optimizer instance.
    Raises:
        NotImplementedError: If the optimizer name is not available in torch.optim.
    """
    optimizer_name = config.optimizer
    if hasattr(optim, optimizer_name):
        optimizer = getattr(optim, optimizer_name)
    else:
        raise NotImplementedError(f"{optimizer_name} is not found in torch.optim")

    return lambda params: optimizer(params, lr=config.learning_rate)


def create_loss(config, reduction: str = "none") -> nn.Module:
    """
    Creates a loss function module from either torch.nn or a custom module.

    Args:
        config: Configuration object containing:
            - config.loss (str): Name of the loss function.
        reduction (str, optional): Reduction method ('none', 'mean', or 'sum'). Defaults to "none".
    Returns:
        nn.Module: A PyTorch loss function module.
    Raises:
        NotImplementedError: If the loss is not found in torch.nn or the custom module.
    """
    loss_name = config.loss
    if hasattr(nn, loss_name):
        return getattr(nn, loss_name)(reduction=reduction)
    else:
        loss_module = importlib.import_module(
            "objectrl.models.basic.loss"
        )  # Import the loss.py module
        if hasattr(loss_module, loss_name):
            loss_class = getattr(loss_module, loss_name)
            return loss_class(config)
        else:
            raise NotImplementedError(
                f"{loss_name} is not found in torch.nn or in loss.py"
            )


class FeatureExtractor(nn.Module):
    """
    Generic shallow MLP for feature extraction.

    Creates a stack of layers with the pattern:
        Linear → (LayerNorm) → Activation

    Args:
        dim_in (int): Input feature dimension.
        depth (int): Number of hidden layers (>= 1).
        width (int): Width of hidden layers.
        act (Literal["relu", "sigmoid"]): Activation function for all hidden layers.
        has_norm (bool): Whether to include LayerNorm after each linear layer.
    """

    def __init__(
        self,
        dim_in: int,
        depth: int,
        width: int,
        act: Literal["relu", "sigmoid"] = "relu",
        has_norm: bool = True,
    ):
        super().__init__()
        assert depth > 0, "Need at least one hidden layer"

        if act == "relu":
            self.activation_fn = nn.ReLU
        elif act == "sigmoid":
            self.activation_fn = nn.Sigmoid
        else:
            raise NotImplementedError(
                f"{act} is not implemented. User should add other activation functions if needed."
            )

        layers = []
        in_dim = dim_in
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            if has_norm:
                layers.append(nn.LayerNorm(width, elementwise_affine=True))
            layers.append(self.activation_fn())
            in_dim = width

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PixelEncoder(nn.Module):
    """
    VGG-style convolutional encoder for pixel observations shaped as HWC or NHWC.
    Uses stacked 3x3 convolutions with max pooling, similar to VGG16 architecture.
    This provides better feature extraction for complex visual tasks like CarRacing.

    HWC denotes a single image, while NHWC denotes a batch of images. The encoder
    applies a series of convolutional layers followed by a projection to a feature space.
    It automatically handles the addition of a batch dimension if the input is a single image.

    Architecture:
        - Block 1: 2x(Conv3x3, 64) + MaxPool → H/2 x W/2 x 64
        - Block 2: 2x(Conv3x3, 128) + MaxPool → H/4 x W/4 x 128
        - Block 3: 2x(Conv3x3, 256) + MaxPool → H/8 x W/8 x 256
        - Block 4: 2x(Conv3x3, 512) + MaxPool → H/16 x W/16 x 512
        - Flatten + Linear to feature_dim
    """

    def __init__(self, obs_shape: tuple[int, int, int], feature_dim: int = 512) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError(
                f"PixelEncoder expects a 3D observation shape (H, W, C), got {obs_shape}"
            )

        self.obs_shape = obs_shape
        in_channels = obs_shape[-1]

        # VGG-style architecture with 4 blocks
        self.conv = nn.Sequential(
            # Block 1: H x W -> H/2 x W/2
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: H/2 x W/2 -> H/4 x W/4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: H/4 x W/4 -> H/8 x W/8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: H/8 x W/8 -> H/16 x W/16
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *obs_shape)
            conv_out = self._forward_conv(sample)
            flat_dim = conv_out.shape[-1]

        self.proj = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
        )
        self.output_dim = feature_dim

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        x = self.conv(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        add_batch_dim = x.ndim == 3
        if add_batch_dim:
            x = x.unsqueeze(0)

        x = self._forward_conv(x)
        x = self.proj(x)

        if add_batch_dim:
            return x.squeeze(0)
        return x


# Keep VGGPixelEncoder as an alias for backward compatibility
VGGPixelEncoder = PixelEncoder


class LightPixelEncoder(nn.Module):
    """
    Lightweight Nature-DQN-style convolutional encoder for pixel observations.

    Three strided conv layers (32→64→64 channels) reduce a 96x96 image to a
    compact feature vector in far fewer operations than the VGG-style encoder,
    making it suitable for short training runs (1K-50K steps) where VGG would
    not yet have converged its large number of parameters.

    Architecture (for 96x96 input):
        Conv(32, 8x8, s4) → ReLU   →  23x23x32
        Conv(64, 4x4, s2) → ReLU   →  10x10x64
        Conv(64, 3x3, s1) → ReLU   →   8x8x64  = 4096
        Flatten → Linear(feature_dim) → ReLU → LayerNorm

    Args:
        obs_shape (tuple[int, int, int]): HWC observation shape, e.g. (96, 96, 3).
        feature_dim (int): Output feature vector size. Default: 256.
    """

    def __init__(self, obs_shape: tuple[int, int, int], feature_dim: int = 256) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError(
                f"LightPixelEncoder expects a 3D observation shape (H, W, C), got {obs_shape}"
            )
        self.obs_shape = obs_shape
        in_channels = obs_shape[-1]

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, in_channels, obs_shape[0], obs_shape[1])
            flat_dim = self.conv(sample).flatten(start_dim=1).shape[-1]

        self.proj = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
        )
        self.output_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        add_batch_dim = x.ndim == 3
        if add_batch_dim:
            x = x.unsqueeze(0)

        x = x.to(dtype=torch.float32)
        # HWC → CHW
        x = x.permute(0, 3, 1, 2) / 255.0
        x = self.conv(x).flatten(start_dim=1)
        x = self.proj(x)

        if add_batch_dim:
            return x.squeeze(0)
        return x


def make_pixel_encoder(
    obs_shape: tuple[int, int, int],
    feature_dim: int,
    encoder_type: str = "light",
) -> "LightPixelEncoder | PixelEncoder":
    """
    Factory for pixel encoders.

    Args:
        obs_shape: HWC tuple, e.g. (96, 96, 3).
        feature_dim: Output dimension.
        encoder_type: ``"light"`` (default) for the Nature-DQN-style 3-conv encoder,
                      ``"vgg"`` for the heavier VGG-style encoder.
    """
    if encoder_type == "vgg":
        return PixelEncoder(obs_shape, feature_dim=feature_dim)
    elif encoder_type == "light":
        return LightPixelEncoder(obs_shape, feature_dim=feature_dim)
    else:
        raise NotImplementedError(f"encoder_type={encoder_type!r} unknown. Use 'light' or 'vgg'.")


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        width: int,
        act: str = "relu",
        has_norm: bool = False,
    ) -> None:
        """
        Constructs a fully connected Multi-Layer Perceptron (MLP).

        Args:
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            depth (int): Total number of layers (>= 1).
            width (int): Width of the hidden layers.
            act (str): Activation function. Options are:
                - "relu": Standard ReLU.
                - "crelu": Concatenated ReLU (doubles width).
            has_norm (bool): If True, applies LayerNorm between layers.
        Raises:
            AssertionError: If depth <= 0.
            NotImplementedError: If unknown activation function is specified.
        """
        super().__init__()
        assert depth > 0, "Need at least one layer"

        if act == "crelu":
            self.activation_fn = CReLU
            width_multiplier = 2
        elif act == "relu":
            self.activation_fn = nn.ReLU
            width_multiplier = 1
        else:
            raise NotImplementedError(
                f"{act} is not implemented. User should add other activation functions if needed."
            )

        effective_width = width * width_multiplier

        layers = []

        if depth == 1:
            layers.append(nn.Linear(dim_in, dim_out))
        else:
            layers.append(nn.Linear(dim_in, width))
            # Hidden layers
            for i in range(depth - 1):
                if has_norm:
                    layers.append(nn.LayerNorm(width, elementwise_affine=False))
                layers.append(self.activation_fn())
                # Last hidden layer connects to output
                if i == depth - 2:
                    layers.append(nn.Linear(effective_width, dim_out))
                else:
                    layers.append(nn.Linear(effective_width, width))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the standard MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim_out).
        """
        return self.model(x)


class BayesianMLP(nn.Module):
    def __init__(  # noqa: C901
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        width: int,
        layer_type: Literal["bbb", "lr", "clt", "cltdet"] = "lr",
        act: Literal["crelu", "relu"] = "relu",
        has_norm: bool = False,
    ) -> None:
        """
        Constructs a Bayesian MLP using probabilistic linear layers.
        Supports various types of Bayesian layers for uncertainty estimation.

        Args:
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            depth (int): Number of layers (>= 1).
            width (int): Width of the hidden layers.
            layer_type (str): Type of Bayesian linear layer. One of:
                - "bbb": Bayes by Backprop.
                - "lr": Local Reparameterization trick.
                - "clt": Central Limit Theorem (probabilistic forward).
                - "cltdet": CLT with deterministic weights.
            act (str): Activation function. One of "relu" or "crelu".
            has_norm (bool): Whether to apply LayerNorm. Not supported for CLT variants.
        Raises:
            AssertionError: If depth <= 0 or incompatible settings.
            NotImplementedError: For unknown layer or activation types.
        """
        super().__init__()
        assert depth > 0, "Need at least one layer"

        # Pick the layer type
        if "clt" in layer_type:
            assert act in [
                "crelu",
                "relu",
            ], "Deterministic uncertainty propagation is only available for 'relu' and 'crelu' activations"
            assert not has_norm, "Not available for CLT-based layers"

        # Identify the chosen layer
        match layer_type:
            case "bbb":
                bnn_layer = BBBLinear
                det_uncertainty = False
            case "lr":
                bnn_layer = LRLinear
                det_uncertainty = False
            case "clt":
                bnn_layer = CLTLinear
                det_uncertainty = True
            case "cltdet":
                bnn_layer = CLTLinearDet
                det_uncertainty = True
            case _:
                raise NotImplementedError(f"{layer_type} is not implemented")

        # Select activation and width multiplier
        if act == "crelu":
            activation_fn = CReLU
            width_multiplier = 2
        elif act == "relu":
            activation_fn = nn.ReLU
            width_multiplier = 1
        else:
            raise NotImplementedError(f"{act} is not implemented")

        effective_width = width * width_multiplier

        layers = []

        # Single-layer case needs to be handled differently
        if depth == 1:
            layers.append(bnn_layer(dim_in, dim_out))
        else:
            layers.append(bnn_layer(dim_in, width))

            for i in range(depth - 1):
                if has_norm and not det_uncertainty:
                    layers.append(nn.LayerNorm(width, elementwise_affine=False))
                if not det_uncertainty:
                    layers.append(activation_fn())
                if i == depth - 2:
                    layers.append(bnn_layer(effective_width, dim_out))
                else:
                    layers.append(bnn_layer(effective_width, width))

        self.model = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the Bayesian MLP.

        Args:
            x (Union[Tensor, Tuple[Tensor, Optional[Tensor]]]):
                - For standard use: input tensor.
                - For CLT-based: tuple of (mean, variance).
        Returns:
            Union[Tensor, Tuple[Tensor, Optional[Tensor]]]: Output in the same format as input.
        """
        # CLT-based layers can have (mean, variance) input
        return self.model(x)

    def get_kl(self) -> tuple[torch.Tensor, int]:
        """
        Get the KL divergence of the Bayesian MLP.
        """
        total_kl = 0.0
        n_params = 0
        for layer in self.model:
            if isinstance(layer, BBBLinear | LRLinear | CLTLinear):
                kl, n = layer.KL()
                total_kl += kl
                n_params += n
        return total_kl, n_params  # ty: ignore (Ty thinks that total_kl stays a float)
