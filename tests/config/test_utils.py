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

from types import SimpleNamespace

import pytest
import torch

from objectrl.config.config import MainConfig
from objectrl.config.utils import setup_config


def test_setup_config_falls_back_to_cpu_when_only_storing_device_overridden(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result_path = tmp_path / "results"
    model_args = {
        "model": {"name": "sac"},
        "logging": {"result_path": str(result_path)},
    }
    tyro_config = MainConfig.from_config(
        {
            "model": {"name": "sac"},
            "system": {"storing_device": "cpu"},
            "logging": {"result_path": str(result_path)},
        }
    )

    config = setup_config(
        config_path=None,
        model_args=model_args,
        tyro_config=tyro_config,
        subset_tyro={"system": {"storing_device": None}},
    )

    assert config.system.storing_device == "cpu"
    assert config.system.device == "cpu"


def test_setup_config_keeps_error_for_explicit_cuda_device(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result_path = tmp_path / "results"
    model_args = {
        "model": {"name": "sac"},
        "logging": {"result_path": str(result_path)},
    }
    tyro_config = MainConfig.from_config(
        {
            "model": {"name": "sac"},
            "system": {"device": "cuda", "storing_device": "cpu"},
            "logging": {"result_path": str(result_path)},
        }
    )

    with pytest.raises(RuntimeError, match="--system.device=cpu"):
        setup_config(
            config_path=None,
            model_args=model_args,
            tyro_config=tyro_config,
            subset_tyro={"system": {"device": None, "storing_device": None}},
        )


def test_setup_config_accepts_explicit_mps_device_when_available(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    result_path = tmp_path / "results"
    model_args = {
        "model": {"name": "sac"},
        "logging": {"result_path": str(result_path)},
    }
    tyro_config = MainConfig.from_config(
        {
            "model": {"name": "sac"},
            "system": {"device": "mps", "storing_device": "cpu"},
            "logging": {"result_path": str(result_path)},
        }
    )

    config = setup_config(
        config_path=None,
        model_args=model_args,
        tyro_config=tyro_config,
        subset_tyro={"system": {"device": None, "storing_device": None}},
    )

    assert config.system.device == "mps"
    assert config.system.storing_device == "cpu"


def test_setup_config_keeps_error_for_explicit_mps_device(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    result_path = tmp_path / "results"
    model_args = {
        "model": {"name": "sac"},
        "logging": {"result_path": str(result_path)},
    }
    tyro_config = MainConfig.from_config(
        {
            "model": {"name": "sac"},
            "system": {"device": "mps", "storing_device": "cpu"},
            "logging": {"result_path": str(result_path)},
        }
    )

    with pytest.raises(RuntimeError, match="--system.device=cpu"):
        setup_config(
            config_path=None,
            model_args=model_args,
            tyro_config=tyro_config,
            subset_tyro={"system": {"device": None, "storing_device": None}},
        )
