from __future__ import annotations
from pathlib import Path
from typing import TypeVar
import yaml
from TRITON_SWMM_toolkit.config.system import system_config
from TRITON_SWMM_toolkit.config.analysis import analysis_config
from TRITON_SWMM_toolkit.config.brand_theme import brand_theme
from TRITON_SWMM_toolkit.config.globus import GlobusTransferSpec
from TRITON_SWMM_toolkit.config.hpc_system import hpc_system_config

_M = TypeVar("_M")


def _load_config(cfg_yaml: Path, model_cls: type[_M]) -> _M:
    raw = yaml.safe_load(cfg_yaml.read_text())
    if raw is None:
        raise ValueError(
            f"YAML config at {cfg_yaml} parsed to None (file empty or top-level null). "
            "Under high parallel I/O this can indicate a concurrent-write race; "
            "see sensitivity_analysis.py::_create_sub_analyses."
        )
    return model_cls.model_validate(raw)


def yaml_to_model(cfg_yaml: Path, model_cls: type[_M]) -> _M:
    """Load a YAML file and validate it against a Pydantic model class."""
    return _load_config(cfg_yaml, model_cls)


def load_system_config_from_dict(cfg_dict: dict) -> system_config:
    return system_config.model_validate(cfg_dict)


def load_system_config(cfg_yaml: Path) -> system_config:
    return _load_config(cfg_yaml, system_config)


def load_analysis_config(cfg_yaml: Path) -> analysis_config:
    return _load_config(cfg_yaml, analysis_config)


def load_hpc_system_config(cfg_yaml: Path) -> hpc_system_config:
    return _load_config(cfg_yaml, hpc_system_config)


def load_brand_theme(cfg_yaml: Path) -> brand_theme:
    return _load_config(cfg_yaml, brand_theme)


def load_transfer_config(cfg_yaml: Path) -> GlobusTransferSpec:
    return _load_config(cfg_yaml, GlobusTransferSpec)
