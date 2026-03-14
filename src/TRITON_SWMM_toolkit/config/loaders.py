from __future__ import annotations
from pathlib import Path
from typing import TypeVar
import yaml
from TRITON_SWMM_toolkit.config.system import system_config
from TRITON_SWMM_toolkit.config.analysis import analysis_config
from TRITON_SWMM_toolkit.config.globus import GlobusTransferSpec

_M = TypeVar("_M")


def _load_config(cfg_yaml: Path, model_cls: type[_M]) -> _M:
    return model_cls.model_validate(yaml.safe_load(cfg_yaml.read_text()))


def load_system_config_from_dict(cfg_dict: dict) -> system_config:
    return system_config.model_validate(cfg_dict)


def load_system_config(cfg_yaml: Path) -> system_config:
    return _load_config(cfg_yaml, system_config)


def load_analysis_config(cfg_yaml: Path) -> analysis_config:
    return _load_config(cfg_yaml, analysis_config)


def load_transfer_config(cfg_yaml: Path) -> GlobusTransferSpec:
    return _load_config(cfg_yaml, GlobusTransferSpec)
