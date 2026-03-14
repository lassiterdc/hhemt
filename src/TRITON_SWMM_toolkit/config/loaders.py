from pathlib import Path
import yaml
from TRITON_SWMM_toolkit.config.system import system_config
from TRITON_SWMM_toolkit.config.analysis import analysis_config
from TRITON_SWMM_toolkit.config.globus import GlobusTransferSpec


def load_system_config_from_dict(cfg_dict: dict) -> system_config:
    cfg = system_config.model_validate(cfg_dict)
    return cfg


def load_system_config(cfg_yaml: Path) -> system_config:
    cfg = yaml.safe_load(cfg_yaml.read_text())
    cfg = system_config.model_validate(cfg)
    return cfg


def load_analysis_config(cfg_yaml: Path) -> analysis_config:
    cfg = yaml.safe_load(cfg_yaml.read_text())
    cfg = analysis_config.model_validate(cfg)
    return cfg


def load_transfer_config(cfg_yaml: Path) -> GlobusTransferSpec:
    cfg = yaml.safe_load(cfg_yaml.read_text())
    return GlobusTransferSpec.model_validate(cfg)
