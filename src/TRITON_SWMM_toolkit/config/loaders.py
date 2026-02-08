from pathlib import Path
import yaml
from TRITON_SWMM_toolkit.config.system import system_config
from TRITON_SWMM_toolkit.config.analysis import analysis_config


def load_system_config_from_dict(cfg_dict):
    cfg = system_config.model_validate(cfg_dict)
    return cfg


def load_system_config(cfg_yaml: Path):
    cfg = yaml.safe_load(cfg_yaml.read_text())
    cfg = system_config.model_validate(cfg)
    return cfg


def load_analysis_config(cfg_yaml: Path):
    cfg = yaml.safe_load(cfg_yaml.read_text())
    cfg = analysis_config.model_validate(cfg)
    return cfg
