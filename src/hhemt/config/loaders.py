from __future__ import annotations
from pathlib import Path
from typing import TypeVar
import yaml
from hhemt.config.system import system_config
from hhemt.config.analysis import analysis_config
from hhemt.config.brand_theme import brand_theme
from hhemt.config.globus import GlobusTransferSpec
from hhemt.config.hpc_system import hpc_system_config

_M = TypeVar("_M")


def _missing_config_message(cfg_yaml: Path, exc: FileNotFoundError) -> str:
    """Explain a missing config path when it sits on a node-local filesystem.

    A LEGIBLE DIAGNOSTIC, never a second guard. The login-node preflight
    ``validation.assert_configs_visible_cross_node`` refuses the cases it can SEE;
    it cannot see a path a rule builds at runtime, a caller that reaches a builder
    without passing a guarded facade, or a run whose operator set the
    acknowledgement env var. In each of those the rule still dies on a compute
    node, and it dies naming a path the operator knows exists on the login node --
    which reads as a broken filesystem rather than a cross-node visibility failure.
    This does not prevent the death; it explains it.

    Attributed rather than unconditional: appending this to an ordinary typo, whose
    message is already complete, teaches operators to skim. Roots include ``/tmp``
    and ``/var/tmp`` beyond ``gettempdir()`` because under SLURM a compute node's
    ``$TMPDIR`` is often a per-job directory, so ``gettempdir()`` there can return
    something else entirely while the offending path is still under ``/tmp``.
    A node-local root that is neither (``/local``, ``/scratch-local``) gets the
    unaugmented message, which is no worse than the pre-existing behaviour.
    """
    import tempfile

    roots = {Path(tempfile.gettempdir()).resolve(), Path("/tmp"), Path("/var/tmp")}
    resolved = cfg_yaml.resolve()
    if not any(resolved.is_relative_to(root) for root in roots):
        return str(exc)
    return (
        f"{exc}\n"
        f"This path is on a NODE-LOCAL filesystem. If it exists on the login node but "
        f"not here, the config was staged where this compute node cannot see it, every "
        f"rule reading it will fail identically, and the allocation is already spent. "
        f"Stage on a shared filesystem (e.g. /scratch/$USER/...), or re-run with "
        f"execution_mode='local'. The login-node preflight that normally refuses this "
        f"is validation.assert_configs_visible_cross_node; it does not see paths a rule "
        f"builds at runtime, and it is bypassed by HHEMT_ALLOW_NODE_LOCAL_CONFIGS=1."
    )


def _load_config(cfg_yaml: Path, model_cls: type[_M]) -> _M:
    try:
        text = cfg_yaml.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(_missing_config_message(cfg_yaml, exc)) from exc
    raw = yaml.safe_load(text)
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
