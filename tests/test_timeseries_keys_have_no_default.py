"""``toggle_consolidate_timeseries`` and ``timeseries_keep_triton_gridded`` have no default: a config states both.

Problem this file solves: a run with neither key set silently consolidated every per-scenario series and kept every
gridded variable, a choice the developer wants stated. Both fields now carry no default, so a config that omits either
is refused at load, naming the field. Two properties follow and each test below pins one against the real code.

- Every analysis config the toolkit tracks states both keys, so none of them depends on a default that no longer
  exists. The population is enumerated, never sampled: every tracked YAML file that is an analysis config, and every
  dict literal in a tracked test or script that builds one.
- A config that omits either key is refused at load, and the refusal names the field.

No solver, no Snakemake and no synthetic analysis is built, so every test is laptop-admissible.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hhemt.config.analysis import analysis_config
from tests.test_config_validation import _minimal_analysis_config_dict

_REPO = Path(__file__).resolve().parent.parent
_KEYS = ("toggle_consolidate_timeseries", "timeseries_keep_triton_gridded")
# The keys every analysis config must carry (neither has a default), so their joint presence marks a file or a dict as
# one. A partial member overlay names the indexer but never the weather record, so the overlay is not marked.
_MARKERS = ("weather_event_indices", "weather_timeseries")


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", *patterns], cwd=_REPO, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return [_REPO / rel for rel in out]


def _tracked_analysis_yaml_files() -> list[Path]:
    found = []
    for path in _tracked("*.yaml", "*.yml"):
        try:
            doc = yaml.safe_load(path.read_text())
        except (yaml.YAMLError, UnicodeDecodeError):
            continue
        if isinstance(doc, dict) and all(m in doc for m in _MARKERS):
            found.append(path)
    return found


def _analysis_dict_literals() -> list[tuple[Path, int, set[str]]]:
    """Every dict literal in a tracked test or script that names both marker keys, with the string keys it carries."""
    found = []
    for path in _tracked("tests/*.py", "tests/**/*.py", "scripts/*.py", "scripts/**/*.py"):
        if path == Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if all(m in keys for m in _MARKERS):
                found.append((path, node.lineno, keys))
    return found


def test_the_enumeration_finds_the_known_analysis_configs():
    """Property: the populations below are not vacuous, so an empty enumeration cannot pass the next two tests.

    Must-fire control: the template and both bundle fixtures are known analysis YAMLs, and the shared minimal builder
    of ``tests/test_config_validation.py`` is a known dict literal.
    """
    names = {p.name for p in _tracked_analysis_yaml_files()}
    assert {"template_analysis_config.yaml", "cfg_analysis.yaml"} <= names
    assert any(p.name == "test_config_validation.py" for p, _, _ in _analysis_dict_literals())


@pytest.mark.parametrize("path", _tracked_analysis_yaml_files(), ids=lambda p: str(p.relative_to(_REPO)))
def test_every_tracked_analysis_yaml_states_both_keys(path):
    """Property: no tracked analysis config YAML leaves either key to a default.

    Killed by: a new or edited analysis YAML that omits either key.
    """
    doc = yaml.safe_load(path.read_text())
    missing = [k for k in _KEYS if k not in doc]
    assert not missing, f"{path.relative_to(_REPO)} does not state {missing}"


@pytest.mark.parametrize(
    "path,line,keys",
    _analysis_dict_literals(),
    ids=lambda v: f"{v.relative_to(_REPO)}" if isinstance(v, Path) else None,
)
def test_every_tracked_analysis_dict_literal_states_both_keys(path, line, keys):
    """Property: no dict literal that builds an analysis config in a tracked test or script omits either key.

    Killed by: a new analysis-config dict that omits either key.
    """
    missing = [k for k in _KEYS if k not in keys]
    assert not missing, f"{path.relative_to(_REPO)}:{line} builds an analysis config without {missing}"


@pytest.mark.parametrize("key", _KEYS)
def test_a_config_that_omits_either_key_is_refused_naming_the_field(tmp_path, key):
    """Property: omitting one key is refused at load with a message that names that field and no other.

    Must-fire control: the same dict with the key restored loads, so the refusal is the omission and not something
    else in the dict. Plausible wrong implementation: restoring ``default=`` on the field, which loads silently.
    """
    cfg = _minimal_analysis_config_dict(tmp_path)
    assert analysis_config.model_validate(cfg) is not None
    omitted = {k: v for k, v in cfg.items() if k != key}
    with pytest.raises(ValidationError) as excinfo:
        analysis_config.model_validate(omitted)
    message = str(excinfo.value)
    assert key in message
    assert all(other not in message for other in _KEYS if other != key)
