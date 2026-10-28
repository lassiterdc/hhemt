"""The config-load gate `assert_both_configs_load`, exercised on real YAML files.

Every other test that reaches the gate replaces it with a no-op stub, so nothing else
shows what it accepts and what it refuses. These tests write real system and analysis
YAML under `tmp_path` and call the gate directly. No solver is compiled or run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hhemt.exceptions import ConfigurationError
from hhemt.validation import assert_both_configs_load
from tests.test_config_validation import _minimal_analysis_config_dict, _minimal_system_config_dict


def _write(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_valid_pair_returns_none(tmp_path):
    system = _write(tmp_path / "system.yaml", _minimal_system_config_dict(tmp_path))
    analysis = _write(tmp_path / "analysis.yaml", _minimal_analysis_config_dict(tmp_path))
    assert assert_both_configs_load(system, analysis) is None


def test_bad_system_config_is_refused_naming_file_and_field(tmp_path):
    bad = _minimal_system_config_dict(tmp_path)
    bad["unexpected_extra"] = 1
    system = _write(tmp_path / "system.yaml", bad)
    analysis = _write(tmp_path / "analysis.yaml", _minimal_analysis_config_dict(tmp_path))
    with pytest.raises(ConfigurationError) as exc:
        assert_both_configs_load(system, analysis)
    assert "system.yaml" in str(exc.value)
    assert "unexpected_extra" in str(exc.value)
    assert "analysis.yaml" not in str(exc.value)


def test_bad_analysis_config_is_refused_naming_file_and_field(tmp_path):
    bad = _minimal_analysis_config_dict(tmp_path)
    bad["unexpected_extra"] = 1
    system = _write(tmp_path / "system.yaml", _minimal_system_config_dict(tmp_path))
    analysis = _write(tmp_path / "analysis.yaml", bad)
    with pytest.raises(ConfigurationError) as exc:
        assert_both_configs_load(system, analysis)
    assert "analysis.yaml" in str(exc.value)
    assert "unexpected_extra" in str(exc.value)
    assert "system.yaml" not in str(exc.value)


def test_missing_config_path_raises_file_not_found_unwrapped(tmp_path):
    analysis = _write(tmp_path / "analysis.yaml", _minimal_analysis_config_dict(tmp_path))
    with pytest.raises(FileNotFoundError):
        assert_both_configs_load(tmp_path / "absent.yaml", analysis)


def test_both_bad_reports_both_files_in_one_error(tmp_path):
    bad_system = _minimal_system_config_dict(tmp_path)
    bad_system["extra_in_system"] = 1
    bad_analysis = _minimal_analysis_config_dict(tmp_path)
    bad_analysis["extra_in_analysis"] = 1
    system = _write(tmp_path / "system.yaml", bad_system)
    analysis = _write(tmp_path / "analysis.yaml", bad_analysis)
    with pytest.raises(ConfigurationError) as exc:
        assert_both_configs_load(system, analysis)
    text = str(exc.value)
    assert "system.yaml" in text and "extra_in_system" in text
    assert "analysis.yaml" in text and "extra_in_analysis" in text
