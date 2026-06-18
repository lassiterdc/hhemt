"""Tests for Globus transfer configuration models.

Tests PostRunTransferConfig validation, WSL path normalization,
and GlobusTransferSpec generation.
"""

from pathlib import Path

import pytest

from hhemt.config.globus import (
    PostRunTransferConfig,
    _normalize_wsl_path,
)
from hhemt.exceptions import ConfigurationError


class TestNormalizeWslPath:
    """Tests for _normalize_wsl_path() helper."""

    def test_windows_backslash_path(self):
        assert _normalize_wsl_path(r"D:\Dropbox\foo") == "/mnt/d/Dropbox/foo"

    def test_windows_forward_slash_path(self):
        assert _normalize_wsl_path("D:/Dropbox/foo") == "/mnt/d/Dropbox/foo"

    def test_lowercase_drive_letter(self):
        assert _normalize_wsl_path(r"c:\Users\test") == "/mnt/c/Users/test"

    def test_posix_path_unchanged(self):
        assert _normalize_wsl_path("/mnt/d/Dropbox/foo") == "/mnt/d/Dropbox/foo"

    def test_relative_path_unchanged(self):
        assert _normalize_wsl_path("relative/path") == "relative/path"

    def test_unc_path_raises(self):
        with pytest.raises(ValueError, match="UNC paths"):
            _normalize_wsl_path(r"\\server\share\path")

    def test_path_with_spaces(self):
        result = _normalize_wsl_path(r"D:\My Documents\file")
        assert result == "/mnt/d/My Documents/file"

    def test_trailing_backslash(self):
        result = _normalize_wsl_path("D:\\Dropbox\\foo\\")
        assert result == "/mnt/d/Dropbox/foo/"


class TestPostRunTransferConfig:
    """Tests for PostRunTransferConfig model validation."""

    def test_valid_config_frontier(self):
        config = PostRunTransferConfig(
            destination_root=r"D:\Dropbox\results",
            system="frontier",
        )
        assert config.system == "frontier"
        assert config.sync_level == 0
        assert config.conflict_policy == "prompt"

    def test_valid_config_uva(self):
        config = PostRunTransferConfig(
            destination_root="/mnt/d/results",
            system="uva",
        )
        assert config.system == "uva"

    def test_unknown_system_raises(self):
        with pytest.raises(ConfigurationError, match="Unknown Globus system"):
            PostRunTransferConfig(
                destination_root="/mnt/d/results",
                system="nonexistent",
            )

    def test_default_exclude_patterns(self):
        config = PostRunTransferConfig(
            destination_root="/mnt/d/results",
            system="frontier",
        )
        assert "subanalyses" in config.exclude_patterns
        assert "out_triton" in config.exclude_patterns

    def test_custom_exclude_patterns(self):
        config = PostRunTransferConfig(
            destination_root="/mnt/d/results",
            system="frontier",
            exclude_patterns=["logs/"],
        )
        assert config.exclude_patterns == ["logs/"]

    def test_include_sims(self):
        config = PostRunTransferConfig(
            destination_root="/mnt/d/results",
            system="frontier",
            include_sims=[0, 3, 7],
        )
        assert config.include_sims == [0, 3, 7]


class TestToTransferSpec:
    """Tests for PostRunTransferConfig.to_transfer_spec()."""

    def test_generates_correct_spec(self):
        config = PostRunTransferConfig(
            destination_root=r"D:\Dropbox\results",
            system="frontier",
        )
        spec = config.to_transfer_spec(
            analysis_dir=Path("/lustre/orion/***REMOVED***/scratch/user/my_analysis"),
            analysis_id="my_analysis",
        )
        assert spec.endpoints.source_uuid == "36d521b3-c182-4071-b7d5-91db5d380d42"
        assert spec.endpoints.destination_uuid == "***REMOVED***"
        assert len(spec.items) == 1
        assert spec.items[0].source_path == "/lustre/orion/***REMOVED***/scratch/user/my_analysis"
        assert spec.items[0].destination_path == "/D/Dropbox/results/my_analysis"
        assert spec.sync_level == 0

    def test_include_sims_generates_per_sim_items(self):
        config = PostRunTransferConfig(
            destination_root="/mnt/d/results",
            system="uva",
            include_sims=[0, 3],
        )
        spec = config.to_transfer_spec(
            analysis_dir=Path("/scratch/user/my_analysis"),
            analysis_id="my_analysis",
        )
        # include_sims produces per-sim items + one recursive top-level item
        assert len(spec.items) == 3
        assert spec.items[0].source_path == "/scratch/user/my_analysis/sims/0"
        assert spec.items[1].source_path == "/scratch/user/my_analysis/sims/3"

    def test_label_auto_generated(self):
        config = PostRunTransferConfig(
            destination_root="/mnt/d/results",
            system="frontier",
        )
        spec = config.to_transfer_spec(
            analysis_dir=Path("/lustre/orion/test"),
            analysis_id="test_run",
        )
        assert "test_run" in spec.label

    def test_custom_label(self):
        config = PostRunTransferConfig(
            destination_root="/mnt/d/results",
            system="frontier",
            label="My custom label",
        )
        spec = config.to_transfer_spec(
            analysis_dir=Path("/lustre/orion/test"),
            analysis_id="test_run",
        )
        assert spec.label == "My custom label"

    def test_destination_path_strips_trailing_slash(self):
        config = PostRunTransferConfig(
            destination_root="/mnt/d/results/",
            system="frontier",
        )
        spec = config.to_transfer_spec(
            analysis_dir=Path("/lustre/test"),
            analysis_id="run1",
        )
        assert spec.items[0].destination_path == "/D/results/run1"
