"""Tests for Globus transfer manager.

Tests GlobusTransferError usage, exclude_dirs parameter, and
destination conflict handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
from TRITON_SWMM_toolkit.config.globus import (
    GlobusEndpoints,
    GlobusTransferItem,
    GlobusTransferSpec,
)
from TRITON_SWMM_toolkit.exceptions import ConfigurationError, GlobusTransferError


class TestGlobusTransferError:
    """Tests for GlobusTransferError exception."""

    def test_attributes(self):
        err = GlobusTransferError(task_id="abc-123", status="FAILED")
        assert err.task_id == "abc-123"
        assert err.status == "FAILED"
        assert "abc-123" in str(err)
        assert "FAILED" in str(err)
        assert "app.globus.org/activity/abc-123" in err.detail_url

    def test_with_message(self):
        err = GlobusTransferError(task_id="xyz", status="CANCELLED", message="User cancelled")
        assert "User cancelled" in str(err)

    def test_inherits_from_base(self):
        from TRITON_SWMM_toolkit.exceptions import TRITONSWMMError

        err = GlobusTransferError(task_id="t", status="FAILED")
        assert isinstance(err, TRITONSWMMError)


class TestHandleDestinationConflict:
    """Tests for _handle_destination_conflict() static method."""

    def test_archive_policy(self, tmp_path):
        dest = tmp_path / "my_analysis"
        dest.mkdir()
        (dest / "file.txt").write_text("data")

        TRITONSWMM_analysis._handle_destination_conflict(dest, "archive")

        assert not dest.exists()
        archived_dir = tmp_path / "archived"
        assert archived_dir.exists()
        archived_items = list(archived_dir.iterdir())
        assert len(archived_items) == 1
        assert archived_items[0].name.startswith("my_analysis_")

    def test_clear_policy(self, tmp_path):
        dest = tmp_path / "my_analysis"
        dest.mkdir()
        (dest / "file.txt").write_text("data")

        TRITONSWMM_analysis._handle_destination_conflict(dest, "clear")

        assert not dest.exists()

    def test_prompt_non_interactive_raises(self, tmp_path):
        dest = tmp_path / "my_analysis"
        dest.mkdir()

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(ConfigurationError, match="interactive terminal"):
                TRITONSWMM_analysis._handle_destination_conflict(dest, "prompt")

    def test_prompt_archive_choice(self, tmp_path):
        dest = tmp_path / "my_analysis"
        dest.mkdir()
        (dest / "file.txt").write_text("data")

        with patch("sys.stdin") as mock_stdin, patch("builtins.input", return_value="a"):
            mock_stdin.isatty.return_value = True
            TRITONSWMM_analysis._handle_destination_conflict(dest, "prompt")

        assert not dest.exists()
        assert (tmp_path / "archived").exists()

    def test_prompt_skip_choice(self, tmp_path):
        dest = tmp_path / "my_analysis"
        dest.mkdir()
        (dest / "file.txt").write_text("data")

        with patch("sys.stdin") as mock_stdin, patch("builtins.input", return_value="s"):
            mock_stdin.isatty.return_value = True
            TRITONSWMM_analysis._handle_destination_conflict(dest, "prompt")

        # Skip means the directory is left in place
        assert dest.exists()


class TestExcludeDirsParameter:
    """Tests for exclude_dirs parameter on GlobusTransferManager.transfer()."""

    def test_exclude_dirs_adds_filter_rules(self):
        """Verify exclude_dirs adds directory-type filter rules to TransferData."""
        spec = GlobusTransferSpec(
            label="test",
            endpoints=GlobusEndpoints(source_uuid="src", destination_uuid="dst"),
            items=[GlobusTransferItem(source_path="/src/path", destination_path="/dst/path")],
        )

        with patch("TRITON_SWMM_toolkit.globus_transfer.globus_sdk") as mock_sdk:
            mock_tdata = MagicMock()
            mock_sdk.TransferData.return_value = mock_tdata
            mock_response = {"task_id": "test-task-id"}
            mock_client = MagicMock()
            mock_client.submit_transfer.return_value = mock_response

            from TRITON_SWMM_toolkit.globus_transfer import GlobusTransferManager

            manager = GlobusTransferManager.__new__(GlobusTransferManager)
            manager.transfer_client = mock_client

            manager.transfer(spec, exclude_dirs=["subanalyses/", "sims/out_triton/"])

            # Verify filter rules were added as directory type
            calls = mock_tdata.add_filter_rule.call_args_list
            assert len(calls) == 2
            assert calls[0].args == ("subanalyses/",)
            assert calls[0].kwargs == {"method": "exclude", "type": "dir"}
            assert calls[1].args == ("sims/out_triton/",)
