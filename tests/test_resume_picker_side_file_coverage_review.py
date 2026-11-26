"""Reviewer tests: the side-file-aware resume picker.

Each test states a property of ``exchange_replay_coverage_s`` or of the picker that the coder's own file
does not pin: a header-only V2 file, a negative node count, a node-less layout, a nested side-file, a cfg
whose time cannot be read, a directory where a file is expected, and the documentation sentence.
No solver is compiled or run; cfgs and side-files are written by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hhemt.coupling_exchange import EXCHANGE_LOG_MAGIC_V2
from hhemt.run_simulation import exchange_replay_coverage_s
from tests.test_resume_picker_side_file_coverage import (  # noqa: E402
    _CFG_TIMES,
    _picked,
    _side_file_path,
    _sim,
    _write_side_file,
)

_REPO = Path(__file__).resolve().parent.parent


def test_a_header_only_v2_file_has_zero_coverage_and_covers_no_checkpoint(tmp_path):
    """Property: a V2 header with no whole record is coverage 0.0 (the solver's cum is 0), not None.

    Every checkpoint is later than 1e-6 s, so the picker returns None (a clean start) and not the newest cfg.
    """
    run, out = _sim(tmp_path)
    side = _side_file_path(out)
    side.write_bytes(np.array([EXCHANGE_LOG_MAGIC_V2, 2, 8], dtype="<i4").tobytes())

    assert exchange_replay_coverage_s(side) == 0.0
    assert _picked(run) is None


def test_a_negative_node_count_is_an_unknown_layout(tmp_path):
    """Property: a negative node count is never guessed at; the coverage is None and the newest cfg is returned."""
    run, out = _sim(tmp_path)
    side = _side_file_path(out)
    side.write_bytes(np.array([EXCHANGE_LOG_MAGIC_V2, -1, 8], dtype="<i4").tobytes() + b"\x00" * 64)

    assert exchange_replay_coverage_s(side) is None
    assert _picked(run) == "config_0012.cfg"


@pytest.mark.parametrize("width", [4, 8])
def test_a_layout_with_no_nodes_is_read_one_dt_per_record(tmp_path, width):
    """Property: with zero coupling nodes a record is one value wide, and the sum still runs over every record."""
    path = _write_side_file(tmp_path / "x_exchange_replay.bin", np.full(64, 0.5), width=width, n_nodes=0)

    assert exchange_replay_coverage_s(path) == 32.0


def test_the_coverage_reader_takes_a_string_path(tmp_path):
    """Property: the signature is ``Path | str``."""
    path = _write_side_file(tmp_path / "x_exchange_replay.bin", np.full(8, 0.25))

    assert exchange_replay_coverage_s(str(path)) == 2.0


def test_a_directory_in_place_of_a_file_is_an_unknown_layout(tmp_path):
    """Property: a path that is not a readable file is None, whatever its size."""
    assert exchange_replay_coverage_s(tmp_path) is None
    assert exchange_replay_coverage_s(tmp_path / "absent_exchange_replay.bin") is None


def test_a_side_file_in_a_nested_swmm_directory_is_found(tmp_path):
    """Property: the picker finds the side-file the way the processing code does, ``**/swmm/*_exchange_replay.bin``."""
    run, out = _sim(tmp_path)
    _write_side_file(out / "deeper" / "swmm" / "n_exchange_replay.bin", np.full(100, 1.0))  # 100 s: covers no cfg

    assert _picked(run) is None


def test_a_cfg_whose_time_cannot_be_read_is_accepted(tmp_path):
    """Property: a cfg with no readable ``sim_start_time`` is not provably uncovered, so it is accepted.

    The newest cfg here has an unparseable time; the side-file covers only 120 s, and the newest cfg is still returned.
    """
    run, out = _sim(tmp_path)
    newest = out / "cfg" / "config_0012.cfg"
    newest.write_text(newest.read_text().replace("sim_start_time=360.00000000000000000", "sim_start_time=unreadable"))
    _write_side_file(_side_file_path(out), np.full(4, 30.0))  # 120 s

    assert _picked(run) == "config_0012.cfg"


def test_coverage_exactly_at_a_checkpoint_covers_it_and_nothing_newer(tmp_path):
    """Property: coverage 240 s covers the 240 s checkpoint (``t <= coverage``) and not the 360 s one."""
    run, out = _sim(tmp_path)
    _write_side_file(_side_file_path(out), np.full(240, 1.0))

    assert exchange_replay_coverage_s(_side_file_path(out)) == 240.0
    assert _picked(run) == "config_0008.cfg"


def test_coverage_past_every_checkpoint_returns_the_newest(tmp_path):
    """Property: when every checkpoint is covered the newest complete cfg is returned."""
    run, out = _sim(tmp_path)
    _write_side_file(_side_file_path(out), np.full(400, 1.0))

    assert _picked(run) == "config_0012.cfg"
    assert set(_CFG_TIMES) == {"config_0004.cfg", "config_0008.cfg", "config_0012.cfg"}


@pytest.mark.parametrize(
    ("page", "sentence"),
    [
        ("docs/explanation/limitations.md", "newest checkpoint that the exchange-replay side-file covers"),
        ("docs/how-to/diagnosing-a-failed-run.md", "newest checkpoint that the exchange-replay side-file covers"),
    ],
)
def test_the_docs_state_the_covered_checkpoint_rule(page, sentence):
    """Property: both pages state the rule, and neither keeps the old "most recent checkpoint" claim (AC-1.7)."""
    text = " ".join((_REPO / page).read_text().split())

    assert sentence in text
    assert "resumes from the most recent checkpoint" not in text
    assert "resumes from the latest checkpoint" not in text
