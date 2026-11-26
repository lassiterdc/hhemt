"""The coupled resume picker skips every checkpoint the exchange-replay side-file does not cover.

THE PROBLEM. A coupled TRITON-SWMM simulation killed at its wall limit can leave a checkpoint
``config_N.cfg`` whose exchange-replay side-file (``*_exchange_replay.bin``) stops short of it. The
picker used to return the newest complete cfg whatever the side-file held, so the solver refused the
resume and the launch was wasted.

THE CHANGE UNDER TEST. ``exchange_replay_coverage_s`` reads the side-file's own header and records and
returns the solver's coverage: the sequential sum of ``dt`` over the whole records, in the file's value
width. The picker returns the newest complete cfg whose ``sim_start_time`` ``t`` satisfies
``t <= coverage + 1e-6`` (the solver's own tolerance), or None when no cfg does. For a layout it does
not know (a V1 magic, a short header, no side-file) and for ``triton`` it returns the newest complete
cfg, as before.

No test here compiles or runs a solver. Cfgs and side-files are written by hand.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from hhemt.coupling_exchange import EXCHANGE_LOG_MAGIC_V1, EXCHANGE_LOG_MAGIC_V2
from hhemt.run_simulation import TRITONSWMM_run, exchange_replay_coverage_s

if TYPE_CHECKING:
    from hhemt.scenario import TRITONSWMM_scenario

#: The hand-written checkpoints: name -> ``sim_start_time`` in seconds.
_CFG_TIMES = {"config_0004.cfg": 120.0, "config_0008.cfg": 240.0, "config_0012.cfg": 360.0}

_EPS = 1e-6


def _cfg_text(step: int, t: float) -> str:
    """One checkpoint cfg. The same line count for every one, so the picker's modal rule keeps all of them."""
    return (
        "\n".join(
            [
                'output_folder="out_tritonswmm"',
                "print_interval=120",
                f"checkpoint_id={step}",
                f"sim_start_time={t:.17f}",
                "sim_duration=172800",
            ]
        )
        + "\n"
    )


def _write_side_file(
    path: Path,
    dts: np.ndarray,
    *,
    width: int = 8,
    magic: int = EXCHANGE_LOG_MAGIC_V2,
    n_nodes: int = 2,
) -> Path:
    """A side-file in the solver's layout: three int32 header fields, then ``dt`` and ``n_nodes`` values per record."""
    dtype = "<f4" if width == 4 else "<f8"
    records = np.zeros((len(dts), 1 + n_nodes), dtype=dtype)
    records[:, 0] = dts
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.array([magic, n_nodes, width], dtype="<i4").tobytes() + records.tobytes())
    return path


def _sim(tmp_path: Path, times: dict[str, float] | None = None, *, side_file: bool = False):
    """A coupled sim directory holding the hand-written cfgs, and a stand-in scenario around it."""
    times = _CFG_TIMES if times is None else times
    sim = tmp_path / "sim"
    out = sim / "out_tritonswmm"
    (out / "cfg").mkdir(parents=True, exist_ok=True)
    (out / "swmm").mkdir(exist_ok=True)
    for name, t in times.items():
        step = int(name.removeprefix("config_").removesuffix(".cfg"))
        (out / "cfg" / name).write_text(_cfg_text(step, t))
    if side_file:
        _write_side_file(out / "swmm" / "hydraulics_exchange_replay.bin", np.full(10, 1.0))
    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(), analysis_paths=SimpleNamespace())
    scen = SimpleNamespace(
        event_iloc=0,
        weather_event_indexers={},
        _analysis=analysis,
        scen_paths=SimpleNamespace(out_tritonswmm=out, out_triton=sim / "out_triton", sim_folder=sim),
    )
    return TRITONSWMM_run(cast("TRITONSWMM_scenario", scen)), out


def _side_file_path(out: Path) -> Path:
    return out / "swmm" / "hydraulics_exchange_replay.bin"


def _picked(run: TRITONSWMM_run, model_type="tritonswmm") -> str | None:
    cfg = run._retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation(model_type)
    return None if cfg is None else cfg.name


def _side_file_reaching(out: Path, coverage_s: float) -> None:
    """A width-8 side-file whose records sum to ``coverage_s`` (up to float rounding), from one dt per record."""
    n = 4
    _write_side_file(_side_file_path(out), np.full(n, coverage_s / n))


# --------------------------------------------------------------------------------------
# (a) the exact sum the solver accumulates, not the exact comparison
# --------------------------------------------------------------------------------------


def test_a_checkpoint_the_solver_accepts_is_not_set_aside_for_a_rounding_error(tmp_path):
    """Property: 200 records of dt 1.2 sum to 239.99999999999915 in float64, and a checkpoint at 240 s is covered.

    The solver accepts it because its rule is ``t <= cum + 1e-6``. An exact comparison against the sum would
    set the 240 s checkpoint aside and resume from the 120 s one.
    """
    run, out = _sim(tmp_path)
    _write_side_file(_side_file_path(out), np.full(200, 1.2), width=8)

    coverage = exchange_replay_coverage_s(_side_file_path(out))

    assert coverage == pytest.approx(239.99999999999915, abs=1e-12)
    assert coverage < 240.0  # the case this test exists for: the sum is below the checkpoint time
    assert _picked(run) == "config_0008.cfg"


# --------------------------------------------------------------------------------------
# (b) nothing covered
# --------------------------------------------------------------------------------------


def test_when_coverage_reaches_no_checkpoint_the_picker_returns_none(tmp_path):
    """Property: a side-file ending before the first checkpoint (120 s) leaves nothing to resume from."""
    run, out = _sim(tmp_path)
    _side_file_reaching(out, 100.0)

    assert _picked(run) is None


# --------------------------------------------------------------------------------------
# (c) a layout the toolkit does not know: today's behaviour
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["v1_magic", "short_header", "no_side_file", "unknown_width", "empty_file"])
def test_an_unknown_layout_picks_the_newest_complete_cfg_as_before(tmp_path, layout):
    """Property: never guess at a layout the reader does not know; the newest complete cfg is returned.

    The records would cover only the 120 s checkpoint if they were read, so a guessed read would change the answer.
    """
    run, out = _sim(tmp_path)
    side = _side_file_path(out)
    dts = np.full(100, 1.0)  # 100 s: reading it would pick nothing
    if layout == "v1_magic":
        _write_side_file(side, dts, magic=EXCHANGE_LOG_MAGIC_V1)
    elif layout == "short_header":
        side.write_bytes(np.array([EXCHANGE_LOG_MAGIC_V2, 2], dtype="<i4").tobytes())
    elif layout == "unknown_width":
        side.write_bytes(np.array([EXCHANGE_LOG_MAGIC_V2, 2, 2], dtype="<i4").tobytes() + b"\x00" * 64)
    elif layout == "empty_file":
        side.write_bytes(b"")

    assert exchange_replay_coverage_s(side) is None
    assert _picked(run) == "config_0012.cfg"


# --------------------------------------------------------------------------------------
# (d) both value widths
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("width", [4, 8])
def test_each_value_width_is_read_in_its_own_width(tmp_path, width):
    """Property: the width field sets how the records decode, and the sum runs in that width.

    Powers of two sum exactly in either width, so the expected coverage is exact: 300 records of 0.5 s is 150 s.
    """
    run, out = _sim(tmp_path)
    _write_side_file(_side_file_path(out), np.full(300, 0.5), width=width, n_nodes=3)

    assert exchange_replay_coverage_s(_side_file_path(out)) == 150.0
    assert _picked(run) == "config_0004.cfg"


def test_a_four_byte_sum_accumulates_in_four_bytes(tmp_path):
    """Property: float32 accumulation, one record at a time, not a float64 or a pairwise sum.

    The expected figure is built with an explicit float32 loop; it differs from the float64 sum of the same dt.
    """
    dts = np.full(5000, 0.1, dtype=np.float32)
    cum = np.float32(0.0)
    for dt in dts:
        cum = np.float32(cum + dt)
    path = _write_side_file(tmp_path / "x_exchange_replay.bin", dts, width=4)

    coverage = exchange_replay_coverage_s(path)

    assert coverage == float(cum)
    assert coverage != float(np.sum(dts.astype(np.float64)))


def test_a_sum_that_spans_several_blocks_equals_the_single_pass_sum(tmp_path, monkeypatch):
    """Property: the block size is a memory bound and does not move the result.

    The running sum is carried across the block seam in the file's width, so a block of 7 records and one block agree.
    """
    import hhemt.run_simulation as rs

    dts = np.random.default_rng(3).uniform(0.01, 2.0, size=1000).astype(np.float32)
    path = _write_side_file(tmp_path / "x_exchange_replay.bin", dts, width=4, n_nodes=1)
    whole = exchange_replay_coverage_s(path)

    monkeypatch.setattr(rs, "_EXCHANGE_REPLAY_COVERAGE_BLOCK_BYTES", 7 * 8)  # 7 records of (dt + 1 value) x 4 bytes
    blocked = exchange_replay_coverage_s(path)

    assert blocked == whole


def test_a_trailing_partial_record_is_not_counted(tmp_path):
    """Property: only whole records count, as the solver reads whole records."""
    path = _write_side_file(tmp_path / "x_exchange_replay.bin", np.full(10, 2.0), n_nodes=2)
    path.write_bytes(path.read_bytes() + b"\x00" * 9)  # a killed write: less than one record of 24 bytes

    assert exchange_replay_coverage_s(path) == 20.0


# --------------------------------------------------------------------------------------
# (e) TRITON alone has no side-file
# --------------------------------------------------------------------------------------


def test_triton_alone_picks_the_newest_complete_cfg(tmp_path):
    """Property: the ``triton`` model has no side-file, so a coupled side-file beside it is never consulted."""
    sim = tmp_path / "sim"
    out_triton = sim / "out_triton"
    (out_triton / "cfg").mkdir(parents=True)
    for name, t in _CFG_TIMES.items():
        (out_triton / "cfg" / name).write_text(_cfg_text(int(name[7:11]), t))
    out_coupled = sim / "out_tritonswmm"
    _side_file_reaching(out_coupled, 100.0)
    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(), analysis_paths=SimpleNamespace())
    scen = SimpleNamespace(
        event_iloc=0,
        weather_event_indexers={},
        _analysis=analysis,
        scen_paths=SimpleNamespace(out_tritonswmm=out_coupled, out_triton=out_triton, sim_folder=sim),
    )
    run = TRITONSWMM_run(cast("TRITONSWMM_scenario", scen))

    assert _picked(run, "triton") == "config_0012.cfg"


def test_two_side_files_are_not_guessed_between(tmp_path):
    """Property: with more than one side-file the coverage is unknown, so the newest complete cfg is returned."""
    run, out = _sim(tmp_path)
    _side_file_reaching(out, 100.0)
    _write_side_file(out / "swmm" / "other_exchange_replay.bin", np.full(4, 25.0))

    assert _picked(run) == "config_0012.cfg"


# --------------------------------------------------------------------------------------
# (f) the property over random coverages and cfg sets
# --------------------------------------------------------------------------------------


def test_the_picked_cfg_is_covered_and_nothing_newer_is(tmp_path):
    """Property: over random coverages and cfg sets the result satisfies ``t <= coverage + 1e-6``, or is None.

    Also that it is the NEWEST such cfg: no complete cfg above it is covered, and None means none is.
    """
    rng = random.Random(20261125)
    for trial in range(60):
        sim_dir = tmp_path / f"trial_{trial}"
        n_cfgs = rng.randint(1, 8)
        steps = sorted(rng.sample(range(1, 40), n_cfgs))
        times = {f"config_{s:04d}.cfg": round(s * rng.choice([60.0, 120.0, 240.0]) + rng.random(), 3) for s in steps}
        # times rise with the step, as the solver writes them
        ordered = sorted(times.values())
        times = dict(zip(times, ordered, strict=True))
        run, out = _sim(sim_dir, times)
        coverage_target = rng.uniform(0.0, max(ordered) * 1.2)
        _side_file_reaching(out, coverage_target)
        coverage = exchange_replay_coverage_s(_side_file_path(out))
        assert coverage is not None

        picked = _picked(run)

        if picked is None:
            assert all(t > coverage + _EPS for t in times.values()), (trial, coverage, times)
        else:
            assert times[picked] <= coverage + _EPS, (trial, coverage, picked, times)
            newer = [t for t in times.values() if t > times[picked]]
            assert all(t > coverage + _EPS for t in newer), (trial, coverage, picked, times)


# --------------------------------------------------------------------------------------
# (g), (h) the boundary the solver's tolerance draws
# --------------------------------------------------------------------------------------


def test_a_coverage_a_nanosecond_short_of_a_checkpoint_still_covers_it(tmp_path):
    """Property: 240 - 1e-9 s is inside the solver's 1e-6 tolerance, so the 240 s checkpoint is accepted."""
    run, out = _sim(tmp_path)
    _write_side_file(_side_file_path(out), np.array([240.0 - 1e-9]), width=8)

    assert _picked(run) == "config_0008.cfg"


def test_a_coverage_a_millisecond_short_of_a_checkpoint_does_not_cover_it(tmp_path):
    """Property: 240 - 1e-3 s is outside the tolerance, so the 240 s checkpoint is refused and the 120 s one chosen."""
    run, out = _sim(tmp_path)
    _write_side_file(_side_file_path(out), np.array([240.0 - 1e-3]), width=8)

    assert _picked(run) == "config_0004.cfg"
