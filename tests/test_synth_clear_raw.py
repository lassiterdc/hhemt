"""Phase 3 regression test: ``_clear_raw_outputs(model_type)`` preserves
top-level files and the ``swmm/`` subdirectory; deletes other subdirs.

Tests the helper invariants directly against a tmp scenario dir so the
test does not depend on a full TRITON-SWMM synth simulation (those are
exercised end-to-end by ``tests/test_synth_01_singlesim.py`` and
``tests/test_synth_02_multisim.py`` under the Phase-3 parameter-retirement
surface).

Per cleanup-rerun-delete-redesign Phase 3 + the user-corrected helper
semantics (the coupled-SWMM ``hydraulics.rpt`` lives at
``out_tritonswmm/swmm/hydraulics.rpt`` — a subdirectory file — so the
``swmm/`` subdir is preserved by the helper to keep the .rpt alive).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from TRITON_SWMM_toolkit.process_simulation import (
    TRITONSWMM_sim_post_processing,
    _CLEAR_RAW_DELETE_SUBDIRS,
)


def _seed_tritonswmm_out_dir(out_dir: Path) -> None:
    """Populate a fake ``out_tritonswmm/`` (or ``out_triton/``) tree.

    Subdirs: H, QX, QY, MH, bin, cfg, performance, swmm (preserve-list).
    Top-level files: performance.txt, log.out.
    The ``swmm/`` subdir contains a ``hydraulics.rpt`` (the file the
    user-corrected semantics protect).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("H", "QX", "QY", "MH", "bin", "cfg", "performance"):
        sub_dir = out_dir / sub
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "marker.bin").write_bytes(b"x")
    (out_dir / "swmm").mkdir(parents=True, exist_ok=True)
    (out_dir / "swmm" / "hydraulics.rpt").write_text("rpt-content\n")
    (out_dir / "performance.txt").write_text("perf-summary\n")
    (out_dir / "log.out").write_text("log-content\n")


def _make_proc(out_dir_attr: str, out_dir_path: Path, swmm_out_file: Path | None = None) -> object:
    """Build a minimal stub satisfying ``_clear_raw_outputs``'s reads."""
    scen_paths = SimpleNamespace(
        out_tritonswmm=None,
        out_triton=None,
        swmm_full_out_file=swmm_out_file,
    )
    setattr(scen_paths, out_dir_attr, out_dir_path)

    # Stub log carries the two LogField-shaped attrs the helper sets. Their
    # values are not asserted by these tests — the goal is just to satisfy
    # the ``getattr(self.log, ...)`` branches without an AttributeError.
    class _LogStub:
        def set(self, _value: bool) -> None:  # pragma: no cover - trivial
            pass

    log = SimpleNamespace(
        raw_TRITON_outputs_cleared=_LogStub(),
        raw_SWMM_outputs_cleared=_LogStub(),
    )

    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.scen_paths = scen_paths
    proc.log = log
    return proc


@pytest.mark.parametrize("out_attr,model_type", [
    ("out_tritonswmm", "tritonswmm"),
    ("out_triton", "triton"),
])
def test_clear_raw_outputs_deletes_subdirs_preserves_files_and_swmm(
    tmp_path: Path, out_attr: str, model_type: str,
):
    """For TRITON/TRITON-SWMM: deletes H/QX/QY/MH/bin/cfg/performance/ but
    preserves top-level files and the ``swmm/`` subdir (per user-corrected
    semantics)."""
    out_dir = tmp_path / out_attr
    _seed_tritonswmm_out_dir(out_dir)

    proc = _make_proc(out_attr, out_dir)
    TRITONSWMM_sim_post_processing._clear_raw_outputs(proc, model_type)  # type: ignore[arg-type]

    for sub in ("H", "QX", "QY", "MH", "bin", "cfg", "performance"):
        assert not (out_dir / sub).exists(), f"{sub}/ should be deleted"

    # Preserved: top-level files
    assert (out_dir / "performance.txt").exists(), "performance.txt must be preserved"
    assert (out_dir / "log.out").exists(), "log.out must be preserved"

    # Preserved: any child NOT in _CLEAR_RAW_DELETE_SUBDIRS — including the
    # swmm/ subdir, which holds hydraulics.rpt under tritonswmm. The seed
    # function creates a swmm/ subdir for every parametrized case, so the
    # assertion below holds for both out_tritonswmm and out_triton seeds
    # even though only the coupled tritonswmm path uses the .rpt downstream.
    assert (out_dir / "swmm").exists(), "swmm/ subdir must be preserved"
    assert (out_dir / "swmm" / "hydraulics.rpt").exists(), (
        "out_tritonswmm/swmm/hydraulics.rpt must survive _clear_raw_outputs"
    )


def test_clear_raw_outputs_swmm_deletes_only_out_file(tmp_path: Path):
    """For standalone SWMM: deletes only the .out file; leaves the .rpt."""
    out_swmm = tmp_path / "out_swmm"
    out_swmm.mkdir(parents=True)
    out_file = out_swmm / "full.out"
    rpt_file = out_swmm / "full.rpt"
    out_file.write_bytes(b"binary-out")
    rpt_file.write_text("rpt-content\n")

    proc = _make_proc("out_tritonswmm", out_swmm, swmm_out_file=out_file)
    TRITONSWMM_sim_post_processing._clear_raw_outputs(proc, "swmm")  # type: ignore[arg-type]

    assert not out_file.exists(), ".out file should be deleted"
    assert rpt_file.exists(), ".rpt file must be preserved"


def test_clear_raw_outputs_preserves_unknown_subdirs(tmp_path: Path):
    """An UNKNOWN subdir (not in the delete-allowlist) survives cleanup.

    Pins the design-recommendation choice of allowlist semantics: a future
    TRITON or coupled-SWMM binary that adds a new output directory family
    under out_*/ has its data preserved by default, not silently deleted.
    The failure mode under future binary-output-set evolution is therefore
    disk pressure (noisy, recoverable), not silent data loss.
    """
    out_dir = tmp_path / "out_tritonswmm"
    _seed_tritonswmm_out_dir(out_dir)
    # Seed an unknown future-output subdir alongside the known families.
    (out_dir / "diagnostics_future").mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics_future" / "marker.bin").write_bytes(b"x")

    proc = _make_proc("out_tritonswmm", out_dir)
    TRITONSWMM_sim_post_processing._clear_raw_outputs(proc, "tritonswmm")  # type: ignore[arg-type]

    # Known delete-list subdirs are gone.
    for sub in ("H", "QX", "QY", "MH", "bin", "cfg", "performance"):
        assert not (out_dir / sub).exists(), f"{sub}/ should be deleted"
    # Unknown subdir survives.
    assert (out_dir / "diagnostics_future").exists(), (
        "unknown subdirs must be preserved (allowlist semantics)"
    )
    assert (out_dir / "diagnostics_future" / "marker.bin").exists()


def test_clear_raw_outputs_handles_missing_dir(tmp_path: Path):
    """Helper no-ops cleanly when the out_dir does not exist."""
    proc = _make_proc("out_triton", tmp_path / "does_not_exist")
    # Should not raise.
    TRITONSWMM_sim_post_processing._clear_raw_outputs(proc, "triton")  # type: ignore[arg-type]


def test_clear_raw_outputs_rejects_unknown_model_type(tmp_path: Path):
    proc = _make_proc("out_triton", tmp_path / "out_triton")
    with pytest.raises(ValueError, match="Unknown model_type"):
        TRITONSWMM_sim_post_processing._clear_raw_outputs(proc, "bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("resolved,model_type,expected", [
    ("none", "tritonswmm", False),
    ("none", "triton", False),
    ("none", "swmm", False),
    ("all", "tritonswmm", True),
    ("all", "triton", True),
    ("all", "swmm", True),
    (["tritonswmm"], "tritonswmm", True),
    (["tritonswmm"], "triton", False),
    (["tritonswmm", "swmm"], "swmm", True),
    (["tritonswmm", "swmm"], "triton", False),
])
def test_should_clear_raw_for_model(resolved, model_type, expected):
    """``_should_clear_raw_for_model`` honors the three ``clear_raw`` shapes."""
    assert (
        TRITONSWMM_sim_post_processing._should_clear_raw_for_model(resolved, model_type)
        is expected
    )
