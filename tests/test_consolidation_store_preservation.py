"""Does a failed rebuild leave the previously-consolidated store intact?

Pure-predicate throughout. No fixture, no solver, no cached tree -- every store here is
created under ``tmp_path`` by the test that reads it, and the one caller-level test drives
``consolidate_to_datatree`` against a duck-typed stand-in rather than a real analysis.

TWO LEVELS, AND THEY HAVE DIFFERENT POWER. Read this before treating any of it as a
regression guard for a particular change.

* ``TestPublisherCharacterization`` characterizes ``utils._publish_store_crash_safe``,
  the primitive both ``write_datatree_zarr`` call sites already route through. Nothing in
  the repo exercised it before this module -- ``grep -rl "_publish_store_crash_safe"
  tests/`` returned zero files -- so this is a genuine first characterization of an
  untested primitive. It is ALSO green on either side of the delete-before-ensure change,
  because the primitive itself was never modified. **It is not that change's witness and
  must not be cited as one.**
* ``test_a_failed_rebuild_leaves_the_prior_consolidated_store_intact`` IS that witness.
  Measured against the tree with the pre-deletes still in place, it fails: the store is
  gone. With them removed it passes. That difference is the whole reason it exists.
"""

from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest

import hhemt.processing_analysis as pa
from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing
from hhemt.utils import _publish_store_crash_safe

_SUMMARY_ATTRS = (
    "output_tritonswmm_triton_summary",
    "output_tritonswmm_node_summary",
    "output_tritonswmm_link_summary",
    "output_triton_only_summary",
    "output_swmm_only_node_summary",
    "output_swmm_only_link_summary",
    "output_tritonswmm_performance_summary",
    "output_triton_only_performance_summary",
)


def _store(root: Path, content: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "zarr.json").write_text(content, encoding="utf-8")
    return root


class _LogField:
    """The two-method surface consolidate_to_datatree uses from a LogField."""

    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class TestPublisherCharacterization:
    """Characterization of _publish_store_crash_safe. Green either side of unit 2."""

    def test_a_failed_write_leaves_the_prior_store_intact(self, tmp_path):
        final = _store(tmp_path / "s.zarr", "ORIGINAL")

        def _boom(dest):
            d = Path(dest)
            d.mkdir(parents=True, exist_ok=True)
            (d / "zarr.json").write_text("PARTIAL", encoding="utf-8")
            raise RuntimeError("write died mid-stream")

        with pytest.raises(RuntimeError):
            _publish_store_crash_safe(_boom, final)

        assert (final / "zarr.json").read_text(encoding="utf-8") == "ORIGINAL", (
            "the primitive must leave the previous complete store in place when the write raises"
        )

    def test_a_successful_write_publishes_and_leaves_no_temporaries(self, tmp_path):
        """Without this arm the arm above passes against a publisher that never writes."""
        final = _store(tmp_path / "s.zarr", "ORIGINAL")

        def _ok(dest):
            d = Path(dest)
            d.mkdir(parents=True, exist_ok=True)
            (d / "zarr.json").write_text("NEW", encoding="utf-8")

        _publish_store_crash_safe(_ok, final)

        assert (final / "zarr.json").read_text(encoding="utf-8") == "NEW"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["s.zarr"], (
            "a successful publish must leave neither .tmp nor .aside behind"
        )

    def test_a_publish_over_an_absent_destination_still_publishes(self, tmp_path):
        """The fresh path, where step 3's rename is skipped because there is nothing to keep."""
        final = tmp_path / "s.zarr"

        def _ok(dest):
            d = Path(dest)
            d.mkdir(parents=True, exist_ok=True)
            (d / "zarr.json").write_text("FRESH", encoding="utf-8")

        _publish_store_crash_safe(_ok, final)
        assert (final / "zarr.json").read_text(encoding="utf-8") == "FRESH"


def test_a_failed_rebuild_leaves_the_prior_consolidated_store_intact(tmp_path, monkeypatch):
    """THE regression witness for the delete-before-ensure removal.

    Drives the real ``consolidate_to_datatree`` down its stale-inputs rebuild arm with a
    ``_retrieve_combined_output`` that raises -- the shape a member hits when its summaries
    are absent, and the exact exception the sensitivity master's ensure loop swallows under
    ``allow_incomplete=True``. The previously-consolidated store must still be on disk when
    the exception surfaces.

    MEASURED DISCRIMINATION, which is the only reason this test earns its place: run
    against the tree with the pre-delete still in the rebuild arm, it FAILS -- the store is
    gone. The publisher-characterization class above cannot see that difference at all.
    """
    analysis_dir = tmp_path / "analysis"
    store = _store(analysis_dir / "analysis_datatree.zarr", "ORIGINAL")

    log = types.SimpleNamespace(
        datatree_consolidation_complete=_LogField(True),
        # Deliberately NOT the value _consolidation_inputs_fingerprint() recomputes, so the
        # stale-inputs rebuild arm is the branch taken.
        consolidation_inputs_fingerprint=_LogField("stale-fingerprint"),
        consolidation_build_stamp=_LogField(None),
        consolidation_version=_LogField(None),
        add_sim_processing_entry=lambda *a, **k: None,
    )
    analysis = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_datatree_zarr=store, analysis_dir=analysis_dir),
        log=log,
        _refresh_log=lambda: None,
        cfg_analysis=types.SimpleNamespace(analysis_id="witness", toggle_consolidate_timeseries=False),
        _get_enabled_model_types=lambda: ["tritonswmm"],
        df_sims=pd.DataFrame(index=[0]),
    )

    class _Scenario:
        def __init__(self, *args, **kwargs):
            self.scen_paths = types.SimpleNamespace(**{a: Path("/nonexistent") for a in _SUMMARY_ATTRS})

    monkeypatch.setattr(pa, "TRITONSWMM_scenario", _Scenario)
    monkeypatch.setattr(
        TRITONSWMM_analysis_post_processing,
        "_retrieve_combined_output",
        lambda self, mode: (_ for _ in ()).throw(FileNotFoundError("summary absent")),
    )

    with pytest.raises(FileNotFoundError):
        TRITONSWMM_analysis_post_processing(analysis).consolidate_to_datatree()

    assert store.exists(), (
        "a rebuild that raises must leave the previously-consolidated store on disk; "
        "deleting it before the replacement is written destroys it with nothing to restore"
    )
    assert (store / "zarr.json").read_text(encoding="utf-8") == "ORIGINAL"
