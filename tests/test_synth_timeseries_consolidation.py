"""Coverage for the opt-in per-scenario timeseries consolidation and the durable ``replay_t`` resume-boundary capture.

Three kinds of test:
  * Unit: ``_parse_replay_t`` extracts the LAST replay marker's ``t=`` value from a model-log string, with no
    pipeline run. A replay marker is the line TRITON prints (``SWMM exchange history replayed to t=``) when a resumed
    coupled run replays SWMM's exchange history; its ``t`` is the simulation time, in seconds, at which the resumed
    run was positioned. One model log can hold several markers, and the LAST is the final resume boundary.
  * Config only: ``test_default_is_off`` reads the toggle from ``synth_multi_sim_builder``, a configured analysis
    that was not run (its docstring in ``tests/conftest.py`` says "no compile, no scenario prep").
  * Golden: the two consolidation tests extract the committed multi-sim golden (``tests/fixtures/golden_trees.py``),
    which already carries the per-scenario SWMM node and link timeseries stores, set ``toggle_consolidate_timeseries``
    in the PERSISTED analysis config, and call ``consolidate_to_datatree`` in-process. No solver is compiled or run
    and no Snakemake process starts.

No golden-reprocess prerequisites are copied. The rows need no Snakefile, no ``python`` on ``PATH`` and no child
interpreter identity shim, because nothing is reprocessed through the workflow. The one environment declaration used
is explained at ``_declare_stale_build``.
"""

from __future__ import annotations

import xarray as xr

from hhemt.analysis_validation import _TRITON_REPLAY_MARKER
from hhemt.processing_analysis import _parse_replay_t
from tests.fixtures.golden_trees import construct_golden_analysis


# ---------------------------------------------------------------------------
# Unit: replay-marker parse
# ---------------------------------------------------------------------------
def test_parse_replay_t_extracts_float():
    line = f"[00:12:34] {_TRITON_REPLAY_MARKER}123.5\n"
    assert _parse_replay_t(line, _TRITON_REPLAY_MARKER) == 123.5


def test_parse_replay_t_returns_none_when_marker_absent():
    assert _parse_replay_t("nothing to see here\n", _TRITON_REPLAY_MARKER) is None


def test_parse_replay_t_keeps_last_marker():
    # The model log is "w"-truncated per exec (one launch of the solver through ``run_simulation_runner``), so it
    # holds only the latest exec's markers; within one exec multiple markers can appear, and the durable value is
    # the LAST one (the final resume boundary).
    text = f"{_TRITON_REPLAY_MARKER}100.0\n... more log ...\n{_TRITON_REPLAY_MARKER}3000.0\n"
    assert _parse_replay_t(text, _TRITON_REPLAY_MARKER) == 3000.0


def test_parse_replay_t_none_on_unparseable_value():
    assert _parse_replay_t(f"{_TRITON_REPLAY_MARKER}not_a_number\n", _TRITON_REPLAY_MARKER) is None


# ---------------------------------------------------------------------------
# Opt-in timeseries consolidation: the toggle default (config only), then the golden rows
# ---------------------------------------------------------------------------
_TS_NODES = ("swmm_node_timeseries", "swmm_link_timeseries")


def test_default_is_off(synth_multi_sim_builder):
    """The toggle must default False so a consolidation that does not opt in stays byte-compatible with trees built
    before the option existed.

    This is asserted on the CONFIG rather than by a second consolidation: the workflow's consolidate step runs as a
    separate process that loads the persisted ``analysis_config.yaml``, so an in-memory flip to False in this process
    would not reach it and such a test would pass for the wrong reason. Established here: the default, and (by the
    toggle-OFF pass of ``test_toggle_change_invalidates_an_already_consolidated_tree``) that ``swmm_node_timeseries``
    is absent after a consolidation with the toggle false. Not established: byte-compatibility with older trees, which
    no test here compares, and the absence of the link timeseries node under the toggle OFF, which that pass does not
    check.
    """
    assert synth_multi_sim_builder.cfg_analysis.toggle_consolidate_timeseries is False


def _golden_with_timeseries_toggle(golden_extractor, *, on: bool):
    """Extract the multi-sim golden and construct its analysis with the toggle written into the persisted config.

    The line is appended, which adds the key exactly once only because the multi-sim golden's ``analysis_config.yaml``
    does not already set ``toggle_consolidate_timeseries`` and ends with a newline (both read from the committed
    ``multisim.zip``).
    """
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(cfg.read_text() + f"toggle_consolidate_timeseries: {'true' if on else 'false'}\n")
    return construct_golden_analysis(system_dir)


def _declare_stale_build(monkeypatch):
    """Make the consolidation build gate defer to the inputs fingerprint.

    A golden's log carries no consolidation build stamp, and a checkout with uncommitted changes never matches one,
    so the build gate would force a rebuild on every call and hide whether the fingerprint alone decides. The
    documented operator escape (``HHEMT_DECLARE_STALE_BUILD=1``) makes the gate reuse a store exactly as it does on a
    matching build, leaving the fingerprint as the only reason to rebuild.

    Setting it is the operator's declaration that the build-stamp mismatch is known and accepted: the gate prints the
    mismatch and reuses the store instead of rebuilding. ``stale`` in the name therefore refers to the BUILD that wrote
    the store, which is the opposite of the staleness of a TREE that
    ``test_toggle_change_invalidates_an_already_consolidated_tree`` exercises, where a changed inputs fingerprint
    forces a rebuild."""
    monkeypatch.setenv("HHEMT_DECLARE_STALE_BUILD", "1")


def test_toggle_on_consolidates_node_and_link_timeseries(golden_extractor):
    # With the opt-in toggle ON, the consolidated DataTree gains the two SWMM node/link timeseries nodes
    # (concatenated along event_iloc), each keeping a time dim. The golden already carries a consolidated tree and a
    # complete log (tests/goldens/SPEC.md section 3 and section 4.4). Its log carries no consolidation build stamp
    # (section 4.4: ``consolidation_build_stamp: null``) and an absent stamp never counts as matching, so the build
    # gate by itself is enough to force the first consolidation to rebuild the tree, which is what builds the
    # timeseries nodes.
    a = _golden_with_timeseries_toggle(golden_extractor, on=True)
    assert a.cfg_analysis.toggle_consolidate_timeseries is True

    a.process.consolidate_to_datatree()

    tree_path = a.analysis_paths.analysis_datatree_zarr
    assert tree_path is not None and tree_path.exists()
    tsg = xr.open_datatree(tree_path, engine="zarr", consolidated=False)["tritonswmm"]
    for node_name in _TS_NODES:
        assert node_name in tsg
        node = tsg[node_name]
        # The per-scenario SWMM timeseries name their time axis `date_time`, not `time`.
        assert "date_time" in node.dims, f"{node_name} missing date_time dim: {dict(node.dims)}"
        assert "event_iloc" in node.dims, f"{node_name} missing event_iloc dim: {dict(node.dims)}"


def test_toggle_change_invalidates_an_already_consolidated_tree(golden_extractor, monkeypatch):
    """The consolidation-inputs fingerprint rebuilds a STALE tree.

    A tree can be present, complete AND built from different inputs than the ones now on disk: the timeseries toggle
    was once flipped ON in every persisted config, consolidation re-ran, and the output was silently unchanged because
    the reuse guard keyed only on existence and log completion. ``regenerate_existing`` does not help, since it is a
    driver-side deletion flag that never reaches ``consolidate_to_datatree``.

    Shape: consolidate with the toggle OFF, rewrite the persisted config with the toggle ON on the same tree without
    wiping, build a fresh analysis object, consolidate again, and assert the timeseries nodes appear and the tree was
    rewritten. The guard lives in ``consolidate_to_datatree``, so calling it directly is sufficient.
    """
    _declare_stale_build(monkeypatch)
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    original = cfg.read_text()

    cfg.write_text(original + "toggle_consolidate_timeseries: false\n")
    a_off = construct_golden_analysis(system_dir)
    assert a_off.cfg_analysis.toggle_consolidate_timeseries is False
    a_off.process.consolidate_to_datatree()
    tree_path = a_off.analysis_paths.analysis_datatree_zarr
    assert tree_path is not None and tree_path.exists()
    tsg_off = xr.open_datatree(tree_path, engine="zarr", consolidated=False)["tritonswmm"]
    assert "swmm_node_timeseries" not in tsg_off, "toggle OFF should not consolidate timeseries"
    mtime_after_pass1 = tree_path.stat().st_mtime

    cfg.write_text(original + "toggle_consolidate_timeseries: true\n")
    a_on = construct_golden_analysis(system_dir)
    assert a_on.cfg_analysis.toggle_consolidate_timeseries is True
    assert tree_path.exists(), "the first tree must survive; otherwise this tests a fresh build, not staleness"

    a_on.process.consolidate_to_datatree()

    tsg_on = xr.open_datatree(tree_path, engine="zarr", consolidated=False)["tritonswmm"]
    for node_name in _TS_NODES:
        assert node_name in tsg_on, (
            f"{node_name} absent after a toggle change on an already-consolidated tree: "
            "the consolidation-inputs fingerprint did not invalidate it"
        )
    assert tree_path.stat().st_mtime > mtime_after_pass1, "tree was not rewritten"
