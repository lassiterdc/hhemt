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
        # The consolidation fingerprint hashes the scenario-id SET (W1 Finding B), so the
        # gate now reaches the same enumeration processing_analysis uses to build the
        # tree's event_id coordinate. Without this the stand-in raises AttributeError and
        # the test below reports a failure that has nothing to do with what it asserts.
        _retrieve_weather_indexer_using_integer_index=lambda i: {"storm": f"evt{i}"},
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


def test_induce_incomplete_analysis_preserves_the_master_tree_by_default():
    """A DEFAULT-LOCK, and deliberately NOT a behavioural witness.

    This asserts the VALUE and KIND of a keyword-only default, not any effect of them. It
    covers exactly one population -- future callers that omit the flag -- and that
    population is empty at authoring time. The census is CALL-scoped,
    `git grep -nE "induce_incomplete_analysis[(]"`, which returns the helper's own
    definition plus exactly one call, and that call passes `delete_master_tree=True`
    explicitly. Call-scoped rather than name-scoped on purpose: a name-scoped count would
    be changed by this very docstring, whereas the pattern above cannot match itself and
    this test never invokes the helper, so the census is stable across its own landing.

    A behavioural witness for the flip IS constructible without a simulation -- it needs a
    duck-typed sensitivity stub with several contact points onto the helper's internals.
    That instrument is stronger and costlier; this is the single-contact-point alternative,
    chosen on COST and not on impossibility. What is genuinely unavailable is an end-to-end
    check: the helper's only consumer, tests/test_synth_08_sensitivity_reprocess.py, is
    simulation-bearing.

    Drop-witness, measured rather than argued: run before the flip, `.default` is True and
    the second assertion below FAILS.
    """
    import inspect

    from tests.fixtures.test_case_builder import induce_incomplete_analysis

    param = inspect.signature(induce_incomplete_analysis).parameters["delete_master_tree"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        "delete_master_tree must stay keyword-only so a caller cannot pass it positionally "
        "and destroy the master tree without naming the flag"
    )
    assert param.default is False, (
        "induce_incomplete_analysis must preserve the master tree unless a caller asks for "
        "its deletion; a destructive default erases the artifact a narrowing test observes"
    )


@pytest.fixture
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidate reuse conjunction TRUE.

    W1 FINDING B LIVES IN THIS MODULE FOR ONE REASON: the defect is a GATE decision, and
    this module already drives the real `consolidate_to_datatree` against a duck-typed
    stand-in with no fixture, no cached tree and no solver. Nothing below simulates
    anything.

    WHY EVERY ARM PINS A CLEAN BUILD. The reuse gate is a four-term conjunction and
    `store_build_mismatch` is one of the terms. On a DIRTY checkout that term is False on
    its own, the gate rebuilds for a reason that has nothing to do with the scenario set,
    and a reproduction attempt reports the defect ABSENT. This fixture forces that term
    true so the only thing under test is the scenario-set term. `consolidate_to_datatree`
    imports `store_build_mismatch` function-locally, so the patch must land on the DEFINING
    module rather than on `processing_analysis`.

    MEASURED DISCRIMINATION, both directions, against the tree at 57ba4e02: pre-fix,
    addition, removal AND substitution all EARLY-RETURN -- the defect -- and the control
    early-returns too, so the control cannot mask them. Post-fix, addition, removal and
    substitution REBUILD while the control still early-returns.
    """
    import hhemt.provenance as _prov

    monkeypatch.setattr(_prov, "store_build_mismatch", lambda store, stamp: None)


def _gate_analysis(root: Path, ilocs, stamped_fingerprint):
    """A duck-typed analysis whose consolidated store is present and log-complete."""
    store = _store(root / "analysis_datatree.zarr", "ORIGINAL")
    log = types.SimpleNamespace(
        datatree_consolidation_complete=_LogField(True),
        consolidation_inputs_fingerprint=_LogField(stamped_fingerprint),
        consolidation_build_stamp=_LogField("clean"),
        consolidation_version=_LogField(None),
        add_sim_processing_entry=lambda *a, **k: None,
    )
    return types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_datatree_zarr=store, analysis_dir=root),
        log=log,
        _refresh_log=lambda: None,
        cfg_analysis=types.SimpleNamespace(analysis_id="w1", toggle_consolidate_timeseries=False),
        _get_enabled_model_types=lambda: ["tritonswmm"],
        df_sims=pd.DataFrame(index=list(ilocs)),
        _retrieve_weather_indexer_using_integer_index=lambda i: {"storm": f"evt{i}"},
    )


def _fingerprint_for(root: Path, ilocs) -> str:
    return TRITONSWMM_analysis_post_processing(_gate_analysis(root, ilocs, None))._consolidation_inputs_fingerprint()


def _gate_verdict(root: Path, stamped_ilocs, live_ilocs, monkeypatch) -> str:
    """EARLY-RETURN when the gate reuses the store, REBUILD when it does not."""

    class _RebuildEntered(Exception):
        pass

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise _RebuildEntered

    monkeypatch.setattr(pa, "TRITONSWMM_scenario", _Boom)
    analysis = _gate_analysis(root, live_ilocs, _fingerprint_for(root / "_fp", stamped_ilocs))
    try:
        TRITONSWMM_analysis_post_processing(analysis).consolidate_to_datatree()
    except _RebuildEntered:
        return "REBUILD"
    return "EARLY-RETURN"


def test_the_consolidation_fingerprint_is_symmetric_in_the_scenario_set(tmp_path):
    """Addition, removal AND substitution must each move the fingerprint.

    Substitution is the arm that rules out a scenario COUNT as the term: remove one and add
    another and the count is identical, which is the exact defect class this closes.
    """
    base = _fingerprint_for(tmp_path / "base", [0])
    assert _fingerprint_for(tmp_path / "add", [0, 1]) != base, "an ADDED scenario must move the fingerprint"
    assert _fingerprint_for(tmp_path / "sub", [1]) != base, "a SUBSTITUTED scenario must move the fingerprint"
    assert _fingerprint_for(tmp_path / "rm", [0]) == base, (
        "an unchanged set must leave the fingerprint alone, or every run rebuilds"
    )
    two = _fingerprint_for(tmp_path / "two", [0, 1])
    assert _fingerprint_for(tmp_path / "rm2", [0]) != two, "a REMOVED scenario must move the fingerprint"


@pytest.mark.parametrize(
    ("stamped", "live", "expected", "why"),
    [
        ([0], [0], "EARLY-RETURN", "an unchanged set must still reuse the store"),
        ([0], [0, 1], "REBUILD", "an ADDED scenario must defeat the reuse gate"),
        ([0, 1], [0], "REBUILD", "a REMOVED scenario must defeat the reuse gate"),
        ([0], [1], "REBUILD", "a SUBSTITUTED scenario must defeat the reuse gate"),
    ],
)
def test_a_changed_scenario_set_defeats_the_consolidate_reuse_gate(
    tmp_path, monkeypatch, _pin_clean_build, stamped, live, expected, why
):
    """THE witness for Finding B, at the gate rather than at the edit.

    The control arm is load-bearing: without it this passes against a gate that never
    reuses anything, which is a different defect with the same green.
    """
    assert _gate_verdict(tmp_path / f"{stamped}-{live}", stamped, live, monkeypatch) == expected, why


def _borrowed_analysis(tmp_path, ilocs, log):
    """A stand-in carrying the two REAL methods under test, bound to itself."""
    from hhemt.analysis import TRITONSWMM_analysis

    class _A:
        pass

    a = _A()
    a.analysis_paths = types.SimpleNamespace(analysis_dir=tmp_path)
    a.df_sims = pd.DataFrame(index=list(ilocs))
    a._retrieve_weather_indexer_using_integer_index = lambda i: {"storm": f"evt{i}"}
    a.log = log
    for name in (
        "_invalidate_consolidate_flag_on_scenario_set_change",
        "_clear_consolidate_completion_signals",
    ):
        fn = getattr(TRITONSWMM_analysis, name, None)
        assert fn is not None, f"TRITONSWMM_analysis.{name} is absent -- the pairing has no single definition"
        setattr(a, name, types.MethodType(fn, a))
    return a


@pytest.mark.parametrize(
    ("prepared", "live", "kind"),
    [(["storm.evt0"], [0, 1], "addition"), (["storm.evt0", "storm.evt1"], [0], "removal")],
)
def test_the_scenario_set_change_invalidator_clears_BOTH_completion_signals(tmp_path, prepared, live, kind):
    """The DEFECT-site clause, and it is deliberately not drawn from the payload.

    A post-condition that greps for the new helper's NAME can only ask whether the payload
    landed; it cannot ask whether the payload was aimed at the right method. This asserts a
    property of `_invalidate_consolidate_flag_on_scenario_set_change` itself -- the method
    the diagnosis named -- so a fix installed anywhere else fails it.
    """
    status_dir = tmp_path / "_status"
    status_dir.mkdir()
    (status_dir / "e_consolidate_complete.flag").write_text("x", encoding="utf-8")
    (status_dir / "e_consolidate_complete.flag.json").write_text("{}", encoding="utf-8")
    for ev in prepared:
        (status_dir / f"b_prepare_evt-{ev}_complete.flag").write_text("x", encoding="utf-8")

    log = types.SimpleNamespace(datatree_consolidation_complete=_LogField(True))
    analysis = _borrowed_analysis(tmp_path, live, log)
    analysis._invalidate_consolidate_flag_on_scenario_set_change()

    assert not (status_dir / "e_consolidate_complete.flag").exists(), f"the flag must be cleared on {kind}"
    assert log.datatree_consolidation_complete.get() is not True, (
        f"the LOG FIELD must be cleared on {kind} too -- a flag cleared without it leaves "
        "consolidate_to_datatree's _log_complete conjunct True and the store is reused"
    )


def test_the_invalidator_is_not_reached_from_reprocess():
    """Round 87's case, and it is why the fingerprint -- not the invalidator -- is the fix.

    The invalidator is called from ONE place, inside `submit_workflow` under
    `pickup_where_leftoff` (which defaults False). `reprocess()` never reaches it, so on
    that route the scenario-set term in the fingerprint is the only defence. If this census
    ever returns more callers, re-read Finding B before widening the invalidator instead.
    """
    import ast
    import inspect

    import hhemt.analysis as _analysis_module

    src = Path(inspect.getfile(_analysis_module)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    needle = "_invalidate_consolidate_flag_on_scenario_set_change("
    callers = sorted(
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        and fn.name != "_invalidate_consolidate_flag_on_scenario_set_change"
        and needle in (ast.get_source_segment(src, fn) or "")
    )
    assert callers == ["submit_workflow"], callers
    assert "reprocess" not in callers
