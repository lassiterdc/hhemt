"""Review tests for the repair of the landing's red tests.

Each repair turned a red test green by editing the test or a stub, so each is paired here with the property the edit
could have silently dropped. None compiles or runs a solver; each reads tracked text or builds a tiny in-memory object.
"""

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


# --- the expected document of the peak-flood-depth fixtures names the generator's own wording ---


def test_the_boundary_long_name_the_fixture_test_expects_is_the_one_the_weather_generator_writes():
    """Property: the wording the repaired fixture test expects equals the wording the generator emits and every
    fixture carries it on each `water_level` reference.

    Class: a constant restated in a test. Why: the repair made the expected document apply one rename, and that
    wording was typed into the test; a typo there would be matched by a fixture typed the same way while the generator
    wrote something else. Kills it: the constant, the generator and the fixtures disagreeing. Control: the number of
    `water_level` references read is nonzero.
    """
    import tests.test_review_clean_slate as clean_slate

    generator = (_ROOT / "src" / "hhemt" / "synthetic_model" / "weather.py").read_text(encoding="utf-8")
    assert f'"long_name": "{clean_slate.BOUNDARY_LONG_NAME}"' in generator
    seen = 0
    for path in sorted(clean_slate.FIXTURES.rglob("peak_flood_depth.manifest.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for artist in document["artists"]:
            for channel in artist["channels"]:
                ref = channel["ref"]
                if ref["variable"] == "water_level":
                    seen += 1
                    assert ref["attrs"]["long_name"] == clean_slate.BOUNDARY_LONG_NAME, path
    assert seen, "control: the fixtures read the water_level variable"


# --- the resource summary keeps counting the two kinds of row apart ---


def _summary(monkeypatch, gpu_rows, swmm_rows, n_gpu_rows):
    import hhemt.analysis_validation as av
    import hhemt.consolidate_workflow as cw

    monkeypatch.setattr(cw, "validate_resource_usage", lambda analysis, logger=None, df_status=None: (True, gpu_rows))
    monkeypatch.setattr(cw, "count_examined_gpu_rows", lambda analysis, df_status=None: n_gpu_rows)
    monkeypatch.setattr(av, "swmm_thread_issues", lambda analysis, df_status=None: swmm_rows)
    return av.check_resource_usage(analysis=None)


_GPU_UNMEASURED = {"resource": "GPU binding", "severity": "info", "actual": "not measured: no gpu_bind artifact"}
_SWMM_UNMEASURED = {"resource": "SWMM threads", "severity": "info", "actual": "not measured: no Number of Threads line"}


def test_a_swmm_thread_row_is_disclosed_apart_from_the_gpu_rows(monkeypatch):
    """Property: an unmeasured SWMM-thread row never inflates the GPU "not measured" count and has its own clause.

    Class: the counting the stub repairs in `test_gpu_bind_guard.py` silence (it silences the SWMM arm). Control: the
    same call without the SWMM row has no SWMM clause.
    """
    both = _summary(monkeypatch, [_GPU_UNMEASURED], [_SWMM_UNMEASURED], 1)
    assert "GPU binding: 1 row not measured" in both.summary
    assert "(SWMM threads: 1 row not measured)" in both.summary
    gpu_only = _summary(monkeypatch, [_GPU_UNMEASURED], [], 1)
    assert "SWMM threads" not in gpu_only.summary
    swmm_only = _summary(monkeypatch, [], [_SWMM_UNMEASURED], 0)
    assert "no GPU model runs" in swmm_only.summary and "(SWMM threads: 1 row not measured)" in swmm_only.summary


def test_a_mismatched_swmm_thread_count_fails_the_resource_check(monkeypatch):
    """Property: a SWMM thread-count row without `info` severity fails the check, and an `info` row does not.

    Control: the passing case in the same test.
    """
    mismatch = {"resource": "SWMM threads", "expected": 2, "actual": 1}
    failed = _summary(monkeypatch, [], [mismatch], 0)
    assert failed.passed is False and failed.summary.startswith("Resource mismatches in 1 scenario")
    assert _summary(monkeypatch, [], [_SWMM_UNMEASURED], 0).passed is True


# --- the observation summary's own deferred block keeps the outer block's single index rebuild ---


def test_a_deferred_block_inside_a_held_one_leaves_one_rebuild_and_keeps_the_store_open(monkeypatch, tmp_path):
    """Property: `deferred_consolidation(store)` opened on a store an outer block holds rebuilds nothing and leaves
    the store in the open set, so the outer block's later appends are still allowed and the index is rebuilt once.

    Class: the repair of `_write_observation_summary` nests such a block in the caller's. Control: a single block
    rebuilds once, so a zero for the nested form is not a store that never rebuilds.
    """
    import hhemt.utils as utils

    store = tmp_path / "tree.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group"}), encoding="utf-8")
    rebuilds = []
    monkeypatch.setattr(utils, "consolidate_zarr_store", lambda s: rebuilds.append(Path(s)))

    with utils.deferred_consolidation(store):
        pass
    assert len(rebuilds) == 1, "control: one block rebuilds the index once"

    rebuilds.clear()
    key = store.resolve()
    with utils.deferred_consolidation(store):
        with utils.deferred_consolidation(store):
            assert key in utils._DEFERRED_STORES
        assert key in utils._DEFERRED_STORES, "the inner exit must not release the outer block's hold"
        assert rebuilds == [], "the inner exit must not rebuild"
    assert key not in utils._DEFERRED_STORES
    assert len(rebuilds) == 1


# --- the pin test's old-sha constant still equals the old pin ---


def test_the_split_old_pin_constant_still_names_the_old_pin_in_the_registry():
    """Property: the assembled `_BB13` of the pin review equals the old pin sha as the registry spells it.

    Why: the repair assembles the constant from two pieces so the file does not carry the sha, and the census that
    uses it reads zero hits for a mistyped piece as readily as for a clean tree. Control: the registry files the
    census allows do carry the full sha, so the assertion needs a nonzero count.
    """
    import tests.test_synth_triton_pin_review as pin_review

    assert len(pin_review._BB13) == 40 and set(pin_review._BB13) <= set("0123456789abcdef")
    carriers = [
        rel
        for rel in ("src/hhemt/model_defects.py", "src/hhemt/scenario.py")
        if pin_review._BB13 in (_ROOT / rel).read_text(encoding="utf-8")
    ]
    assert carriers, "the old pin as assembled is carried by none of the registry files"


# --- the synthetic default model differs from the earlier one by its matched outfall alone ---


def test_the_default_synthetic_model_has_the_matched_outfall_and_no_pollutant():
    """Property: the re-taken default digest describes a model with the matched outfall and, with the pollutant knob
    off, no pollutant frame, so the pin still guards the knob.

    Class: a re-taken regression pin. Why: a pin re-taken from the generator's own output also accepts a change that
    came with the knob; this reads the two facts the digest stands for from the frames the emission is built from.
    Control: the knob-on parameters do build a pollutant frame, and the matched outfall row is found.
    """
    from hhemt.synthetic_model._build import SyntheticModelParams
    from hhemt.synthetic_model.swmm_model import MATCHED_OUTFALL, _outfalls_df, _pollutants_df

    outfalls = _outfalls_df(SyntheticModelParams())
    assert list(outfalls.index).count(MATCHED_OUTFALL) == 1, "control: the matched outfall row is found"
    assert outfalls.loc[MATCHED_OUTFALL, "OutfallType"] == "TIMESERIES"
    assert _pollutants_df(SyntheticModelParams()) is None
    assert _pollutants_df(SyntheticModelParams(pollutant_name="TSS")) is not None, "control: the knob emits a frame"


# --- the consolidation test's cleanup restores the shared directory ---


def test_the_cleanup_removes_what_the_consolidation_added_and_nothing_it_found(tmp_path):
    """Property: `_remove_entries_added_since` removes the added directories, files and links and leaves each
    entry that was listed before.

    Class: the cleanup that keeps the version-state tests' directory in the state they expect. Control: the added
    entries exist before the call.
    """
    import tests.test_synth_01_singlesim as singlesim

    (tmp_path / "log.json").write_text("{}", encoding="utf-8")
    (tmp_path / "kept_dir").mkdir()
    before = {entry.name for entry in tmp_path.iterdir()}
    (tmp_path / "analysis_datatree.zarr").mkdir()
    (tmp_path / "analysis_datatree.zarr" / "zarr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ro-crate-metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "link_to_kept").symlink_to(tmp_path / "kept_dir", target_is_directory=True)
    assert {entry.name for entry in tmp_path.iterdir()} - before == {
        "analysis_datatree.zarr",
        "ro-crate-metadata.json",
        "link_to_kept",
    }
    singlesim._remove_entries_added_since(tmp_path, before)
    assert {entry.name for entry in tmp_path.iterdir()} == before
    assert (tmp_path / "kept_dir").is_dir(), "the link was removed, not what it points to"


@pytest.mark.parametrize("name", ["_validate_swmm_threads", "_validate_external_boundaries"])
def test_the_validators_the_per_member_tests_silence_are_called_by_preflight(name):
    """Property: each validator the repaired wiring tests silence is one `preflight_validate` really calls.

    Why: a silenced name that preflight no longer calls would turn the silencing into dead weight that hides a
    removed validator. Control: the source text of `preflight_validate` is nonempty.
    """
    import inspect

    import hhemt.validation as validation

    source = inspect.getsource(validation.preflight_validate)
    assert source
    assert f"{name}(" in source
