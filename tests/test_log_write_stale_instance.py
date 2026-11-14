"""A log instance that is stale against a peer's write must never republish the peer's field.

Problem this file guards (`TRITONSWMM_log.write()` and `refresh()` in `src/hhemt/log.py`):
`write()` publishes only the fields its instance CHANGED relative to a baseline
and overlays every other field from disk. Before the repair the write path re-baselined on the
merged document while memory kept its pre-write values, so a field this instance never touched
read as "changed by me" at its NEXT write and the stale in-memory value was published over a
peer's newer one. `refresh()` carried a second defect: it bound the instance's fields to a
throwaway reloaded instance, so every later `.set()` wrote through that throwaway.

Reachability, both-states anchor and invocation isolation, stated once for every test below:
- Reachability: every test builds two `from_json` instances over ONE temp file (the pattern
  `TRITONSWMM_system.__init__` and every runner's own system-log load produce), lets the peer
  publish, then drives the stale instance through two writes. The population is the four log
  classes that share the single `write()`.
- Both-states anchor: each assertion reads the value on DISK (or the field object's identity),
  a property that exists identically in the pre-repair and the repaired module; none asserts on
  wording or on a warning the repair introduces.
- Invocation isolation: every instance is constructed from a temp path inside the test, and
  the module under test is resolved from `hhemt.log` once per process; no path, environment
  variable or interpreter flag participates in any assertion.

Tests authored here: Class REGRESSION means red on the pre-repair `log.py` (the committed
version before this change); Class CONTROL means green on both versions and named with the
change that reds it. Each docstring names, as a "variant", a one-line change to the repaired
module that the test fails under; a variant that repairs `write()` but keeps the old
`refresh()` is called the write-only variant. The wider arms of `_absorb` that these tests do
not reach (a default-constructed instance whose field is None, a plain non-LogField field, the
typed-value adoption) are covered in `tests/test_review_log_write_stale_instance.py`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hhemt.log as L
from hhemt.log import (
    LogField,
    TRITONSWMM_model_log,
    TRITONSWMM_scenario_log,
    TRITONSWMM_system_log,
)

SHA = "a146fb6ed7c22790fb9a15ff854c446db4519ad2"


def _fresh(cls, tmp_path: Path, name: str):
    """Write a default document for `cls` at `tmp_path/name` and return its path."""
    path = tmp_path / name
    cls(logfile=path).write()
    return path


def _disk(path: Path) -> dict:
    """Return the JSON document currently on disk at `path`."""
    return json.loads(path.read_text())


def test_scalar_survives_two_stale_writes(tmp_path):
    """A `LogField[str]` a peer set survives two writes by a stale instance.

    Property: after a peer publishes `triton_head_sha`, a stale instance writing two other
    fields leaves that sha on disk. Class: REGRESSION (the pre-repair module publishes None
    on the second write). Why required: the consolidation stamp reads this field, and losing
    it leaves the consolidated tree unstamped. Killed by: a variant that restores
    `self._baseline = merged` in `write()`. Second correct implementation: per-field dirty
    tracking (a dirty set written by each setter, everything else overlaid from disk) also
    passes, because the assertion reads disk and not how the writer decided what to publish.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    stale = TRITONSWMM_system_log.from_json(p)
    TRITONSWMM_system_log.from_json(p).triton_head_sha.set(SHA)
    stale.dem_processed.set(True)
    stale.mannings_processed.set(True)
    assert _disk(p)["triton_head_sha"] == SHA


def test_stale_memory_reads_the_peers_value_after_writes(tmp_path):
    """After its writes the stale instance's own memory holds the peer's value.

    Property: `stale.triton_head_sha.get()` equals the peer's value once `stale` has written.
    Class: REGRESSION (the pre-repair memory stays None). Why required: in-process readers of
    the instance (the same object a runner keeps) would otherwise read a superseded value after
    a successful write. Killed by: a variant that drops the `_absorb` call in `write()`.
    Second correct implementation: calling `self.refresh()` at the end of `write()` also leaves
    memory at the peer's value, so the assertion does not pin the absorb mechanism.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    stale = TRITONSWMM_system_log.from_json(p)
    TRITONSWMM_system_log.from_json(p).triton_head_sha.set(SHA)
    stale.dem_processed.set(True)
    stale.mannings_processed.set(True)
    assert stale.triton_head_sha.get() == SHA


def test_control_single_stale_write_preserves_peer_value(tmp_path):
    """A stale instance that writes ONCE leaves the peer's value on disk.

    Property: the first write after a load overlays unchanged fields from disk. Class:
    CONTROL (green on both versions; it varies the number of writes, the property the defect
    needs to be two). Why required: separates the defect from the baseline read-modify-write
    design that already worked. Killed by: a variant whose overlay skips unchanged keys on the
    first write. Second correct implementation: any merge that overlays disk for unchanged keys.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    stale = TRITONSWMM_system_log.from_json(p)
    TRITONSWMM_system_log.from_json(p).triton_head_sha.set(SHA)
    stale.dem_processed.set(True)
    assert _disk(p)["triton_head_sha"] == SHA


def test_control_own_change_wins_over_the_peers(tmp_path):
    """A field the writer itself sets wins over the peer's value for that field.

    Property: `stale.triton_head_sha.set("ffff")` after the peer set the sha leaves "ffff" on
    disk. Class: CONTROL (green on both versions; varies WHICH instance changed the field).
    Why required: the repair must narrow what is republished, never what the writer owns.
    Killed by: a variant whose absorb runs before changed-key detection and erases the
    writer's change. Second correct implementation: field-granular dirty tracking passes too.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    stale = TRITONSWMM_system_log.from_json(p)
    TRITONSWMM_system_log.from_json(p).triton_head_sha.set(SHA)
    stale.dem_processed.set(True)
    stale.triton_head_sha.set("ffff")
    assert _disk(p)["triton_head_sha"] == "ffff"


def test_control_deliberate_reset_to_an_earlier_value_wins(tmp_path):
    """A deliberate set back to a value the writer held earlier still wins.

    Property: the writer publishes False, a peer sets True, the writer writes again (absorbing
    True), then sets False; disk must read False. Class: CONTROL (green on both versions; the
    input is a set-back, differently positioned from the fresh value in the previous test).
    Why required: the alternative repair that re-baselines on the stale memory instead of the
    absorbed one fails exactly here. Killed by: a variant that re-baselines on the stale
    memory. Second correct implementation: dirty-key tracking marks the set and passes without
    any absorb.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    w = TRITONSWMM_system_log.from_json(p)
    w.dem_processed.set(False)
    TRITONSWMM_system_log.from_json(p).dem_processed.set(True)
    w.mannings_processed.set(True)
    w.dem_processed.set(False)
    assert _disk(p)["dem_processed"] is False


def test_logfielddict_survives_stale_writes(tmp_path):
    """A `LogFieldDict` a peer set survives two writes by a stale scenario log.

    Property: the peer's `swmm_rainfall_dat_files` entry is on disk after two stale writes.
    Class: REGRESSION (the pre-repair module publishes `{}`). Why required: the field kind
    differs from a scalar and the absorb copies `.value` into a different class. Killed by: a
    variant that restores the pre-repair write path (`self._baseline = merged`, no absorb),
    measured red by running it. Not established: no single-arm variant of the absorb reds this
    test, because the absorb's final arm installs the validated field and binds it to this
    instance for any kind the earlier arms skip; a variant that restricts the absorb to
    `LogField` was run and this test stays green under it. Second correct implementation:
    dirty-key tracking passes.
    """
    p = _fresh(TRITONSWMM_scenario_log, tmp_path, "scenario_log.json")
    stale = TRITONSWMM_scenario_log.from_json(p)
    TRITONSWMM_scenario_log.from_json(p).swmm_rainfall_dat_files.set({"g1": "/x/a.dat"})
    stale.extbc_loc_created.set(True)
    stale.hyg_locs_created.set(True)
    assert _disk(p)["swmm_rainfall_dat_files"] == {"g1": "/x/a.dat"}


def test_logfielddict_own_update_merges_with_the_absorbed_entry(tmp_path):
    """After the absorb, the writer's own dict update merges with the peer's entry.

    Property: stale sets `{"g2": ...}` after absorbing `{"g1": ...}`; disk carries both keys.
    Class: REGRESSION (the pre-repair module loses g1). Why required: shows the absorb moved
    the peer's entries into memory, which is what narrows the residual where two writers
    change the same dict field. Killed by: a variant whose absorb leaves the dict empty in
    memory. Second correct implementation: a per-key dict merge at publish time would also
    yield both keys.
    """
    p = _fresh(TRITONSWMM_scenario_log, tmp_path, "scenario_log.json")
    stale = TRITONSWMM_scenario_log.from_json(p)
    TRITONSWMM_scenario_log.from_json(p).swmm_rainfall_dat_files.set({"g1": "/x/a.dat"})
    stale.extbc_loc_created.set(True)
    stale.hyg_locs_created.set(True)
    stale.swmm_rainfall_dat_files.set({"g2": "/x/b.dat"})
    assert _disk(p)["swmm_rainfall_dat_files"] == {"g1": "/x/a.dat", "g2": "/x/b.dat"}


def test_optional_model_field_none_in_stale_memory_survives(tmp_path):
    """A peer-set Optional model-log `LogField` on a loaded instance survives two stale writes.

    Property: a peer attaches `TRITON_summary_written=True` as a `LogField`; a stale
    model-log instance loaded with `from_json` (whose memory holds `LogField(None)` for that
    field, not None) writes twice and the True stays on disk. Class: REGRESSION. Why required:
    covers an Optional field that a peer set after the stale instance loaded, through the
    absorb's copy into an existing `LogField`. Not established here: the arm where the
    destination field is itself None (a default-constructed instance) and the line that
    re-binds a taken field to this instance are NOT reached by this test; they are covered in
    `tests/test_review_log_write_stale_instance.py`. Killed by: a variant that restores the
    pre-repair write path (re-baseline on the merged document, no absorb). Second correct
    implementation: eagerly registering every optional field as a LogField at load passes
    without any absorb change.
    """
    p = _fresh(TRITONSWMM_model_log, tmp_path, "log_tritonswmm.json")
    stale = TRITONSWMM_model_log.from_json(p)
    peer = TRITONSWMM_model_log.from_json(p)
    peer.TRITON_summary_written = LogField(None)
    peer.TRITON_summary_written.set_log(peer)
    peer.TRITON_summary_written.set(True)
    stale.simulation_completed.set(True)
    stale.sim_run_time_minutes.set(3.0)
    assert _disk(p).get("TRITON_summary_written") is True


def test_list_field_survives_stale_writes(tmp_path):
    """A `LogField[list]` a peer set survives two writes by a stale model log.

    Property: `resume_reporting_tsteps == [3, 7]` stays on disk. Class: REGRESSION.
    Why required: list values are a third field kind with value equality by content.
    Killed by: a variant that restores the pre-repair write path (`self._baseline = merged`,
    no absorb), measured red by running it. Not established: no single-arm variant of the
    absorb reds this test, because the absorb's final arm installs the validated field for
    any kind the earlier arms skip. Second correct implementation: dirty-key tracking
    passes.
    """
    p = _fresh(TRITONSWMM_model_log, tmp_path, "log_tritonswmm.json")
    stale = TRITONSWMM_model_log.from_json(p)
    TRITONSWMM_model_log.from_json(p).resume_reporting_tsteps.set([3, 7])
    stale.simulation_completed.set(True)
    stale.n_resumes.set(1)
    assert _disk(p)["resume_reporting_tsteps"] == [3, 7]


def test_processing_entries_survive_stale_writes(tmp_path):
    """`Processing` entries a peer added survive two writes by a stale model log.

    Property: the peer's `summary.zarr` entry is under `processing_log.outputs` on disk.
    Class: REGRESSION. Why required: `Processing` is a pydantic model, not a LogField, and
    takes the absorb's `dst.outputs = src.outputs` branch. Killed by: a variant that restores
    the pre-repair write path (`self._baseline = merged`, no absorb), measured red by running
    it. Not established: no single-arm variant of the absorb reds this test, because the
    absorb's final arm installs the validated field for any kind the earlier arms skip; a
    variant that removes the `dst.outputs = src.outputs` branch was run and this test stays
    green under it. Second correct implementation: a whole-object replace bound back with
    `set_log` passes.
    """
    p = _fresh(TRITONSWMM_model_log, tmp_path, "log_tritonswmm.json")
    stale = TRITONSWMM_model_log.from_json(p)
    TRITONSWMM_model_log.from_json(p).add_sim_processing_entry(Path("/x/summary.zarr"), 1.0, 2.0, True)
    stale.simulation_completed.set(True)
    stale.n_resumes.set(1)
    assert "summary.zarr" in _disk(p)["processing_log"]["outputs"]


def test_tuple_field_survives_stale_writes(tmp_path):
    """A `LogField[tuple]` (`dem_shape`) a peer set survives two stale writes.

    Property: `dem_shape` reads `[10, 20]` on disk. Class: REGRESSION. Why required: tuples
    round-trip through JSON as lists, so a type-strict absorb could drop them. Killed by: a
    variant that restores the pre-repair write path (`self._baseline = merged`, no absorb),
    measured red by running it. Not established: no single-arm variant of the absorb reds this
    test, because the absorb's final arm installs the validated field for any kind the earlier
    arms skip; a variant whose absorb compares `type(src) is type(dst)` on the value was run
    and this test stays green under it. Second correct implementation: dirty-key tracking
    passes.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    stale = TRITONSWMM_system_log.from_json(p)
    TRITONSWMM_system_log.from_json(p).dem_shape.set((10, 20))
    stale.dem_processed.set(True)
    stale.mannings_processed.set(True)
    assert _disk(p)["dem_shape"] == [10, 20]


def test_refresh_keeps_deferred_writes_to_one_publish(tmp_path, monkeypatch):
    """After `refresh()`, two sets inside `deferred_writes()` are ONE publish.

    Property: the count of `write_json` calls across the batching context is 1. Class:
    REGRESSION (the pre-repair module counts 2, because the fields write through a throwaway
    instance whose batching is not on). Why required: batching on a refreshed instance is
    otherwise silently off. Killed by: a variant that restores the `setattr(self, key, value)`
    loop in `refresh()`. Second correct implementation: re-registering each field with
    `set_log(self)` after the swap.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    w = TRITONSWMM_system_log.from_json(p)
    w.refresh()
    calls = {"n": 0}
    real = L.write_json

    def counting(obj, path, *a, **k):
        """Count each `write_json` call this instance makes, then delegate to the real writer."""
        calls["n"] += 1
        return real(obj, path, *a, **k)

    monkeypatch.setattr(L, "write_json", counting)
    with w.deferred_writes():
        w.dem_processed.set(True)
        w.mannings_processed.set(True)
    assert calls["n"] == 1


def test_refresh_leaves_fields_bound_to_self(tmp_path):
    """After `refresh()` a field's log reference is the instance itself.

    Property: `w.dem_processed._log is w`. Class: REGRESSION (the pre-repair module binds the
    throwaway). Why required: it is the invariant that makes the instance's baseline the one
    consulted. Killed by: a variant that restores the `setattr` swap. Second correct
    implementation: any refresh that copies values into the existing objects passes.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    w = TRITONSWMM_system_log.from_json(p)
    w.refresh()
    assert w.dem_processed._log is w


def test_absorb_failure_still_keeps_the_peers_value(tmp_path, monkeypatch, caplog):
    """When the published document cannot be validated, the peer's value survives.

    Property: with `model_validate` forced to raise AFTER the instances are loaded, two stale
    writes still leave the peer's sha on disk. Class: NEW CAPABILITY (the plausible wrong
    implementation re-baselines on the merged document when the absorb fails; it is red under
    it). Why required: a version-skewed document must cost freshness, never a peer's value.
    Killed by: a variant that sets `self._baseline = merged` after a failed absorb. Second
    correct implementation: re-baselining on memory in a `finally` block passes the same
    assertion.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    stale = TRITONSWMM_system_log.from_json(p)
    TRITONSWMM_system_log.from_json(p).triton_head_sha.set(SHA)

    def boom(cls, *a, **k):
        """Stand in for `model_validate` and always fail, so the absorb cannot read the document back."""
        raise ValueError("forced")

    monkeypatch.setattr(TRITONSWMM_system_log, "model_validate", classmethod(boom))
    with caplog.at_level(logging.WARNING, logger=L.__name__):
        stale.dem_processed.set(True)
        stale.mannings_processed.set(True)
    assert _disk(p)["triton_head_sha"] == SHA


def test_control_document_logfile_never_redirects_the_instance(tmp_path):
    """A document carrying another `logfile` never redirects where the instance writes.

    Property: after two writes `w.logfile == p`. Class: CONTROL (green on both versions; it
    varies the document's `logfile` value, the one key the absorb must refuse). Why required:
    `logfile` names where THIS instance writes. Killed by: a variant that removes the
    `k == "logfile"` skip in the absorb. Second correct implementation: excluding `logfile`
    from the keys set passed to the absorb.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    doc = _disk(p)
    doc["logfile"] = "/elsewhere/system_log.json"
    p.write_text(json.dumps(doc))
    w = TRITONSWMM_system_log.from_json(p)
    w.dem_processed.set(True)
    w.mannings_processed.set(True)
    assert w.logfile == p


def _peer_stamped_document(p: Path) -> None:
    """Publish a peer-authored `last_writer` and sha straight to disk."""
    doc = _disk(p)
    doc["last_writer"] = {
        "slurm_jobid": None,
        "hostname": "node-B",
        "restart_count": 0,
        "pid": 1,
        "written_at": "t",
    }
    doc["triton_head_sha"] = SHA
    p.write_text(json.dumps(doc))


def test_control_refresh_adopts_the_disk_stamp_without_a_spurious_warning(tmp_path, caplog):
    """A write right after `refresh()` does not warn "written by ANOTHER process".

    Property: no log record containing "ANOTHER" is emitted by the write that follows a
    refresh with no intervening peer. Class: CONTROL (green on the repaired module and the
    pre-repair one; red under a variant whose `refresh()` skips `last_writer`, an earlier
    draft of this repair that was measured red on this test). Why required: the second-writer
    detector would otherwise fire on every refreshed instance. Killed by: that variant.
    Second correct implementation: copying `last_writer` explicitly after the loop in
    `refresh()`.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    w = TRITONSWMM_system_log.from_json(p)
    _peer_stamped_document(p)
    w.refresh()
    with caplog.at_level(logging.WARNING, logger=L.__name__):
        caplog.clear()
        w.dem_processed.set(True)
    assert not [r for r in caplog.records if "ANOTHER" in r.getMessage()]


def test_refresh_reads_the_peers_value(tmp_path):
    """`refresh()` loads the peer's published value into the instance.

    Property: after the refresh, `w.triton_head_sha.get() == SHA`. Class: CONTROL (green on
    both versions). Why required: pairs the previous test so a refresh that adopts nothing
    cannot pass it by emitting no warning. Killed by: a `refresh()` that returns early.
    Second correct implementation: the pre-repair swap loop passes it.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    w = TRITONSWMM_system_log.from_json(p)
    _peer_stamped_document(p)
    w.refresh()
    assert w.triton_head_sha.get() == SHA


def test_refresh_then_write_keeps_the_peers_latest_value(tmp_path):
    """Refresh, set, peer writes X, set, peer writes Y, direct `write()` keeps Y.

    Property: the final sha on disk is the peer's LATEST value. Class: REGRESSION against the
    write-only variant (the pre-repair module yields Y, the write-only variant yields the
    superseded X, the full repair yields Y); this is the one test that distinguishes the
    two-edit repair from the `write()` edit alone. Why required: `refresh()` and `write()`
    must be repaired together. Killed by: a variant that restores the old `refresh()`.
    Second correct implementation: dirty-key tracking that also binds refreshed fields to
    self passes.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    w = TRITONSWMM_system_log.from_json(p)
    w.refresh()
    w.dem_processed.set(True)
    TRITONSWMM_system_log.from_json(p).triton_head_sha.set("X")
    w.mannings_processed.set(True)
    TRITONSWMM_system_log.from_json(p).triton_head_sha.set("Y")
    w.write()
    assert _disk(p)["triton_head_sha"] == "Y"


def test_field_object_identity_survives_writes(tmp_path):
    """A field reference taken before two writes `is` the field after them.

    Property: `stale.dem_processed is ref` after two stale writes that absorb a peer's value.
    Class: NEW CAPABILITY (the plausible wrong implementation replaces the field object
    with the reloaded one; it is red under it). Why required: callers hold field references
    across writes and those must keep writing through this instance. Killed by: a variant
    that does `setattr(self, k, src)` for a LogField. Second correct implementation: copying
    `.value` into the existing object, as the repair does, is the only one that keeps
    identity, and any in-place update passes.
    """
    p = _fresh(TRITONSWMM_system_log, tmp_path, "system_log.json")
    stale = TRITONSWMM_system_log.from_json(p)
    ref = stale.dem_processed
    TRITONSWMM_system_log.from_json(p).triton_head_sha.set(SHA)
    stale.dem_processed.set(True)
    stale.mannings_processed.set(True)
    assert stale.dem_processed is ref
