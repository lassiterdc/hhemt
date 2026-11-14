"""Stale-log-instance tests for the field kinds `tests/test_log_write_stale_instance.py` does not reach.

Problem this file guards: `TRITONSWMM_log.write()` must leave THIS instance holding the values
it just published, with every field object bound to itself, and `_stamp_triton_provenance`
must warn for every recorded TRITON build that lost its sha. The existing stale-instance tests
build every stale instance with `from_json`, in which every Optional model-log field is already
a `LogField(None)` and every plain field already holds a value of its own kind. Four arms of
`TRITONSWMM_log._absorb` and two of the stamp's predicate are therefore executed by no test
there; each test below names the one-line change to production code that it fails under and
the existing tests pass.

Conditions that hold for every test below:
- Reachability: a default-constructed log is what `TRITONSWMM_system.__init__` builds when
  `system_log.json` is absent (the `except FileNotFoundError` arm in `system.py`) and what
  `scenario.get_log` builds when the model log is absent; a peer can create the file in the
  window before the first write, so a stale instance whose memory is the model default is a
  reachable input. `recovered_from_unparseable` is written by `write()`'s own
  quarantine-and-reconstruct arm.
- Both-states anchor: every assertion reads the document on DISK, the memory of the stale
  instance, a `write_json` call count, or captured stdout. Each exists identically before and
  after the repair; none asserts on wording the repair introduces.
- Invocation isolation: every path is under `tmp_path`; no environment variable, working
  directory or interpreter flag participates.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import xarray as xr

import hhemt.log as L
from hhemt.log import TRITONSWMM_model_log, TRITONSWMM_system_log
from hhemt.processing_analysis import _stamp_triton_provenance


def _disk(path: Path) -> dict:
    """Return the JSON document currently on disk at `path`."""
    return json.loads(path.read_text())


def _model_log_pair(tmp_path: Path):
    """Return `(path, stale, peer)`: `stale` is DEFAULT-constructed, `peer` is loaded."""
    p = tmp_path / "log_tritonswmm.json"
    TRITONSWMM_model_log(logfile=p).write()
    stale = TRITONSWMM_model_log(logfile=p)
    peer = TRITONSWMM_model_log.from_json(p)
    return p, stale, peer


def test_default_instance_takes_an_optional_field_a_peer_set(tmp_path):
    """Optional model-log field that is None in stale memory: the peer's value lands in both places.

    Property: with `stale` default-constructed (its `TRITON_summary_written` is None, not a
    LogField) and a peer setting it True, two writes by `stale` leave True on disk AND leave
    `stale.TRITON_summary_written.get()` True. Class: REGRESSION (red on the pre-change tree:
    the second write publishes None over the peer's True). Why required: `scenario.get_log`
    builds this default instance whenever the model log is absent, and a field that is None in
    memory is the one input kind where the absorb must install a field object rather than copy
    a value. Killed by: an absorb that skips every key whose memory is None; the disk
    assertion still passes under it (the baseline is memory), so the memory assertion is what
    fails. Second correct implementation: registering every Optional field as an empty
    LogField at construction and then copying into it passes both assertions.
    """
    p, stale, peer = _model_log_pair(tmp_path)
    peer.TRITON_summary_written.set(True)
    stale.simulation_completed.set(True)
    stale.sim_run_time_minutes.set(3.0)
    assert _disk(p)["TRITON_summary_written"] is True
    field = stale.TRITON_summary_written
    assert field is not None
    assert field.get() is True


def test_field_taken_into_a_default_instance_writes_through_that_instance(tmp_path, monkeypatch):
    """A field object the absorb installs where memory held None writes through THIS instance.

    Property: after `stale` takes the peer's `TRITON_summary_written`, setting it inside
    `stale.deferred_writes()` together with another field is ONE publish. Class: REGRESSION
    (pre-change the field is None, so the reference is unusable at all). Why required: a field
    object must be bound to the instance whose baseline `write()` consults; one left bound to
    the throwaway instance the absorb validated into publishes at once, outside this
    instance's batching and baseline. Killed by: an absorb that installs the validated field
    without calling its `set_log(self)`; the existing tests execute no such install. Second
    correct implementation: copying the peer's value into a new `LogField` that this instance
    constructs and binds passes, because the assertion counts publishes and names no binding
    call.
    """
    p, stale, peer = _model_log_pair(tmp_path)
    peer.TRITON_summary_written.set(True)
    stale.simulation_completed.set(True)
    stale.sim_run_time_minutes.set(3.0)
    field = stale.TRITON_summary_written
    assert field is not None
    calls = {"n": 0}
    real = L.write_json

    def counting(obj, path, *a, **k):
        """Count one publish, then delegate to the real `write_json`."""
        calls["n"] += 1
        return real(obj, path, *a, **k)

    monkeypatch.setattr(L, "write_json", counting)
    with stale.deferred_writes():
        field.set(False)
        stale.simulation_completed.set(False)
    assert calls["n"] == 1
    assert _disk(p)["TRITON_summary_written"] is False


def test_plain_field_a_peer_published_survives_and_is_read_back(tmp_path):
    """A plain (non-LogField) field a peer published survives two writes and is read back.

    Property: a peer's quarantine-and-reconstruct write publishes `recovered_from_unparseable`;
    two later writes by a stale instance (memory None for it) leave that record on disk and
    leave `stale.recovered_from_unparseable` equal to it. Class: REGRESSION (pre-change the
    second write publishes None over the record, which `validate_analysis` reads from disk).
    Why required: a field this instance did not change must read the latest published value
    for every field kind, and no existing write-path test puts a plain field in the overlay
    (only `refresh()` reaches that arm, through `last_writer`). Killed by: a `write()` that
    skips a key whose memory is None; the disk assertion still passes under it because the
    baseline is memory, so the memory assertion is the one that fails. Second correct
    implementation: calling `self.refresh()` at the end of the write passes, because both
    assertions read state that refresh also produces.
    """
    p = tmp_path / "system_log.json"
    TRITONSWMM_system_log(logfile=p).write()
    stale = TRITONSWMM_system_log.from_json(p)
    peer = TRITONSWMM_system_log.from_json(p)
    p.write_text("{this is not json")
    peer.dem_processed.set(True)
    record = _disk(p)["recovered_from_unparseable"]
    assert record
    stale.mannings_processed.set(True)
    stale.dem_shape.set((1, 2))
    assert _disk(p)["recovered_from_unparseable"] == record
    assert stale.recovered_from_unparseable == record


def test_default_instance_reads_a_peers_tuple_with_the_typed_value(tmp_path):
    """A default-constructed instance reads a peer's typed field with the field's own type.

    Property: `stale.dem_shape.get()` equals the tuple `(10, 20)` after two writes by a
    DEFAULT-constructed system log (its LogField has no expected type) while a peer set it.
    Class: REGRESSION (pre-change memory is None). Why required: `TRITONSWMM_system.__init__`
    builds this untyped default instance, and the same field loaded with `from_json` returns a
    tuple, so a reader must not see a list because of which constructor ran. Killed by: an
    absorb that copies `.value` without adopting the validated field's expected type; the
    existing tests execute no such adoption. Second correct implementation: installing the
    validated, already-typed field object bound to this instance passes. Not established:
    that a raw-copy implementation is incorrect rather than merely different; this test pins
    the behaviour of the landed absorb, which adopts the type.
    """
    p = tmp_path / "system_log.json"
    TRITONSWMM_system_log(logfile=p).write()
    stale = TRITONSWMM_system_log(logfile=p)
    TRITONSWMM_system_log.from_json(p).dem_shape.set((10, 20))
    stale.dem_processed.set(True)
    stale.mannings_processed.set(True)
    assert stale.dem_shape.get() == (10, 20)


def _stamp(sys_log) -> xr.DataTree:
    """Stamp an empty tree from `sys_log` the way both consolidation sites do."""
    analysis = SimpleNamespace(_system=SimpleNamespace(log=sys_log))
    tree = xr.DataTree.from_dict({"/": xr.Dataset()})
    _stamp_triton_provenance(tree, analysis)
    return tree


_BUILD_FLAGS = [
    "compilation_tritonswmm_cpu_successful",
    "compilation_tritonswmm_gpu_successful",
    "compilation_triton_cpu_successful",
    "compilation_triton_gpu_successful",
]


@pytest.mark.parametrize("flag", _BUILD_FLAGS)
def test_each_recorded_triton_build_with_no_sha_prints_the_warning(tmp_path, capsys, flag):
    """Every one of the four build flags, alone, makes the missing sha loud.

    Property: with exactly one of the four compile flags True, no `triton_head_sha`, and no
    `sif_sha256`, stdout carries `[Provenance] WARNING` naming that flag. Class: NEW CAPABILITY
    (the plausible wrong implementation lists fewer than four flags and is red for the flag it
    omits). Why required: each flag names a distinct build target whose experiment would
    otherwise lose its stamp in silence, and `tests/test_provenance_stamp_loud_absent.py`
    sets only one of them. Killed by: removing any one flag from the tuple the stamp scans.
    Second correct implementation: iterating `type(log).model_fields` for names that start
    with `compilation_` and end with `_successful` passes without naming the four.
    """
    sys_log = TRITONSWMM_system_log(logfile=tmp_path / "system_log.json")
    getattr(sys_log, flag).set(True)
    tree = _stamp(sys_log)
    out = capsys.readouterr().out
    assert "[Provenance] WARNING" in out
    assert flag in out
    assert "triton_producing_sha" not in tree.attrs


def test_failed_builds_with_no_sha_stay_silent(tmp_path, capsys):
    """Compile flags recorded as False (an attempted build that failed) do not warn.

    Property: all four flags False, no sha and no digest prints nothing. Class: CONTROL (green
    on the pre-change tree; it varies a flag's VALUE between None and False, the distinction
    the predicate rests on). Why required: the warning means a recorded SUCCESSFUL build lost
    its stamp; a failed compile has no sha to lose and must not raise a false alarm. Killed by:
    testing the flag for `is not None` instead of `is True`; the other stamp tests only use
    None. Second correct implementation: `bool(flag.get())` passes.
    """
    sys_log = TRITONSWMM_system_log(logfile=tmp_path / "system_log.json")
    for flag in _BUILD_FLAGS:
        getattr(sys_log, flag).set(False)
    tree = _stamp(sys_log)
    assert capsys.readouterr().out == ""
    assert "triton_producing_sha" not in tree.attrs
