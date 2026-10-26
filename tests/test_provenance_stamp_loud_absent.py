"""Consolidation must say so when it cannot stamp the TRITON version it was built with.

Problem this file guards (`_stamp_triton_provenance` in `src/hhemt/processing_analysis.py`):
it omitted `triton_producing_sha` silently when the system log carried no sha, so a lost
stamp surfaced two hours later at an experiment-driver gate with no record of what
consolidation had read. It now prints one `[Provenance] WARNING` line when the log records a
TRITON build (native: a compile flag) or an image (container: `sif_sha256`) and no sha, and it
still never raises.

Reachability, both-states anchor and invocation isolation, stated once for every test below:
- Reachability: each test builds a real `TRITONSWMM_system_log` over a temp file (the last
  test a `SimpleNamespace` standing in for a log), wraps it as `analysis._system.log`, and
  calls `_stamp_triton_provenance` on an empty DataTree, the same call both consolidation
  sites make.
- Both-states anchor: each assertion reads captured stdout or the tree's `attrs`, which exist
  identically before and after the change; none asserts on warning wording beyond the
  `[Provenance] WARNING` prefix and the field names the line must carry.
- Invocation isolation: every path is under `tmp_path`; no environment variable or working
  directory participates.

Class labels: NEW CAPABILITY names the plausible wrong implementation the test is red under;
CONTROL is green on the pre-change tree and names the change that reds it. A "variant" is a
one-line change to the repaired function that the named test fails under. The arm that tests
each of the four compile flags separately, and the arm where every flag is recorded False,
are covered in `tests/test_review_log_write_stale_instance.py`."""

from __future__ import annotations

from types import SimpleNamespace

import xarray as xr

from hhemt.log import TRITONSWMM_system_log
from hhemt.processing_analysis import _stamp_triton_provenance

SHA = "cafebabecafebabecafebabecafebabecafebabe"


def _stamp(sys_log) -> xr.DataTree:
    """Stamp an empty tree from `sys_log` the way both consolidation sites do."""
    analysis = SimpleNamespace(_system=SimpleNamespace(log=sys_log))
    tree = xr.DataTree.from_dict({"/": xr.Dataset()})
    _stamp_triton_provenance(tree, analysis)
    return tree


def test_native_build_with_no_sha_prints_the_warning(tmp_path, capsys):
    """Native arm: a True compile flag and no sha prints the warning and stamps nothing.

    Property: stdout carries `[Provenance] WARNING` naming the flag and `triton_head_sha`,
    and the tree has no `triton_producing_sha`. Class: NEW CAPABILITY (the plausible wrong
    implementation keeps the silent omission; it is red under it, pre-change output pasted
    in the build return). Why required: the print is the only record of what consolidation
    read. Killed by: deleting the else-branch. Second correct implementation: a
    `warnings.warn` with the same fields would fail only the stdout capture, so the stdout
    contract is the assertion's one deliberate coupling.
    """
    sys_log = TRITONSWMM_system_log(logfile=tmp_path / "system_log.json")
    sys_log.compilation_tritonswmm_gpu_successful.set(True)
    tree = _stamp(sys_log)
    out = capsys.readouterr().out
    assert "[Provenance] WARNING" in out
    assert "compilation_tritonswmm_gpu_successful" in out
    assert "triton_head_sha" in out
    assert "triton_producing_sha" not in tree.attrs


def test_container_image_with_no_sha_prints_the_warning(tmp_path, capsys):
    """Container arm: no compile flag, a recorded `sif_sha256` and no sha prints the warning.

    Property: stdout names `sif_sha256`, and no attribute is stamped. Class: NEW CAPABILITY
    (the plausible wrong implementation tests the compile flags only; it stays silent here
    and is red). Why required: container mode sets no compile flag, so the a6000 benchmarking
    arm would otherwise never warn. Killed by: dropping the `sif_sha256` term. Second correct
    implementation: testing `sif_sha256` first and the flags second yields the same line.
    """
    sys_log = TRITONSWMM_system_log(logfile=tmp_path / "system_log.json")
    sys_log.sif_sha256.set("sha256:" + "ab" * 32)
    tree = _stamp(sys_log)
    out = capsys.readouterr().out
    assert "[Provenance] WARNING" in out
    assert "sif_sha256" in out
    assert "triton_producing_sha" not in tree.attrs


def test_control_nothing_recorded_stays_silent(tmp_path, capsys):
    """No compile flag, no image digest and no sha prints nothing.

    Property: stdout is empty and no attribute is stamped. Class: CONTROL (green on the
    pre-change tree; it varies whether a build or image was recorded, the predicate the
    warning rests on). Why required: a reprex bundle or a legacy log must not warn.
    Killed by: an else-branch that prints unconditionally. Second correct implementation:
    any predicate over the same two recorded facts.
    """
    sys_log = TRITONSWMM_system_log(logfile=tmp_path / "system_log.json")
    tree = _stamp(sys_log)
    assert capsys.readouterr().out == ""
    assert "triton_producing_sha" not in tree.attrs


def test_control_stamped_log_sets_the_attribute_and_prints_nothing(tmp_path, capsys):
    """A log carrying the sha stamps the tree and prints nothing, even with a build.

    Property: with a True compile flag AND a sha, `triton_producing_sha` is set and stdout is
    empty. Class: CONTROL (green on the pre-change tree; it varies sha presence while the
    build flag stays True, the discriminating input for the else-branch). Why required: the
    warning must never fire on a correctly stamped tree. Killed by: printing outside the
    else-branch. Second correct implementation: computing the warning first and emitting it
    only when the sha is None.
    """
    sys_log = TRITONSWMM_system_log(logfile=tmp_path / "system_log.json")
    sys_log.compilation_tritonswmm_gpu_successful.set(True)
    sys_log.triton_head_sha.set(SHA)
    tree = _stamp(sys_log)
    assert tree.attrs["triton_producing_sha"] == SHA
    assert capsys.readouterr().out == ""


def test_control_incomplete_log_object_never_raises():
    """A log object lacking `logfile` and `last_writer` makes the stamp return, not raise.

    Property: with a True compile flag and an absent sha on a `SimpleNamespace` log that has
    neither `logfile` nor `last_writer`, `_stamp_triton_provenance` returns and stamps
    nothing. Class: CONTROL (green on the pre-change tree; red under the variant that builds
    the message outside the `try`, because the f-string then raises AttributeError out of the
    function). Why required: the function's documented contract is "Never raises" and both
    consolidation sites rely on it. Killed by: moving the `print` out of the try block.
    Second correct implementation: a message built from `getattr(..., "unknown")` defaults
    prints a degraded line and still returns, so the assertion pins only the no-raise
    contract and the absent attribute, never what is printed.
    """

    class _Field:
        def __init__(self, value):
            self._value = value

        def get(self):
            """Return the value this stand-in field was constructed with."""
            return self._value

    fake_log = SimpleNamespace(
        triton_head_sha=_Field(None),
        compilation_tritonswmm_gpu_successful=_Field(True),
        compilation_tritonswmm_cpu_successful=_Field(None),
        compilation_triton_cpu_successful=_Field(None),
        compilation_triton_gpu_successful=_Field(None),
        sif_sha256=_Field(None),
    )
    tree = _stamp(fake_log)
    assert "triton_producing_sha" not in tree.attrs
