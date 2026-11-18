"""The work-branch solver tier of the open-water outfall tests is provisioned by the warm step and not by a chunk.

LAPTOP-SAFE: nothing here clones a solver, compiles or runs one. The provisioning call is stubbed at the clone (the
`subprocess.run` of `provision_borrower`) or at `provision_borrower` itself, and every path is a temporary directory.

The defect these tests grade: under the suite harness a chunk never provisions a shared solver tier, the harness's
warm step (`tests/test_synth_00_compile_models.py`, run outside a chunk) provisioned only the standard tier, and so
every test of `tests/test_synth_open_water_outfall_review.py` raised `refusing to provision ... from a suite chunk:
the tier is ABSENT`. Three properties close it, and each test below names the one it grades.
"""

from __future__ import annotations

import pytest

from tests.fixtures import _triton_source_cache as tsc
from tests.fixtures import outfall_tier

_CHUNK_REFUSAL = "refusing to provision"


@pytest.fixture
def stubbed_clone(tmp_path, monkeypatch):
    """`provision_borrower` with the canonical and the git commands stubbed: it records each command, runs none."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(tsc._DISABLE_ENV, raising=False)
    monkeypatch.delenv(tsc._SUITE_ROLE_ENV, raising=False)
    monkeypatch.setattr(tsc, "ensure_canonical", lambda *, pin=tsc.TRITON_PIN: tmp_path / "canonical")
    commands: list[list[str]] = []

    def record(cmd, **_kwargs):
        commands.append([str(word) for word in cmd])

    monkeypatch.setattr(tsc.subprocess, "run", record)
    return commands


def test_the_warm_target_provisions_the_tier_at_the_pin_and_directory_the_outfall_tests_use(monkeypatch):
    """Property: the warm target calls `provision_borrower` once, for the outfall tier's directory and its pin.

    Class: REGRESSION -- the warm step provisioned only the standard tier, so the work-branch tier was absent for
    every chunk. The plausible wrong implementation adds a warm test that provisions the standard tier again, or
    the right tier at the wrong pin or directory.
    Why required: a chunk finds the tier PRESENT only if the warm provisioned the same pin at the same directory
    the tests' system config names.
    What kills it: a warm target that does not call the provisioner, or calls it for another directory or pin.
    Second correct implementation: a warm target that calls `provision_outfall_tier` through another helper still
    reaches `provision_borrower` with the same two arguments, and the assertion holds.
    """
    from tests import test_synth_00_compile_models as warm

    calls: list[tuple] = []

    def record(dest, *, pin):
        calls.append((dest, pin))
        return dest

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(tsc._DISABLE_ENV, raising=False)
    monkeypatch.setattr(tsc, "provision_borrower", record)
    monkeypatch.setattr(tsc, "borrower_is_healthy", lambda tree, pin: True)

    warm.test_provision_the_open_water_outfall_solver_tier()

    assert calls == [(outfall_tier.outfall_software_directory(), outfall_tier.OUTFALL_SOLVER_SHA)], calls


def test_a_chunk_context_call_does_not_provision_and_a_warm_context_call_does(tmp_path, monkeypatch, stubbed_clone):
    """Property: with the tier absent, `provision_outfall_tier` clones nothing in a chunk and clones it outside one.

    Class: REGRESSION -- the chunk side is the refusal the ten reds raised, and the warm side is what the warm
    target relies on. The plausible wrong implementation makes the tier's helper bypass the role gate, which would
    let N chunks clone into one path.
    Why required: the fix moves provisioning to the warm step; it must not move it into a chunk.
    What kills it: a helper that provisions when `HHEMT_SUITE_ROLE` is `chunk`, or one that never provisions.
    Second correct implementation: a helper that raises its own error in a chunk before reaching the gate would
    also clone nothing, and the chunk half holds.
    """
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(tsc.cache_paths, "slug_runs_root", lambda slug: runs_root / slug)
    dest = outfall_tier.outfall_software_directory()
    assert runs_root in dest.parents, dest

    monkeypatch.setenv(tsc._SUITE_ROLE_ENV, "chunk")
    with pytest.raises(RuntimeError, match=_CHUNK_REFUSAL):
        outfall_tier.provision_outfall_tier()
    assert stubbed_clone == [] and not dest.exists()

    monkeypatch.delenv(tsc._SUITE_ROLE_ENV)
    assert outfall_tier.provision_outfall_tier() == dest
    clone = [c for c in stubbed_clone if c[:2] == ["git", "clone"]]
    assert len(clone) == 1 and clone[0][-1] == str(dest), stubbed_clone
    assert ["git", "-C", str(dest), "checkout", outfall_tier.OUTFALL_SOLVER_SHA] in stubbed_clone, stubbed_clone


def test_the_outfall_tests_build_their_case_on_the_directory_and_pin_the_warm_step_provisions(tmp_path, monkeypatch):
    """Property: the system config the outfall tests build carries the warm step's directory and the warm step's pin.

    Class: REGRESSION -- the two statements of the tier, one in the tests and one in the warm step, drifting apart
    so that the chunk looks for a tier the warm never made. The plausible wrong implementation leaves a copy of the
    pin or the path in the test file when it moves the rest.
    Why required: `retrieve_synth_TRITON_SWMM_test_case` provisions the directory named in
    `TRITONSWMM_software_directory` at the pin named in `TRITONSWMM_branch_key`; a standalone session of the ten tests
    therefore provisions exactly what the warm step provisions, and a chunk finds exactly that.
    What kills it: a different pin or directory in the test file's `_build_case` than in `outfall_tier`.
    Second correct implementation: a test file that takes both from a fixture in `outfall_tier` passes the same
    kwargs and the assertion holds.
    """
    from tests import test_synth_open_water_outfall_review as outfall_tests
    from tests.fixtures import test_case_builder

    seen: dict = {}

    def record(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(test_case_builder, "retrieve_synth_TRITON_SWMM_test_case", record)

    outfall_tests._build_case(tmp_path, "driven")
    assert seen["additional_system_configs"] == {
        "TRITONSWMM_branch_key": outfall_tier.OUTFALL_SOLVER_SHA,
        "TRITONSWMM_software_directory": str(outfall_tier.outfall_software_directory()),
    }, seen["additional_system_configs"]

    seen.clear()
    outfall_tests._build_case(tmp_path, "none_pin")
    assert seen["additional_system_configs"] == {}, seen["additional_system_configs"]
