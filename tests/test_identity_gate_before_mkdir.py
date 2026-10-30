"""The running-identity gate runs BEFORE the analysis constructor writes anything.

`TRITONSWMM_analysis.__init__` used to `mkdir` the per-analysis `sims/` directory and then
run the `[Q331]` identity gate, so a construction REFUSED on identity had already created
part of the analysis tree — and the two published statements that claim otherwise ("before
any I/O" in `analysis.py`, "before doing anything else" on the `hhemt_sha` field
description) were false.

The first arm below is a REGRESSION: it fails on the pre-change tree, where the refused
construction leaves `sims/` behind, and passes once the gate moves above the `mkdir`.
Pre-change output is pasted in the chunk's round file.

BOTH the fixture's own construction and the measured one are driven through STUBBED
identities. That is not convenience: the fixture builds a real analysis, so an unstubbed
run refuses on a dirty worktree before the arm can measure anything — the stub is what
makes this arm about the mkdir rather than about the tree's cleanliness.
"""

from __future__ import annotations

import shutil

import pytest

from hhemt import analysis as an
from hhemt import validation
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.exceptions import ConfigurationError
from hhemt.validation import RunningIdentity
from tests.fixtures.test_case_builder import _TEST_RUNS_ROOT_OVERRIDE_ENV
from tests.fixtures.test_case_catalog import Local_TestCases

# The identity the fixture's config declares, and one it does not. Both are 40-hex so the
# field's own pattern gate passes and the refusal under test is the IDENTITY comparison.
DECLARED_SHA = "a" * 40
FOREIGN_SHA = "0" * 40


def _stub(sha: str):
    return lambda: RunningIdentity(sha=sha, shape="checkout", dirty=False)


def _build_case(tmp_path, monkeypatch):
    """A constructed synthetic case, with both identity seams stubbed to the DECLARED sha."""
    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
    monkeypatch.setattr(validation, "running_identity", _stub(DECLARED_SHA))
    monkeypatch.setattr(an, "running_identity", _stub(DECLARED_SHA))
    # skip_run=True: the fixture builds and validates a synthetic case; nothing compiles
    # and no solver executes, so this arm is solver-free.
    return Local_TestCases.retrieve_synth_multi_sim_test_case(skip_run=True).analysis


def test_a_refused_construction_leaves_no_sims_directory(tmp_path, monkeypatch):
    """Property: an identity-refused construction creates no `sims/` directory.

    Class: every construction refused by the `[Q331]` identity gate — a wrong
    `hhemt_sha`, a dirty checkout, an unresolvable toolkit. Why required: the gate exists
    to stop a run that is not the declared commit BEFORE it touches the tree; a gate that
    runs after the first mkdir has already created part of an analysis tree under a
    refused identity, which is the observable the two corrected statements deny. Killed
    by: leaving the `mkdir` above the gate, which recreates `sims/` on the refused path.
    Second correct implementation: computing every path first and running the gate before
    ANY write of any kind, rather than moving it one statement.

    Reachability: the failure is the `mkdir` executing before the gate in
    `TRITONSWMM_analysis.__init__`; artifact: the two `exists()` assertions, one before and
    one after the call, so a directory that already existed cannot pass for one the call
    made.
    Both-states anchor: whether the directory exists after a refused call is a property of
    both the current and the proposed code, not of any wording.
    Invocation not isolatable from this tree — discrimination shown instead: moving the gate
    above the `mkdir` greens this test, and the pre-change run left `sims/` present with
    the exact path pasted in the round file.
    """
    built = _build_case(tmp_path, monkeypatch)
    analysis_config_yaml = built.analysis_config_yaml
    system = built._system
    sims = built.analysis_paths.simulation_directory

    # Remove the directory the fixture's own construction made, so what the assertions
    # below observe is only what the REFUSED call creates.
    shutil.rmtree(sims, ignore_errors=True)
    assert not sims.exists(), "the fixture set-up left sims/ behind; the arm would be vacuous"

    monkeypatch.setattr(an, "running_identity", _stub(FOREIGN_SHA))
    with pytest.raises(ConfigurationError) as exc:
        TRITONSWMM_analysis(analysis_config_yaml, system)
    assert "hhemt_sha" in str(exc.value)

    assert not sims.exists(), f"the refused construction created {sims} — the identity gate runs AFTER the mkdir"


def test_the_gate_still_refuses_at_all(tmp_path, monkeypatch):
    """Property: the gate still refuses a mismatched identity, so the move did not disarm it.

    Class: any construction whose config names a commit the running toolkit is not. Why
    required: relocating a gate is exactly the edit class that can move it somewhere it
    never runs, and without this arm a fix that DELETED the gate would pass the arm above.
    Killed by: placing the gate below a branch that returns before it, or removing it.
    Second correct implementation: the same refusal raised from a helper the constructor
    calls first.
    """
    built = _build_case(tmp_path, monkeypatch)
    monkeypatch.setattr(an, "running_identity", _stub(FOREIGN_SHA))
    with pytest.raises(ConfigurationError) as exc:
        TRITONSWMM_analysis(built.analysis_config_yaml, built._system)
    assert "hhemt_sha" in str(exc.value)
