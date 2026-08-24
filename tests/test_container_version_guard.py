"""The container-mode preflight refuses a SIF whose TRITON version disagrees with the pin.

WHY THIS GUARD EXISTS. Container mode SKIPS the compile, and the compile is the only place
``_verify_tritonswmm_pin`` fires. So before this guard, a containerized run had NO pin
verification of any kind: an image built at the wrong TRITON ran silently, produced plausible
numbers, and nothing downstream said so. This is not hypothetical -- an estate config on
Rivanna names an image whose ``org.hhemt.triton_sha`` is ``15eb18a5``, at which
``model_defects`` records all three registry defects PRESENT.

WHY THE ARMS ARE WHAT THEY ARE. Each test below pins one decision, and the decisions are not
interchangeable:

* A -- a version MISMATCH is refused. The core case.
* B -- a MATCH produces ZERO errors. Keep this one. A guard that fires on the violating input
  AND on the satisfying input is worthless, and nothing else in this file would notice.
* C -- a MISSING label is refused (fail-closed). "Cannot prove a match" must not read as
  "passes" in the guard whose whole purpose is to stop an unverified image.
* D -- a NON-SHA pin is refused. Container mode has no clone, so a branch name cannot be
  resolved locally; resolving it remotely would verify against a MOVING target. This is
  stricter than native mode deliberately, and it keeps preflight agreeing with
  ``build_sifs_uva.sh``, which already requires the pin to be a ref TIP.
* E -- a RENAMED or COPIED image is caught, by comparing the filename's sha against the
  label's. This is the user requirement verbatim. The label is authoritative; the filename
  corroborates. Neither alone catches this case.
* F -- native mode is a NO-OP. The in-scope synthetic experiments run native, so this arm is
  what proves the guard is inert for them.
* G -- the guard walks ``sif_paths_by_arch``, not just ``sif_path``. The loop is control flow
  the guard added; without this, a multi-arch experiment's per-arch images go unchecked.

PRECEDENCE IS A BEHAVIOURAL DECISION, NOT AN ACCIDENT. See
``test_version_mismatch_takes_precedence_over_filename_disagreement``. Arm A's fixture is
BOTH version-mismatched and filename-disagreeing, and exactly ONE error fires. Reporting both
would send an operator to rename a file whose real problem is that it is the wrong build. Do
not "improve" this by removing the ``continue`` -- two errors is not more informative here,
it is differently wrong.

WHY REAL FIXTURES RATHER THAN A MOCKED READER. The label reader's three-valued failure
classification (readable / unreadable / readable-but-unlabelled) is part of what is being
pinned. A test that mocks ``read_container_labels`` would pass whether or not the shipped
reader distinguishes those states. These fixtures are real apptainer sandbox directories --
a ``.singularity.d/labels.json`` is exactly what apptainer writes -- so the sandbox branch of
the real reader executes, with no subprocess and no apptainer binary required.
"""

from __future__ import annotations

import json

import pytest

from hhemt.config.hpc_system import ContainerSpec
from hhemt.validation import ValidationResult, _validate_container_config

PIN = "5d2ad1e8adf9a85d7df14e885b76e59a10f9a98b"
OTHER_SHA = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"
MAIN_SHA = "21e666d6e0efc3383344813853386aaba1474785"


class _Analysis:
    """Minimal stand-in: the validator reads execution_environment via getattr only."""

    def __init__(self, execution_environment="container"):
        self.execution_environment = execution_environment


class _HpcSystem:
    def __init__(self, container):
        self.container = container


class _System:
    """Minimal stand-in carrying only the field the guard reads off the system config."""

    def __init__(self, branch_key):
        self.TRITONSWMM_branch_key = branch_key


def _sandbox(tmp_path, name, sha):
    """Build a real apptainer SANDBOX directory carrying (or omitting) the triton label.

    Keyed on the ``.singularity.d/`` marker apptainer writes at every sandbox root, which is
    also what the existence check upstream of the guard requires -- so these fixtures clear
    that check and reach the guard, rather than failing earlier for an unrelated reason.
    """
    d = tmp_path / name
    (d / ".singularity.d").mkdir(parents=True)
    labels = {"org.hhemt.triton_sha": sha} if sha else {}
    (d / ".singularity.d" / "labels.json").write_text(json.dumps(labels))
    return d


def _run(sif_path, pin, *, execution_environment="container", by_arch=None):
    """Invoke the REAL preflight validator, not a re-implementation of its rule.

    Do NOT substitute a local copy of the comparison here: a test that restates the rule it
    is checking passes whether or not the shipped code agrees with it.
    """
    result = ValidationResult()
    cspec = ContainerSpec(sif_path=str(sif_path), sif_paths_by_arch=by_arch or {})
    _validate_container_config(_Analysis(execution_environment), _HpcSystem(cspec), result, _System(pin))
    return result


def _messages(result):
    return [issue.message for issue in result.errors]


# --------------------------------------------------------------------------- #
# A -- version mismatch is refused
# --------------------------------------------------------------------------- #
def test_version_mismatch_is_refused(tmp_path):
    result = _run(_sandbox(tmp_path, "img", OTHER_SHA), PIN)
    assert len(result.errors) == 1
    assert "was built at TRITON 15eb18a5d25a" in _messages(result)[0]
    assert "5d2ad1e8adf9" in _messages(result)[0]


# --------------------------------------------------------------------------- #
# B -- a MATCH produces zero errors. The over-firing arm; do not delete.
# --------------------------------------------------------------------------- #
def test_matching_version_produces_no_error(tmp_path):
    result = _run(_sandbox(tmp_path, "img", PIN), PIN)
    assert result.errors == [], f"guard fired on a correct image: {_messages(result)}"


def test_short_pin_validates_against_full_label(tmp_path):
    """An 8-char config pin identifies the same commit as the 40-char label.

    Guards the comparison's floor from being tightened into an equality test, which would
    reject every abbreviated pin while looking like a harmless simplification.
    """
    result = _run(_sandbox(tmp_path, "img", PIN), PIN[:8])
    assert result.errors == []


# --------------------------------------------------------------------------- #
# C -- missing label is refused (fail-closed)
# --------------------------------------------------------------------------- #
def test_missing_label_is_refused_not_trusted(tmp_path):
    result = _run(_sandbox(tmp_path, "img", None), PIN)
    assert len(result.errors) == 1
    assert "carries no org.hhemt.triton_sha label" in _messages(result)[0]


# --------------------------------------------------------------------------- #
# D -- non-sha pin is refused
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_pin", ["main", "triton-swmm", "v1.0"])
def test_non_sha_pin_is_refused_in_container_mode(tmp_path, bad_pin):
    result = _run(_sandbox(tmp_path, "img", PIN), bad_pin)
    assert len(result.errors) == 1
    assert "requires TRITONSWMM_branch_key to be a git SHA" in _messages(result)[0]


# --------------------------------------------------------------------------- #
# E -- a renamed or copied image is caught (the user requirement, verbatim)
# --------------------------------------------------------------------------- #
def test_renamed_image_is_caught_by_filename_label_disagreement(tmp_path):
    """Filename says one sha, label says another, and the LABEL matches the pin.

    The run would be CORRECT and every human reading the config would be misled. Only the
    filename/label pair sees this; neither source alone does.
    """
    result = _run(_sandbox(tmp_path, f"hhemt_uva_cuda_{MAIN_SHA[:8]}", PIN), PIN)
    assert len(result.errors) == 1
    msg = _messages(result)[0]
    assert "has a filename naming TRITON 21e666d6" in msg
    assert "renamed or copied" in msg


def test_filename_without_a_sha_is_not_treated_as_disagreement(tmp_path):
    """The legacy images carry no sha in the name at all, so there is nothing to corroborate.

    Absence must degrade to "label alone decides", not to an error -- otherwise the guard
    would reject every correctly-built image whose filename happens to omit the sha.
    """
    result = _run(_sandbox(tmp_path, "hhemt_uva_cuda", PIN), PIN)
    assert result.errors == []


# --------------------------------------------------------------------------- #
# PRECEDENCE -- one error, and it is the version one
# --------------------------------------------------------------------------- #
def test_version_mismatch_takes_precedence_over_filename_disagreement(tmp_path):
    """A fixture that is BOTH wrong-version and wrong-name yields exactly ONE error.

    The version arm ``continue``s before the filename check reaches it. That ordering is
    deliberate: the remedies differ, and telling an operator to rename a file whose real
    problem is that it is the wrong build sends them to the wrong fix. If someone removes the
    ``continue`` to "report more", this test is what fails.
    """
    # name says 5d2ad1e8 (== pin), label says 15eb18a5 -> mismatch AND disagreement
    result = _run(_sandbox(tmp_path, f"hhemt_{PIN[:8]}", OTHER_SHA), PIN)
    assert len(result.errors) == 1, f"expected exactly one error, got {_messages(result)}"
    assert "was built at TRITON" in _messages(result)[0]
    assert "renamed or copied" not in _messages(result)[0]


# --------------------------------------------------------------------------- #
# F -- native mode is a no-op
# --------------------------------------------------------------------------- #
def test_native_mode_is_a_no_op(tmp_path):
    """The in-scope synthetic experiments run native; this is what proves them unaffected."""
    result = _run(_sandbox(tmp_path, "img", OTHER_SHA), PIN, execution_environment="native")
    assert result.errors == []


def test_absent_pin_does_not_fire(tmp_path):
    """No pin declared => nothing to compare against; the guard must not invent a failure."""
    result = _run(_sandbox(tmp_path, "img", OTHER_SHA), None)
    assert result.errors == []


# --------------------------------------------------------------------------- #
# G -- the guard walks sif_paths_by_arch, not just sif_path
# --------------------------------------------------------------------------- #
def test_per_arch_image_is_checked_too(tmp_path):
    """A multi-arch experiment resolves its SIM image from sif_paths_by_arch.

    Checking only sif_path would leave every per-arch image unverified while reporting green.
    """
    good = _sandbox(tmp_path, "default", PIN)
    bad = _sandbox(tmp_path, "a100", OTHER_SHA)
    result = _run(good, PIN, by_arch={"a100": str(bad)})
    assert len(result.errors) == 1
    assert "container.sif_paths_by_arch[a100]" == result.errors[0].field
    assert "was built at TRITON 15eb18a5d25a" in _messages(result)[0]
