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
* H/I -- the STANDALONE SWMM version is checked, and a missing label is refused. The coupled
  model's SWMM is vendored inside TRITON and travels with the TRITON pin, so arm A already
  covers it; these arms cover the OTHER SWMM, which no other check reaches.
* J/K/L -- the TOOLKIT baked into the image is checked against the RUNNING one. This is the
  arm that would have caught the mixed-version split: the three ``process_*`` rules run
  ``python -m hhemt.…`` INSIDE the image, so a driver at one commit and an image at another
  split one campaign across two toolkits with nothing downstream saying so. K pins that the
  UNSUBSTITUTED placeholder reads as a MISS rather than as a version -- the whole reason the
  placeholder is an unmistakable literal instead of a plausible default. L pins that when the
  running sha is unknowable (a wheel install), the guard WARNS: not-compared must not render
  as compared-and-equal.
* M -- an UNREADABLE image is refused, and is a distinct state from an unlabelled one. They
  send an operator to different remedies, which is why the reader is three-valued.

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
SWMM_TAG = "v5.2.4"
#: The sha the RUNNING toolkit reports under the autouse fixture below. Pinned to a
#: literal rather than read from git so these tests are deterministic in any checkout.
TOOLKIT_SHA = "aabbccdd11223344556677889900aabbccddeeff"


@pytest.fixture(autouse=True)
def _pin_running_toolkit(monkeypatch):
    """Pin the RUNNING toolkit's sha so the toolkit arm is deterministic.

    This monkeypatches an ENVIRONMENT PROBE, not the rule under test. The module's
    no-mocked-reader doctrine above targets ``read_container_labels`` -- the component
    whose three-valued classification is part of what these tests pin -- and that
    reader still executes for real. ``_running_toolkit_sha`` reads the checkout the
    test happens to run in, which is not a property of the guard and would make every
    assertion below depend on the tester's git state.
    """
    # raising=False so this fixture cannot ERROR on a tree lacking the symbol. A setup
    # error and an assertion failure look alike in a summary but prove different things,
    # and a control run that only proves "the symbol is absent" is not a control on the RULE.
    monkeypatch.setattr("hhemt.validation._running_toolkit_sha", lambda: TOOLKIT_SHA, raising=False)


class _Analysis:
    """Minimal stand-in: the validator reads execution_environment via getattr only."""

    def __init__(self, execution_environment="container"):
        self.execution_environment = execution_environment


class _HpcSystem:
    def __init__(self, container):
        self.container = container


class _System:
    """Minimal stand-in carrying only the field the guard reads off the system config."""

    def __init__(self, branch_key, swmm_tag_key=SWMM_TAG):
        self.TRITONSWMM_branch_key = branch_key
        self.SWMM_tag_key = swmm_tag_key


def _sandbox(tmp_path, name, sha, *, swmm=SWMM_TAG, hhemt=TOOLKIT_SHA):
    """Build a real apptainer SANDBOX directory carrying (or omitting) the triton label.

    Keyed on the ``.singularity.d/`` marker apptainer writes at every sandbox root, which is
    also what the existence check upstream of the guard requires -- so these fixtures clear
    that check and reach the guard, rather than failing earlier for an unrelated reason.
    """
    d = tmp_path / name
    (d / ".singularity.d").mkdir(parents=True)
    labels = {"org.hhemt.triton_sha": sha} if sha else {}
    # A REAL image carries all three provenance labels, so the default fixture does too.
    # Without this every TRITON-arm test below would ALSO trip the SWMM and toolkit
    # fail-closed arms, and would then be asserting on three unrelated decisions at once.
    # Pass ``swmm=None`` / ``hhemt=None`` to omit one deliberately.
    if swmm is not None:
        labels["org.hhemt.swmm_version"] = swmm
    if hhemt is not None:
        labels["org.hhemt.hhemt_sha"] = hhemt
    (d / ".singularity.d" / "labels.json").write_text(json.dumps(labels))
    return d


def _run(sif_path, pin, *, execution_environment="container", by_arch=None, swmm_tag_key=SWMM_TAG):
    """Invoke the REAL preflight validator, not a re-implementation of its rule.

    Do NOT substitute a local copy of the comparison here: a test that restates the rule it
    is checking passes whether or not the shipped code agrees with it.
    """
    result = ValidationResult()
    cspec = ContainerSpec(sif_path=str(sif_path), sif_paths_by_arch=by_arch or {})
    _validate_container_config(_Analysis(execution_environment), _HpcSystem(cspec), result, _System(pin, swmm_tag_key))
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


# --------------------------------------------------------------------------- #
# H -- standalone SWMM version mismatch is refused
# --------------------------------------------------------------------------- #
def test_swmm_version_mismatch_is_refused(tmp_path):
    result = _run(_sandbox(tmp_path, "img", PIN, swmm="v5.2.3"), PIN)
    assert len(result.errors) == 1
    msg = _messages(result)[0]
    assert "standalone SWMM v5.2.3" in msg
    assert "v5.2.4" in msg
    # The message must say WHICH SWMM it governs. Two SWMMs are live in a coupled
    # analysis and only one of them is SWMM_tag_key; a reader who conflates them
    # will re-pin the wrong thing.
    assert "vendored inside TRITON" in msg


# --------------------------------------------------------------------------- #
# I -- a missing SWMM label is refused, not trusted
# --------------------------------------------------------------------------- #
def test_missing_swmm_label_is_refused_not_trusted(tmp_path):
    result = _run(_sandbox(tmp_path, "img", PIN, swmm=None), PIN)
    assert len(result.errors) == 1
    assert "no org.hhemt.swmm_version label" in _messages(result)[0]


def test_absent_swmm_pin_does_not_fire(tmp_path):
    """No SWMM_tag_key declared -> nothing to compare, so the arm is inert.

    Mirrors ``test_absent_pin_does_not_fire`` for TRITON. Without this, adding the SWMM
    arm would have made the guard fire on every config that simply does not pin SWMM.
    """
    result = _run(_sandbox(tmp_path, "img", PIN, swmm=None), PIN, swmm_tag_key=None)
    assert result.errors == []


# --------------------------------------------------------------------------- #
# J -- the toolkit baked into the image must match the RUNNING toolkit
# --------------------------------------------------------------------------- #
def test_toolkit_sha_mismatch_is_refused(tmp_path):
    result = _run(_sandbox(tmp_path, "img", PIN, hhemt=OTHER_SHA), PIN)
    assert len(result.errors) == 1
    msg = _messages(result)[0]
    assert "bakes hhemt 15eb18a5d25a" in msg
    assert "RUNNING toolkit is aabbccdd1122" in msg


# --------------------------------------------------------------------------- #
# K -- an UNSUBSTITUTED placeholder is a MISS, never a version
# --------------------------------------------------------------------------- #
def test_unsubstituted_placeholder_reads_as_missing_not_as_a_version(tmp_path):
    """The build script failed to substitute -> the label names no commit.

    This is the arm the placeholder's SHAPE exists for. A plausible-looking default
    would compare cleanly against nothing and report a version mismatch, sending the
    operator to re-pin a config when the real defect is in the build. The error must
    name the BUILD.
    """
    from hhemt.container_labels import HHEMT_SHA_PLACEHOLDER

    result = _run(_sandbox(tmp_path, "img", PIN, hhemt=HHEMT_SHA_PLACEHOLDER), PIN)
    assert len(result.errors) == 1
    msg = _messages(result)[0]
    assert "no usable org.hhemt.hhemt_sha label" in msg
    assert HHEMT_SHA_PLACEHOLDER in result.errors[0].fix_hint


# --------------------------------------------------------------------------- #
# L -- an unknowable RUNNING sha WARNS; it does not pass and it does not error
# --------------------------------------------------------------------------- #
def test_unknown_running_toolkit_warns_rather_than_erroring(tmp_path, monkeypatch):
    """A wheel install has no sha. That is the intended fallback, not a failure.

    But it is also not a PASS, and the distinction is the whole point: the guard
    reports the comparison as UNPERFORMED so a reader cannot mistake silence for
    agreement. Erroring here would refuse every non-git install; passing silently
    would reintroduce exactly the unverified-version state this guard exists to stop.
    """
    monkeypatch.setattr("hhemt.validation._running_toolkit_sha", lambda: None)
    result = _run(_sandbox(tmp_path, "img", PIN, hhemt=OTHER_SHA), PIN)
    assert result.errors == []
    assert len(result.warnings) == 1
    assert "UNPERFORMED rather than passing" in result.warnings[0].message


# --------------------------------------------------------------------------- #
# M -- an UNREADABLE image is refused, and is distinct from an unlabelled one
# --------------------------------------------------------------------------- #
def test_unreadable_image_is_refused(tmp_path):
    """COVERAGE, not a new-feature test -- and the distinction is load-bearing.

    This branch SHIPPED UNTESTED. Measured by controlled pair: with the SWMM and toolkit
    checks reverted, the other five new arms FAIL and this one PASSES, because it pins
    behaviour that already existed. Recorded here so a later reader does not cite this
    test as evidence for the version-label work; it is evidence about the reader.
    """
    d = tmp_path / "img"
    (d / ".singularity.d").mkdir(parents=True)
    (d / ".singularity.d" / "labels.json").write_text("{not json")
    result = _run(d, PIN)
    assert len(result.errors) == 1
    msg = _messages(result)[0]
    assert "could not read provenance labels" in msg
    # Distinct remedy from the unlabelled case: the IMAGE is the problem, not the build.
    assert "the IMAGE is the problem" in result.errors[0].fix_hint
