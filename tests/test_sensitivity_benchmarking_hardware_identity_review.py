"""Reviewer tests for work chunk ``2a.2`` — the CPU hardware identity and its two silent-failure arms.

Two findings are carried here, each as a test that FAILS against the chunk as returned:

- **F1** — the disagreement rule DETECTS and does not RECORD. ``HardwareIdentity.disagreement``
  is defined and read by nothing in the tree, and the module produces no ``CheckResult`` at all,
  so a declared token contradicted by the solver's own run header reaches no surface. The amended
  stipulation requires the opposite in as many words: *"A declared token and a derived name that
  disagree are not silently reconciled: the declared value carries the column and both values are
  recorded on the errors-and-warnings surface."*
- **F2** — the refusal gate's single-hardware exemption is written as a COUNT of distinct resolved
  tokens (``len(resolved_tokens) <= 1``), and its stated ground — "every member resolves to the
  same token" — is false at zero. A master whose every member is unresolved therefore renders
  silently with the axis collapsed, which is the exact defect the package exists to remove.

Both tests accept a raise OR a captured warning as the visible outcome, because the remedy for a
disagreement is explicitly non-fatal and the remedy for the zero-token arm could be either; a test
that demanded one channel would fail a correct implementation that chose the other.

Stub-only: no solver, no HPC, ``tmp_path`` alone.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from hhemt.report_renderers import sensitivity_benchmarking as sb


class _Spec:
    def __init__(self, hardware):
        self.hardware = hardware


class _Hpc:
    def __init__(self, hardware):
        self.partitions = {"standard": _Spec(hardware)}


class _CfgAnalysis:
    def __init__(self, partition="standard"):
        self.hpc_ensemble_partition = partition


class _Paths:
    def __init__(self, d):
        self.analysis_dir = d


class _Sub:
    def __init__(self, hpc, d):
        self.cfg_hpc_system = hpc
        self.cfg_analysis = _CfgAnalysis()
        self.analysis_paths = _Paths(d)


class _Sens:
    def __init__(self, members):
        self.members = members


class _Master:
    def __init__(self, members):
        self.sensitivity = _Sens(members)


def _member(tmp_path: Path, name: str, declared: str | None = None, cpu_line: str | None = None) -> _Sub:
    """One stub member; ``cpu_line`` writes a TRITON run header carrying that CPU name."""
    d = tmp_path / name
    log_dir = d / "sims" / "s0" / "out_tritonswmm"
    log_dir.mkdir(parents=True, exist_ok=True)
    if cpu_line is not None:
        (log_dir / "log.out").write_text(f"Machine : node01\nCPU : {cpu_line}\nnTasks : 4\n")
    return _Sub(_Hpc(declared), d)


def _observable(messages: list[str], needles: tuple[str, ...]) -> bool:
    """True when some captured message carries every needle."""
    return any(all(n.lower() in m.lower() for n in needles) for m in messages)


def test_a_disagreement_is_recorded_and_not_merely_detected(tmp_path):
    """Finding F1 — a declared token contradicted by the run header must reach a surface.

    Property: resolving a member whose partition DECLARES one token while its run header names a
    DIFFERENT CPU leaves the record carrying both raw values, and the resolution-then-gate path
    makes that pair observable (a warning naming both, or a refusal — the latter would be wrong
    here but is accepted so the test is not coupled to one channel).

    Class: REGRESSION — red on the chunk as returned. ``HardwareIdentity.disagreement`` has exactly
    ONE occurrence in the tree and it is a comment about the property; nothing calls it, and the
    module constructs no ``CheckResult``, so the pair reaches no report surface.

    Why this is required: the amended stipulation's `hardware` clause ends *"A declared token and a
    derived name that disagree are not silently reconciled: the declared value carries the column
    and both values are recorded on the errors-and-warnings surface."* The record half is the half
    that does not exist.

    What kills it: the current code — the pair lives only in the frozen record and dies there.

    A second correct implementation that still passes: one that pushes the pair as a
    ``CheckResult`` row through the report's warnings pipeline and also emits a
    ``warnings.warn`` at resolution time, or one that routes it through whichever collector the
    consumer ends up owning — the assertion is over an observable message naming BOTH tokens, not
    over the channel or the wording.
    """
    sub = _member(tmp_path, "declared_epyc", declared="AMD EPYC 9454", cpu_line="Intel Xeon Gold 6248")

    identity = sb._resolve_member_hardware(sub)
    assert identity.token == "AMD EPYC 9454", f"the declared value must carry the column; got token={identity.token!r}"
    assert identity.derived is not None and identity.derived != identity.token, (
        f"the fixture must produce a genuine disagreement; got derived={identity.derived!r}"
    )
    assert identity.disagreement is True, "the record must report the disagreement it holds"

    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            sb._resolve_hardware_identities(_Master({"declared_epyc": sub}))
            sb._require_identifiable_hardware({"declared_epyc": identity})
        except sb.HardwareIdentityRefused as exc:  # a refusal would also make it visible
            messages.append(str(exc))
        messages.extend(str(w.message) for w in caught)

    assert _observable(messages, (identity.token, identity.derived)), (
        "a declared token contradicted by the solver's own run header reached NO surface: the pair "
        f"is held in the record and read by nothing. declared={identity.token!r} "
        f"derived={identity.derived!r} captured={messages!r}"
    )


def test_a_master_whose_members_all_lack_identity_is_not_silent(tmp_path):
    """Finding F2 — zero resolved tokens is not "single-hardware"; it is unknown-hardware.

    Property: ``_require_identifiable_hardware`` must not return silently for a master in which NO
    member's hardware resolves. Visibility may be a refusal or a recorded warning; silence is the
    failure.

    Class: REGRESSION — red on the chunk as returned. The exemption is ``len(resolved_tokens) <= 1``
    and the set is empty, so the function returns.

    Why this is required: the gate's own docstring states its exempting ground — *"every member
    resolves to the same token, so a member with no identity belongs to that same column by
    elimination"* — and that premise is FALSE when nothing resolves. The population is reachable
    and routine: the same docstring calls an absent run header a legitimate state because
    ``clear_raw`` may have purged the raw tier, so a CPU master rendered after clearing raw with
    no declared `hardware` is exactly this population.

    What kills it: the current ``<= 1`` test. It exempts three populations, of which only the first
    has the stated property — all members resolved and agreeing; no members resolved; and one
    token resolved beside unresolved members.

    A second correct implementation that still passes: one that REFUSES on zero resolved tokens, or
    one that warns and records — both are captured here. A remedy that only covers the zero-token
    arm and leaves the one-token-plus-unresolved arm silent also passes this test, which is why
    that arm is reported in prose as ROUTED rather than asserted here.
    """
    unresolved = sb._resolve_member_hardware(_member(tmp_path, "no_header"))
    assert unresolved.token is None, "the fixture must produce a member with no identity"
    identities = {"m1": unresolved, "m2": unresolved}

    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            sb._require_identifiable_hardware(identities)
        except sb.HardwareIdentityRefused as exc:
            messages.append(str(exc))
        messages.extend(str(w.message) for w in caught)

    assert messages, (
        "a master in which NO member's hardware resolved passed the gate silently: the axis "
        "collapses to one column with nothing raised and nothing logged, which is the defect this "
        "package exists to remove, restored. The exemption's own ground (every member resolves to "
        "the same token) does not hold when zero resolve."
    )
