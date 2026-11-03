"""Reviewer tests for chunk 1b.3: solver-date capture, per-solver aggregation and the combine refusal.

The coder's own checks exercise ONE position per class (one arm undated, equal author and
committer dates, a single-store root). These tests take the DIFFERENTLY-POSITIONED input of each
class. Every test here is green on the tree as built; each is a HELD test, and each docstring
names the production change that reds it and a second correct implementation that stays green.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle import _combine as C
from hhemt.bundle import _combine_merge as M
from hhemt.exceptions import ConfigurationError
from hhemt.log import TRITONSWMM_system_log
from hhemt.report_renderers import cross_solver_comparison as R
from hhemt.system import TRITONSWMM_system

_CPU = {
    "run_mode": "clean",
    "n_mpi_procs": 1,
    "n_omp_threads": 1,
    "n_gpus": 0,
    "n_nodes": 1,
    "hpc.partition": "standard",
}
_PIN_A = "a38338b09e62e57c936f51516bbdbe495d89a546"
_PIN_B = "b1ff2689" + "0" * 32
_PIN_E = "e53c2fa0" + "0" * 32
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "PATH": "/usr/bin:/bin",
}


def _bundle(
    root: Path, pin: str, arm_group: str, *, sha: str | None, date: str | None, store: str = "sensitivity_datatree.zarr"
) -> Path:
    """One single-arm bundle: a pinned ``cfg_system.yaml`` and a consolidated tree stamped with the given root attrs."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg_system.yaml").write_text(f"TRITONSWMM_branch_key: {pin}\n")
    attrs = {}
    if sha is not None:
        attrs["triton_producing_sha"] = sha
    if date is not None:
        attrs["triton_producing_commit_date"] = date
    groups = {
        "/": xr.Dataset(attrs=attrs),
        "/member_0": xr.Dataset(attrs=_CPU),
        f"/member_0/{arm_group}": xr.Dataset(),
        f"/member_0/{arm_group}/performance": xr.Dataset({"Total": ("Rank", np.asarray([10.0]))}),
    }
    xr.DataTree.from_dict(groups).to_zarr(root / store, mode="w", consolidated=False)
    return root


def _solver(
    tmp_path: Path, tag: str, pin: str, *, sha: str | None, date_first: str | None, date_second: str | None
) -> list[Path]:
    """Both model arms of one solver; the capture state of each arm is set independently."""
    return [
        _bundle(tmp_path / f"{tag}_coupled", pin, "tritonswmm", sha=sha, date=date_first),
        _bundle(tmp_path / f"{tag}_triton", pin, "triton_only", sha=sha, date=date_second),
    ]


def _readmodel(tmp_path: Path, roots: list[Path], name: str = "combined") -> dict:
    out = tmp_path / name
    out.mkdir(exist_ok=True)
    C._write_cross_solver_comparison(out, roots)
    return json.loads((out / R.READ_MODEL_FILENAME).read_text())


def _figure_facts(payload: dict) -> dict:
    _fig, facts = R._build_figure(R._normalize_cells(payload), payload, R.ProvenanceLog())
    return facts


def _capture(clone: Path, logfile: Path) -> TRITONSWMM_system_log:
    log = TRITONSWMM_system_log(logfile=logfile)
    fake = SimpleNamespace(cfg_system=SimpleNamespace(TRITONSWMM_software_directory=clone), log=log)
    TRITONSWMM_system._capture_tritonswmm_provenance(fake, verbose=False)
    return log


@pytest.mark.parametrize(
    ("author_iso", "committer_iso", "machine_tz", "expected"),
    [
        ("2026-01-01T12:00:00+0000", "2026-10-04T12:00:00+0000", "UTC", "20261004"),
        ("2026-10-06T12:00:00+0000", "2026-10-04T12:00:00+0000", "UTC", "20261004"),
        ("2026-01-01T12:00:00+0000", "2026-10-04T23:30:00-0500", "Asia/Tokyo", "20261004"),
    ],
)
def test_capture_stamps_the_committer_date_whatever_the_author_date_and_machine_zone(
    tmp_path, monkeypatch, author_iso, committer_iso, machine_tz, expected
):
    """Property: the captured date is the COMMITTER date, in the commit's own zone, on any machine.

    Class: commits whose author date differs from the committer date (a rebase or cherry-pick makes
    them differ) and whose committer zone differs from the machine zone. Shape row: Settlement 1 and
    DoD 1b ("stamps the clone's committer date"). Why required: the figure orders series by this date
    and a rebased commit's author date predates the build by months; a machine-zone-dependent date
    would let the capture node and the estate's name disagree on a commit near midnight.
    Killed by: switching the capture's ``%cd`` to ``%ad`` (author date), or its ``--date=format:`` to
    ``format-local:``. Second correct implementation: reading ``%cI`` and slicing its first ten
    characters, which stays in the commit's own zone and so also stays green.
    Class: NEW CAPABILITY on this chunk's reviewer path (the plausible wrong implementation is the
    author-date one; the coder's own check sets both dates equal and cannot tell the two apart).
    """
    monkeypatch.setenv("TZ", machine_tz)
    clone = tmp_path / "clone"
    env = {**_GIT_ENV, "GIT_AUTHOR_DATE": author_iso, "GIT_COMMITTER_DATE": committer_iso}
    subprocess.run(["git", "init", "-q", str(clone)], check=True, env=env)
    subprocess.run(["git", "-C", str(clone), "commit", "-q", "--allow-empty", "-m", "x"], check=True, env=env)
    log = _capture(clone, tmp_path / "system_log.json")
    assert log.triton_head_commit_date.get() == expected


@pytest.mark.parametrize(
    "stores",
    [
        ["analysis_datatree.zarr"],
        ["sensitivity_datatree.zarr"],
        ["experiment_datatree.zarr"],
        ["analysis_datatree.zarr", "sensitivity_datatree.zarr"],
        ["experiment_datatree.zarr", "analysis_datatree.zarr"],
    ],
)
def test_provenance_is_read_from_the_store_the_merge_opens(tmp_path, stores):
    """Property: the provenance attrs come from the SAME root store the combine's root-tree check opens.

    The check is ``_combine._check_root_tree``.

    Class: every root-store subset a bundle can carry, including the two-store roots on which two
    resolvers once disagreed. Shape row: Settlement 3 ("the tree ``_combine_merge._open_experiment_tree``
    resolves"). Why required: a read from a different store than the one the combine checks and reads is a date
    and sha that describe other data than the figure draws. Killed by: replacing the read's resolver with a
    fixed first-existing walk over a different name order. Second correct implementation: calling
    ``_combine_merge._resolve_root_tree`` directly instead of ``utils.resolve_experiment_tree``.
    Every store here carries a DISTINCT sha, so a read from the wrong store is visible.
    Class: REGRESSION guard (green on the tree as built; a held test).
    """
    root = tmp_path / "b"
    for i, name in enumerate(stores):
        _bundle(root, _PIN_A, "tritonswmm", sha=f"{i}" * 40, date=f"2026010{i + 1}", store=name)
    stamps = {name: (f"{i}" * 40, f"2026010{i + 1}") for i, name in enumerate(stores)}
    assert len(set(stamps.values())) == len(stores)  # the stores are distinguishable, so the assertion discriminates
    assert C._root_provenance_attrs(root) == stamps[M._resolve_root_tree(root).name]


@pytest.mark.parametrize("undated_arm", ["first", "second", "both"])
@pytest.mark.parametrize("attr", ["date", "sha"])
def test_a_missing_capture_in_either_arm_changes_neither_the_series_nor_the_not_run_count(tmp_path, undated_arm, attr):
    """Property: which arm lacks a capture, and which attribute, cannot change the series or ``not_run`` counts.

    Class: capture state of the two arms of one solver, crossed over arm position and over the
    attribute (date or sha). Shape row: DoD 1b ("one solver with one arm dated draws the same series
    count and the same ``not_run`` count as the both-dated control"). Why required: the cell identity is the
    pin, so no capture state may split a solver into two series or manufacture a ``not_run`` marker; the
    coder's own check covers only the first arm lacking the DATE. Killed by: deriving a solver's date or
    sha from its first bundle only, or from a bundle chosen by sort order. Second correct implementation:
    taking the date from whichever bundle carries it, in any order, which stays green.
    A solver whose arms are ALL undated is the fallback and keeps its pin identity, one series.
    Class: REGRESSION guard (a held test; green on the tree as built).
    """
    sha_first = None if (attr == "sha" and undated_arm in ("first", "both")) else _PIN_A
    date_first = None if (attr == "date" and undated_arm in ("first", "both")) else "20260828"
    date_second = None if (attr == "date" and undated_arm in ("second", "both")) else "20260828"
    roots = [
        _bundle(tmp_path / "a_coupled", _PIN_A, "tritonswmm", sha=sha_first, date=date_first),
        _bundle(
            tmp_path / "a_triton",
            _PIN_A,
            "triton_only",
            sha=None if (attr == "sha" and undated_arm in ("second", "both")) else _PIN_A,
            date=date_second,
        ),
    ] + _solver(tmp_path, "b", _PIN_B, sha=_PIN_B, date_first="20261002", date_second="20261002")
    payload = _readmodel(tmp_path, roots)
    facts = _figure_facts(payload)
    assert (facts["solver_count"], facts["not_run_cell_count"]) == (2, 0)
    assert {c["solver"] for c in payload["cells"]} == {_PIN_A, _PIN_B}
    info = payload["solver_info"][_PIN_A]
    both_date_attrs_gone = attr == "date" and undated_arm == "both"
    assert info["date"] == (None if both_date_attrs_gone else "20260828")
    assert info["label"] == (_PIN_A[:8] if both_date_attrs_gone else "a38338b0_20260828")


@pytest.mark.parametrize("attr", ["date", "sha"])
def test_two_agreeing_arms_do_not_outvote_a_third_that_differs(tmp_path, attr):
    """Property: any present disagreement under one pin raises, however many bundles agree with each other.

    Class: bundle sets under one ``TRITONSWMM_branch_key`` of size three or more, where a MAJORITY agrees
    and a minority differs. Shape row: Settlement 3 ("two of one solver's bundles carry two *present*
    values that differ ... raises ``ConfigurationError`` ... rather than picking one"). Why required: a
    majority vote would publish the majority's build under a label that also covers the minority's build,
    which is the wrong-build-at-a-plausible-label defect the refusal exists to stop. Killed by: replacing the
    raise with a most-common-value pick. Second correct implementation: comparing every present value to
    the first present value, which also raises. The control arm (the same three bundles with the third
    absent rather than different) must NOT raise.
    Class: NEW CAPABILITY (plausible wrong implementation: majority vote; the coder's checks use two bundles).
    """
    bad = "20260901" if attr == "date" else "b" * 40

    def trio(third_date, third_sha):
        roots = []
        for i, (d, s) in enumerate([("20260828", _PIN_A), ("20260828", _PIN_A), (third_date, third_sha)]):
            roots.append(
                _bundle(tmp_path / f"{attr}_{third_date}_{third_sha}_{i}", _PIN_A, "tritonswmm", sha=s, date=d)
            )
        return roots

    third_date, third_sha = ("20260828", _PIN_A)
    if attr == "date":
        third_date = bad
    else:
        third_sha = bad
    with pytest.raises(ConfigurationError) as exc:
        C._solver_provenance(trio(third_date, third_sha))
    message = str(exc.value)
    assert bad in message and ("20260828" in message or _PIN_A in message)
    assert "_2" in message  # the deviating bundle is named, not only the agreeing pair
    absent = trio(None if attr == "date" else "20260828", None if attr == "sha" else _PIN_A)
    assert (
        C._solver_provenance(absent)[_PIN_A]["date"] == "20260828"
    )  # control: absent is absorbed, never a disagreement


@pytest.mark.parametrize("second_pin", [_PIN_E, "a38338b0" + "1" * 32], ids=["distinct-prefix", "shared-8-char-prefix"])
@pytest.mark.parametrize("capture_state", ["dated", "undated", "mixed"])
def test_distinct_solver_count_counts_pins_and_is_blind_to_capture_state_and_label_equality(
    tmp_path, capture_state, second_pin
):
    """Property: the solver count and the series count equal the number of distinct ``TRITONSWMM_branch_key`` values.

    Class: pin sets crossed with capture state and with LABEL equality (two different pins whose first
    eight characters agree, committed on one day, produce one identical legend label). Shape row: DoD 1b's
    inertness clause and Settlement 3 ("``distinct_solver_count`` stays keyed on ``TRITONSWMM_branch_key``,
    so 1b.4's predicate contract is untouched"; "the cell identity stays the pin"). Why required: 1b.4
    skips the renderer when the count is below two, so a count taken over labels or dates would suppress a
    two-solver figure whose legend labels happen to coincide, and a series keyed by label would merge two
    builds. Killed by: counting distinct labels, distinct dates, or dated solvers only. Second correct
    implementation: counting the keys of ``solver_info`` instead of ``len(solvers)``, which stays equal by construction.
    Class: NEW CAPABILITY (plausible wrong implementation: count distinct labels; the coder's check covers one pin).
    """
    same_day = "20260828" if capture_state != "undated" else None
    second_arm = None if capture_state == "mixed" else same_day
    two_pins = _solver(tmp_path, "a", _PIN_A, sha=_PIN_A, date_first=same_day, date_second=second_arm) + _solver(
        tmp_path, "e", second_pin, sha=second_pin, date_first=same_day, date_second=same_day
    )
    two = _readmodel(tmp_path, two_pins, "two")
    assert two["distinct_solver_count"] == 2
    assert set(two["solver_info"]) == {_PIN_A, second_pin}
    assert _figure_facts(two)["solver_count"] == 2
    one = _readmodel(tmp_path, two_pins[:2], "one")
    assert one["distinct_solver_count"] == 1


@pytest.mark.parametrize("declare_solver_split", [False, True])
@pytest.mark.parametrize("attr", ["date", "sha"])
def test_a_one_pin_two_builds_refusal_precedes_the_merge_and_every_write_and_no_flag_waives_it(
    tmp_path, monkeypatch, attr, declare_solver_split
):
    """Property: a one-pin disagreement stops ``combine_bundle`` before the root-tree check and the emit.

    No flag waives it.

    Class: disagreeing attribute (date or sha) crossed with ``declare_solver_split``. Shape row: Settlement 3
    ("sits in ``combine_bundle`` beside the compatibility gate ... before any write to the output path ... No
    flag waives it"). Why required: the emit's first write is the output directory, so a refusal that fires
    later leaves a partly written bundle, and a waiver would publish one series that mixes two builds.
    Killed by: moving the agreement call after the root-tree check (``_check_root_tree``) or into the
    read-model writer, or skipping it when ``declare_solver_split`` is true. Second correct
    implementation: running the agreement call before ``check_bundle_compatibility``, which also leaves
    nothing reached and stays green.
    Two-arm differential: the violating set (below) reaches neither step; the agreeing control set, positioned
    differently (one arm's attribute absent, not different), reaches both.
    Class: REGRESSION guard (a held test; adopted from the coder's scratch harness after evaluating that it
    asserts the required ORDER and not merely that a raise occurs).
    """
    differing = {"date": ("20260828", "20260901"), "sha": (_PIN_A, "b" * 40)}[attr]
    first = _bundle(tmp_path / "a1", _PIN_A, "tritonswmm", sha=_PIN_A, date="20260828")
    second = _bundle(
        tmp_path / "a2",
        _PIN_A,
        "triton_only",
        sha=differing[1] if attr == "sha" else _PIN_A,
        date=differing[1] if attr == "date" else "20260828",
    )
    reached: list[str] = []
    ids = iter(f"exp_{i}" for i in range(10))
    monkeypatch.setattr(C, "_experiment_id", lambda root: next(ids))
    monkeypatch.setattr(C, "_check_root_tree", lambda root: reached.append("root-check"))
    monkeypatch.setattr(C, "_emit_combined_bundle", lambda *a, **k: reached.append("emit"))
    monkeypatch.setattr(C.CombinedBundle, "from_directory", classmethod(lambda cls, p: "handle"))
    out = tmp_path / "combined_out"
    with pytest.raises(ConfigurationError, match="built from different solver commits"):
        C.combine_bundle([first, second], out, declare_solver_split=declare_solver_split)
    assert reached == [] and not out.exists()

    absent = _bundle(
        tmp_path / "a3",
        _PIN_A,
        "triton_only",
        sha=None if attr == "sha" else _PIN_A,
        date=None if attr == "date" else "20260828",
    )
    assert (
        C.combine_bundle(
            [first, absent], tmp_path / "control_out", declare_solver_split=declare_solver_split, unzipped=True
        )
        == "handle"
    )
    assert reached == ["root-check", "root-check", "emit"]
