"""Coder tests for package ``1b`` — the cross-sim identity instrument's three defects.

The file under test is ``hhemt.eda.cross_sim_identity``, the instrument that answers *did the
same physics run twice produce the same numbers*, plus the read path
(``analysis_validation._read_persisted_eda_verdicts``) that carries its verdict to the sidebars. It
had three independent ways of reporting — or failing to report — a reproducibility result it had not
measured:

1. **No resolution partition.** Members were partitioned by hardware family alone, so a GPU arm
   carrying 3.5 / 1.10 / 0.35 m members inside ONE family compared across a resolution boundary
   and reported the resolution difference as a reproducibility failure. Different resolutions are
   different physics; the comparison has to be drawn within a ``(hardware family, DEM resolution)``
   group.
2. **The false green.** The aggregate verdict collapsed zero comparisons to PASS, so a run that
   compared nothing reported success in wording byte-identical to a measured agreement. The third
   applicability state — ``applicable=False`` — is what distinguishes "not evidence" from
   "evidence that agreed".
3. **The state did not survive the round trip.** Defect 2's fix is only worth what survives to the
   reader: the persisted verdict is a ``dataclasses.asdict(CheckResult)``, and the in-tree reader
   rebuilt the object from a SUBSET of its fields, leaving ``applicable``, ``instrument`` and
   ``detection_floor`` at their defaults. The artifact then said N/A while the sidebar rendered a
   green row — the same false green, one hop downstream, and the reason a package that closes it at
   the writer must also close it at the reader.

Five tests, because two of the three defects need more than one arm: the gate's POSITIVE arm exists
because a one-sided test cannot tell a correct gate from one that is stuck
(``test_a_run_with_a_real_comparison_stays_applicable``), and the reader's TOLERANCE arm exists
because the fix can itself delete rows from older artifacts
(``test_a_payload_written_before_the_carried_fields_still_loads``).

These tests are the PACKAGE's own. The reviewing seat's modules
(``tests/test_eda_cross_sim_identity_partition_review.py`` and
``tests/test_eda_cross_sim_identity_group_table_review.py``) are separate by construction — a test
authored by the party that wrote the code checks a different failure mode from one authored by the
party that reviews it — and nothing here duplicates their assertions. Where a property is already
carried there, the test below names it rather than restating it.

Every test is stub-only: no solver build, no HPC, ``tmp_path`` alone. The stub surface is the
three attributes ``check_cross_sim_identity`` touches on a sub (``cfg_analysis`` for ``_ref_rank``,
``process._MODE_CONFIG`` + ``_retrieve_combined_output`` for ``_enabled_modes``,
``analysis_paths.analysis_dir`` for the emitted provenance sources), one it touches on the master
(``cfg_analysis.toggle_sensitivity_analysis`` and ``sensitivity.members``), and ``_system.cfg_system``
— which is where a real ``TRITONSWMM_analysis`` keeps ``target_dem_resolution``, and the omission of
which would make every resolution unreadable and silently collapse the partition (see
``tests/test_eda_cross_sim_identity_partition_review.py``, whose docstring records that trap).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import xarray as xr

from hhemt.analysis_validation import CheckResult, _read_persisted_eda_verdicts
from hhemt.eda import cross_sim_identity as csi

_DEPTH_MODE = "tritonswmm"
_LINK_MODE = "tritonswmm_swmm_link"

#: One float32 ULP at 1.0 — the magnitude a cross-backend float32 summary difference takes, and the
#: magnitude this module uses to make two members differ by the SMALLEST amount an identity check
#: can see. Pinning the smallest value is deliberate: a comparison drawn across a resolution
#: boundary must be refused at ANY magnitude, and a gross perturbation would let a tolerance-based
#: mistake pass for a partition.
_ULP32 = float(np.finfo(np.float32).eps)


class _Cfg:
    """Minimal stand-in for a sub's ``cfg_analysis`` — only the compute attrs ``_ref_rank`` reads."""

    def __init__(self, run_mode, n_gpus=0, n_mpi_procs=0, n_omp_threads=0, n_nodes=0):
        self.run_mode = run_mode
        self.n_gpus = n_gpus
        self.n_mpi_procs = n_mpi_procs
        self.n_omp_threads = n_omp_threads
        self.n_nodes = n_nodes


class _SystemCfg:
    def __init__(self, target_dem_resolution):
        self.target_dem_resolution = target_dem_resolution


class _System:
    """The LIVE attribute path: ``TRITONSWMM_analysis._system.cfg_system``."""

    def __init__(self, target_dem_resolution):
        self.cfg_system = _SystemCfg(target_dem_resolution)


def _summaries(depth_bump: float = 0.0, *, with_depth: bool = True):
    """The flat per-scenario summaries a sub returns.

    ``with_depth=False`` yields ONLY the link summary, and that variant is the fixture for the
    false green: the link summary carries ``max_flow_cms`` — a ``max`` PREFIX column with no
    ``_last`` partner and no raster grid — so ``compared_columns_for`` routes it to its
    fail-closed third branch and the run compares NOTHING. That is a different vacuity source from
    "no two members share a group" and from "a member has no summaries at all", which is why it can
    carry the same property without restating either test that already holds one.
    """
    out: dict[str, xr.Dataset] = {}
    if with_depth:
        base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        out[_DEPTH_MODE] = xr.Dataset(
            {"max_wlevel_m": (("event_iloc", "y", "x"), (base + np.float32(depth_bump))[None, :, :])},
            coords={"event_iloc": [0], "y": [0, 1], "x": [0, 1]},
        )
    out[_LINK_MODE] = xr.Dataset(
        {"max_flow_cms": (("event_iloc", "link_id"), np.array([[5.0, 6.0]], dtype="float64"))},
        coords={"event_iloc": [0], "link_id": ["c1", "c2"]},
    )
    return out


class _Process:
    """Stands in for ``TRITONSWMM_analysis_post_processing``.

    ``_MODE_CONFIG`` must be reachable via the live ``.process`` instance (never module-level), and
    ``_retrieve_combined_output`` must raise ``FileNotFoundError`` on an absent mode — that is the
    existence guard ``_enabled_modes`` keys on, and it is what lets a fixture enable one mode and
    not the other.
    """

    _MODE_CONFIG = {_DEPTH_MODE: None, _LINK_MODE: None}

    def __init__(self, datasets):
        self._datasets = datasets

    def _retrieve_combined_output(self, mode):
        if mode not in self._datasets:
            raise FileNotFoundError(mode)
        return self._datasets[mode]


class _Paths:
    def __init__(self, d):
        self.analysis_dir = d


class _Sub:
    def __init__(self, cfg, datasets, d, resolution):
        self.cfg_analysis = cfg
        self.process = _Process(datasets)
        self.analysis_paths = _Paths(d)
        self._system = _System(resolution)


class _MasterCfg:
    toggle_sensitivity_analysis = True


class _Sens:
    def __init__(self, subs):
        self.members = subs


class _Master:
    def __init__(self, subs, d):
        self.cfg_analysis = _MasterCfg()
        self.sensitivity = _Sens(subs)
        self.analysis_paths = _Paths(d)


def _master(tmp_path, spec, no_depth=()):
    """Build a stub sensitivity master from ``{member_id: (cfg, resolution, depth_bump)}``.

    ``no_depth`` names members whose depth summary is absent, which leaves only the link summary
    enabled for them. That is the state the false-green test needs and it is reached through the
    same existence guard (``_enabled_modes``) a real partially-processed member takes, rather than
    by editing ``_MODE_CONFIG``.

    Each sub gets a real on-disk ``analysis_dir`` because the emitted provenance source path is
    ``{member_dir}/analysis_datatree.zarr``; that path need not exist (the source gate accepts a
    non-existent path, and the ``.zarr`` suffix clears the directory-as-source clause). Both summary
    KINDS are materialized because ``target_processed_output_type`` defaults to ``zarr`` and the
    emitter refuses an empty source set.
    """
    subs = {}
    for member_id, (cfg, resolution, bump) in spec.items():
        d = tmp_path / member_id
        d.mkdir(parents=True, exist_ok=True)
        proc = d / "sims" / "s0" / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        (proc / "TRITONSWMM_TRITON_summary.nc").touch()
        zarr_store = proc / "TRITONSWMM_SWMM_link_summary.zarr"
        zarr_store.mkdir(exist_ok=True)
        (zarr_store / ".zgroup").write_text("{}")
        datasets = _summaries(bump, with_depth=member_id not in no_depth)
        subs[member_id] = _Sub(cfg, datasets, d, resolution)
    root = tmp_path / "master"
    root.mkdir(parents=True, exist_ok=True)
    return _Master(subs, root)


def _group_rows(verdict):
    """The group rows in ``details``, identified by the two keys only they carry."""
    return [row for row in verdict.details if "group_family" in row and "group_resolution" in row]


def test_a_run_that_compares_nothing_is_not_evidence(tmp_path):
    """Defect 2 (REGRESSION) — zero comparisons must not read as a measured agreement.

    Property: on a sensitivity master where every enabled mode routes to ``compared_columns_for``'s
    fail-closed branch — so no ``(member, mode, variable)`` tuple is evaluated at all — the verdict
    reports ``applicable is False``, a summed ``evaluated_tuples`` of 0, and a summary containing no
    bit-identity claim; and it reports ``passed is True``.

    Class: REGRESSION. Red on the tree before package ``1b`` — the summary read "All tracked
    variables bit-identical within every … group across N compared members" over zero comparisons
    and ``applicable`` defaulted ``True``. The failing output is pasted in the round file against
    the pre-chunk module.

    Why it is required, as a consequence a reader can check against the code: ``passed`` is a
    conjunction over the tuples that were EVALUATED, so it is vacuously true over none; the field
    that answers "is this verdict evidence at all" is ``applicable``, which the renderer greys and
    shows as ``N/A``. Collapsing the two is what produced a green row on a surface an acceptance
    decision rests on.

    What kills it: making the summary assertion depend only on ``all_identical`` — reverting the
    run-level gate — reds the ``applicable`` and ``bit-identical`` assertions; forcing
    ``passed = False`` in the vacuous branch reds the ``passed`` assertion, which is the reason the
    ``passed`` conjunct is asserted here rather than left to the reviewer's held test.

    A second correct implementation that still passes: one that tests the count directly
    (``if not any(row["evaluated_tuples"] for row in group_rows)``) rather than summing it, or one
    that emits a distinct ``N/A`` summary wording — this test asserts the CONJUNCTION of the four
    observables, not the branch structure or the sentence.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_3p5_a": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_3p5_b": (_Cfg("gpu", n_gpus=1), 3.5, _ULP32),
        },
        no_depth=("gpu_1g_3p5_a", "gpu_1g_3p5_b"),
    )
    verdict = csi.check_cross_sim_identity(master).verdict

    # NON-VACUITY CONTROL, stated on evidence present in BOTH the pre-chunk and the fixed verdicts
    # so that the assertions that follow fail on the CLAIM rather than on a missing construct: the
    # fixture must have produced rows that disclose a NOT-EVALUATED mode, and no row asserting a
    # comparison. Without this the zero-comparison state below would be an assumption.
    assert any(row.get("verdict") == csi.NOT_EVALUATED for row in verdict.details), (
        "the fixture must route every enabled mode to the fail-closed derivation, so a "
        f"NOT-EVALUATED row is expected; details={verdict.details}"
    )
    assert not [row for row in verdict.details if "variable" in row], (
        f"the fixture must compare nothing, yet a divergence row exists: {verdict.details}"
    )

    assert verdict.applicable is False, (
        "a verdict over zero evaluated tuples is not evidence and must say so; "
        f"applicable={verdict.applicable!r}, summary={verdict.summary!r}"
    )
    assert "bit-identical" not in verdict.summary.lower(), (
        f"the summary makes a bit-identity claim over zero comparisons: {verdict.summary!r}"
    )
    assert verdict.passed is True, (
        "passed is a conjunction over EVALUATED tuples and is vacuously true here; flipping it "
        "would assert that a comparison disagreed, which is a different and stronger claim, and "
        f"would red ValidationReport.overall_passed. got passed={verdict.passed!r}"
    )

    # The run-level count, asserted LAST because it is the construct this package ADDS: it belongs
    # in this test, but ordering it first would make the pre-chunk run fail on its absence rather
    # than on the false green, which is the defect.
    rows = _group_rows(verdict)
    assert rows, f"no group row was emitted: details={verdict.details}"
    assert sum(row["evaluated_tuples"] for row in rows) == 0, (
        "the disclosed count must agree that nothing was compared; got "
        f"evaluated_tuples={[row['evaluated_tuples'] for row in rows]}"
    )


def test_two_resolutions_are_never_placed_in_one_group(tmp_path):
    """Defect 1 (REGRESSION) — the group table separates members by DEM resolution.

    Property: on a GPU master whose two members differ ONLY in ``target_dem_resolution`` (3.5 m vs
    0.35 m), the emitted group table holds TWO rows for that hardware family, each resolving one
    member, and the two members are not in the same row.

    Class: NEW CAPABILITY, and the class is the standard's, not a preference. Every assertion here
    is over ``group_family`` / ``group_resolution`` — a construct that does not exist before package
    ``1b`` — so the pre-chunk run is a CONSTRUCTION ERROR and carries no information as a
    regression signal (a missing row reds this test whatever the partition does). The discriminator
    is therefore the plausible wrong implementation below, not the pre-chunk tree.

    Plausible wrong implementation: keying the table on the hardware family alone — the
    pre-``1b.1`` key. That implementation emits ONE row for this fixture, with both members in
    ``members`` and one resolution named in ``group_resolution``; it reds the row count, the
    resolution set, and the distinct-row assertion together.

    Why it is required: shape §12.5 names "two members differing only in
    ``system.target_dem_resolution`` are not placed in the same group" as this package's own test
    obligation, and the group table is the surface a reader checks it on — an implementation that
    partitioned correctly but reported the groups pooled would pass every comparison-side test and
    still tell a reader the arm had one group.

    What kills it: collapsing the key back to the family, or emitting one row per family with a
    list of resolutions rather than one row per pair.

    A second correct implementation that still passes: one that spells the resolution differently on
    the row (``"3.5"`` rather than ``3.5``) — the assertion resolves each declared resolution through
    ``float()``, so a numerics-equal spelling passes — and one that keys on a pre-rendered
    ``"{family}@{resolution}"`` string, because the assertions are over the pair each row declares,
    not over how the table was keyed.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_0p35": (_Cfg("gpu", n_gpus=1), 0.35, 0.0),
            "gpu_1g_3p5": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
        },
    )
    verdict = csi.check_cross_sim_identity(master).verdict
    rows = _group_rows(verdict)

    assert len(rows) == 2, (
        f"two members at different DEM resolutions are two groups, not one; got {len(rows)} row(s): {rows}"
    )
    assert {row["group_family"] for row in rows} == {"gpu"}, f"both members share a hardware family: {rows}"
    assert {float(row["group_resolution"]) for row in rows} == {0.35, 3.5}, (
        f"the table must name both resolutions, not pool them: {rows}"
    )

    row_of = {member: index for index, row in enumerate(rows) for member in row["members"]}
    assert row_of["gpu_1g_0p35"] != row_of["gpu_1g_3p5"], (
        f"members differing ONLY in resolution were placed in one group: {rows}"
    )


def test_a_run_with_a_real_comparison_stays_applicable(tmp_path):
    """The gate's POSITIVE arm (REGRESSION) — a genuine comparison must still count as evidence.

    Property: on a GPU master whose two members share a hardware family AND a resolution and agree
    exactly, the verdict reports ``applicable is True``, a summed ``evaluated_tuples`` of at least
    1, and ``passed is True``.

    Class: NEW CAPABILITY on its count half, for the same reason as the table test above:
    ``evaluated_tuples`` does not exist before package ``1b``, so the pre-chunk run is a
    construction error and carries no information there. Its ``applicable is True`` half IS
    both-states, and that half is what makes this test's discriminator the gate rather than the
    table.

    Plausible wrong implementation: ``applicable = False`` unconditionally, or an
    ``evaluated_tuples`` field that is always 0. Both red the count assertion; the first also reds
    ``applicable is True`` here while the defect-2 test above stays green, which is exactly the pair
    of independent failures this test exists to supply.

    Why it is required, and why it is a SEPARATE test rather than a second assertion inside the
    defect-2 test: a one-sided test cannot distinguish a correct gate from a gate that is stuck.
    The defect-2 test forbids ``applicable`` being stuck True; this one forbids it being stuck
    False, and the two fail INDEPENDENTLY.

    What kills it: ``applicable = False`` unconditionally, or a tuple count that never increments.
    The complementary mutation — ``applicable = True`` unconditionally — is killed by the defect-2
    test above.

    A second correct implementation that still passes: one that counts per EVENT rather than per
    ``(member, mode, variable)``, or that derives the count from the length of the emitted
    diff/identical arrays — this test asserts a LOWER BOUND of 1, deliberately, so the unit's
    granularity is not over-constrained here and stays the shape's to fix.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_3p5_a": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_3p5_b": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
        },
    )
    verdict = csi.check_cross_sim_identity(master).verdict

    rows = _group_rows(verdict)
    assert rows, f"no group row was emitted: details={verdict.details}"
    total = sum(row["evaluated_tuples"] for row in rows)
    assert total >= 1, (
        "two members at the same family and resolution MUST be compared, so the run holds at "
        f"least one evaluated tuple; got {total} across rows={rows}"
    )
    assert verdict.applicable is True, (
        "a run that really compared two distinct members IS evidence; "
        f"applicable={verdict.applicable!r}, summary={verdict.summary!r}"
    )
    assert verdict.passed is True, f"the two members agree exactly, so the verdict must pass: {verdict.summary!r}"


def test_the_third_applicability_state_survives_the_persisted_round_trip(tmp_path):
    """Defect 3 (REGRESSION) — a persisted ``applicable: False`` must be re-read as ``False``.

    Property: over a verdict JSON written as ``dataclasses.asdict(CheckResult)`` with
    ``applicable=False``, ``instrument="summary_tier"`` and a float ``detection_floor``, the reader
    returns a ``CheckResult`` carrying all three unchanged — and still carrying the five fields it
    already read, so the test cannot pass by the reader failing wholesale.

    Class: REGRESSION. Red on the tree before this chunk, where the rebuild named five of the
    dataclass's fields and left ``applicable``, ``instrument`` and ``detection_floor`` at their
    defaults. Output pasted in the round file.

    Why it is required, as a consequence a reader can check against the code: each dropped field's
    DEFAULT is a claim rather than an absence. ``applicable=True`` re-labels a persisted "not
    evidence" as applicable, and ``errors_and_warnings._partition_applicable`` splits the sidebar on
    exactly that field — so the artifact says N/A and the rendered row says green. The other two
    erase the instrument provenance the same dataclass carries to keep a summary-tier pass from
    rendering identically to a raw-raster one.

    What kills it: rebuilding from the subset — the pre-chunk body — leaves ``applicable`` at
    ``True`` and reds this test, with the other two assertions red beside it.

    A second correct implementation that still passes: one that constructs the dataclass through
    ``CheckResult(**{k: payload[k] for k in (...)} )`` over an explicit field list, or one that
    drops and re-adds fields via ``dataclasses.replace`` — the assertion is over the three values
    returned, not over how the rebuild spells them.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_3p5_a": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_3p5_b": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
        },
    )
    eda_dir = Path(master.analysis_paths.analysis_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    floor = float(np.finfo(np.float32).eps)
    persisted = CheckResult(
        name="Cross-sim byte-identity",
        level="aggregate",
        passed=True,
        summary="N/A — no (member, mode, variable) tuple was evaluated",
        details=[],
        applicable=False,
        instrument="summary_tier",
        detection_floor=floor,
    )
    (eda_dir / "eda_cross_sim_identity.verdict.json").write_text(json.dumps(dataclasses.asdict(persisted), default=str))

    read = _read_persisted_eda_verdicts(master)
    assert len(read) == 1, f"the fixture's one verdict must load; got {len(read)}: {read}"
    got = read[0]

    # CONTROL: the fields the pre-chunk rebuild already carried. Without this the test would pass
    # against a reader that dropped everything and returned a default-constructed CheckResult.
    assert got.passed is True and got.summary == persisted.summary, (
        f"the reader must still carry the leading fields: passed={got.passed!r}, summary={got.summary!r}"
    )

    assert got.applicable is False, (
        "a persisted `applicable: False` was re-read as applicable — the sidebar then renders a "
        f"green row where the artifact records N/A. got applicable={got.applicable!r}"
    )
    assert got.instrument == "summary_tier", (
        f"the instrument provenance was dropped, so the detection floor becomes unreadable: got {got.instrument!r}"
    )
    assert got.detection_floor == floor, (
        f"the detection floor was dropped, erasing what the verdict could see: got {got.detection_floor!r}"
    )


def test_a_payload_written_before_the_carried_fields_still_loads(tmp_path):
    """Defect 3's tolerance arm (HELD-CHARACTERIZATION) — an older artifact must not VANISH.

    Property: over a verdict JSON carrying only ``name``, ``level``, ``passed``, ``summary`` and
    ``details`` — the shape a pre-existing artifact has — the reader still returns ONE
    ``CheckResult``, with the three carried fields at the dataclass's own defaults.

    Class: HELD-CHARACTERIZATION. Green before this chunk and required to stay green; the named
    mutation that would red it is below.

    Why it is required, as a consequence a reader can check against the code: the reader's
    ``except (OSError, KeyError, ValueError): continue`` treats a missing key as an UNREADABLE
    artifact and skips it. So a fix that carried the three fields with SUBSCRIPTION
    (``payload["applicable"]``) would not merely default them for old artifacts — it would delete
    the verdict from the sidebars, which is the same silent-drop failure class this package exists
    to close, arriving through the fix rather than around it.

    What kills it: subscripting any of the three instead of ``.get`` — the read then returns ``[]``
    and the count assertion fails. ``dataclasses.asdict`` always emits all eight keys, so this is
    reachable only from artifacts written before the fields existed, which is exactly the
    population that must keep loading.

    A second correct implementation that still passes: one that validates against a field allowlist
    and fills the missing entries from ``dataclasses.fields(CheckResult)`` defaults explicitly,
    rather than relying on the constructor's defaults — the assertion is over the loaded row and its
    values, not over the mechanism.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_3p5_a": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_3p5_b": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
        },
    )
    eda_dir = Path(master.analysis_paths.analysis_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    legacy = {
        "name": "Cross-sim byte-identity",
        "level": "aggregate",
        "passed": True,
        "summary": "All tracked variables bit-identical within every group.",
        "details": [],
    }
    (eda_dir / "eda_cross_sim_identity.verdict.json").write_text(json.dumps(legacy))

    read = _read_persisted_eda_verdicts(master)
    assert len(read) == 1, (
        "an artifact written before the carried fields existed must still LOAD; a reader that "
        "subscripts those keys raises KeyError, the except clause swallows it, and the verdict "
        f"DISAPPEARS from the sidebars. got {len(read)} row(s): {read}"
    )
    got = read[0]
    assert got.applicable is True, f"the dataclass default for a legacy artifact must apply: got {got.applicable!r}"
    assert got.instrument is None and got.detection_floor is None, (
        "absent fields must default to None, not to a fabricated value: "
        f"got {got.instrument!r} / {got.detection_floor!r}"
    )
