"""Reviewer tests for work chunk ``1b.2`` — the group table and the run-level ``applicable`` gate.

The instrument under test is the strict path of ``hhemt.eda.cross_sim_identity.check_cross_sim_identity``:
one ``details`` row per ``(hardware family, DEM resolution)`` pair, plus the run-level gate that
reports ``applicable=False`` when nothing was evaluated. Each test names the finding it carries
under ``Finding:``.

The stub harness is duplicated from ``tests/test_eda_cross_sim_identity_partition_review.py``
rather than imported, so this module collects on a tree that predates that file and stays
independent of pytest's import-mode configuration. All of its callables are documented; the
duplication is a fixture, not a second statement of any rule.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from hhemt.eda import cross_sim_identity as csi

_DEPTH_MODE = "tritonswmm"
_LINK_MODE = "tritonswmm_swmm_link"
_BUMP = 1.0

#: The row's declared field set — shape §12.5 (i-a), plus the `members` and `reference_member_id`
#: fields the same row names.
_ROW_FIELDS = frozenset(
    {
        "group_family",
        "group_resolution",
        "members",
        "reference_member_id",
        "evaluated_tuples",
        "not_evaluated_modes",
    }
)

#: Keys the row must NOT carry, each for a consumer-derived reason rather than on taste.
_FORBIDDEN_ROW_KEYS = {
    "variable": 'the divergence count is len([d for d in details if "variable" in d])',
    "member": "compare_arms keys its own rows on `member`",
    "artifact": "compare_arms keys its own rows on `artifact`",
}


class _Cfg:
    """Only the compute attributes ``_ref_rank`` reads."""

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
    def __init__(self, target_dem_resolution):
        self.cfg_system = _SystemCfg(target_dem_resolution)


def _summaries(depth_bump: float = 0.0):
    """Flat summaries for the two modes; ``max_wlevel_m`` is float32 as on disk."""
    base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    depth = xr.Dataset(
        {"max_wlevel_m": (("event_iloc", "y", "x"), (base + np.float32(depth_bump))[None, :, :])},
        coords={"event_iloc": [0], "y": [0, 1], "x": [0, 1]},
    )
    link = xr.Dataset(
        {"max_flow_cms": (("event_iloc", "link_id"), np.array([[5.0, 6.0]], dtype="float64"))},
        coords={"event_iloc": [0], "link_id": ["c1", "c2"]},
    )
    return {_DEPTH_MODE: depth, _LINK_MODE: link}


class _Process:
    """``_MODE_CONFIG`` must be instance-reachable and an absent mode must raise."""

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


def _master(tmp_path, spec, no_summaries=()):
    """Build a stub master from ``{member_id: (cfg, resolution, depth_bump)}``.

    ``no_summaries`` names members whose summary tier is absent, which is the routine
    partially-complete state of this pipeline and the state the totality requirement exists for.
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
        datasets = {} if member_id in no_summaries else _summaries(bump)
        subs[member_id] = _Sub(cfg, datasets, d, resolution)
    root = tmp_path / "master"
    root.mkdir(parents=True, exist_ok=True)
    return _Master(subs, root)


def _group_rows(verdict):
    """The ``details`` rows that are group rows, identified by their own field set."""
    return [row for row in verdict.details if "group_family" in row and "group_resolution" in row]


def test_a_run_with_no_comparable_pair_is_not_evidence(tmp_path):
    """Finding A — a master with ONE member per group reports ``applicable`` True and claims agreement.

    Property: on a master whose every ``(family, resolution)`` group holds exactly one member
    (a resolution-only sweep, which the shape's own `dem_resolution` reporting set makes a real
    experiment shape), no two distinct members are ever compared. The verdict must therefore not
    present itself as evidence: ``applicable`` is ``False`` and the summary makes no bit-identity
    claim.

    Class: REGRESSION — this is RED on both the pre-chunk tree and the coder's tree. Pre-chunk it
    is red because ``applicable`` defaults True and no gate exists; on the coder's tree it is red
    because ``evaluated_tuples`` counts a member's SELF-compare, so the run-level sum is non-zero
    and the gate never fires. It is raised as an OPEN finding, not greened.

    Why this is required: shape §12.4's PC-0 states "compared-tuple count = 0 reported as
    NOT-EVALUATED, never passed=True", and §12.3 motivates the whole chunk with the measurement of
    a verdict reading "All tracked variables bit-identical ... across 24 compared members" over
    zero comparisons. A self-compare is that same zero-comparison state one vacuity source over,
    and the module's own comment already states the rule — "a self-compare is not a comparison" —
    while applying it only to the primary reference.

    What kills it: counting a tuple only when the member and its group reference are DIFFERENT
    members. Under the current code this test fails on the assertion below.

    A second correct implementation that still passes: one that keeps the self-compare counted but
    gates on the number of groups whose ``evaluated_tuples`` came from distinct members, provided
    the observable — ``applicable`` False and no bit-identity wording — is unchanged.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_0p35": (_Cfg("gpu", n_gpus=1), 0.35, 0.0),
            "gpu_1g_1p10": (_Cfg("gpu", n_gpus=1), 1.10, 0.0),
            "gpu_1g_3p5": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
        },
    )
    verdict = csi.check_cross_sim_identity(master).verdict

    # Non-vacuity control: the fixture really does hold one member per group and no two members
    # share a group, so there is no pair that COULD be compared.
    rows = _group_rows(verdict)
    assert rows, "the fixture produced no group rows, so the property below is vacuous"
    assert all(len(row["members"]) == 1 for row in rows), (
        f"the fixture must give every group exactly one member; got {[row['members'] for row in rows]}"
    )

    assert verdict.applicable is False, (
        "no two distinct members were compared, so this verdict is not evidence; "
        f"applicable={verdict.applicable!r}, summary={verdict.summary!r}, "
        f"evaluated_tuples={[row['evaluated_tuples'] for row in rows]}"
    )
    assert "bit-identical" not in verdict.summary.lower(), (
        f"the summary asserts agreement over zero comparisons: {verdict.summary!r}"
    )


def test_the_group_table_is_total_over_the_arms_pairs(tmp_path):
    """Finding B (closure of a round-1 erratum) — a rung whose only member has no summaries still gets a row.

    Property: the set of ``(group_family, group_resolution)`` keys in ``details`` equals the set of
    pairs the ARM carries, including a pair all of whose members lack summaries. A missing row and
    a zero row are different facts, and only the second one is a fact about the arm.

    Class: REGRESSION — red on the pre-chunk tree, where no group row exists at all.

    Why this is required: §12.5 (i-b) is a SET EQUALITY against "the arm's own pairs", and the
    round-1 erratum recorded that deriving that set from the present-summaries map makes the
    comparison self-satisfying. This test pins the arm-side population to the members the arm
    carries rather than to the members that happened to produce output.

    What kills it: seeding the table from the present-summaries map instead of from every member.

    A second correct implementation that still passes: one that emits the row with
    ``reference_member_id`` null and ``evaluated_tuples`` 0, or one that emits an explicit
    zero-tuple row with a named reason field — the assertion is over the KEY SET, not the row body.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_3p5_a": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_3p5_b": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_0p35_c": (_Cfg("gpu", n_gpus=1), 0.35, 0.0),
        },
        no_summaries=("gpu_1g_0p35_c",),
    )
    verdict = csi.check_cross_sim_identity(master).verdict

    arm_pairs = {
        (sub.cfg_analysis.run_mode and "gpu", sub._system.cfg_system.target_dem_resolution)
        for sub in master.sensitivity.members.values()
    }
    row_pairs = {(row["group_family"], row["group_resolution"]) for row in _group_rows(verdict)}

    assert row_pairs == arm_pairs, (
        f"the group table must be total over the arm's pairs; arm={sorted(arm_pairs)}, "
        f"rows={sorted(row_pairs)} — a missing row is a failure, a zero row is not"
    )


def test_the_group_row_carries_its_fields_and_none_of_the_forbidden_keys(tmp_path):
    """Finding C — the row's key set, checked against the three consumers that read it.

    Property: every group row carries all six declared fields, and carries none of ``variable``,
    ``member`` or ``artifact``.

    Class: NEW CAPABILITY — on the pre-chunk tree no group row exists, so the pre-fix run is a
    construction error. The discriminator is the plausible wrong implementation below.

    Plausible wrong implementation: a row that reuses ``member`` for its member list (which reads
    naturally and collides with ``compare_arms``), or one that carries the failing ``variable`` so a
    reader can see what diverged — the second is the one that silently inflates the divergence
    count, because that count is a substring test over the row.

    Why this is required: the divergence count is ``len([d for d in details if "variable" in d])``
    and is read both here and by the renderers; ``compare_arms`` keys its own rows on ``member`` and
    ``artifact``.

    What kills it: adding any of the three keys, or dropping any of the six declared fields.

    A second correct implementation that still passes: one that orders the row's fields differently,
    adds further disclosure fields, or names its member list ``group_members`` — the assertion is
    over presence and the three forbidden names, not over the exact key set.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_3p5_a": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_3p5_b": (_Cfg("gpu", n_gpus=1), 3.5, _BUMP),
            "gpu_1g_0p35_c": (_Cfg("gpu", n_gpus=1), 0.35, 0.0),
        },
    )
    verdict = csi.check_cross_sim_identity(master).verdict
    rows = _group_rows(verdict)
    assert rows, "no group row was emitted"

    for row in rows:
        missing = _ROW_FIELDS - set(row)
        assert not missing, f"group row is missing declared field(s) {sorted(missing)}: {row}"
        for key, why in _FORBIDDEN_ROW_KEYS.items():
            assert key not in row, f"group row carries forbidden key {key!r} ({why}): {row}"

    # The divergence count must not have absorbed the table: this fixture plants one within-group
    # divergence, so the count is 1 and not 1 + the number of group rows.
    divergences = [row for row in verdict.details if "variable" in row]
    assert len(divergences) == 1, (
        f"the divergence count must be unaffected by the group table; got {len(divergences)}: {divergences}"
    )


def test_a_zero_tuple_row_does_not_report_an_empty_skipped_mode_list(tmp_path):
    """Finding D — ``not_evaluated_modes`` is ``[]`` beside ``evaluated_tuples: 0`` on a summary-less rung.

    Property: a group row whose ``evaluated_tuples`` is 0 must not ALSO report an empty
    ``not_evaluated_modes``. The empty list means "nothing was skipped"; beside a zero count it
    means "everything was skipped", so the two fields contradict each other and the skimmer reads
    the empty list as a clean row.

    Class: REGRESSION — red on the coder's tree, and it is raised as an OPEN finding, not greened.

    Why this is required: the coder's own comment states this exact invariant — "two fields of one
    row saying opposite things, and the empty list is the reading a skimmer takes as 'every mode
    was evaluated'" — and the mitigation they wrote reaches only the case where the group's
    reference is present in the reference map. On a rung whose only member has no summaries the
    map has no entry, the seed is skipped, and the empty list ships.

    What kills it: seeding ``not_evaluated_modes`` from the group's REFERENCE where one exists and
    otherwise from the modes of the group's members, or reporting null.

    A second correct implementation that still passes: one that reports ``None`` (JSON null) for an
    unknowable mode set, or one that names the skipped reason in a separate field — the assertion
    forbids only the empty-list-beside-zero combination.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_3p5_a": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_3p5_b": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_0p35_c": (_Cfg("gpu", n_gpus=1), 0.35, 0.0),
        },
        no_summaries=("gpu_1g_0p35_c",),
    )
    verdict = csi.check_cross_sim_identity(master).verdict
    zero_rows = [row for row in _group_rows(verdict) if row["evaluated_tuples"] == 0]
    assert zero_rows, (
        "the fixture must leave at least one group with a zero evaluated-tuple count, "
        f"or the property below is vacuous; rows={_group_rows(verdict)}"
    )
    for row in zero_rows:
        assert row["not_evaluated_modes"] != [], (
            f"a group with members but zero evaluated tuples reports an EMPTY skipped-mode list, "
            f"which reads as 'every mode was evaluated': {row}"
        )


def test_passed_is_not_repurposed_by_the_applicability_gate(tmp_path):
    """Finding E (HELD) — ``passed`` keeps its meaning on both arms of the vacuity question.

    Property: on a run with zero real comparisons ``passed`` is still ``True`` (a conjunction over
    no conjuncts is vacuously true, and ``applicable`` is the field that carries "this is not
    evidence"); on a run with a genuine within-group divergence ``passed`` is ``False``.

    Class: HELD-CHARACTERIZATION — green on the pre-chunk tree and required to REMAIN green. Named
    mutation that would red it: setting ``passed = False`` in the applicability branch, or
    ``passed = True`` unconditionally.

    Why this is required: it is the guard on the SHAPE of the finding-A fix. A fix that made this
    run look inapplicable by flipping ``passed`` would also flip
    ``ValidationReport.overall_passed`` and would assert "a comparison disagreed", which is a
    different and stronger claim than "no comparison was made". 1b.3(a) states the ``passed=True``
    half explicitly.

    A second correct implementation that still passes: any implementation that leaves ``passed``
    derived from ``all_identical`` alone, whatever it does to ``applicable`` and the summary.
    """
    vacuous = csi.check_cross_sim_identity(
        _master(
            tmp_path,
            {
                "gpu_1g_0p35": (_Cfg("gpu", n_gpus=1), 0.35, 0.0),
                "gpu_1g_3p5": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            },
        )
    ).verdict
    assert vacuous.passed is True, (
        f"passed must stay a conjunction over EVALUATED tuples and is vacuously true here; "
        f"got {vacuous.passed!r} (summary {vacuous.summary!r})"
    )

    diverging = csi.check_cross_sim_identity(
        _master(
            tmp_path / "diverging",
            {
                "gpu_1g_3p5_a": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
                "gpu_1g_3p5_b": (_Cfg("gpu", n_gpus=1), 3.5, _BUMP),
            },
        )
    ).verdict
    assert diverging.passed is False, (
        f"a within-group divergence must still fail the verdict; got passed={diverging.passed!r}"
    )
