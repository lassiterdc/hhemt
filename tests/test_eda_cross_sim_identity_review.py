"""Reviewer-seat tests for the per-artifact-family compared-set derivation.

THE PROBLEM THIS FILE SOLVES. ``eda/cross_sim_identity.py`` is the instrument that answers
*did the resumed run reproduce the clean run bit-for-bit*. An instrument that reports
agreement it did not measure is worse than no instrument, because a reader cannot tell the
two apart from the verdict: the summary sentence a zero-comparison run emits is byte-identical
to the one a fully-compared agreeing run emits. The approved shape closes that hole with one
conjunct -- `G1`, NON-EMPTY, which it states as UNIVERSAL and evaluated PER COMPARED ARTIFACT,
with an empty compared set resolving to NOT-EVALUATED and never to a pass. These tests hold the
module to that conjunct at BOTH of its consumer sites, and hold the newly-widened TRITON
compared set inside the hardware-family boundary its own module docstring says the comparison
is anchored within.

This file is the REVIEWER's write set for Build work chunk 1 and contains no production code.
The coder's tests live in ``tests/test_eda_cross_sim_identity.py``; nothing here imports from
that module, deliberately -- a reviewer test coupled to the coder's fixtures reds when the
coder refactors a helper, which reports on the helper rather than on the behaviour.

Deliberately NOT built on the real on-disk summary stores: the stub surface below is the three
attributes ``check_cross_sim_identity`` touches on a member and the two it touches on a master,
so every test here resolves to ``tmp_path`` alone. Reaching for a fixture that builds a real
analysis would pull in the compile-bearing fixture family, which is prohibited on this host.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hhemt.eda.cross_sim_identity import NOT_EVALUATED, check_cross_sim_identity, compared_columns_for

# ---------------------------------------------------------------------------
# The stub surface. Five attributes total, named where each is read.
# ---------------------------------------------------------------------------

#: An arbitrary mode key. The real `_MODE_CONFIG` keys are irrelevant here because
#: `_enabled_modes` iterates whatever `.process._MODE_CONFIG` exposes, so a stub key
#: exercises the same code path without binding these tests to the production mode set.
_MODE = "stub_mode"

#: The eight data_vars of the real `TRITONSWMM_TRITON_summary`, measured on the local run
#: cache. Exactly ONE of the eight (`max_wlevel_m`) was reachable by the retired hand-named
#: floor; the other seven are what the all-vars derivation newly admits, and three of those
#: seven are argmax-derived and therefore tie-break candidates across compute configurations.
_TRITON_VARS = (
    "final_surface_flood_volume_m3",
    "max_velocity_mps",
    "max_wlevel_m",
    "time_of_max_velocity_min",
    "time_of_max_wlevel_min",
    "velocity_x_mps_at_time_of_max_velocity",
    "velocity_y_mps_at_time_of_max_velocity",
    "wlevel_m_last_tstep",
)

#: The seven the hand-named floor could never reach. Membership is derived rather than
#: transcribed so that adding a variable to `_TRITON_VARS` cannot silently leave it untested.
_NEWLY_ADMITTED = tuple(v for v in _TRITON_VARS if v != "max_wlevel_m")


class _Cfg:
    """The five compute attributes `_ref_rank` and `_b4b_family_key` read off a member."""

    def __init__(self, run_mode, n_gpus=0, n_mpi_procs=0, n_omp_threads=0, n_nodes=0):
        self.run_mode = run_mode
        self.n_gpus = n_gpus
        self.n_mpi_procs = n_mpi_procs
        self.n_omp_threads = n_omp_threads
        self.n_nodes = n_nodes


def _unroutable_summary(bump: float = 0.0) -> xr.Dataset:
    """A perf-shaped artifact: no ``_max``/``_last`` pair, no ``x``/``y`` grid.

    Takes the fail-closed third branch and yields the empty compared set. ``bump`` shifts
    EVERY column, so a member built with a non-zero bump differs from the reference in every
    variable it carries -- the input on which an instrument that measures nothing is
    indistinguishable from one that measured and agreed.
    """
    return xr.Dataset(
        {
            name: (("event_iloc",), np.array([1.0 + i + bump], dtype="float64"))
            for i, name in enumerate(("Total", "Simulation", "Init"))
        },
        coords={"event_iloc": [0]},
    )


def _pair_summary(bump: float = 0.0) -> xr.Dataset:
    """A SWMM-shaped artifact carrying one ``_max``/``_last`` pair, so it routes to the
    pair-signature branch and yields a NON-empty compared set.

    This is the control side of the differential below: a non-empty compared set is a state in
    which an unqualified bit-identity verdict is CORRECT, and an assertion that fires on it has
    encoded "always disclose" rather than the conjunct `G1` actually states.
    """
    return xr.Dataset(
        {
            "flow_cms_max": (("event_iloc", "link_id"), np.array([[1.0 + bump, 2.0]], dtype="float64")),
            "flow_cms_last": (("event_iloc", "link_id"), np.array([[3.0, 4.0]], dtype="float64")),
        },
        coords={"event_iloc": [0], "link_id": ["c1", "c2"]},
    )


def _triton_summary(perturb: str | None = None, delta: float = 1.0) -> xr.Dataset:
    """A TRITON-shaped artifact on the ``x``/``y`` raster grid, so it routes to all-vars.

    ``perturb`` names ONE variable to shift by ``delta``. Shifting one variable rather than all
    of them is what makes a per-variable claim checkable: the detail rows name the variable, so
    a test can assert WHICH variable reached the verdict and not merely that something did.
    """
    data = {}
    for i, name in enumerate(_TRITON_VARS):
        value = 10.0 + i + (delta if name == perturb else 0.0)
        data[name] = (("event_iloc", "y", "x"), np.array([[[value, value + 0.5]]], dtype="float64"))
    return xr.Dataset(data, coords={"event_iloc": [0], "y": [0], "x": [0, 1]})


class _Process:
    """Stands in for ``TRITONSWMM_analysis_post_processing``.

    ``_MODE_CONFIG`` is read off the live ``.process`` instance and ``_retrieve_combined_output``
    raises ``FileNotFoundError`` on an absent mode, because that raise IS the existence guard
    ``_enabled_modes`` keys on -- returning ``None`` instead would make every mode read as
    present and the member as having summaries it does not have.
    """

    _MODE_CONFIG = {_MODE: None}

    def __init__(self, dataset):
        self._datasets = {_MODE: dataset}

    def _retrieve_combined_output(self, mode):
        if mode not in self._datasets:
            raise FileNotFoundError(mode)
        return self._datasets[mode]


class _Paths:
    def __init__(self, analysis_dir):
        self.analysis_dir = analysis_dir


class _Member:
    def __init__(self, cfg, dataset, analysis_dir):
        self.cfg_analysis = cfg
        self.process = _Process(dataset)
        self.analysis_paths = _Paths(analysis_dir)


class _MasterCfg:
    toggle_sensitivity_analysis = True


class _Sensitivity:
    def __init__(self, members):
        self.members = members


class _Master:
    def __init__(self, members, analysis_dir):
        self.cfg_analysis = _MasterCfg()
        self.sensitivity = _Sensitivity(members)
        self.analysis_paths = _Paths(analysis_dir)


def _master(tmp_path, spec: dict) -> _Master:
    """Build a stub sensitivity master from ``{member_id: (cfg, dataset)}``.

    Each member gets a real on-disk directory carrying one ``.nc`` file and one ``.zarr``
    directory under its ``sims/s0/processed/`` tree, because the emitted provenance artifact
    globs that layout for its declared source paths and an empty glob fails the non-empty
    source-path gate. The file CONTENTS are never read -- ``_Process`` hands the comparison its
    datasets directly -- so a touch and an empty group marker are sufficient.
    """
    members = {}
    for member_id, (cfg, dataset) in spec.items():
        member_dir = tmp_path / member_id
        processed = member_dir / "sims" / "s0" / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        (processed / "stub_summary.nc").touch()
        store = processed / "stub_other_summary.zarr"
        store.mkdir(exist_ok=True)
        (store / ".zgroup").write_text("{}")
        members[member_id] = _Member(cfg, dataset, member_dir)
    root = tmp_path / "master"
    root.mkdir(parents=True, exist_ok=True)
    return _Master(members, root)


def _discloses_non_evaluation(verdict) -> bool:
    """True iff this verdict tells a reader that something was NOT evaluated.

    DISJUNCTIVE over the three shapes a correct implementation may take, because the shape
    settles THAT an empty compared set is not a pass and does not settle HOW the refusal is
    surfaced: `compare_arms` already does it two ways in one function (a per-artifact
    NOT-EVALUATED detail row, and a whole-comparison NOT-EVALUATED verdict when nothing was
    compared at all). An assertion naming only one of those shapes would red on a correct
    implementation that chose the other, which is the upper-bound failure a reviewer test must
    not commit.
    """
    if verdict.passed is not True:
        return True
    if NOT_EVALUATED in str(verdict.summary):
        return True
    return any(NOT_EVALUATED in str(row.get(key, "")) for row in verdict.details for key in ("verdict", "detail"))


# ---------------------------------------------------------------------------
# FINDING A -- the `G1` conjunct is honoured at one consumer site and not the other
# ---------------------------------------------------------------------------


def test_a_zero_comparison_run_must_not_return_an_unqualified_bit_identity_verdict(tmp_path):
    """`G1` is UNIVERSAL, so a compared set of nothing is NOT-EVALUATED at EVERY consumer.

    Property: for every artifact family whose compared set is empty, the verdict
    ``check_cross_sim_identity`` returns discloses that nothing was evaluated -- quantified over
    the family of artifacts the derivation routes to its fail-closed third branch, instantiated
    here on the perf shape.

    Class: NEW CAPABILITY. The plausible wrong implementation is the shipped one, which binds
    the provenance dict to ``_col_prov`` and never reads ``g1_non_empty``, so ``all_identical``
    stays at its initialised ``True`` and the strict path returns ``passed=True`` with the
    summary "All tracked variables bit-identical ...". Red under it at the first assertion.
    The same assertion is also red against the pre-change tree, so this names a hole the chunk
    did not open; what the chunk adds is a docstring at the third branch asserting the hole is
    closed ("yields nothing and is refused by `G1`"), which is true at ``compare_arms`` and
    false here.

    Why it is required, as a consequence a reader can check against the code: ``compare_arms``
    guards this with ``if total_compared == 0: return _not_evaluated(...)`` and
    ``check_cross_sim_identity`` has no counterpart, so one of the two consumers of the same
    derivation reports agreement over zero comparisons. The two verdicts are then byte-identical
    on the page for "every variable agreed" and "no variable was looked at", which is the one
    reading the campaign's acceptance decision rests on.

    What kills it: consulting ``_col_prov["g1_non_empty"]`` at the call site and recording a
    NOT-EVALUATED row, or counting compared variables and refusing an unqualified pass at zero.

    A second correct implementation under which it still passes: one that keeps
    ``passed=True`` and instead appends a NOT-EVALUATED detail row naming the empty compared
    set, exactly as ``compare_arms`` does per artifact. The predicate is disjunctive for that
    reason and does not demand ``passed=False``.
    """
    # (1) VIOLATING INPUT. Every member's only artifact is unroutable, and the two members
    #     differ in EVERY column they carry, so a verdict of agreement here is agreement
    #     nothing measured.
    cols, prov = compared_columns_for(_unroutable_summary())
    assert cols == () and prov["g1_non_empty"] is False, "fixture must reach the fail-closed branch"

    blind = _master(
        tmp_path / "blind",
        {
            "serial_0_r1": (_Cfg("serial", n_mpi_procs=1, n_omp_threads=1), _unroutable_summary()),
            "mpi_8_r1": (_Cfg("mpi", n_mpi_procs=8, n_omp_threads=1), _unroutable_summary(bump=1000.0)),
        },
    )
    res_blind = check_cross_sim_identity(blind)
    assert res_blind.skipped is False, "the master has present summaries, so this is not the skip path"
    assert _discloses_non_evaluation(res_blind.verdict), (
        "zero columns were compared and the verdict claims bit-identity: "
        f"passed={res_blind.verdict.passed!r} summary={res_blind.verdict.summary!r} "
        f"details={res_blind.verdict.details!r}"
    )

    # (2) DIFFERENTLY-POSITIONED SATISFYING INPUT. A NON-empty compared set over members that
    #     genuinely agree is a state in which an unqualified bit-identity pass is CORRECT. An
    #     assertion that also fired here would have encoded "always disclose" rather than `G1`.
    cols_ok, prov_ok = compared_columns_for(_pair_summary())
    assert prov_ok["g1_non_empty"] is True and len(cols_ok) == 2, "control fixture must be routable"

    sighted = _master(
        tmp_path / "sighted",
        {
            "serial_0_r1": (_Cfg("serial", n_mpi_procs=1, n_omp_threads=1), _pair_summary()),
            "mpi_8_r1": (_Cfg("mpi", n_mpi_procs=8, n_omp_threads=1), _pair_summary()),
        },
    )
    res_sighted = check_cross_sim_identity(sighted)
    assert res_sighted.verdict.passed is True, res_sighted.verdict.summary
    assert not _discloses_non_evaluation(res_sighted.verdict), (
        "a genuine agreement over a non-empty compared set must NOT be reported as "
        "not-evaluated; this predicate is scoped to the empty case"
    )


# ---------------------------------------------------------------------------
# HELD TEST -- green before this turn and must remain green
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variable", _NEWLY_ADMITTED)
def test_the_all_vars_widening_stays_inside_one_hardware_family(tmp_path, variable):
    """Every variable all-vars newly admits is compared WITHIN a hardware family only.

    Property: for each of the seven TRITON variables the all-vars derivation newly admits, a
    difference between members of DIFFERENT hardware families does not reach the strict
    verdict, while the same difference between members of the SAME family does. Quantified over
    the newly-admitted set, parametrized so each member of it is its own case rather than one
    case that stops at the first.

    Class: PRESERVATION -- green before this turn and required to remain green. It is a HELD
    test, not a finding, and it is enumerated as one in the reviewer's round file with the
    mutation that reds it.

    Why it is required, as a consequence a reader can check against the code: the repair takes
    the TRITON compared set from one variable to eight, and three of the seven it adds are
    argmax-derived (``time_of_max_*``, ``velocity_*_at_time_of_max_velocity``) and therefore
    tie-break candidates across compute configurations. The module's own docstring states that a
    GPU-versus-serial-CPU float32 difference at one eps is EXPECTED PHYSICS rather than a
    reproducibility failure, and that the comparison is anchored per family for exactly that
    reason. Before the repair that anchoring carried one variable; after it, seven more ride on
    it, and nothing in the chunk's own tests holds the boundary while the payload grows.

    What kills it: replacing ``fam_ref_id = ref_by_family[fam_of[member_id]]`` with the global
    ``ref_id``, or collapsing ``_b4b_family_key`` so every member lands in one family.

    A second correct implementation under which it still passes: one partitioning by GPU
    hardware rather than by device class, since both put a CPU member and a GPU member in
    different buckets and this test asserts nothing about how GPUs are grouped among themselves.
    """
    # Two GPU members that agree with EACH OTHER and both differ from the serial-CPU reference
    # in the newly-admitted variable. Under the family anchoring the GPU pair is measured
    # against the GPU reference, so the cross-boundary difference is never compared.
    master = _master(
        tmp_path / "cross",
        {
            "serial_0_r1": (_Cfg("serial", n_mpi_procs=1, n_omp_threads=1), _triton_summary()),
            "gpu_1_r1": (_Cfg("gpu", n_gpus=1), _triton_summary(perturb=variable)),
            "gpu_2_r1": (_Cfg("gpu", n_gpus=2), _triton_summary(perturb=variable)),
        },
    )
    res = check_cross_sim_identity(master)
    assert res.verdict.passed is True, (
        f"a cross-family difference in {variable} reached the strict verdict: {res.verdict.summary} "
        f"{res.verdict.details}"
    )

    # CONTROL, and it is what stops the assertion above from passing vacuously: the SAME
    # magnitude in the SAME variable, moved INSIDE the GPU family, must still be fatal. Without
    # it "the boundary held" and "nothing is compared at all" are the same green.
    master_within = _master(
        tmp_path / "within",
        {
            "serial_0_r1": (_Cfg("serial", n_mpi_procs=1, n_omp_threads=1), _triton_summary()),
            "gpu_1_r1": (_Cfg("gpu", n_gpus=1), _triton_summary()),
            "gpu_2_r1": (_Cfg("gpu", n_gpus=2), _triton_summary(perturb=variable)),
        },
    )
    res_within = check_cross_sim_identity(master_within)
    assert res_within.verdict.passed is False, (
        f"a within-GPU-family difference in {variable} must still fail: {res_within.verdict.summary}"
    )
    assert {row["variable"] for row in res_within.verdict.details if "variable" in row} == {variable}
