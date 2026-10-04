"""Tests for the ADR-9 cross-sim byte-identity EDA check (eda/cross_sim_identity.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from hhemt.eda import EdaResult, check_cross_sim_identity
from hhemt.eda.cross_sim_identity import _ref_rank, _references_by_family, compare_variable_exact

# ---- Fast tier (no build): non-sensitivity skip + graceful-absent + kernel ----


def test_non_sensitivity_returns_skipped(synth_multi_sim_analysis_cached):
    """A non-sensitivity analysis yields a skipped N/A verdict, no artifact."""
    result = check_cross_sim_identity(synth_multi_sim_analysis_cached)
    assert isinstance(result, EdaResult)
    assert result.skipped is True
    assert result.artifact_path is None
    assert result.verdict is not None
    assert result.verdict.passed is True
    assert result.verdict.level == "aggregate"
    assert "N/A" in result.verdict.summary


@pytest.mark.requires_snakemake_subprocess
@pytest.mark.slow
def test_validate_analysis_graceful_absent(synthetic_multisim_completed):
    """No eda/ dir -> no EDA row; validate_analysis is unchanged from today.

    Uses the built (post-consolidate) multisim fixture because validate_analysis
    runs the 7 core checks (check_system_setup reads the DEM); the merge of EDA
    verdicts is graceful-absent when no eda/ dir exists, so the report carries no
    Cross-sim byte-identity row."""
    from hhemt.analysis_validation import validate_analysis

    report = validate_analysis(synthetic_multisim_completed)
    assert not any(c.name == "Cross-sim byte-identity" for c in report.checks)


def test_compare_variable_exact_identical_and_divergent():
    """Kernel-level proof of BOTH outcomes with no solver build (the passed=False
    case the sensitivity-master test cannot deterministically force).

    Identical arrays (incl. matched NaN) -> identical True, max_abs_diff 0.0.
    A single perturbed cell -> identical False, finite max_abs_diff, dtype/coord
    still matched (the divergence is value-only, mirroring last-ULP FP drift)."""
    coords = {"x": [0, 1, 2], "y": [0, 1]}
    base = xr.DataArray(
        np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]),
        dims=("x", "y"),
        coords=coords,
    )
    # Identical (matched NaN counts as equal under equal_nan=True).
    res_same = compare_variable_exact(base, base.copy(deep=True))
    assert res_same["identical"] is True
    assert res_same["dtype_match"] is True
    assert res_same["coord_match"] is True
    assert res_same["max_abs_diff"] == 0.0

    # Divergent: perturb one cell by one ULP-scale delta.
    perturbed = base.copy(deep=True)
    perturbed.values[1, 0] = 2.0 + 1e-12
    res_diff = compare_variable_exact(base, perturbed)
    assert res_diff["identical"] is False
    assert res_diff["coord_match"] is True
    assert res_diff["dtype_match"] is True
    assert res_diff["max_abs_diff"] > 0.0


def test_compare_variable_exact_coord_mismatch_fails_closed():
    """A different coordinate set fails closed (coord_match False, not comparable)."""
    a = xr.DataArray(np.array([1.0, 2.0]), dims=("link_id",), coords={"link_id": [10, 11]})
    b = xr.DataArray(np.array([1.0, 2.0]), dims=("link_id",), coords={"link_id": [10, 99]})
    res = compare_variable_exact(a, b)
    assert res["identical"] is False
    assert res["coord_match"] is False


def test_combine_cells_single_and_multi():
    """Regression for the artifact-assembly helper _combine_cells (the stitch the
    operator 2-row validation run tripped on Rivanna). A single 1x1 (member_id, event_iloc)
    cell — the minimal native+container suite, one non-reference sub + one event — is the
    DEGENERATE combine_by_coords case (raises "Could not find any dimension coordinates"
    on the Rivanna py3.11 xarray). The helper returns the lone cell directly there and
    still hypercube-stitches N>=2. The slow synthetic_sensitivity_completed fixture has
    multiple subs, so it never exercised the single-cell path."""
    from hhemt.eda.cross_sim_identity import _combine_cells

    cell = xr.DataArray(0.0).expand_dims({"sa_id": ["container"], "event_iloc": [0]})
    # Single cell: must return the lone array intact (no combine_by_coords degeneracy).
    single = _combine_cells([cell])
    assert single.sel(sa_id="container", event_iloc=0).item() == 0.0
    assert list(single["sa_id"].values) == ["container"]

    # N>=2 across member_id: assembled into the (member_id, event_iloc) grid (manual build, no
    # combine_by_coords) with values placed at the right coords.
    cell2 = xr.DataArray(1.0).expand_dims({"sa_id": ["native_dup"], "event_iloc": [0]})
    multi = _combine_cells([cell, cell2])
    assert set(multi["sa_id"].values) == {"container", "native_dup"}
    assert multi.sel(sa_id="container", event_iloc=0).item() == 0.0
    assert multi.sel(sa_id="native_dup", event_iloc=0).item() == 1.0

    # N>=2 across event_iloc for one sub: grid spans both events.
    ev0 = xr.DataArray(0.0).expand_dims({"sa_id": ["c"], "event_iloc": [0]})
    ev1 = xr.DataArray(2.5).expand_dims({"sa_id": ["c"], "event_iloc": [1]})
    grid = _combine_cells([ev0, ev1])
    assert list(grid["event_iloc"].values) == [0, 1]
    assert grid.sel(sa_id="c", event_iloc=1).item() == 2.5

    # Bool dtype (the `identical__{var}` artifact) is preserved through the manual build.
    b0 = xr.DataArray(True).expand_dims({"sa_id": ["c"], "event_iloc": [0]})
    b1 = xr.DataArray(False).expand_dims({"sa_id": ["c"], "event_iloc": [1]})
    bgrid = _combine_cells([b0, b1])
    assert bgrid.dtype == bool
    assert bool(bgrid.sel(sa_id="c", event_iloc=0)) is True
    assert bool(bgrid.sel(sa_id="c", event_iloc=1)) is False


# ---- Slow tier (one real build, session-cached): summaries-present sensitivity ----


@pytest.mark.requires_snakemake_subprocess
@pytest.mark.slow
def test_sensitivity_master_identical_passes(synthetic_sensitivity_completed):
    """On a benchmarking sensitivity master whose subs vary ONLY compute config,
    every tracked variable is bit-identical and the verdict passes; the artifact
    + verdict JSON are written under {analysis_dir}/eda/.

    Substrate: synthetic_sensitivity_completed (conftest.py) runs the synth
    sensitivity master once per session to the f_consolidate_experiment_complete
    state, materializing per-member summaries on disk. Per the plan's bit-repro
    empirical precondition, if the synth solver is NOT bit-reproducible across
    the 4 compute modes this assertion is re-scoped to 'check ran + well-formed
    verdict/artifact' (plan Empirical Testing decision rule)."""
    analysis = synthetic_sensitivity_completed.experiment
    result = check_cross_sim_identity(analysis)
    assert result.skipped is False
    assert result.verdict is not None
    assert result.verdict.passed is True, result.verdict.summary
    assert result.plot_id == "eda_cross_sim_identity"
    assert result.artifact_path is not None and result.artifact_path.exists()
    # Manifest sidecar present + plot_id stamped + source paths declared.
    manifest = result.artifact_path.parent / f"{result.plot_id}.manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["plot_id"] == result.plot_id
    assert payload["output_format"] == "data"
    assert payload["source_paths_relative"]
    # Verdict JSON round-trips the CheckResult fields.
    verdict_json = result.artifact_path.parent / f"{result.plot_id}.verdict.json"
    assert verdict_json.exists()
    vp = json.loads(verdict_json.read_text())
    assert vp["name"] == "Cross-sim byte-identity"
    assert vp["passed"] is True
    # This check is the one PRECISION-BEARING verdict in the suite: it compares the
    # FLAT per-scenario summaries, where max_wlevel_m is float32. It must declare
    # that, or the renderer shows an unqualified green tick meaning "identical to
    # within float32 rounding" -- the defect that shipped. An unstamped check renders
    # a plain pass by design, so losing this stamp fails SILENTLY; that is precisely
    # why it is pinned here and not left to the renderer's default.
    assert vp["instrument"] == "summary_tier"
    assert vp["detection_floor"] == pytest.approx(float(np.finfo(np.float32).eps))
    assert result.verdict.applicable is True


@pytest.mark.requires_snakemake_subprocess
@pytest.mark.slow
def test_sensitivity_master_across_family_characterizes(synthetic_sensitivity_completed):
    """ADR-4 across-family (within_family=False): the verdict NEVER asserts equality.

    Whether or not the subs are bit-identical, the across-family verdict is
    passed=True and its summary discloses the bounded divergence (the boundary IS
    the contribution). The persisted artifact + verdict JSON are still written
    under {analysis_dir}/eda/, and the verdict's name/contract is unchanged — only
    passed/summary/details semantics branch on within_family."""
    analysis = synthetic_sensitivity_completed.experiment
    result = check_cross_sim_identity(analysis, within_family=False)
    assert result.skipped is False
    assert result.verdict is not None
    # Disclosed divergence is always a PASS under ADR-4 across-family semantics.
    assert result.verdict.passed is True, result.verdict.summary
    assert "haracterized divergence" in result.verdict.summary
    assert result.verdict.name == "Cross-sim byte-identity"
    assert result.artifact_path is not None and result.artifact_path.exists()
    # The persisted verdict JSON round-trips the (passed=True) across-family verdict.
    verdict_json = result.artifact_path.parent / f"{result.plot_id}.verdict.json"
    assert verdict_json.exists()
    vp = json.loads(verdict_json.read_text())
    assert vp["passed"] is True


@pytest.mark.requires_snakemake_subprocess
@pytest.mark.slow
def test_verdict_surfaces_in_validate_analysis(synthetic_sensitivity_completed):
    """A persisted EDA verdict is merged into validate_analysis()'s ValidationReport."""
    from hhemt.analysis_validation import validate_analysis

    analysis = synthetic_sensitivity_completed.experiment
    check_cross_sim_identity(analysis)
    report = validate_analysis(analysis)
    eda_checks = [c for c in report.checks if c.name == "Cross-sim byte-identity"]
    assert len(eda_checks) == 1
    assert eda_checks[0].level == "aggregate"


@pytest.mark.requires_snakemake_subprocess
@pytest.mark.slow
def test_identity_group_partition_persisted(synthetic_sensitivity_completed):
    """The additive byte-identity PARTITION (identity_group) is persisted into the artifact.

    Contract (plan R1 producer): identity_group is int32, dims (member_id,), computed from the
    FLAT summaries via compare_variable_exact (NOT the consolidated tree), emitted over the
    NON-reference member_id coord (purely additive), with the reference's own label carried in the
    reference_group attr. Two subs share a label iff byte-identical on the config-diff
    variables at every event -- so on a bit-identical master (verdict passed) every sub +
    the reference collapse to ONE group."""
    analysis = synthetic_sensitivity_completed.experiment
    result = check_cross_sim_identity(analysis)
    ds = xr.open_zarr(result.artifact_path, consolidated=False)

    # Structural contract.
    assert "identity_group" in ds
    assert ds["identity_group"].dtype == np.int32
    assert ds["identity_group"].dims == ("sa_id",)
    assert (ds["identity_group"].values >= 0).all()
    # Purely additive: identity_group shares the existing artifact vars' (non-reference) member_id
    # coord; the reference is carried separately as an attr, not in the label array.
    assert "reference_member_id" in ds.attrs
    ref_id = str(ds.attrs["reference_member_id"])
    assert ref_id not in {str(s) for s in ds["sa_id"].values}
    assert "reference_group" in ds.attrs
    assert int(ds.attrs["reference_group"]) >= 0

    # Behavioral contract, gated on bit-reproducibility (same decision rule as the identical
    # test): when the verdict is passed (all subs byte-identical to the reference), the whole
    # partition -- including the reference's own label -- collapses to a single group.
    if result.verdict.passed:
        labels = set(int(v) for v in ds["identity_group"].values)
        labels.add(int(ds.attrs["reference_group"]))
        assert labels == {int(ds.attrs["reference_group"])}, (
            f"bit-identical master must be one identity group, got labels {labels}"
        )


def test_tracked_vars_are_actually_emitted_names() -> None:
    """TRACKED_VARS must name EMITTED data_vars, not cf_conventions attribute keys.

    Defect (2026-07-21): the tuple carried ``max_full_flow_ratio`` /
    ``max_full_depth_ratio`` — defined in cf_conventions.py:121,127 but emitted
    NOWHERE. Because the comparison loops ``continue`` past absent vars, all three
    consumers (check_cross_sim_identity, compute_sensitivity, the [Q8] REQ-1
    reproduction check) silently compared 2 of 4 variables, and a [Q8] cross-hardware
    DoD was certified without conduit capacity ever being compared. The emitted names
    live in constants.LST_COL_HEADERS_LINK_FLOW_SUMMARY and are what the renderers
    consume (per_sim_conduit_flow.py:120,555).
    """
    from hhemt.constants import LST_COL_HEADERS_LINK_FLOW_SUMMARY
    from hhemt.eda.cross_sim_identity import TRACKED_VARS

    for dead in ("max_full_flow_ratio", "max_full_depth_ratio"):
        assert dead not in TRACKED_VARS, (
            f"{dead} is a cf_conventions attribute key, not an emitted data_var; "
            f"including it makes the identity check silently skip that variable"
        )
    for live in ("max_over_full_flow", "max_over_full_depth"):
        assert live in TRACKED_VARS, f"{live} is emitted but not tracked"
        assert live in LST_COL_HEADERS_LINK_FLOW_SUMMARY, (
            f"{live} is tracked but is not in the emitted-name registry — re-ground "
            f"TRACKED_VARS against list(ds.data_vars) of a real summary zarr"
        )


def test_every_tracked_var_has_cf_attributes() -> None:
    """A tracked variable with no CF entry ships unitless in a CF-1.13 dataset.

    ``cf_conventions.apply_cf_attributes`` gives unmapped vars a ``long_name`` ONLY
    (auto-generated by ``_auto_long_name``), so a variable can be reproduction-critical
    and still publish with no ``units`` and no ``standard_name``. Found 2026-07-21: the
    EMITTED conduit-capacity names had zero entries while curated metadata sat under
    two names emitted nowhere, so every published dataset declared Conventions=CF-1.13
    while shipping those variables unlabelled. Pin the two registries together.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP
    from hhemt.eda.cross_sim_identity import TRACKED_VARS

    missing = [v for v in TRACKED_VARS if v not in _CF_VARIABLE_MAP]
    assert not missing, (
        f"tracked but unlabelled in _CF_VARIABLE_MAP: {missing} — these publish with "
        f"an auto-generated long_name and no units"
    )


class _StubCfg:
    """Minimal stand-in for a sub's cfg_analysis — only the compute attrs _ref_rank reads."""

    def __init__(self, run_mode, n_gpus=0, n_mpi_procs=0, n_omp_threads=0, n_nodes=0):
        self.run_mode = run_mode
        self.n_gpus = n_gpus
        self.n_mpi_procs = n_mpi_procs
        self.n_omp_threads = n_omp_threads
        self.n_nodes = n_nodes


class _StubSub:
    def __init__(self, cfg):
        self.cfg_analysis = cfg


def test_reference_rank_selects_serial_over_lexicographically_earlier_gpu():
    """N1: the reference is the SERIAL-CPU sub, not the lexicographically-first member_id.

    The retired rule sorted on member_id alone, which on the real compute-config sweep selected
    `gpu_0_r1` — making every reported difference a difference-from-a-GPU-run rather than
    from the serial oracle. This fixture reproduces that adversarial shape deliberately:
    the GPU member_id sorts FIRST lexicographically and the serial member_id sorts LAST, so the
    assertion discriminates the new rule from the old one rather than passing under both.
    """
    items = [
        ("a_gpu_0_r1", _StubSub(_StubCfg("gpu", n_gpus=1))),
        ("m_mpi_8_r1", _StubSub(_StubCfg("mpi", n_mpi_procs=8, n_omp_threads=1))),
        ("z_serial_0_r1", _StubSub(_StubCfg("serial", n_mpi_procs=1, n_omp_threads=1))),
    ]
    ordered = sorted(items, key=_ref_rank)
    assert ordered[0][0] == "z_serial_0_r1", (
        f"reference must be the serial-CPU sub even when its member_id sorts last; got {ordered[0][0]}"
    )
    # Pre-fix control: member_id-only ordering picks the GPU sub, so the two rules disagree on
    # this fixture. Without this the test could not distinguish "serial won" from "serial
    # happened to sort first anyway".
    assert sorted(items, key=lambda kv: kv[0])[0][0] == "a_gpu_0_r1"


def test_reference_rank_tiebreaks_are_ordered_as_documented():
    """Among non-serial subs: ascending nodes, then GPUs, then MPI x OMP, then member_id.

    Pins the rule's LATER terms. Without this only the leading serial term is covered, and
    a regression in the tiebreak would move the starred reference between renders of the
    same data while the serial assertion above still passed.
    """
    items = [
        ("b_gpu_3", _StubSub(_StubCfg("gpu", n_gpus=3))),
        ("a_gpu_1", _StubSub(_StubCfg("gpu", n_gpus=1))),
        ("c_gpu_2", _StubSub(_StubCfg("gpu", n_gpus=2))),
    ]
    assert [member for member, _ in sorted(items, key=_ref_rank)] == ["a_gpu_1", "c_gpu_2", "b_gpu_3"]

    # member_id is the FINAL tiebreak, reached only when every compute term ties.
    tied = [
        ("z_gpu_1", _StubSub(_StubCfg("gpu", n_gpus=1))),
        ("a_gpu_1", _StubSub(_StubCfg("gpu", n_gpus=1))),
    ]
    assert [member for member, _ in sorted(tied, key=_ref_rank)] == ["a_gpu_1", "z_gpu_1"]


# ---- EW-4: per-family reference selection (strict path) ----
#
# Fast tier, no solver build and no HPC: the harness stubs the three surfaces
# check_cross_sim_identity touches on a sub (`cfg_analysis`, `process._MODE_CONFIG` +
# `process._retrieve_combined_output`, `analysis_paths.analysis_dir`) and the two it touches on the
# master (`cfg_analysis.toggle_sensitivity_analysis`, `sensitivity.members`). Divergence has to
# be synthetic for these arms either way — no real solver emits a controlled one-ULP difference on
# demand — so a fast stub is strictly better than a slow fixture here: it runs on every change, and
# CI is CPU-only so a GPU family cannot be materialized any other way.
#
# Why the existing slow tier does not cover this: BOTH real fixtures are single-hardware-family.
# _write_synth_sensitivity_csv is mpi/openmp/hybrid/serial with n_gpus=[0,0,0,0] -> one 'cpu' family;
# container_validation_suite.csv is four rows all run_mode=gpu -> one 'gpu' family. In a single
# family the per-family reference IS the global reference, so every slow test in this file is a
# provable no-op under EW-4. Their green is a regression check, never evidence the change works.

_DEPTH_MODE = "tritonswmm"
_LINK_MODE = "tritonswmm_swmm_link"

#: One float32 ULP at 1.0 — the EXACT magnitude of all 24 GPU-vs-serial-CPU tuples EW-4 stops
#: failing, and (see test_within_family_cpu_divergence_still_fails) the magnitude that must STILL be
#: fatal within a family. Pinning one number to both roles is what makes "we re-referenced" and "we
#: widened tolerance" distinguishable outcomes rather than two stories about the same green suite.
_ULP32 = float(np.finfo(np.float32).eps)


def _summaries(depth_bump: float = 0.0):
    """The flat per-scenario summaries a sub returns, optionally perturbed by `depth_bump`.

    Two modes so the partition signature (_PARTITION_VARS = max_wlevel_m, max_flow_cms) is fully
    populated. max_wlevel_m is float32 to match the on-disk dtype the detection floor is derived
    from; the link-mode var is float64, as in the real pipeline.
    """
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


class _StubProcess:
    """Stands in for TRITONSWMM_analysis_post_processing.

    _MODE_CONFIG must be reachable via the live `.process` instance (never module-level) and
    _retrieve_combined_output must raise FileNotFoundError on an absent mode — that is the
    existence guard _enabled_modes keys on.
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
    def __init__(self, cfg, datasets, d):
        self.cfg_analysis = cfg
        self.process = _StubProcess(datasets)
        self.analysis_paths = _Paths(d)


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


def _master(tmp_path, spec):
    """Build a stub sensitivity master from {member_id: (cfg, depth_bump)}.

    Each sub gets a real on-disk analysis_dir because the emitted provenance source path is
    `{member_dir}/analysis_datatree.zarr`; that file need not exist (_validate_source_path
    accepts a non-existent path, and the `.zarr` suffix clears the directory-as-source gate anyway).
    """
    subs = {}
    for member_id, (cfg, bump) in spec.items():
        d = tmp_path / member_id
        d.mkdir(parents=True, exist_ok=True)
        # Materialize the flat summary tier at its real relative layout
        # ({sim}/processed/{STEM}_summary.{ext}, scenario.py:372,444) so the
        # declaration _summary_paths globs is non-empty. Contents are irrelevant:
        # _validate_source_path gates on path SHAPE, and nothing in this file's
        # fast tier opens a summary file -- _StubProcess returns datasets directly.
        # BOTH KINDS are materialized: a .nc FILE and a .zarr STORE, so each clause
        # of _summary_paths' predicate has a witness here.
        _proc = d / "sims" / "s0" / "processed"
        _proc.mkdir(parents=True, exist_ok=True)
        (_proc / "TRITONSWMM_TRITON_summary.nc").touch()
        # One summary as a zarr STORE -- a DIRECTORY -- because
        # target_processed_output_type defaults to "zarr" (config/analysis.py:692).
        # Without this every test exercises only the is_file() clause of
        # _summary_paths' predicate, and narrowing it back to is_file() alone --
        # which declares NOTHING on the default config -- stays green.
        _zarr = _proc / "TRITONSWMM_SWMM_link_summary.zarr"
        _zarr.mkdir(exist_ok=True)
        (_zarr / ".zgroup").write_text("{}")
        subs[member_id] = _Sub(cfg, _summaries(bump), d)
    root = tmp_path / "master"
    root.mkdir(parents=True, exist_ok=True)
    return _Master(subs, root)


def test_references_by_family_partitions_cpu_and_gpu():
    """Each hardware family gets its OWN _ref_rank winner (the CPU one is not the GPU one).

    Adversarial by construction, in the same style as the _ref_rank tests above: the GPU member_id sorts
    FIRST lexicographically and the serial member_id sorts LAST, so a rule that merely returned the
    global winner would produce {'cpu': 'z_serial_0_r1'} alone and fail the equality. The trailing
    control pins that the global winner is still serial, so the two rules are visibly different
    on this fixture rather than coincidentally equal.
    """
    items = [
        ("a_gpu_0_r1", _StubSub(_StubCfg("gpu", n_gpus=1))),
        ("b_gpu_1_r1", _StubSub(_StubCfg("gpu", n_gpus=4))),
        ("m_mpi_8_r1", _StubSub(_StubCfg("mpi", n_mpi_procs=8, n_omp_threads=1))),
        ("z_serial_0_r1", _StubSub(_StubCfg("serial", n_mpi_procs=1, n_omp_threads=1))),
    ]
    ordered = sorted(items, key=_ref_rank)
    assert _references_by_family(ordered) == {"cpu": "z_serial_0_r1", "gpu": "a_gpu_0_r1"}
    assert ordered[0][0] == "z_serial_0_r1"


def test_references_by_family_single_family_is_todays_reference():
    """On a SINGLE-family master the per-family reference IS the pre-EW-4 global reference.

    Both real fixtures are single-family, so this is the guard that keeps their verdicts bit-identical
    if the family predicate is ever changed. The two frames are transcribed from the fixtures rather
    than invented: the synth compute-config sweep (_write_synth_sensitivity_csv) and the
    container-validation suite CSV. The container case additionally pins that _ref_rank's final
    lexicographic tiebreak still selects `container_1g` once serial-first and device count both tie.
    """
    synth = [
        ("0", _StubSub(_StubCfg("mpi", n_mpi_procs=2, n_omp_threads=1, n_nodes=1))),
        ("1", _StubSub(_StubCfg("openmp", n_mpi_procs=1, n_omp_threads=2, n_nodes=1))),
        ("2", _StubSub(_StubCfg("hybrid", n_mpi_procs=2, n_omp_threads=2, n_nodes=1))),
        ("3", _StubSub(_StubCfg("serial", n_mpi_procs=1, n_omp_threads=1, n_nodes=1))),
    ]
    assert _references_by_family(sorted(synth, key=_ref_rank)) == {"cpu": "3"}

    container = [
        ("native_1g", _StubSub(_StubCfg("gpu", n_gpus=1, n_mpi_procs=1, n_omp_threads=1, n_nodes=1))),
        ("container_1g", _StubSub(_StubCfg("gpu", n_gpus=1, n_mpi_procs=1, n_omp_threads=1, n_nodes=1))),
        ("native_2g", _StubSub(_StubCfg("gpu", n_gpus=2, n_mpi_procs=2, n_omp_threads=1, n_nodes=1))),
        ("container_2g", _StubSub(_StubCfg("gpu", n_gpus=2, n_mpi_procs=2, n_omp_threads=1, n_nodes=1))),
    ]
    ordered_c = sorted(container, key=_ref_rank)
    assert _references_by_family(ordered_c) == {"gpu": "container_1g"}
    assert ordered_c[0][0] == "container_1g"


def test_within_family_cpu_divergence_still_fails(tmp_path):
    """P7 GUARD — DO NOT DELETE OR RELAX. A CPU-vs-CPU divergence MUST still fail the verdict.

    This is a PRESERVATION test: it is green both before and after EW-4, and that is the point. EW-4
    re-references the comparison; it does not widen it. Without this arm, "the false GPU FAILs are
    gone" and "the strict path was silently widened" are indistinguishable outcomes — both produce a
    passing verdict on the campaign, and the second is exactly what rejecting a within_family=False
    flip was meant to avoid.

    The magnitude is load-bearing: the perturbation is ONE float32 ULP, bit-for-bit the same
    max_abs_diff as every GPU-family tuple EW-4 stops failing. So the pair of assertions states the
    fix's actual semantics — the same magnitude that is no longer compared ACROSS families is still
    fatal WITHIN one. Any tolerance added to the strict path turns this red.
    """
    master = _master(
        tmp_path,
        {
            "serial_0_r1": (_StubCfg("serial", n_mpi_procs=1, n_omp_threads=1), 0.0),
            "mpi_8_r1": (_StubCfg("mpi", n_mpi_procs=8, n_omp_threads=1), _ULP32),
        },
    )
    result = check_cross_sim_identity(master)
    assert result.skipped is False
    assert result.verdict.passed is False, result.verdict.summary
    rows = [d for d in result.verdict.details if d.get("variable") == "max_wlevel_m"]
    assert rows, result.verdict.details
    assert {r["sa_id"] for r in rows} == {"mpi_8_r1"}
    # The divergent row names the family reference it was measured against, not just a member_id.
    assert {r["ref_member_id"] for r in rows} == {"serial_0_r1"}
    ds = xr.open_zarr(result.artifact_path, consolidated=False)
    mad = float(np.max(ds["max_abs_diff__max_wlevel_m"].sel(sa_id="mpi_8_r1").values))
    assert mad == pytest.approx(_ULP32)


def test_cross_family_gpu_divergence_no_longer_fails(tmp_path):
    """A GPU sub diverging from serial-CPU by one float32 ULP no longer fails — and KEEPS its row.

    Pre-EW-4 this verdict was `passed=False` with '1 (member, event, variable) tuple(s) diverged from
    reference member_id=serial_0_r1' — the two false FAIL cells on the combined report's
    errors-and-warnings page, in miniature.

    The second half is the part that matters structurally. The lone GPU sub is its own family's
    reference, so it self-compares and stays in the artifact's member_id coord with identical=True /
    max_abs_diff=0.0. Had the change excluded EVERY family reference from the loop instead of only
    the primary one, this sub would vanish from the coord, _config_diff._identity_labels would have
    no label for the 1-GPU group, and — because _config_diff re-references every GPU group to that
    group — the whole GPU half of the config-diff identity column would render 'differs'. These
    assertions are what make that design decision falsifiable from the test suite.
    """
    master = _master(
        tmp_path,
        {
            "serial_0_r1": (_StubCfg("serial", n_mpi_procs=1, n_omp_threads=1), 0.0),
            "mpi_8_r1": (_StubCfg("mpi", n_mpi_procs=8, n_omp_threads=1), 0.0),
            "gpu_0_r1": (_StubCfg("gpu", n_gpus=1), _ULP32),
        },
    )
    result = check_cross_sim_identity(master)
    assert result.verdict.passed is True, result.verdict.summary
    ds = xr.open_zarr(result.artifact_path, consolidated=False)
    member_ids = {str(s) for s in np.atleast_1d(ds["sa_id"].values)}
    assert "gpu_0_r1" in member_ids
    assert bool(np.all(ds["identical__max_wlevel_m"].sel(sa_id="gpu_0_r1").values))
    assert float(np.max(ds["max_abs_diff__max_wlevel_m"].sel(sa_id="gpu_0_r1").values)) == 0.0


def test_artifact_member_id_coord_excludes_only_the_primary_reference(tmp_path):
    """EXACTLY ONE sub is excluded from the artifact's member_id coord, whatever the family count.

    Stated as a coord-membership invariant on purpose. The downstream symptom of breaking it is a
    false 'differs' in a renderer two modules away (_config_diff), which nobody would trace back to
    this loop; the invariant is checkable right here. The companion assertion pins VMS 8's
    per-family reference map, which is pure disclosure — _config_diff still folds back only the
    single scalar `reference_member_id`, and that contract is deliberately unchanged.
    """
    master = _master(
        tmp_path,
        {
            "serial_0_r1": (_StubCfg("serial", n_mpi_procs=1, n_omp_threads=1), 0.0),
            "mpi_8_r1": (_StubCfg("mpi", n_mpi_procs=8, n_omp_threads=1), 0.0),
            "gpu_0_r1": (_StubCfg("gpu", n_gpus=1), _ULP32),
            "gpu_1_r1": (_StubCfg("gpu", n_gpus=4), _ULP32),
        },
    )
    result = check_cross_sim_identity(master)
    ds = xr.open_zarr(result.artifact_path, consolidated=False)
    member_ids = {str(s) for s in np.atleast_1d(ds["sa_id"].values)}
    all_ids = {"serial_0_r1", "mpi_8_r1", "gpu_0_r1", "gpu_1_r1"}
    assert member_ids == all_ids - {str(ds.attrs["reference_member_id"])}
    assert json.loads(ds.attrs["reference_member_id_by_family"]) == {
        "cpu": "serial_0_r1",
        "gpu": "gpu_0_r1",
    }


def test_across_family_path_keeps_one_global_reference(tmp_path):
    """within_family=False is NOT partitioned — it keeps ONE global reference.

    That arm exists precisely to measure the cross-boundary bound, so partitioning it would make it
    measure WITHIN-family divergence and label the result 'across-family'. The disclosed bound would
    silently become a different quantity under the same name — a wrong number, not a missing one.
    This is the arm most likely to be lost to a future 'why is the family logic conditional?'
    simplification, which is why the map is asserted to be exactly {'all': <global ref>} rather than
    merely non-empty, and why the bound is asserted to be strictly positive.
    """
    master = _master(
        tmp_path,
        {
            "serial_0_r1": (_StubCfg("serial", n_mpi_procs=1, n_omp_threads=1), 0.0),
            "gpu_0_r1": (_StubCfg("gpu", n_gpus=1), _ULP32),
        },
    )
    result = check_cross_sim_identity(master, within_family=False)
    assert result.verdict.passed is True
    assert "haracterized divergence" in result.verdict.summary
    ds = xr.open_zarr(result.artifact_path, consolidated=False)
    assert json.loads(ds.attrs["reference_member_id_by_family"]) == {"all": "serial_0_r1"}
    bounds = [d for d in result.verdict.details if d.get("variable") == "max_wlevel_m"]
    assert bounds and bounds[0]["max_abs_diff"] > 0.0


def test_declared_sources_are_the_summary_tier_not_the_consolidated_store(tmp_path):
    """The SHIPPED provenance record names files this member opens, not one it never opens.

    Pre-fix RED on the middle assertion: the declaration is exactly
    `{member_dir}/analysis_datatree.zarr`, which `check_cross_sim_identity` never opens
    (the flat-summary stipulation forbids it). Anchored on the ABSENCE of that store -- a
    property that exists in both the pre-fix and post-fix worlds -- rather than on the
    summary filenames, which cannot exist pre-fix and would make this green by construction
    whatever the code did.

    Asserts `opened is a subset of declared`, never the converse: `_summary_paths` globs the
    sub's whole summary tier and therefore over-declares by design (~4x), so a
    declared-subset-of-opened assertion would be red on correct code.

    Reads the manifest sidecar rather than the returned EdaResult, for two reasons: the
    dataclass carries no source_paths field, and the sidecar is the artifact the bundle
    harvest actually consumes. Asserts over the raw relative STRINGS rather than Path
    objects because this module imports json but not pathlib.Path, and an [ADD] spec
    carries one anchor and so cannot also reach the import block.
    """
    master = _master(tmp_path, {"serial_0_r1": (_StubCfg("serial", n_mpi_procs=1, n_omp_threads=1), 0.0)})
    result = check_cross_sim_identity(master)

    sidecar = result.artifact_path.with_suffix(".manifest.json")
    assert sidecar.exists(), f"no manifest sidecar beside {result.artifact_path}"
    declared = json.loads(sidecar.read_text())["source_paths_relative"]

    assert declared, "the member declared no sources at all"
    assert not any(p.endswith("analysis_datatree.zarr") for p in declared), (
        f"declares the consolidated store this member never opens: {declared}"
    )
    assert all("processed" in p and "_summary" in p for p in declared), (
        f"declared a path outside the flat summary tier: {declared}"
    )


# ---- §8.7 conjunct (A): the CROSS-ARM comparison instrument ----
#
# Fast tier, no solver build and no HPC. Both arm roots are materialized on tmp_path with
# REAL zarr summary stores whose data_var set reproduces the real link summary's — the
# at-risk pair-derived reductions AND the structurally-immune `.rpt` max-PREFIX columns
# CO-RESIDENT in one Dataset, which is the property that makes the enumerated instrument
# blind rather than empty (measured on
# ~/.cache/hhemt/.../processed/TRITONSWMM_SWMM_link_summary.zarr, 14 data_vars).

_PIN_P = "b1ff26892d95"
_PIN_OTHER = "e53c2fa01a64"

#: The real link summary's data_var set, split by whether the pair rule admits it.
_AT_RISK_BASES = ("capacity_setting", "flow_cms", "link_depth_m", "velocity_mps")
_IMMUNE_COLS = ("max_flow_cms", "max_over_full_depth", "max_over_full_flow", "max_velocity_mps")


def _link_summary(*, truncate: str | None = None, drop_pair: str | None = None) -> xr.Dataset:
    """A link summary shaped like the real one.

    ``truncate`` names ONE at-risk base whose ``_max`` is LOWERED while its ``_last`` is left
    untouched — which IS the resume-defect signature: the true maximum fell outside the
    retained window, so the reduction is low while the final-step value is unaffected. It is
    not a synthetic one-ULP bump; it is the mechanism under test.

    ``drop_pair`` names ONE at-risk base whose ``_max``/``_last`` pair is OMITTED, producing
    an artifact whose PAIR-DERIVED set is a strict subset of the other side's. That is the
    model-path spelling asymmetry (``capacity_setting`` vs ``capacity``) in its smallest
    reproducible form, and it is what a one-sided derivation NARROWS over instead of
    refusing.
    """
    data = {}
    for i, base in enumerate(_AT_RISK_BASES):
        if base == drop_pair:
            continue
        mx = 10.0 + i
        if base == truncate:
            mx = 6.0 + i  # the maximum that survived a 36-step window
        data[f"{base}_max"] = (("event_iloc", "link_id"), np.array([[mx, mx + 0.5]], dtype="float64"))
        data[f"{base}_last"] = (("event_iloc", "link_id"), np.array([[1.0 + i, 1.5 + i]], dtype="float64"))
    for j, col in enumerate(_IMMUNE_COLS):
        # IMMUNE: written from the .rpt TLinkStats family, which provably round-trips, so
        # these are IDENTICAL on both sides even when the reduction is truncated. That is
        # exactly why an instrument enumerating them cannot fail on this defect.
        data[col] = (("event_iloc", "link_id"), np.array([[100.0 + j, 101.0 + j]], dtype="float64"))
    return xr.Dataset(data, coords={"event_iloc": [0], "link_id": ["c1", "c2"]})


def _write_arm(root, layout, *, pin, suffixes=("0", "1"), truncate_on=(), drop_pair=None, stamp_only=None):
    """Materialize an arm root at the NAMED layout with per-member stamped stores.

    ``drop_pair`` omits one at-risk pair from EVERY member's summary on this side, so the two
    sides' pair-derived sets are UNEQUAL. ``stamp_only`` restricts the stamp to the named
    suffixes, producing the PARTIALLY-stamped arm whose stamp is uniform over the members
    that carry one and absent on the rest.
    """
    from hhemt.eda.cross_sim_identity import _LAYOUT_PREFIX, MEMBER_LAYOUTS

    top = MEMBER_LAYOUTS[layout].split("/")[0]
    for suffix in suffixes:
        member = root / top / f"{_LAYOUT_PREFIX[layout]}{suffix}"
        proc = member / "sims" / "event_index.0" / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        ds = _link_summary(
            truncate=truncate_on[0] if (suffix in truncate_on[1:] and truncate_on) else None,
            drop_pair=drop_pair,
        )
        ds.to_zarr(proc / "TRITONSWMM_SWMM_link_summary.zarr", mode="w", consolidated=False)
        # The G3 read target: zarr v3 per-member store, attributes under `attributes` in
        # zarr.json. A `.zattrs` read raises FileNotFoundError here, which is the false-ABSENT
        # the specified read avoids.
        store = member / "analysis_datatree.zarr"
        store.mkdir(parents=True, exist_ok=True)
        payload = {"zarr_format": 3, "node_type": "group", "attributes": {}}
        if pin is not None and (stamp_only is None or suffix in stamp_only):
            payload["attributes"]["triton_producing_sha"] = pin
        (store / "zarr.json").write_text(json.dumps(payload))
    return root


def test_enumerated_instrument_is_BLIND_to_the_truncation_the_derived_one_catches(tmp_path):
    """THE FAILING DEMONSTRATION, and it is the load-bearing one.

    A repair shown only to pass has not been shown to be a repair. This pins the defect's
    signature and asserts BOTH halves against the SAME artifact pair:

      * the ENUMERATED four (today's shipped ``TRACKED_VARS``) are all equal, so an
        instrument keyed on them reports AGREEMENT on a truncated member;
      * the DERIVED set catches it.

    Asserting only the second half would leave "the derived set fails" indistinguishable
    from "any instrument would have failed here", which is the distinction the whole §8.7.1
    finding turns on.
    """
    from hhemt.eda.cross_sim_identity import TRACKED_VARS, compared_columns_for

    ref = _link_summary()
    truncated = _link_summary(truncate="flow_cms")

    # (1) The enumerated instrument is BLIND: every one of today's four names that is PRESENT
    #     on this artifact is bitwise equal across the truncated pair.
    enumerated_present = [v for v in TRACKED_VARS if v in ref.data_vars]
    assert enumerated_present, "fixture must exercise the enumerated names, else this proves nothing"
    for var in enumerated_present:
        assert compare_variable_exact(ref[var], truncated[var])["identical"], (
            f"{var} differs — fixture no longer reproduces the defect signature (an immune "
            f"column must be IDENTICAL for the blindness demonstration to mean anything)"
        )

    # (2) The derived instrument CATCHES it.
    columns, prov = compared_columns_for(ref)
    assert prov["n_at_risk"] == 4 and prov["n_derived"] == 8, prov
    differing = [v for v in columns if not compare_variable_exact(ref[v], truncated[v])["identical"]]
    assert differing == ["flow_cms_max"], f"the derived set must catch exactly the truncated reduction, got {differing}"


def test_compare_arms_DISAGREES_on_a_truncated_member(tmp_path):
    """The failing demonstration end-to-end through ``compare_arms``, same pin on both sides."""
    from hhemt.eda.cross_sim_identity import compare_arms

    clean = _write_arm(tmp_path / "cleanPR", "members", pin=_PIN_P)
    resumed = _write_arm(tmp_path / "resumePS", "members", pin=_PIN_P, truncate_on=("flow_cms", "1"))

    res = compare_arms(arm_root=resumed, arm_layout="members", reference_root=clean, reference_layout="members")
    assert res.verdict == "DISAGREE", f"{res.verdict}: {res.reason}"
    assert res.n_compared > 0 and res.n_members_compared == 2
    bad = [d for d in res.details if d.get("verdict") == "DISAGREE"]
    assert {d["variable"] for d in bad} == {"flow_cms_max"}, bad
    assert {d["member"] for d in bad} == {"1"}, "only the truncated member may disagree"
    # The published figures §8.7.2's G1 requires be OUTPUTS rather than assumptions.
    assert all(p["n_at_risk"] == 4 for p in res.provenance)
    assert res.pins["arm"]["pin"] == res.pins["reference"]["pin"] == _PIN_P


def test_compare_arms_AGREES_on_an_untruncated_pair(tmp_path):
    """The PASSING demonstration — run second, and only meaningful because the first exists."""
    from hhemt.eda.cross_sim_identity import compare_arms

    clean = _write_arm(tmp_path / "cleanPR", "members", pin=_PIN_P)
    resumed = _write_arm(tmp_path / "resumePS", "members", pin=_PIN_P)
    res = compare_arms(arm_root=resumed, arm_layout="members", reference_root=clean, reference_layout="members")
    assert res.verdict == "AGREE", f"{res.verdict}: {res.reason}"
    # 11 columns x 2 members = 22; the number is pinned so a silently-narrowing set is a RED,
    # not a quieter green.
    assert res.n_compared == 22, res.n_compared


# ---- The refusal semantics: NOT-EVALUATED is a first-class THIRD outcome ----


def test_absent_stamp_is_not_evaluated_never_agreement(tmp_path):
    """G3's ABSENT half — the ONLY property discriminating the unstamped run-root decoy.

    That tree passes member enumeration at 30, passes path resolution, and passes the
    file-set cardinality guard at 30 == 30 == 30. The pin is the one thing it fails.
    """
    from hhemt.eda.cross_sim_identity import NOT_EVALUATED, compare_arms

    clean = _write_arm(tmp_path / "decoy", "members", pin=None)  # unstamped
    resumed = _write_arm(tmp_path / "resumePS", "members", pin=_PIN_P)
    res = compare_arms(arm_root=resumed, arm_layout="members", reference_root=clean, reference_layout="members")
    assert res.verdict == NOT_EVALUATED and "G3" in res.reason
    assert res.pins["reference"]["n_stamped"] == 0
    assert res.pins["reference"]["n_stores"] == 2, "the stores must EXIST and merely lack the value"


def test_differing_stamp_is_not_evaluated_and_the_control_can_opt_out(tmp_path):
    """A cross-pin comparison is refused by default; the A1-vs-A4 control opts out explicitly."""
    from hhemt.eda.cross_sim_identity import NOT_EVALUATED, compare_arms

    other = _write_arm(tmp_path / "cleanP", "members", pin=_PIN_OTHER)
    resumed = _write_arm(tmp_path / "resumePS", "members", pin=_PIN_P)
    gated = compare_arms(arm_root=resumed, arm_layout="members", reference_root=other, reference_layout="members")
    assert gated.verdict == NOT_EVALUATED and "cross-pin" in gated.reason
    assert gated.pins["arm"]["pin"] == _PIN_P and gated.pins["reference"]["pin"] == _PIN_OTHER

    reported = compare_arms(
        arm_root=resumed,
        arm_layout="members",
        reference_root=other,
        reference_layout="members",
        require_pin_identity=False,
    )
    assert reported.verdict == "AGREE"
    assert reported.pins["arm"]["pin"] != reported.pins["reference"]["pin"], "both stamps still published"


def test_layout_asymmetry_does_not_pass_vacuously(tmp_path):
    """One glob on both sides matches N on one and 0 on the other and a zipper then passes.

    Named per side, the archive layout compares correctly against a live arm; named WRONGLY,
    the result is NOT-EVALUATED rather than agreement.
    """
    from hhemt.eda.cross_sim_identity import NOT_EVALUATED, compare_arms, member_suffixes

    archive = _write_arm(tmp_path / "archive", "subanalyses", pin=_PIN_P)
    live = _write_arm(tmp_path / "live", "members", pin=_PIN_P)
    # Mapping is identity on the suffix.
    assert member_suffixes(archive, "subanalyses") == member_suffixes(live, "members") == ("0", "1")

    ok = compare_arms(arm_root=live, arm_layout="members", reference_root=archive, reference_layout="subanalyses")
    assert ok.verdict == "AGREE", ok.reason

    wrong = compare_arms(arm_root=live, arm_layout="members", reference_root=archive, reference_layout="members")
    assert wrong.verdict == NOT_EVALUATED, f"a wrong per-side layout must REFUSE, got {wrong.verdict}"


def test_empty_derived_and_empty_member_sets_are_not_evaluated(tmp_path):
    """G1 per artifact, and the empty member intersection — the two remaining triggers."""
    from hhemt.eda.cross_sim_identity import NOT_EVALUATED, compare_arms, compared_columns_for

    # G1: an artifact carrying neither a pair nor a named-floor member yields nothing.
    barren = xr.Dataset(
        {"time_of_max_flow_min": (("event_iloc", "link_id"), np.array([[1.0, 2.0]]))},
        coords={"event_iloc": [0], "link_id": ["c1", "c2"]},
    )
    cols, prov = compared_columns_for(barren)
    assert cols == () and prov["g1_non_empty"] is False and prov["n_compared"] == 0

    # Empty member intersection: disjoint suffixes.
    a = _write_arm(tmp_path / "a", "members", pin=_PIN_P, suffixes=("0", "1"))
    b = _write_arm(tmp_path / "b", "members", pin=_PIN_P, suffixes=("8", "9"))
    res = compare_arms(arm_root=a, arm_layout="members", reference_root=b, reference_layout="members")
    assert res.verdict == NOT_EVALUATED and "empty member intersection" in res.reason


def test_g2_is_scoped_to_the_pair_derived_set_not_the_union(tmp_path):
    """G2 evaluated over the UNION would fire on the correct configuration.

    §8.7.2 scopes it to the pair-derived operands precisely because the hand-named floor
    enters from OUTSIDE the rule. This pins the scoping: the union is ODD (11) on a real link
    summary shape while G2 still reads True.
    """
    from hhemt.eda.cross_sim_identity import compared_columns_for, pair_rule_is_self_consistent

    cols, prov = compared_columns_for(_link_summary())
    assert len(cols) % 2 == 1, "the union is odd — which is why G2 must not range over it"
    assert prov["g2_pair_even"] is True
    assert pair_rule_is_self_consistent(frozenset(), frozenset()) is True, "and it holds on the zero case"


# ---- W1/W2/W4/W5/W6: the repair set from testing-specialist:4's withhold ----
#
# Each of the first four is a POSITIVE CONTROL: it was RED at 08948124 before the repair in
# the same session that turned it green, and the pre-fix verdict is recorded in the assertion
# message so a future reader sees what the instrument USED to return rather than only that it
# now refuses.


def test_a_one_sided_derived_set_REFUSES_rather_than_narrowing(tmp_path):
    """W1 — §8.7.2's property asserted at the VERDICT layer, not as detail rows.

    §8.7.2 states the pair rule *"yields the SAME eight on both sides, which is the property
    conjunct (A) actually needs."* Deriving the compared set from the REFERENCE alone does
    not assert that property: a column absent on the arm side becomes a per-column
    NOT-EVALUATED detail row, the loop CONTINUES, and the surviving columns all agree — so a
    live column-name asymmetry (the ``capacity_setting`` vs ``capacity`` model-path spelling,
    measured co-live on one host in one run) silently NARROWS the comparison and returns
    AGREE over 9 columns where 11 were owed.

    PRE-FIX at 08948124 this returned ``AGREE`` with ``n_compared == 18``.
    """
    from hhemt.eda.cross_sim_identity import NOT_EVALUATED, compare_arms

    clean = _write_arm(tmp_path / "cleanPR", "members", pin=_PIN_P)
    # The arm's summaries carry one FEWER at-risk pair than the reference's.
    resumed = _write_arm(tmp_path / "resumePS", "members", pin=_PIN_P, drop_pair="velocity_mps")

    res = compare_arms(arm_root=resumed, arm_layout="members", reference_root=clean, reference_layout="members")
    assert res.verdict == NOT_EVALUATED, (
        f"a one-sided derived set must REFUSE at the verdict layer, not narrow to the "
        f"intersection; got {res.verdict}: {res.reason}"
    )
    assert "asymmetric compared set" in res.reason, res.reason
    # The refusal names BOTH sides' cardinalities and the symmetric difference, so a reader
    # can tell an asymmetry from an empty set without re-deriving either.
    assert "velocity_mps_max" in res.reason and "velocity_mps_last" in res.reason, res.reason


def test_an_absent_pin_is_not_evaluated_even_when_pin_identity_is_waived(tmp_path):
    """W2 — the `G3` opt-out waives pin INEQUALITY only, never the ABSENT half.

    Spanning two pins presupposes two pins, so §8.7.3's one exemption — the `A1`-versus-`A4`
    control — cannot coherently reach an arm carrying no stamp at all. The module's own
    in-code comment calls the absent half *"the sole discriminator"* against the unstamped
    run-root decoy, which passes member enumeration, path resolution and the 30 == 30 == 30
    cardinality guard and fails on the pin alone.

    PRE-FIX at 08948124 this returned ``AGREE``: both absent-OR-non-uniform and differing
    were nested inside ``if require_pin_identity:``, so ``False`` waived both.
    """
    from hhemt.eda.cross_sim_identity import NOT_EVALUATED, compare_arms

    decoy = _write_arm(tmp_path / "decoy", "members", pin=None)
    resumed = _write_arm(tmp_path / "resumePS", "members", pin=_PIN_P)

    res = compare_arms(
        arm_root=resumed,
        arm_layout="members",
        reference_root=decoy,
        reference_layout="members",
        require_pin_identity=False,
    )
    assert res.verdict == NOT_EVALUATED, (
        f"require_pin_identity=False must waive INEQUALITY only; an unstamped comparand is "
        f"NOT-EVALUATED under both settings. Got {res.verdict}: {res.reason}"
    )
    assert "pin absent or non-uniform" in res.reason, res.reason
    assert res.pins["require_pin_identity"] is False, "the waiver is still published"

    # And the control the exemption EXISTS for is unaffected: two arms, two DIFFERENT stamps,
    # both present -> the cross-pin comparison still runs and is reported.
    other = _write_arm(tmp_path / "cleanP", "members", pin=_PIN_OTHER)
    reported = compare_arms(
        arm_root=resumed,
        arm_layout="members",
        reference_root=other,
        reference_layout="members",
        require_pin_identity=False,
    )
    assert reported.verdict == "AGREE", f"the A1-vs-A4 control must still run: {reported.reason}"


def test_a_partially_stamped_arm_yields_no_pin(tmp_path):
    """W4 — ``n_stamped != n_members`` is NOT-EVALUATED, in EITHER direction.

    ``sorted(set(values))`` collapses a partially-stamped arm onto the one value its stamped
    members carry, so 1-of-30 stamped is indistinguishable from 30-of-30 at the return. The
    denominator must be the DECLARED population, which is the member count — the same
    argument ``scripts/experiments/synth_compute_config.py::check_arm_pin_stamp`` already
    makes for the mint-time check, and the same reason: a verdict derived as "no bad stores
    were found" cannot distinguish "examined 30, all good" from "examined 1".

    PRE-FIX at 08948124 ``read_arm_pin`` returned ``_PIN_P`` here and ``compare_arms``
    returned ``AGREE``.
    """
    from hhemt.eda.cross_sim_identity import NOT_EVALUATED, compare_arms, read_arm_pin

    partial = _write_arm(tmp_path / "partial", "members", pin=_PIN_P, suffixes=("0", "1"), stamp_only=("0",))
    pin, prov = read_arm_pin(partial, "members")
    assert prov["n_members"] == 2 and prov["n_stores"] == 2 and prov["n_stamped"] == 1, prov
    assert prov["distinct_pins"] == (_PIN_P,), "one uniform VALUE, over a partial population"
    assert pin is None, "a partially-stamped arm must not read as uniform"

    # A member directory carrying NO store at all is the other direction of the same gap: it
    # is matched by no glob, so it is neither examined nor reported missing.
    full = _write_arm(tmp_path / "full", "members", pin=_PIN_P)
    res = compare_arms(arm_root=partial, arm_layout="members", reference_root=full, reference_layout="members")
    assert res.verdict == NOT_EVALUATED and "pin absent or non-uniform" in res.reason, res.reason

    # The guard is PRESENCE-keyed, not value-keyed: a fully-stamped arm still reads its pin.
    assert read_arm_pin(full, "members")[0] == _PIN_P


def test_g2_odd_pair_derived_set_is_consulted_by_compare_arms(tmp_path, monkeypatch):
    """W5 — `G2` is a GATE, not only a published figure.

    ``compared_columns_for`` computes ``g2_pair_even`` and ``compare_arms`` consulted
    ``g1_non_empty`` alone, so the EVEN conjunct was provenance a reader could check and the
    instrument could not act on. `G2` is provably vacuous on every set the pair rule
    produces — ``IMMUNE`` is the injective image of ``AT_RISK`` with a disjoint range — so it
    is unreachable without a source edit to ``derive_compared_columns``, and that edit is
    exactly what it exists to catch. The monkeypatch IS that edit, applied for one test.

    PRE-FIX at 08948124 this returned ``AGREE``.
    """
    import hhemt.eda.cross_sim_identity as csi

    clean = _write_arm(tmp_path / "cleanPR", "members", pin=_PIN_P)
    resumed = _write_arm(tmp_path / "resumePS", "members", pin=_PIN_P)

    real = csi.derive_compared_columns

    def _hand_added(ds):
        at_risk, immune, _derived = real(ds)
        # A future edit that hand-adds a `_max` BY NAME to a pair-derived set: the member
        # enters from OUTSIDE the rule, so the pair identity no longer holds.
        derived = at_risk | immune | frozenset({"max_over_full_flow"})
        return at_risk, immune, derived

    monkeypatch.setattr(csi, "derive_compared_columns", _hand_added)
    assert csi.compared_columns_for(_link_summary())[1]["g2_pair_even"] is False, "the patch must break G2"

    res = csi.compare_arms(arm_root=resumed, arm_layout="members", reference_root=clean, reference_layout="members")
    assert res.verdict == csi.NOT_EVALUATED, f"an odd PAIR-DERIVED set must refuse; got {res.verdict}: {res.reason}"
    assert "G2" in res.reason, res.reason


def test_float32_eps_is_the_coarsest_floor_over_the_admitted_float_dtype_set() -> None:
    """W6's accepted residual, pinned as a CLASS claim rather than left as prose.

    ``check_cross_sim_identity`` publishes a HARDCODED ``detection_floor`` of
    ``np.finfo(np.float32).eps`` beside a population derived at run time by
    ``compared_columns_for``. The constant is CORRECT and is NOT derived, and the hazard the
    reviewer names is that the next author reads a correct constant as a derived one. The
    claim that makes it correct — *"float32 eps is the coarsest over today's admitted set"* —
    quantifies over a class, so it is pinned here rather than asserted in a comment.

    MEASURED 2026-10-04 over ``src/``: ``float16`` has ZERO occurrences and the only float
    dtypes named anywhere in the package are ``float32`` (66 occurrences) and ``float64``
    (80). ``float16`` is the one standard float dtype COARSER than ``float32``
    (eps 9.77e-04 vs 1.19e-07), so its absence is what makes the constant the ceiling. This
    test fails the moment a coarser float dtype enters the package, which is the moment the
    published floor starts UNDERSTATING what the verdict can see.
    """
    import pathlib

    floor = float(np.finfo(np.float32).eps)
    admitted = ("float32", "float64")
    for name in admitted:
        assert float(np.finfo(np.dtype(name)).eps) <= floor, (
            f"{name} is coarser than the published detection_floor {floor:g}"
        )

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "hhemt"
    assert src.is_dir(), src
    coarser = sorted(
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if any(tok in p.read_text() for tok in ("float16", "np.half", "'<f2'", '"<f2"'))
    )
    assert coarser == [], (
        f"a float dtype coarser than float32 entered the package ({coarser}); the hardcoded "
        f"detection_floor in check_cross_sim_identity now UNDERSTATES the verdict's "
        f"resolution and must be derived from the compared columns' dtypes"
    )
