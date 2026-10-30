"""Reviewer tests for work chunk ``1b.1`` — the ``(hardware family, DEM resolution)`` partition key.

The instrument under test is the INTRA-arm comparison partition inside
``hhemt.eda.cross_sim_identity.check_cross_sim_identity``. Before this chunk the partition was
the HARDWARE FAMILY alone, so a GPU arm carrying 3.5 / 1.10 / 0.35 m members inside ONE family
compared across a resolution boundary and reported the resolution difference as a
reproducibility failure — a false FAIL by construction, not by occurrence.

Each test carries exactly one finding, named in its docstring under ``Finding:``.

The stub harness mirrors ``tests/test_eda_cross_sim_identity.py``'s fast tier — no solver build
and no HPC — stubbing the three surfaces ``check_cross_sim_identity`` touches on a sub
(``cfg_analysis`` for ``_ref_rank``, ``process._MODE_CONFIG`` + ``_retrieve_combined_output``
for ``_enabled_modes``, ``analysis_paths.analysis_dir`` for the emitted provenance sources) plus
one surface that harness has no reason to carry and every test here needs: ``_system.cfg_system``,
which is where ``TRITONSWMM_analysis`` actually keeps ``target_dem_resolution``
(``analysis.py:3658`` and ``:3683`` both reach it through ``a._system.cfg_system``; the analysis
exposes no ``cfg_system`` attribute of its own, and defines no ``__getattr__``).

That last point is why the harness cannot be reused as-is: a stub sub with no ``_system`` yields
an unreadable resolution for EVERY member, which collapses the partition back to family-only
with no exception raised and no line logged — so a suite built on such stubs cannot tell the
widened key from the family-only one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from hhemt.eda import cross_sim_identity as csi

_DEPTH_MODE = "tritonswmm"
_LINK_MODE = "tritonswmm_swmm_link"

#: A gross, unambiguous depth perturbation. Only the ORDERING matters here (a member differs
#: from its reference or it does not), so there is no epsilon to pick and nothing to tune.
_BUMP = 1.0


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


def _summaries(depth_bump: float = 0.0):
    """The flat per-scenario summaries a sub returns, optionally perturbed by ``depth_bump``.

    Two modes so the partition signature (``_PARTITION_VARS`` = ``max_wlevel_m``,
    ``max_flow_cms``) is fully populated. ``max_wlevel_m`` is float32 to match the on-disk dtype
    the detection floor is derived from; the link-mode variable is float64, as in the real
    pipeline.
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


class _Process:
    """Stands in for ``TRITONSWMM_analysis_post_processing``.

    ``_MODE_CONFIG`` must be reachable via the live ``.process`` instance (never module-level) and
    ``_retrieve_combined_output`` must raise ``FileNotFoundError`` on an absent mode — that is the
    existence guard ``_enabled_modes`` keys on.
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


def _master(tmp_path, spec):
    """Build a stub sensitivity master from ``{member_id: (cfg, resolution, depth_bump)}``.

    Each sub gets a real on-disk ``analysis_dir`` because the emitted provenance source path is
    ``{member_dir}/analysis_datatree.zarr``; that file need not exist (the source gate accepts a
    non-existent path, and the ``.zarr`` suffix clears the directory-as-source clause anyway).
    Both summary KINDS are materialized — a ``.nc`` FILE and a ``.zarr`` STORE — because
    ``target_processed_output_type`` defaults to ``zarr`` and the emitter refuses an empty
    source set.
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
        subs[member_id] = _Sub(cfg, _summaries(bump), d, resolution)
    root = tmp_path / "master"
    root.mkdir(parents=True, exist_ok=True)
    return _Master(subs, root)


def _partition_key():
    """``cross_sim_identity._partition_key``, resolved lazily.

    Lazy rather than module-level so that this file COLLECTS against the pre-chunk tree: a
    module-level ``from ... import _partition_key`` raises ``ImportError`` at collection and
    would take the REGRESSION test below down with it, destroying the only pre-change red this
    file can produce.
    """
    return csi._partition_key


def test_a_cross_resolution_difference_is_not_a_reproducibility_failure(tmp_path):
    """Finding 1 (REGRESSION) — a within-family, across-resolution difference must not fail.

    Property: on a GPU master whose two members differ ONLY in ``target_dem_resolution`` (3.5 m
    vs 0.35 m) and in the depth field that resolution implies, the verdict carries no divergence
    row and ``passed`` is ``True``.

    Class: REGRESSION. Red on the pre-chunk tree, output pasted in the round file — the family-only
    partition put both members in one group, compared the 3.5 m member against the 0.35 m one, and
    recorded a ``variable``-carrying divergence row under ``passed=False``.

    Why this is required: different DEM resolutions are different physics. A comparison drawn
    across that boundary reports the resolution as a reproducibility failure, and the arm this
    run exists to benchmark carries 3.5 / 1.10 / 0.35 m inside ONE hardware family.

    What kills it: reverting the comparison partition to ``_family_key`` (one group per hardware
    family) reinstates the cross-resolution comparison and reds both assertions.

    A second correct implementation that still passes: one that groups on a pre-rendered
    ``"{family}@{resolution}"`` STRING key rather than on the ``(str, float)`` tuple, and that
    selects each group's reference by its own sort rather than by first-encounter order — it draws
    the same partition, so no cross-resolution tuple is produced and this test stays green.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_0p35": (_Cfg("gpu", n_gpus=1), 0.35, _BUMP),
            "gpu_1g_3p5": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
        },
    )
    result = csi.check_cross_sim_identity(master)

    divergences = [row for row in result.verdict.details if "variable" in row]
    assert divergences == [], (
        f"a difference that is exactly the DEM resolution must not be reported as a "
        f"reproducibility failure; got {len(divergences)} divergence row(s): {divergences}"
    )
    assert result.verdict.passed is True, f"verdict failed on a resolution-only difference: {result.verdict.summary}"


def test_partition_key_reads_the_resolution_from_the_live_system_attribute():
    """Finding 2 (NEW CAPABILITY) — the key's second dimension comes off ``_system.cfg_system``.

    Property: for a sub carrying ONLY the live attribute path (``sub._system.cfg_system``, and no
    direct ``sub.cfg_system``), the partition key is ``(family, resolution)`` with the resolution
    read as a float.

    Class: NEW CAPABILITY. The function does not exist on the pre-chunk tree, so its pre-fix run
    is a construction error and carries no information; the plausible-wrong-implementation arm is
    the discriminator.

    Plausible wrong implementation: reading ``getattr(sub, "cfg_system", None)`` and stopping
    there. ``TRITONSWMM_analysis`` has no such attribute, so that arm returns
    ``("gpu", None)`` for EVERY member on a real master, which collapses the partition to
    family-only with no exception raised and no line logged — the failure this key exists to
    prevent, arriving silently.

    Why this is required: the criterion names the resolution's source as each member's in-memory
    ``cfg_system.target_dem_resolution``, and the attribute that reaches it on a real analysis is
    ``_system.cfg_system`` (``analysis.py:3658``, ``:3683``).

    What kills it: a ``_resolution_key`` that reads no attribute, or reads a different one, and
    therefore returns ``None`` here.

    A second correct implementation that still passes: a key function that resolves the path once
    via a cached helper, or one that prefers ``sub._system.cfg_system`` FIRST and falls back to a
    direct ``cfg_system`` — the order is not what this test pins; the value is.
    """
    sub = _Sub(_Cfg("gpu", n_gpus=1), _summaries(), Path("."), 0.35)
    assert not hasattr(sub, "cfg_system"), (
        "the fixture must expose the resolution ONLY through _system.cfg_system, or this test "
        "cannot distinguish the live path from a direct attribute"
    )
    assert _partition_key()(sub) == ("gpu", 0.35), (
        f"expected ('gpu', 0.35) read through _system.cfg_system; got {_partition_key()(sub)!r} — "
        f"a None resolution here means the partition silently collapses to family-only"
    )


def test_one_reference_per_family_and_resolution_group():
    """Finding 3 (NEW CAPABILITY) — the reference map is keyed per PARTITION, not per family.

    Property: over a fixture spanning two families and two resolutions, the widened key yields
    three entries — one per ``(family, resolution)`` pair — and the SAME fixture under the
    default (family) key yields two. The trailing assertion is the control: without it the test
    could pass under a rule that never reads the resolution at all.

    Class: NEW CAPABILITY, discriminator = the plausible wrong implementation below.

    Plausible wrong implementation: a ``partition_key`` that calls ``_family_key`` and discards
    the resolution — green on the control, red on the three-entry equality.

    Why this is required: the comparison and the disclosure are two maps built by ONE function,
    so a wrong key here re-anchors every member of a multi-resolution arm onto one member.

    What kills it: collapsing the key back to the family, or keying the map on the member_id.

    A second correct implementation that still passes: one that builds the per-group map by
    iterating groups explicitly instead of ``dict.setdefault`` over a ``_ref_rank``-ordered list,
    provided the winner within each group is still that group's ``_ref_rank`` winner.
    """
    items = [
        ("cpu_serial", _Sub(_Cfg("serial", n_mpi_procs=1, n_omp_threads=1), _summaries(), Path("."), 3.5)),
        ("gpu_a", _Sub(_Cfg("gpu", n_gpus=1), _summaries(), Path("."), 3.5)),
        ("gpu_b", _Sub(_Cfg("gpu", n_gpus=1), _summaries(), Path("."), 0.35)),
    ]
    ordered = sorted(items, key=csi._ref_rank)

    assert csi._references_by_family(ordered, partition_key=_partition_key()) == {
        ("cpu", 3.5): "cpu_serial",
        ("gpu", 3.5): "gpu_a",
        ("gpu", 0.35): "gpu_b",
    }, "the reference map must carry one entry per (hardware family, DEM resolution) group"

    assert csi._references_by_family(ordered) == {"cpu": "cpu_serial", "gpu": "gpu_a"}, (
        "control: the family-only map on this fixture pools both GPU members into one group, "
        "which is what makes the widened map distinguishable from it rather than coincidentally equal"
    )


def test_every_compared_tuple_shares_its_reference_group(tmp_path):
    """Finding 4 (REGRESSION, the ATTRIBUTION half of Finding 1) — no tuple crosses the boundary.

    Property: every divergence row names a ``ref_member_id`` whose ``target_dem_resolution``
    equals its own ``sa_id``'s. Asserted over the divergence population, which the companion
    assertion proves non-empty so the property cannot pass vacuously.

    Class: REGRESSION — red on the pre-chunk tree, where the single GPU family anchored every
    member on one reference and the 0.35 m member's rows carried a 3.5 m reference (or vice
    versa, depending on which member ``_ref_rank`` selected). Output pasted in the round file.

    Why this is required and why it is a SEPARATE test from Finding 1: Finding 1 asserts no
    divergence is ATTRIBUTED across the boundary; this one asserts no COMPARISON is drawn across
    it. They fail independently — an implementation that suppressed the divergence row while still
    comparing across the boundary would pass Finding 1 and fail this one. The pair also pins that
    the change did not simply stop comparing: a within-group divergence must still FAIL.

    What kills it: reverting the key to the family, which re-anchors the 0.35 m member on the
    3.5 m reference; or widening a tolerance instead of re-partitioning, which would green
    ``passed`` while leaving the attribution wrong.

    A second correct implementation that still passes: one that records the reference as a
    ``(family, resolution)`` pair rather than a member_id and resolves it to a member at read
    time — the resolution it resolves to is the same one this test recomputes.
    """
    master = _master(
        tmp_path,
        {
            "gpu_1g_0p35_c": (_Cfg("gpu", n_gpus=1), 0.35, _BUMP),
            "gpu_1g_3p5_a": (_Cfg("gpu", n_gpus=1), 3.5, 0.0),
            "gpu_1g_3p5_b": (_Cfg("gpu", n_gpus=1), 3.5, _BUMP),
        },
    )
    result = csi.check_cross_sim_identity(master)

    resolution_of = {
        member: sub._system.cfg_system.target_dem_resolution for member, sub in master.sensitivity.members.items()
    }
    rows = [row for row in result.verdict.details if "ref_member_id" in row]
    assert rows, (
        "no divergence row carried a reference attribution, so the property below is vacuous — "
        "the fixture must produce at least one INTRA-group divergence"
    )
    for row in rows:
        assert resolution_of[row["ref_member_id"]] == resolution_of[row["sa_id"]], (
            f"member {row['sa_id']!r} at {resolution_of[row['sa_id']]} m was compared against "
            f"reference {row['ref_member_id']!r} at {resolution_of[row['ref_member_id']]} m — "
            f"a cross-resolution comparison, which is a false FAIL by construction"
        )
    assert result.verdict.passed is False, (
        "the within-group divergence this fixture plants must still fail the verdict; a PASS here "
        "means the comparison was disabled rather than re-partitioned"
    )
