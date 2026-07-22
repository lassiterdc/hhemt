"""Synth-tier LOCAL, network-free run-proof for the DOI-ingestion path (ADR-13 C9).

Closes the gap that Phase-1's ``from_doi`` tests prove CONSTRUCTION only, while
the sole run-proof (Phase-3 e2e) is operator-gated (credentials + live
Zenodo/HydroShare). Here the REAL self-contained emit -> ingest -> RUN path is
exercised end to end with the network fetch mocked and the toolchain reused:

  emit  : ``rendered_synth_multi_sim`` -> ``bundle_report_data`` (to tmp)
  ingest: ``TRITON_SWMM_experiment.from_doi`` (``_fetch_bundle_zip`` mocked)
  run   : ``exp.analysis.test()`` (ADR-8), ``software_dir`` symlinked at the
          pre-compiled ``tritonswmm_cpu_compiled`` tree so NO git-clone and NO
          recompile fire -- ``compilation_cpu_successful`` reads the on-disk
          ``build_tritonswmm_cpu/compilation.log`` markers through the symlink.

SCOPE LIMIT (defect-10): this proof runs a NATIVE bundle and supplies a
pre-compiled host tree, so it is structurally incapable of exercising the
CONTAINER prep path -- where no host build exists at all. Container-mode
``prepare_scenario`` coverage lives in ``test_synth_container_mode.py``
(fast tier, no compile). Do not read a green here as container-mode coverage.

The run happens entirely under a tmp ``bundle_root/_test/`` -- never the session
fixture's snapshotted analysis tree -- so the ``_assert_no_sha_drift`` finalizer
is not implicated. File-scoped ``requires_snakemake_subprocess`` + ``slow``: this
launches Snakemake as a subprocess and is a compile-tier test, so it is
deselected from the default ``-m "not slow"`` fast set and serialized under xdist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut
from hhemt.experiments import TRITON_SWMM_experiment

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    pytest.mark.slow,
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(),
        reason="Local coupled run-proof; do not launch on an HPC scheduler node.",
    ),
]


@pytest.fixture
def _prebuilt_software_dir(tritonswmm_cpu_compiled, tmp_path) -> Path:
    """A private tmp ``software_dir`` whose ``tritonswmm``/``swmm`` subdirs are
    SYMLINKS into the shared ``tritonswmm_cpu_compiled`` synth tree.

    ``from_doi`` maps ``software_dir`` to ``software_dir/'tritonswmm'`` +
    ``software_dir/'swmm'`` (bundle/_emit.py). The synth family compiles into
    ``_software_root/'triton'`` + ``_software_root/'swmm'`` (test_case_builder.py),
    so the symlinks perform the ``triton -> tritonswmm`` rename while pointing at
    the REAL compiled build dirs (RPATH-safe: execution resolves to the physical
    path the binary was linked against). Depending on ``tritonswmm_cpu_compiled``
    guarantees the CPU build exists AND inherits its toolchain-absent skip /
    ``HHEMT_REQUIRE_COMPILE_TIER`` fail-closed gate.
    """
    from tests.fixtures.test_case_catalog import Local_TestCases

    case = Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False)
    cfg = case.analysis._system.cfg_system
    compiled_triton = Path(cfg.TRITONSWMM_software_directory)
    compiled_swmm = Path(cfg.SWMM_software_directory)
    assert (compiled_triton / "build_tritonswmm_cpu" / "compilation.log").exists(), (
        "tritonswmm_cpu_compiled did not leave a CPU compilation.log at the "
        f"expected path under {compiled_triton} -- the recompile-skip precondition "
        "is unmet; the run-proof would git-clone + recompile."
    )

    software_dir = tmp_path / "software"
    software_dir.mkdir()
    (software_dir / "tritonswmm").symlink_to(compiled_triton, target_is_directory=True)
    (software_dir / "swmm").symlink_to(compiled_swmm, target_is_directory=True)
    return software_dir


@pytest.fixture
def _reprex_bundle_zip(rendered_synth_multi_sim, tmp_path) -> Path:
    """Emit a REAL self-contained reprex bundle from the rendered synth multi_sim
    analysis to a tmp path (read-only consumption of the session fixture; mirrors
    ``test_experiments_from_doi.py::self_contained_bundle``). Emitting to tmp --
    never ``analysis_dir/render_bundle/`` -- keeps the session fixture's no-SHA
    -drift finalizer out of the picture."""
    out = tmp_path / "emit"
    out.mkdir()
    return rendered_synth_multi_sim.bundle_report_data(out / "bundle.zip")


def test_from_doi_reconstituted_bundle_runs_locally(
    _reprex_bundle_zip, _prebuilt_software_dir, monkeypatch, tmp_path
):
    """emit -> from_doi (fetch mocked) -> analysis.test() proves a RECONSTITUTED
    bundle RUNS end to end, network-free and without a recompile."""
    # Mock the network fetch: return the pre-built zip; from_doi still runs the
    # real extract + reconstitute + fail-closed gates against it.
    monkeypatch.setattr(
        TRITON_SWMM_experiment,
        "_fetch_bundle_zip",
        classmethod(lambda cls, *args, **kwargs: Path(_reprex_bundle_zip)),
    )

    exp = TRITON_SWMM_experiment.from_doi(
        doi="10.5281/zenodo.123456",
        host="zenodo",
        target_dir=tmp_path / "ingest",
        software_dir=_prebuilt_software_dir,
    )

    # Guard the network-free precondition: the toolchain the run will use must be
    # the symlinked pre-built tree, NOT bundle_root/software (which would trigger
    # a git-clone + full recompile). Fail fast + legibly if this ever regresses.
    resolved_triton = Path(exp.system.cfg_system.TRITONSWMM_software_directory).resolve()
    assert resolved_triton == Path(_prebuilt_software_dir / "tritonswmm").resolve(), (
        "from_doi did not adopt the pre-built software_dir; the run would "
        f"git-clone + recompile. Got {resolved_triton}."
    )

    # RUN the reconstituted analysis's strict least-demanding subset end to end
    # (compile[SKIPPED via on-disk log markers] -> run -> process -> consolidate
    # -> report). dry_run defaults to False -- this is a real run.
    result = exp.analysis.test(execution_mode="local", verbose=False)

    # "It ran" proof set (mirrors test_analysis_test_end_to_end.py):
    assert (exp.bundle_root / "_test").exists(), "no _test subtree materialized"
    assert result.subanalyses, "analysis.test() produced no _test sub-analyses"
    for sub_result in result.subanalyses:
        # validate_analysis's 7 checks inspect REAL consolidated outputs -- a
        # Snakemake --dry-run could never satisfy them.
        tst_ut.assert_analysis_workflow_completed_successfully(sub_result.analysis)
