"""Phase 3 (R8) — scoped reprocess-delete runner + Snakefile-builder tests.

Environment-independent coverage (no GPU / no real simulation pipeline):

- The three NEW scoped reprocess-delete runners exercised via their public
  ``main()`` entrypoints with on-disk tmp trees:
  - ``delete_processed_runner``        — deletes ``sims/{eid}/processed/`` ONLY,
                                          preserving sibling raw ``out_*``.
  - ``delete_reprocess_zarr_runner``   — deletes the consolidated zarr(s) ONLY,
                                          preserving report/plots/_status.
  - ``delete_subanalysis_reprocess_runner`` (D-scope Option C) — deletes the
                                          sub's ``sims/*/processed/`` (only with
                                          ``--delete-processed``) + the sub's
                                          ``analysis_datatree.zarr``.
  Each: writes the submission sentinel under ``SLURM_JOB_ID`` and cleans it in
  ``finally`` on both clean-return and exception paths.

- The ``_build_reprocess_delete_snakefile_content`` builder (uses lightweight
  constructed analyses — no sims run):
  - non-sensitivity ``start_with='process'`` emits per-event
    ``delete_processed_*`` rules + a ``delete_reprocess_zarr_consolidation`` rule.
  - sensitivity (D-scope Option C) emits ONE ``delete_subanalysis_reprocess_{sa}``
    rule per sub (NOT per-(sa,event)) + a master consolidation rule.
  - the reprocess-delete namespace is isolated to ``_deleting_reprocess/``.

The synthetic end-to-end deletion tests (``test_synth_07`` / ``test_synth_08``
process-stage rebuild) require the GPU/compile synth pipeline and run on HPC.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def slurm_env(monkeypatch):
    """Set SLURM_JOB_ID so the runners take the sentinel-writing branch."""
    monkeypatch.setenv("SLURM_JOB_ID", "999999")
    monkeypatch.setenv("SLURM_JOB_NAME", "test-job")
    yield


# ---------------------------------------------------------------------------
# delete_processed_runner — processed/-only deletion, sibling raw preserved
# ---------------------------------------------------------------------------


def _seed_scenario(analysis_dir: Path, event_id: str) -> Path:
    sim = analysis_dir / "sims" / event_id
    (sim / "processed").mkdir(parents=True)
    (sim / "processed" / "summary.zarr").mkdir()
    (sim / "out_triton").mkdir(parents=True)  # sibling raw rebuild source
    (sim / "out_triton" / "h_0001.bin").write_text("raw")
    return sim


def test_delete_processed_runner_removes_processed_preserves_raw(tmp_path, slurm_env):
    from hhemt import delete_processed_runner as runner

    analysis_dir = tmp_path / "analysis"
    sim = _seed_scenario(analysis_dir, "evt_1")

    rc = runner.main(["--event-id", "evt_1", "--analysis-dir", str(analysis_dir)])
    assert rc == 0
    assert not (sim / "processed").exists(), "processed/ must be deleted"
    assert (sim / "out_triton" / "h_0001.bin").exists(), "sibling raw out_* must survive"

    flag = analysis_dir / "_status" / "_deleting_reprocess" / "processed_evt-evt_1.flag"
    assert flag.exists(), "completion flag must be written"
    sentinel = analysis_dir / "_status" / "_submitted" / "delete_processed_evt_1.json"
    assert not sentinel.exists(), "submission sentinel must be cleaned in finally"


def test_delete_processed_runner_cleans_sentinel_on_exception(tmp_path, slurm_env):
    from hhemt import delete_processed_runner as runner

    analysis_dir = tmp_path / "analysis"
    _seed_scenario(analysis_dir, "evt_1")

    with patch.object(runner, "fast_rmtree", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            runner.main(["--event-id", "evt_1", "--analysis-dir", str(analysis_dir)])

    sentinel = analysis_dir / "_status" / "_submitted" / "delete_processed_evt_1.json"
    assert not sentinel.exists(), "sentinel must be cleaned by finally on exception"


def test_delete_processed_runner_no_sentinel_without_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    from hhemt import delete_processed_runner as runner

    analysis_dir = tmp_path / "analysis"
    _seed_scenario(analysis_dir, "evt_1")
    runner.main(["--event-id", "evt_1", "--analysis-dir", str(analysis_dir)])

    submitted = analysis_dir / "_status" / "_submitted"
    assert not submitted.exists() or not any(submitted.iterdir())


# ---------------------------------------------------------------------------
# delete_reprocess_zarr_runner — consolidated zarr(s) only, never report/plots
# ---------------------------------------------------------------------------


def test_delete_reprocess_zarr_runner_removes_zarrs_preserves_report(tmp_path, slurm_env):
    from hhemt import delete_reprocess_zarr_runner as runner

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "analysis_datatree.zarr").mkdir()
    (analysis_dir / "sensitivity_datatree.zarr").mkdir()
    (analysis_dir / "analysis_report.html").write_text("<html/>")
    (analysis_dir / "plots").mkdir()
    (analysis_dir / "plots" / "fig.png").write_text("png")

    rc = runner.main(["--analysis-dir", str(analysis_dir)])
    assert rc == 0
    assert not (analysis_dir / "analysis_datatree.zarr").exists()
    assert not (analysis_dir / "sensitivity_datatree.zarr").exists()
    assert (analysis_dir / "analysis_report.html").exists(), "report must be preserved"
    assert (analysis_dir / "plots" / "fig.png").exists(), "plots must be preserved"

    flag = analysis_dir / "_status" / "_deleting_reprocess" / "reprocess_consolidation.flag"
    assert flag.exists()


def test_delete_reprocess_zarr_runner_noop_on_absent_zarr(tmp_path, slurm_env):
    from hhemt import delete_reprocess_zarr_runner as runner

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    # only one of the two canonical zarrs present
    (analysis_dir / "analysis_datatree.zarr").mkdir()

    rc = runner.main(["--analysis-dir", str(analysis_dir)])
    assert rc == 0
    assert not (analysis_dir / "analysis_datatree.zarr").exists()
    flag = analysis_dir / "_status" / "_deleting_reprocess" / "reprocess_consolidation.flag"
    assert flag.exists()


# ---------------------------------------------------------------------------
# delete_subanalysis_reprocess_runner (Option C) — per-sub processed + zarr
# ---------------------------------------------------------------------------


def _seed_subanalysis(sub_dir: Path) -> None:
    for eid in ("evt_1", "evt_2"):
        (sub_dir / "sims" / eid / "processed").mkdir(parents=True)
        (sub_dir / "sims" / eid / "out_triton").mkdir(parents=True)
        (sub_dir / "sims" / eid / "out_triton" / "h.bin").write_text("raw")
    (sub_dir / "analysis_datatree.zarr").mkdir()


def test_subanalysis_reprocess_runner_deletes_processed_and_zarr(tmp_path, slurm_env):
    from hhemt import delete_subanalysis_reprocess_runner as runner

    sub_dir = tmp_path / "subanalyses" / "sa_3"
    _seed_subanalysis(sub_dir)

    rc = runner.main(
        ["--sa-id", "3", "--analysis-dir", str(sub_dir), "--delete-processed"]
    )
    assert rc == 0
    assert not (sub_dir / "sims" / "evt_1" / "processed").exists()
    assert not (sub_dir / "sims" / "evt_2" / "processed").exists()
    assert (sub_dir / "sims" / "evt_1" / "out_triton" / "h.bin").exists(), "raw preserved"
    assert not (sub_dir / "analysis_datatree.zarr").exists()

    flag = sub_dir / "_status" / "_deleting_reprocess" / "subanalysis_reprocess.flag"
    assert flag.exists()
    sentinel = sub_dir / "_status" / "_submitted" / "delete_subanalysis_reprocess_3.json"
    assert not sentinel.exists()


def test_subanalysis_reprocess_runner_preserves_processed_without_flag(tmp_path, slurm_env):
    """Without --delete-processed (start_with != 'process'), processed/ survives;
    only the sub's consolidated zarr is removed."""
    from hhemt import delete_subanalysis_reprocess_runner as runner

    sub_dir = tmp_path / "subanalyses" / "sa_3"
    _seed_subanalysis(sub_dir)

    rc = runner.main(["--sa-id", "3", "--analysis-dir", str(sub_dir)])
    assert rc == 0
    assert (sub_dir / "sims" / "evt_1" / "processed").exists(), "processed/ preserved"
    assert not (sub_dir / "analysis_datatree.zarr").exists(), "sub zarr removed"


def test_subanalysis_reprocess_runner_cleans_sentinel_on_exception(tmp_path, slurm_env):
    from hhemt import delete_subanalysis_reprocess_runner as runner

    sub_dir = tmp_path / "subanalyses" / "sa_3"
    _seed_subanalysis(sub_dir)

    with patch.object(runner, "fast_rmtree", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            runner.main(
                ["--sa-id", "3", "--analysis-dir", str(sub_dir), "--delete-processed"]
            )

    sentinel = sub_dir / "_status" / "_submitted" / "delete_subanalysis_reprocess_3.json"
    assert not sentinel.exists()


# ---------------------------------------------------------------------------
# _submit_delete_snakemake default-preservation (byte-for-byte caller safety)
# ---------------------------------------------------------------------------


def test_submit_delete_snakemake_defaults_preserve_analysis_delete_callers():
    """The new working_subdir/logfile_name params default to analysis.delete()'s
    namespace, so existing callers are byte-for-byte unchanged."""
    import inspect

    from hhemt.workflow import SnakemakeWorkflowBuilder

    params = inspect.signature(SnakemakeWorkflowBuilder._submit_delete_snakemake).parameters
    assert params["working_subdir"].default == ".snakemake_delete"
    assert params["logfile_name"].default == "snakemake_delete.log"


def test_pre_delete_guards_defaults_preserve_analysis_delete_namespace():
    """_pre_delete_guards parametrizes the lock-check namespace; defaults preserve
    analysis.delete()'s Snakefile.delete + .snakemake_delete/ (F-I Flag 2 — the
    reprocess-delete path overrides both so the guard inspects the correct lock)."""
    import inspect

    from hhemt.workflow import SnakemakeWorkflowBuilder

    params = inspect.signature(SnakemakeWorkflowBuilder._pre_delete_guards).parameters
    assert params["snakefile_name"].default == "Snakefile.delete"
    assert params["working_subdir"].default == ".snakemake_delete"


# ---------------------------------------------------------------------------
# Snakefile builder — non-sensitivity + sensitivity (Option C) + isolation
# ---------------------------------------------------------------------------


def test_build_reprocess_delete_snakefile_non_sensitivity(norfolk_multi_sim_analysis):
    """Non-sensitivity start_with='process' emits per-event delete_processed_*
    rules + a delete_reprocess_zarr_consolidation rule, all in the
    _deleting_reprocess/ namespace."""
    from hhemt.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_multi_sim_analysis
    builder = SnakemakeWorkflowBuilder(analysis)
    content = builder._build_reprocess_delete_snakefile_content(start_with="process")

    assert "rule delete_processed_" in content
    assert "hhemt.delete_processed_runner" in content
    assert "rule delete_reprocess_zarr_consolidation:" in content
    assert "hhemt.delete_reprocess_zarr_runner" in content
    # isolation: flags land in the scoped reprocess namespace, never _deleting/
    assert "_deleting_reprocess/" in content
    assert "_status/_deleting/" not in content
    # non-sensitivity must NOT emit per-sub rules
    assert "delete_subanalysis_reprocess_" not in content


def test_delete_rules_declare_a_log_and_redirect_into_it(norfolk_multi_sim_analysis):
    """Every scoped-delete rule carries BOTH halves of log capture.

    `log:` alone does not capture stdout/stderr — Snakemake only declares the path
    and exempts it from delete-on-failure; the shell must redirect into it. A
    directive without a redirect yields an empty file and a false sense of coverage,
    which is why both halves are asserted here rather than just the directive.

    Without this, a failing delete rule is undiagnosable: on 2026-07-20 the scoped
    delete failed 112 times and the ONLY surviving evidence was the outer snakemake
    log's "command exited with non-zero exit code" — no runner stderr anywhere. The
    SLURM executor's per-job log is not a fallback: `sacct` showed zero delete jobs
    for that run, so no per-job log was ever created, and under local `--cores`
    execution none exists by construction either.

    The `{log}` must reach the generated Snakefile LITERALLY (the emitter is an
    f-string, so it is written `{{log}}`); a single brace would interpolate the
    Python path instead and produce a redirect Snakemake does not manage.
    """
    from hhemt.workflow import SnakemakeWorkflowBuilder

    builder = SnakemakeWorkflowBuilder(norfolk_multi_sim_analysis)
    content = builder._build_reprocess_delete_snakefile_content(start_with="process")

    # Both emitters reachable on the non-sensitivity path declare a log...
    assert content.count("    log:\n") >= 2, "delete rules must declare `log:`"
    assert "logs/delete_reprocess/" in content, "log path must be toolkit-owned, not executor-owned"
    # ...and every rule must redirect into the Snakemake-managed {log}, literally.
    assert "> {log} 2>&1" in content, (
        "shell must redirect into {log}; a `log:` directive alone captures nothing"
    )
    assert content.count("> {log} 2>&1") == content.count("    log:\n"), (
        "every rule declaring a log must also redirect into it"
    )


def test_build_reprocess_delete_snakefile_sensitivity_option_c(norfolk_sensitivity_analysis):
    """D-scope Option C: sensitivity emits ONE delete_subanalysis_reprocess_{sa}
    rule per sub (NOT per-(sa,event)) + a master consolidation rule."""
    from hhemt.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_sensitivity_analysis
    sub_ids = [str(k) for k in analysis.sensitivity.sub_analyses.keys()]
    assert len(sub_ids) >= 1, "fixture must construct >=1 sub-analysis"

    builder = SnakemakeWorkflowBuilder(analysis)
    content = builder._build_reprocess_delete_snakefile_content(start_with="process")

    # exactly one per-sub rule per sub-analysis (Option C granularity)
    n_sub_rules = content.count("rule delete_subanalysis_reprocess_")
    assert n_sub_rules == len(sub_ids), (
        f"expected one per-sub rule per sub ({len(sub_ids)}), got {n_sub_rules}"
    )
    assert "hhemt.delete_subanalysis_reprocess_runner" in content
    # start_with='process' threads --delete-processed into the per-sub runner
    assert "--delete-processed" in content
    # master consolidation rule deletes the sensitivity master zarr
    assert "rule delete_reprocess_zarr_consolidation:" in content
    # Option C must NOT degenerate to per-(sa,event) processed rules at master level
    assert "rule delete_processed_" not in content
    assert "_deleting_reprocess/" in content


def test_build_reprocess_delete_snakefile_consolidate_skips_processed(norfolk_multi_sim_analysis):
    """start_with='consolidate' emits NO per-event processed rules (processed/
    deletion is a process-stage-only concern) but still deletes the zarr."""
    from hhemt.workflow import SnakemakeWorkflowBuilder

    builder = SnakemakeWorkflowBuilder(norfolk_multi_sim_analysis)
    content = builder._build_reprocess_delete_snakefile_content(start_with="consolidate")

    assert "rule delete_processed_" not in content
    assert "rule delete_reprocess_zarr_consolidation:" in content


def test_reprocess_phase3_self_methods_are_defined_on_class():
    """Guard against the 'call site lands but method definition does not' failure
    mode — which compiles + imports cleanly yet AttributeErrors at reprocess time.
    (This is exactly how a missing `_delete_processed_outputs_for_reprocess` slipped
    past py_compile + import smoke + the R8 unit tests and only surfaced once the
    GPU-compile-gated synth suite could finally run reprocess() end-to-end.)"""
    import inspect
    import re

    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
    from hhemt.workflow import SnakemakeWorkflowBuilder

    # The specific method that went missing.
    assert hasattr(TRITONSWMM_analysis, "_delete_processed_outputs_for_reprocess")

    # Generic: every self._method(...) referenced in these Phase-3 bodies must
    # resolve on the class.
    for cls, meth in [
        (TRITONSWMM_analysis, "reprocess"),
        (TRITONSWMM_sensitivity_analysis, "reprocess"),
        (SnakemakeWorkflowBuilder, "submit_reprocess_delete_workflow"),
        (SnakemakeWorkflowBuilder, "_build_reprocess_delete_snakefile_content"),
    ]:
        src = inspect.getsource(getattr(cls, meth))
        called = set(re.findall(r"self\.(_[a-z][a-zA-Z0-9_]*)\(", src))
        missing = sorted(m for m in called if not hasattr(cls, m))
        assert not missing, f"{cls.__name__}.{meth} references undefined methods: {missing}"


def test_delete_rules_declare_the_processing_partition_on_hpc(norfolk_multi_sim_analysis):
    """Every scoped-delete rule must name the CPU processing partition on HPC.

    Without an explicit slurm_partition these rules inherit `default-resources`'
    slurm_partition={hpc_ensemble_partition} (workflow.py:3177) — the GPU
    partition — and are submitted with zero GRES, which a GPU-only partition
    rejects at sbatch. Observed 2026-07-20: 112 uniform failures on gpu-a6000.
    The rules must target hpc_setup_and_analysis_processing_partition, matching
    the report-tail plot-rule precedent at workflow.py:359-366.

    Local mode must stay byte-identical (no slurm_partition key at all), so both
    directions are asserted.
    """
    from hhemt.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_multi_sim_analysis
    builder = SnakemakeWorkflowBuilder(analysis)
    cfg = builder.cfg_analysis

    # --- local mode: no partition key, byte-identical to the pre-fix literal ---
    for field, value in (("multi_sim_run_method", "local"),):
        try:
            setattr(cfg, field, value)
        except (AttributeError, TypeError, ValueError):
            object.__setattr__(cfg, field, value)
    local_content = builder._build_reprocess_delete_snakefile_content(start_with="process")
    assert "slurm_partition" not in local_content, "local-mode delete Snakefile must emit no slurm_partition"
    assert "resources: cpus_per_task=1, mem_mb=4096, runtime=120" in local_content

    # --- batch_job: every rule names the PROCESSING partition, never the ensemble one ---
    for field, value in (
        ("multi_sim_run_method", "batch_job"),
        ("hpc_ensemble_partition", "gpu-a6000"),
        ("hpc_setup_and_analysis_processing_partition", "standard"),
    ):
        try:
            setattr(cfg, field, value)
        except (AttributeError, TypeError, ValueError):
            object.__setattr__(cfg, field, value)
    hpc_content = builder._build_reprocess_delete_snakefile_content(start_with="process")

    n_rules = hpc_content.count("    resources: ")
    assert n_rules >= 2, "expected at least the per-event + consolidation delete rules"
    n_part = hpc_content.count('slurm_partition="standard"')
    assert n_part == n_rules, f"EVERY delete rule must carry the processing partition; got {n_part} of {n_rules}"
    assert "gpu-a6000" not in hpc_content, (
        "delete rules must NOT target the GPU ensemble partition — that is the "
        "inherited default this fix exists to override"
    )
