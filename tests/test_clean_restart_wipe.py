"""Unit test for ``TRITONSWMM_analysis._clean_restart_wipe`` — the targeted
clean-restart capability landed for the ``pure_triton_arms_multi_resume_b4b`` Phase-7
sweep recovery (surfaced to ``run()`` as ``override_clean_restart_member_ids``).

The capability recovers a resume-sweep sub whose coupled exchange-replay side-file was
truncated by a hard kill: it wipes the named member dir's RUNTIME children (sim
outputs incl. the side-file, per-model logs incl. ``n_resumes``, per-sub status) and
delegates the per-member completion-flag deletion to the already-tested
``_delete_flags_for_force_rerun`` — so a subsequent ``run(from_scratch=False)`` re-fires
ONLY those subs fresh from ``checkpoint_id=0`` while every other (good) sub is untouched.

The member DIRECTORY and its setup-generated per-sub config ``member_{id}.yaml`` are
PRESERVED by design. That config is written by ``sensitivity_analysis._create_members``
at master ``__init__`` — before this wipe fires — and is read by the prepare runner via
``--analysis-config``. A whole-dir removal would delete it, and because the surviving
``a_setup_target_N_complete.flag`` makes snakemake skip setup, nothing regenerates it, so
prepare aborts with "[ERROR] Analysis config not found". The config-preservation assertion
below is what pins that behaviour: it fails against any variant that removes the whole dir.
"""
from __future__ import annotations

import types


def test_clean_restart_wipe_removes_only_named_subs_and_delegates_flag_delete(tmp_path):
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.workflow import ResolvedForceRerunSpec

    analysis_dir = tmp_path / "master"
    (analysis_dir / "members" / "member_1" / "out_tritonswmm").mkdir(parents=True)
    (analysis_dir / "members" / "member_2" / "out_tritonswmm").mkdir(parents=True)
    # Non-empty dirs (realistic sim outputs) so the wipe is exercised on real content.
    (analysis_dir / "members" / "member_1" / "out_tritonswmm" / "swmm_replay.bin").write_bytes(b"\x00" * 8)
    (analysis_dir / "members" / "member_2" / "out_tritonswmm" / "swmm_replay.bin").write_bytes(b"\x00" * 8)
    # The setup-generated per-sub configs, written at master __init__ before the wipe fires.
    # Their survival is the behaviour the implementation's config-outside-wipe branch exists for.
    (analysis_dir / "members" / "member_1" / "member_1.yaml").write_text("member: 1\n", encoding="utf-8")
    (analysis_dir / "members" / "member_2" / "member_2.yaml").write_text("member: 2\n", encoding="utf-8")

    # The sub's ANALYSIS-LEVEL model runtime logs live under the MASTER's logs/sims/, not
    # under the sub dir — model_logfile_for routes every sim of a sweep to one directory.
    # Seed both subs so the wipe's scoping is observable: member_1's log must go, member_2's stays.
    simlogs = analysis_dir / "logs" / "sims"
    (simlogs / "_walltime").mkdir(parents=True)
    for _sa in ("1", "2"):
        (simlogs / f"model_tritonswmm_member_{_sa}_evt0.log").write_text("Simulation ends\n", encoding="utf-8")
        (simlogs / "_walltime" / f"model_tritonswmm_member_{_sa}_evt0.jsonl").write_text("{}\n", encoding="utf-8")

    recorded: list = []

    class _RecordingBuilder:
        def _delete_flags_for_force_rerun(self, spec):
            recorded.append(spec)

    stub = types.SimpleNamespace(
        # simlog_directory is a REQUIRED (non-Optional) AnalysisPaths field (paths.py:57),
        # always constructed as {analysis_log_directory}/sims (analysis.py:274). A stub
        # omitting it is unrealistic, not a signal — the wipe legitimately reads it.
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir, simlog_directory=simlogs),
        _workflow_builder=_RecordingBuilder(),
    )

    # Invoke the (new) capability's helper with the stub as ``self``.
    TRITONSWMM_analysis._clean_restart_wipe(stub, ["1"])

    member_1 = analysis_dir / "members" / "member_1"

    # The named sub's RUNTIME children are gone...
    assert not (member_1 / "out_tritonswmm").exists(), "runtime sim outputs were not wiped"

    # ...but the dir and its setup-generated config survive. This is the load-bearing
    # assertion: a whole-dir removal deletes member_1.yaml, snakemake skips setup because
    # a_setup_target_N_complete.flag survives, and prepare aborts "Analysis config not found".
    assert member_1.is_dir(), "the member dir must survive the wipe"
    assert (member_1 / "member_1.yaml").read_text(encoding="utf-8") == "member: 1\n", (
        "the setup-generated per-sub config must survive the wipe byte-intact"
    )

    # The named sub's analysis-level model log and its _walltime ledger sibling are gone.
    # Leaving the log behind would reproduce, at member granularity, the stale-evidence skip that
    # motivated the model_logfile_for relocation: model_run_completed's raw-marker fallback
    # would find this "Simulation ends" and skip the sim whose outputs were just cleared.
    # Leaving the ledger behind would double-count into wall_clock_ledger_s on the re-run.
    assert not (simlogs / "model_tritonswmm_member_1_evt0.log").exists(), (
        "the named sub's model log must be removed, else the re-run skips on stale evidence"
    )
    assert not (simlogs / "_walltime" / "model_tritonswmm_member_1_evt0.jsonl").exists(), (
        "the walltime ledger must go with its log, else the re-run double-counts"
    )

    # The good sub (member_2) is wholly untouched — including its log and ledger, which share
    # the master's one logs/sims/ directory with member_1's. This is what pins the SCOPING:
    # a glob that dropped the member token would take every sub's log in the sweep.
    assert (analysis_dir / "members" / "member_2" / "out_tritonswmm" / "swmm_replay.bin").exists()
    assert (analysis_dir / "members" / "member_2" / "member_2.yaml").exists()
    assert (simlogs / "model_tritonswmm_member_2_evt0.log").exists(), "the good sub's log must survive"
    assert (simlogs / "_walltime" / "model_tritonswmm_member_2_evt0.jsonl").exists()

    # Per-member flag deletion was delegated exactly once with a scope="member" spec naming member_1.
    assert len(recorded) == 1
    assert isinstance(recorded[0], ResolvedForceRerunSpec)
    assert recorded[0].scope == "member"
    assert recorded[0].tokens == ("1",)
