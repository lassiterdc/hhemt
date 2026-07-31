"""Unit test for ``TRITONSWMM_analysis._clean_restart_wipe`` — the targeted
clean-restart capability landed for the ``pure_triton_arms_multi_resume_b4b`` Phase-7
sweep recovery (surfaced to ``run()`` as ``override_clean_restart_sa_ids``).

The capability recovers a resume-sweep sub whose coupled exchange-replay side-file was
truncated by a hard kill: it wipes the named sub-analysis dir's RUNTIME children (sim
outputs incl. the side-file, per-model logs incl. ``n_resumes``, per-sub status) and
delegates the per-sa completion-flag deletion to the already-tested
``_delete_flags_for_force_rerun`` — so a subsequent ``run(from_scratch=False)`` re-fires
ONLY those subs fresh from ``checkpoint_id=0`` while every other (good) sub is untouched.

The sub-analysis DIRECTORY and its setup-generated per-sub config ``sa_{id}.yaml`` are
PRESERVED by design. That config is written by ``sensitivity_analysis._create_sub_analyses``
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
    (analysis_dir / "subanalyses" / "sa_1" / "out_tritonswmm").mkdir(parents=True)
    (analysis_dir / "subanalyses" / "sa_2" / "out_tritonswmm").mkdir(parents=True)
    # Non-empty dirs (realistic sim outputs) so the wipe is exercised on real content.
    (analysis_dir / "subanalyses" / "sa_1" / "out_tritonswmm" / "swmm_replay.bin").write_bytes(b"\x00" * 8)
    (analysis_dir / "subanalyses" / "sa_2" / "out_tritonswmm" / "swmm_replay.bin").write_bytes(b"\x00" * 8)
    # The setup-generated per-sub configs, written at master __init__ before the wipe fires.
    # Their survival is the behaviour the implementation's config-outside-wipe branch exists for.
    (analysis_dir / "subanalyses" / "sa_1" / "sa_1.yaml").write_text("sa: 1\n", encoding="utf-8")
    (analysis_dir / "subanalyses" / "sa_2" / "sa_2.yaml").write_text("sa: 2\n", encoding="utf-8")

    recorded: list = []

    class _RecordingBuilder:
        def _delete_flags_for_force_rerun(self, spec):
            recorded.append(spec)

    stub = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        _workflow_builder=_RecordingBuilder(),
    )

    # Invoke the (new) capability's helper with the stub as ``self``.
    TRITONSWMM_analysis._clean_restart_wipe(stub, ["1"])

    sa_1 = analysis_dir / "subanalyses" / "sa_1"

    # The named sub's RUNTIME children are gone...
    assert not (sa_1 / "out_tritonswmm").exists(), "runtime sim outputs were not wiped"

    # ...but the dir and its setup-generated config survive. This is the load-bearing
    # assertion: a whole-dir removal deletes sa_1.yaml, snakemake skips setup because
    # a_setup_target_N_complete.flag survives, and prepare aborts "Analysis config not found".
    assert sa_1.is_dir(), "the sub-analysis dir must survive the wipe"
    assert (sa_1 / "sa_1.yaml").read_text(encoding="utf-8") == "sa: 1\n", (
        "the setup-generated per-sub config must survive the wipe byte-intact"
    )

    # The good sub (sa_2) is wholly untouched.
    assert (analysis_dir / "subanalyses" / "sa_2" / "out_tritonswmm" / "swmm_replay.bin").exists()
    assert (analysis_dir / "subanalyses" / "sa_2" / "sa_2.yaml").exists()

    # Per-sa flag deletion was delegated exactly once with a scope="sa" spec naming sa_1.
    assert len(recorded) == 1
    assert isinstance(recorded[0], ResolvedForceRerunSpec)
    assert recorded[0].scope == "sa"
    assert recorded[0].tokens == ("1",)
