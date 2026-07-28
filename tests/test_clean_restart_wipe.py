"""Unit test for ``TRITONSWMM_analysis._clean_restart_wipe`` — the targeted
clean-restart capability landed for the ``pure_triton_arms_multi_resume_b4b`` Phase-7
sweep recovery (surfaced to ``run()`` as ``override_clean_restart_sa_ids``).

The capability recovers a resume-sweep sub whose coupled exchange-replay side-file was
truncated by a hard kill: it wipes ONLY the named sub-analysis dir(s) (sim outputs incl.
the side-file, per-model logs incl. ``n_resumes``, per-sub status) and delegates the
per-sa completion-flag deletion to the already-tested
``_delete_flags_for_force_rerun`` — so a subsequent ``run(from_scratch=False)`` re-fires
ONLY those subs fresh from ``checkpoint_id=0`` while every other (good) sub is untouched.

This test verifies the dir-scoping (only the named sub is removed) and the delegated
spec (``scope="sa"``, the named token). Pre-fix — before the capability landed —
``_clean_restart_wipe`` does not exist, so this test fails with ``AttributeError``.
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

    # Only the named sub (sa_1) was wiped; the good sub (sa_2) is untouched.
    assert not (analysis_dir / "subanalyses" / "sa_1").exists()
    assert (analysis_dir / "subanalyses" / "sa_2" / "out_tritonswmm" / "swmm_replay.bin").exists()

    # Per-sa flag deletion was delegated exactly once with a scope="sa" spec naming sa_1.
    assert len(recorded) == 1
    assert isinstance(recorded[0], ResolvedForceRerunSpec)
    assert recorded[0].scope == "sa"
    assert recorded[0].tokens == ("1",)
