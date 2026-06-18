"""cProfile harness for the P2-V1 wall-clock investigation (v2).

Mirrors the FULL test body of `test_multisim_rule_all_input_symmetry`:
- generate_snakefile_content called twice (matplotlib + plotly backends)
- _assert_symmetry per backend
- _rule_structure cross-backend equality
- _structural_diff non-ext-swap check

This surfaces whether the 5390s pytest cost lives in the SECOND generate
call, in the regex helpers, or somewhere else that the harness's single-call
v1 missed.

Run: `conda run --no-capture-output -n hhemt python -m scripts.profile.profile_p2v1_builder`
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path


def main() -> None:
    import tests.fixtures.test_case_catalog as cases
    from tests.test_workflow_snakefile_extension_consistency import (
        _assert_symmetry,
        _generate_multisim_snakefile_text,
        _rule_structure,
        _structural_diff,
    )

    t0 = time.perf_counter()
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=True,
        skip_run=True,
    )
    analysis = case.analysis
    t_setup = time.perf_counter() - t0
    print(f"[profile_p2v1_builder] fixture-equivalent setup: {t_setup:.3f}s")

    # Replicate pytest's monkeypatch semantics: setattr the report-backend hook
    # on the builder for each backend in turn (matching `_generate_multisim_snakefile_text`).
    class _MonkeypatchStub:
        def __init__(self) -> None:
            self._undo: list = []

        def setattr(self, target, name, value):
            prior = getattr(target, name)
            self._undo.append((target, name, prior))
            setattr(target, name, value)

    mp = _MonkeypatchStub()

    profile_path = Path("/tmp/profile_p2v1_first_call.pstats")
    top_path = Path("/tmp/p2v1_profile_top.txt")

    pr = cProfile.Profile()
    pr.enable()

    snakefiles: dict[str, str] = {}
    for backend in ("matplotlib", "plotly"):
        t_call = time.perf_counter()
        text = _generate_multisim_snakefile_text(analysis, backend, mp)
        elapsed = time.perf_counter() - t_call
        print(f"[profile_p2v1_builder] generate_snakefile_content({backend!r}): {elapsed:.3f}s, {len(text)} chars")
        snakefiles[backend] = text

    t_check = time.perf_counter()
    for backend, text in snakefiles.items():
        _assert_symmetry(text, consumer_rule="all")
    elapsed_assert = time.perf_counter() - t_check
    print(f"[profile_p2v1_builder] _assert_symmetry x 2: {elapsed_assert:.3f}s")

    t_struct = time.perf_counter()
    struct_eq = _rule_structure(snakefiles["matplotlib"]) == _rule_structure(snakefiles["plotly"])
    elapsed_struct = time.perf_counter() - t_struct
    print(f"[profile_p2v1_builder] _rule_structure equality ({struct_eq}): {elapsed_struct:.3f}s")

    t_diff = time.perf_counter()
    diffs = _structural_diff(snakefiles["matplotlib"], snakefiles["plotly"])
    elapsed_diff = time.perf_counter() - t_diff
    print(f"[profile_p2v1_builder] _structural_diff ({len(diffs)} diffs): {elapsed_diff:.3f}s")

    pr.disable()
    pr.dump_stats(str(profile_path))

    print(f"[profile_p2v1_builder] pstats dump: {profile_path}")

    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats(pstats.SortKey.CUMULATIVE)
    buf.write("\n=== top 30 by CUMULATIVE time ===\n")
    stats.print_stats(30)
    stats = pstats.Stats(pr, stream=buf).sort_stats(pstats.SortKey.TIME)
    buf.write("\n=== top 30 by INTERNAL (tottime) time ===\n")
    stats.print_stats(30)

    top_path.write_text(buf.getvalue())
    print(f"[profile_p2v1_builder] top-30 reports: {top_path}")
    print(buf.getvalue())


if __name__ == "__main__":
    main()
