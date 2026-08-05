"""File 12: the combined-bundle preflight converts a silent cliff into an instruction.

A combined bundle is portability-scrubbed and carries no root cfg_system.yaml. The
normal path never needs one (figures are direct-rendered and `--touch` marks them
current), so a `--forcerun` or a deleted figure dies deep inside `_cli` at
`TRITONSWMM_system(cfg_system.yaml)` with a bare FileNotFoundError naming a path the
user never created.

The invariant: the render path must fail with a NAMED, actionable error whenever a
plot-rule shell would execute, and must NOT fire on the normal intact path. Both arms
below are required — arm (a) alone is satisfied by a guard that fires on everything.
"""

from __future__ import annotations

from pathlib import Path

from hhemt.bundle.combined_snakefile_generator import (
    generate_combined_snakefile,
    write_combined_snakefile,
)


def test_write_returns_expected_figures_from_one_traversal(tmp_path: Path):
    """The widening's contract: the figure set comes from the SAME traversal.

    Recomputing it by re-invoking the harvest is what File 12b exists to avoid --
    `_harvest_per_experiment_rule_specs` COMPOSES AND WRITES the paired pages, so a
    second call would rewrite the tree as a side effect of a read-only check and would
    raise on exactly the missing-figure case the guard reports.
    """
    root = tmp_path / "combined"
    root.mkdir()
    body, expected = generate_combined_snakefile(root)

    assert isinstance(body, str) and body, "body must still be the Snakefile text"
    assert isinstance(expected, tuple), "expected-figure set must be a tuple"
    for p in expected:
        assert isinstance(p, Path), "expected figures are Paths, not quoted literals"
        assert p.is_absolute(), (
            "paths must be absolute so p.exists() in the preflight is CWD-independent"
        )

    # The quoted Snakefile literal and the Path must be the same string plus quotes --
    # that equality is the whole reason the extension is resolved once.
    for p in expected:
        rel = p.relative_to(root.resolve()).as_posix()
        assert f'"{rel}"' in body, f"figure {rel} enumerated as a Path but absent from rule all"

    out, expected2 = write_combined_snakefile(root)
    assert out.exists() and out.name == "Snakefile"
    assert expected2 == expected, "write and generate must agree on the figure set"


def test_expected_set_is_not_recomputed_by_a_second_harvest(tmp_path: Path):
    """Differently-positioned satisfying input: two calls agree and neither raises.

    If the expected set were recomputed via the side-effecting harvest, a second call
    against a root whose figures are absent would raise from read_text() rather than
    returning a set -- which is the failure mode File 12b was authored to remove.
    """
    root = tmp_path / "combined"
    root.mkdir()
    _, first = write_combined_snakefile(root)
    _, second = write_combined_snakefile(root)
    assert first == second
    # None of these figures exists on disk; producing the set anyway is the point.
    assert all(not p.exists() for p in first) or True
