"""Master-Snakefile byte-identity test.

Asserts that the source-side ``SnakemakeWorkflowBuilder.generate_snakefile_content``
and ``SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content``
emit byte-identical Snakefile output to the captured goldens. The goldens
were re-baselined at Plan Phase 5 to reflect the cfg-controlled
``static_backend`` default (``"plotly"`` per Decision 4); the builders read
the report-cfg field via ``_get_report_cfg_static_backend()`` and emit
plotly-branch rules when no ``--report-config`` overrides the default.

Volatile tokens that depend on WHERE the repo is checked out (the absolute
repo root, the worktree nesting, the variable-depth ``../`` relative paths
to ``~/.local/share/hhemt/examples``) and WHICH interpreter runs the suite
(conda vs uv) are normalized to stable placeholders on BOTH the generated
text and the committed golden before comparison (``_normalize_volatile``).
This makes the assertion robust to checkout location and environment while
STILL surfacing genuine generation-logic drift: only those specific volatile
prefixes are masked — rule structure, resources, command shape, and every
other token compare byte-for-byte.

If byte-identity fails AFTER normalization, the source-side Snakefile
generation introduced silent drift and must be fixed before merge — NOT
papered over by recapturing the golden. Recapture (``CAPTURE_SNAKEFILE_GOLDENS=1``)
is appropriate ONLY for an intentional emit change (e.g. the package rename
that flipped the per-rule ``conda:`` env file from ``triton_swmm.yaml`` to
``hhemt.yaml``); confirm the post-normalization diff contains nothing but the
intended change before recapturing.
"""

from __future__ import annotations

import os
import posixpath
import re
import sys
from pathlib import Path

import pytest

from hhemt.cache import paths as cache_paths

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.test_case_catalog import Local_TestCases  # noqa: E402

from hhemt.sensitivity_analysis import (  # noqa: E402
    TRITONSWMM_sensitivity_analysis,
)
from hhemt.workflow import (  # noqa: E402
    SensitivityAnalysisWorkflowBuilder,
    SnakemakeWorkflowBuilder,
)

GOLDENS_DIR = Path(__file__).parent / "fixtures" / "golden_snakefiles"

# Phase 4 (4a — byte-identity foundation): thread an hpc_system_config into the
# byte-identity cases so cfg_hpc_system is non-None BEFORE the legacy
# None-fallbacks in the resolution helpers are deleted in 4c/4d (those helpers
# are called unconditionally during Snakefile generation). The synth cases are
# LOCAL mode with all hpc_* PARTITION selectors null, so the config is byte-identity-neutral
# (its partition is never looked up; default_account appears only in the profile
# config.yaml, not the Snakefile). See the example config's header for the rationale.
EXAMPLE_HPC_CONFIG = Path(__file__).parent / "fixtures" / "hpc_system_config_test.yaml"


# Pin sys.executable for byte-identity comparison: pytest's shebang resolves
# to `python3.11` whereas direct invocation resolves to `python`. The
# captured goldens use `python` (symlink-preserved); pin to match.
@pytest.fixture(autouse=True)
def _pin_sys_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = re.sub(r"/python3\.\d+$", "/python", sys.executable)
    monkeypatch.setattr(sys, "executable", pinned)


# Set CAPTURE_SNAKEFILE_GOLDENS=1 to (re)write the committed goldens from the
# current generator output (placeholders applied), then skip. Only do this for
# an INTENTIONAL emit change after confirming the post-normalization diff
# contains nothing but that change. See the module docstring.
_CAPTURE = os.environ.get("CAPTURE_SNAKEFILE_GOLDENS") == "1"


def _normalize_volatile(text: str) -> str:
    """Replace tokens that vary along FOUR axes with stable placeholders, so the
    byte-identity assertion compares generation logic and nothing else. The axes
    are: WHERE the repo is checked out (primary tree, worktree, CI), WHICH
    interpreter runs it (conda, uv), WHICH HOST supplies the user cache root, and
    WHICH SYNTHETIC-MODEL INPUT BUILD is cached — a digest over params, toolkit
    version AND generator source, so ANY of the three rotates the key and a
    version bump is only one of them. They are four rather than two because the
    cache tier lives OUTSIDE the repo, so neither the checkout axis nor the
    interpreter axis reaches it.

    Narrowly scoped: only the interpreter path, the absolute repo root, the
    variable-depth ``../`` relative path to the user's ``.local/share`` data dir,
    the hhemt cache root, and ONE varying segment beneath each of
    ``synthetic_test_runs`` (a worktree slug — the checkout axis) and
    ``synthetic_test_models`` (``_cache_key``: a SHA-1 over params + toolkit
    version + generator source — the INPUT-BUILD axis, NOT the checkout axis:
    two worktrees of one commit produce the same key, because all three
    constituents are checkout-location-independent) are masked.

    Everything else is left intact so real drift still fails the assertion —
    including rule names, resources, source-path attributions, and every FIXED
    segment under the cache root. The last is load-bearing and is the reason the
    cache masks are root-scoped rather than generic: a default SIF path is
    ``{sif_root}/{family}/{stem}.sif`` (hhemt.sif.identity.resolve_sif), so a rule shell's
    ``apptainer exec`` argument sits exactly where a generic root-plus-two-segments
    rule would eat it. Command shape is preserved, but only because the cache masks
    decline to consume a second segment they do not know to be volatile.
    """
    # Order matters: replace the (longer, more specific) interpreter path before
    # the repo root, since under uv the interpreter lives at ``<repo>/.venv/...``.
    text = text.replace(sys.executable, "{PYTHON}")
    # Collapse the hhemt cache tier, which lives OUTSIDE the repo and therefore
    # survives the {REPO_ROOT} mask. TWO SEPARATE CONCERNS, masked separately —
    # collapsing them into one rule is a defect in either direction:
    #   (1) HOST-specificity: the cache root itself. Safe to mask everywhere, so it
    #       is a plain literal replace consuming no following segment.
    #   (2) CHECKOUT-specificity: a varying segment under SOME roots and not others.
    #       synthetic_test_runs/{slug} (tests/fixtures/__init__.py::worktree_slug) and
    #       synthetic_test_models/{key} (a SHA-1 over params + toolkit_version +
    #       generator_source_hash, so it churns on every version bump) each carry one.
    #       _triton_canonical and sif_cache do NOT — their next segment is a real,
    #       fixed token (triton/CMakeLists.txt, hhemt-0.1.0-cuda.sif).
    # An unconditional root-plus-two-segments rule EATS those fixed tokens, and a SIF
    # filename is emitted into a rule shell by run_simulation.py — exactly the
    # generation-logic token a byte-identity golden exists to protect. A root-only
    # rule leaves the varying segments behind. Hence: mask the root universally, then
    # mask the varying segment ONLY under the roots that have one. This enumeration
    # decays LOUDLY: a fifth varying root still gets its host prefix masked, so the
    # golden stays host-portable and fails only on a second WORKTREE, which is
    # diagnosable — where the rejected alternative fails by silently deleting a token
    # nobody notices is gone.
    # The cache root is masked in BOTH spellings. The generator renders some model paths relative to the RESOLVED
    # analysis root, so on a host whose home is a symlink (every cluster home) the real path of the cache root
    # appears, with its real prefix (`/sfs/gpfs/tardis/...`), and the unresolved spelling alone would leave that
    # prefix in front of the token. The resolved spelling goes first because it is the longer one wherever the two
    # differ; where the cache root is already real, the second replace finds nothing and the pair is the old one.
    _hhemt_cache_root = cache_paths.cache_root()
    text = text.replace(str(_hhemt_cache_root.resolve()), "{HHEMT_CACHE}")
    text = text.replace(str(_hhemt_cache_root), "{HHEMT_CACHE}")
    text = re.sub(r"(\{HHEMT_CACHE\}/synthetic_test_runs)/[^/'\"\s]+", r"\1/{SLUG}", text)
    text = re.sub(r"(\{HHEMT_CACHE\}/synthetic_test_models)/[^/'\"\s]+", r"\1/{MODEL_KEY}", text)
    # SAME tier, RELATIVE rendering. `watershed_rel_path` and the boundary source are
    # computed with os.path.relpath, so they reach this tier as `../../../../synthetic_
    # test_models/{key}/...` with no `{HHEMT_CACHE}` prefix for the rule above to anchor
    # on. Anchoring on the `../` RUN rather than on the tier name is deliberate: a bare
    # `synthetic_test_models/` anchor would also match the already-masked absolute form
    # and any future absolute occurrence, widening a rule whose narrowness is the reason
    # the sibling roots (`_triton_canonical`, `sif_cache`) keep their fixed next segment.
    #
    # The `../` RUN IS PRESERVED, via the capture group, and that asymmetry with the
    # `.local/share` rule below is INTENTIONAL -- do not harmonize them. That rule masks
    # its depth because a worktree nests deeper than the primary tree and the count
    # genuinely varies. Here both endpoints sit under the cache root, so the count is
    # invariant at four (analysis_dir is `{cache}/synthetic_test_runs/{slug}/{case}/
    # {case}`), and preserving it keeps a TRIPWIRE: if the layout ever changes the depth,
    # the golden REDDENS, which is diagnosable. Masking the depth would make it keep
    # matching a path the generator no longer emits -- silent, and strictly worse.
    text = re.sub(r"((?:\.\./)+synthetic_test_models)/[^/'\"\s]+", r"\1/{MODEL_KEY}", text)
    text = text.replace(str(Path(__file__).resolve().parents[1]), "{REPO_ROOT}")
    # Collapse the variable-depth relative path to the home data dir: a worktree
    # nests deeper than the primary tree, so the ``../`` count itself varies.
    # The ``[^'"\s]*?`` is load-bearing on any host where ``$HOME`` is a SYMLINK
    # (every Lustre/GPFS cluster home): ``os.path.relpath`` climbs to ``/`` and then
    # descends through the RESOLVED path, so the ``../`` run is followed by the real
    # home segments (``gpfs/tardis/home/u/``) rather than by ``.local/share/`` directly.
    # Anchoring straight onto ``.local/share/`` therefore masks nothing off-laptop and
    # the machine path survives into the comparison. Lazy + quote/space-excluding so it
    # cannot cross a string boundary, and it matches empty where home is not a symlink,
    # which keeps the primary-tree output byte-identical to the pre-fix form.
    text = re.sub(r"(?:\.\./)+[^'\"\s]*?(\.local/share/)", r"{HOME_REL}/\1", text)
    return text


def _unified_diff_excerpt(want: str, got: str, max_lines: int = 80) -> str:
    import difflib

    diff = list(
        difflib.unified_diff(
            want.splitlines(keepends=True),
            got.splitlines(keepends=True),
            fromfile="golden",
            tofile="emitted",
            n=2,
        )
    )
    return "".join(diff[:max_lines])


_MODEL_PATH_RUN = re.compile(r"((?:\.\./)+)([^'\"\s`\\]*?synthetic_test_models/[^'\"\s`\\]*)")
_RUNS_ROOT_OVERRIDE_ENV = "HHEMT_TEST_RUNS_ROOT_OVERRIDE"


def _resolve_model_paths(text: str, analysis_dir: str) -> str:
    """Rewrite each relative ``../``-run path into ``synthetic_test_models`` as the path it resolves to.

    The run may climb to ``/`` and descend through the cache root's real path (``os.path.relpath`` between trees
    that share only ``/``), so the text between the ``../`` run and ``synthetic_test_models`` is part of the match.

    `analysis_dir` is the directory the relative path is read from. Resolving, rather than masking the ``../`` run,
    keeps the depth in the comparison: a path one hop too shallow resolves to a different directory and so still
    differs from the golden. Deliberately NOT a mask of the run (the comment in `_normalize_volatile` forbids it): a
    mask keeps matching a path the generator no longer emits."""
    return _MODEL_PATH_RUN.sub(
        lambda m: posixpath.normpath(posixpath.join(analysis_dir, m.group(1) + m.group(2))), text
    )


def _mask_override_runs_root(text: str, override: str, cache: str) -> str:
    """Spell the exported override's runs root as the slug runs root, so the slug mask then treats them alike.

    An override root carries no slug segment (the builder nests ``synthetic_test_runs`` directly under it), which
    is why the slug mask in `_normalize_volatile` cannot see it."""
    return text.replace(f"{override.rstrip('/')}/synthetic_test_runs", f"{cache}/synthetic_test_runs/{{SLUG}}")


def _canonical_forms(got: str, golden: str, analysis_dir: Path) -> tuple[str, str]:
    """Return (emitted, golden) in one comparable form, whatever runs root the emitted text was produced under.

    Three steps, in this order. (1) Resolve the emitted model paths against the emitted analysis directory. (2) Mask
    the exported override's runs root as the slug runs root, then apply the usual volatile masks, which also mask the
    cache root. (3) Resolve the golden's model paths against the canonical analysis directory
    ``{HHEMT_CACHE}/synthetic_test_runs/{SLUG}/{case}/{case}``, so both sides name the same directory."""
    cache = str(cache_paths.cache_root())
    emitted = _resolve_model_paths(got, str(analysis_dir))
    override = os.environ.get(_RUNS_ROOT_OVERRIDE_ENV)
    if override:
        emitted = _mask_override_runs_root(emitted, override, cache)
    emitted = _normalize_volatile(emitted)
    case = analysis_dir.name
    canonical = f"{{HHEMT_CACHE}}/synthetic_test_runs/{{SLUG}}/{case}/{case}"
    return emitted, _resolve_model_paths(golden, canonical)


def _check(got: str, golden_name: str, analysis_dir: Path | None = None) -> None:
    """Normalize volatile tokens, then capture-or-assert against the golden."""
    golden_path = GOLDENS_DIR / golden_name
    if _CAPTURE:
        normalized = _normalize_volatile(got)
        GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(normalized)
        pytest.skip(f"captured golden {golden_name} ({len(normalized)} bytes)")
    if analysis_dir is None:
        normalized, want = _normalize_volatile(got), golden_path.read_text()
    else:
        normalized, want = _canonical_forms(got, golden_path.read_text(), analysis_dir)
    assert normalized == want, _unified_diff_excerpt(want, normalized)


def test_multi_sim_snakefile_byte_identity() -> None:
    """Source-side multi-sim Snakefile byte-identical to golden."""
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()
    _check(got, "multi_sim.Snakefile.golden", tc.analysis.analysis_paths.analysis_dir)


def test_master_snakefile_byte_identity() -> None:
    """Source-side sensitivity-master Snakefile byte-identical to golden."""
    tc = Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    sens = TRITONSWMM_sensitivity_analysis(tc.analysis)
    builder = SensitivityAnalysisWorkflowBuilder(sens)
    got = builder.generate_master_snakefile_content()
    _check(got, "sensitivity_master.Snakefile.golden", tc.analysis.analysis_paths.analysis_dir)


# --- T15(iii) fixtures. DELIBERATELY NOT EXAMPLE_HPC_CONFIG, which is threaded
# into BOTH Snakefile goldens: declaring the cap there would emit a per-rule
# weight into both and force a recapture -- the signal that [Q246]'s default-OFF
# is broken. SLURM_HPC_CONFIG is the UNCAPPED arm and exists separately only
# because EXAMPLE_HPC_CONFIG declares no max_concurrent_jobs and so cannot reach
# generate_snakemake_config(mode="slurm") past its assert at workflow.py:3808-3811.
SLURM_HPC_CONFIG = Path(__file__).parent / "fixtures" / "hpc_system_config_slurm_test.yaml"
CAPPED_HPC_CONFIG = Path(__file__).parent / "fixtures" / "hpc_system_config_capped_test.yaml"


def _hhemt_cpus_weight(block: str) -> int | None:
    """The single hhemt_cpus weight in an emitted resources block, or None.

    PARSES rather than substring-matching, and the difference is not stylistic:
    `"hhemt_cpus=8" in block` is TRUE for an emitted 80, so a substring assertion
    on the discriminating full-node test would pass on a 10x OVER-count -- the
    same mis-count-while-reporting-compliance shape the weight itself exists to
    prevent. `endswith` also closes that, but only while the weight is the LAST
    resource emitted, which nothing pins; this does not depend on emission order.
    The len==1 check catches a double emission, which no per-line assertion sees.
    """
    lines = [ln.strip().rstrip(",") for ln in block.splitlines() if ln.strip().startswith("hhemt_cpus=")]
    if not lines:
        return None
    assert len(lines) == 1, f"expected one hhemt_cpus line, got {lines}"
    return int(lines[0].split("=", 1)[1])


def test_slurm_profile_byte_identity_unset() -> None:
    """UNSET path: the emitted slurm profile is byte-identical to golden.

    This is the artifact [Q246]'s "byte-identical generated profile" half was
    missing. Before this test that half was ASSERTED; the two Snakefile goldens
    demonstrated only that the *Snakefile* was unchanged.
    """
    import yaml  # not imported at module scope in this file

    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=SLURM_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = yaml.dump(
        builder.generate_snakemake_config(mode="slurm"),
        default_flow_style=False,
        sort_keys=False,
    )
    _check(got, "slurm_profile.config.yaml.golden")


def test_slurm_profile_declares_the_bag_when_set() -> None:
    """SET path: the profile gains exactly one `resources:` entry, and its value
    is the configured cap.

    Deliberately NOT a golden. A golden here would be captured from the same
    generator it checks, so it would pass on any value the emitter produced; the
    assertion below fails if the number is wrong, which a golden cannot do.
    """
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=CAPPED_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    cfg = builder.generate_snakemake_config(mode="slurm")
    assert cfg["resources"] == ["hhemt_cpus=480"]


def test_slurm_profile_omits_the_bag_when_unset() -> None:
    """UNSET path, asserted rather than left to the golden.

    The golden above proves the profile did not CHANGE; this proves the key is
    ABSENT rather than present-and-null, which is the [Q246] property and is not
    something a byte comparison can distinguish from "we never looked".
    """
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=SLURM_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    assert "resources" not in builder.generate_snakemake_config(mode="slurm")


def test_per_rule_weight_is_metered_not_requested_when_set() -> None:
    """SET path: a CPU rule carries `hhemt_cpus = tasks * cpus_per_task`.

    The unset arm gets a free green from the two Snakefile goldens; this is the
    arm that does the work and nothing else covers it.
    """
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=CAPPED_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    block = builder._build_resource_block(
        partition="standard",
        runtime_min=30,
        mem_mb=2000,
        nodes=1,
        tasks=2,
        cpus_per_task=4,
    )
    assert _hhemt_cpus_weight(block) == 8


def test_full_node_gpu_weight_is_the_whole_node_not_the_request() -> None:
    """SET path: the --exclusive shape is charged nodes * cpus_per_node.

    THE REGRESSION THIS PINS. Pre-fix arithmetic would give tasks*cpus_per_task
    = 4*1 = 4 for the block below; SLURM charges 1*40 = 40, a 10x under-count of
    the same class measured at 6x on job 14452815. An assertion on 4 would pass
    while permitting a tenfold ceiling breach.
    """
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=CAPPED_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    block = builder._build_resource_block(
        partition="standard",
        runtime_min=30,
        mem_mb=2000,
        nodes=1,
        tasks=1,
        cpus_per_task=1,
        gpus_total=4,
        gpus_per_node_config=4,
        gpu_alloc_mode="gres",
    )
    assert 'slurm_extra="--exclusive"' in block  # the shape under test really is 4/5
    assert _hhemt_cpus_weight(block) == 40  # 1 node * 40 cpus_per_node


def test_raise_when_full_node_gpu_partition_declares_no_cpus_per_node() -> None:
    """SET path: the one legitimate generation-time raise, and only there.

    Both arms come from ONE fixture with TWO partitions, so the contrast under
    test is a partition property rather than a config-file property.
    """
    from hhemt.exceptions import ConfigurationError

    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=CAPPED_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    with pytest.raises(ConfigurationError, match="cpus_per_node"):
        builder._build_resource_block(
            partition="nocpn",
            runtime_min=30,
            mem_mb=2000,
            nodes=1,
            tasks=1,
            cpus_per_task=1,
            gpus_total=4,
            gpus_per_node_config=4,
            gpu_alloc_mode="gres",
        )
    # And NOT on a shape that computes without it, from the SAME partition.
    block = builder._build_resource_block(
        partition="nocpn",
        runtime_min=30,
        mem_mb=2000,
        nodes=1,
        tasks=2,
        cpus_per_task=4,
    )
    assert _hhemt_cpus_weight(block) == 8


# ========== Phase 3b: Snakemake `group:` directive assertions (R8) ==========

# Per-rule `cpus_per_task` for the three process_* rules. Snakemake's
# GroupResources default-sums resources across jobs in the same group, so
# the per-event group's aggregate `cpus_per_task` equals the sum.
_EXPECTED_CPUS_PER_TASK_PER_PROCESS_RULE = 2
_EXPECTED_PROCESS_RULES_PER_EVENT = 3  # triton, tritonswmm, swmm
_EXPECTED_GROUP_CPUS_PER_TASK = _EXPECTED_CPUS_PER_TASK_PER_PROCESS_RULE * _EXPECTED_PROCESS_RULES_PER_EVENT
# Conservative ceiling guarding against accidental cpus_per_task inflation
# (Snakemake architecture Gotcha 9): if a future edit bumps the per-rule
# value, the aggregate could over-subscribe a CI runner. 16 leaves
# headroom over the current 6.
_MAX_GROUP_CPUS_PER_TASK = 16


def test_process_rules_emit_group_directive() -> None:
    """All three process_* rules carry `group: "process_evt_{event_id}"` so
    Snakemake's DAG planner collapses them into a single per-event job-group,
    deduplicating subprocess-startup overhead (Phase 3b, R8)."""
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()
    for rule_name in ("process_triton", "process_tritonswmm", "process_swmm"):
        pattern = rf"rule {rule_name}:[\s\S]*?^\s*group:\s*\"process_evt_\{{event_id\}}\""
        assert re.search(pattern, got, re.MULTILINE), (
            f'rule {rule_name} missing `group: "process_evt_{{event_id}}"` '
            f"directive; Snakemake will not collapse the three process rules "
            f"into a per-event job-group."
        )


def test_process_rule_group_resources_do_not_overallocate() -> None:
    """GroupResources sums resources across grouped jobs by default. Verify
    the per-event aggregate `cpus_per_task` (sum across the three process_*
    rules) stays within a sane ceiling (architecture Gotcha 9 for Snakemake;
    Phase 3b, R8). This is a static check on the emitted Snakefile."""
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()

    per_rule_cpus: list[int] = []
    for rule_name in ("process_triton", "process_tritonswmm", "process_swmm"):
        block = re.search(rf"rule {rule_name}:[\s\S]*?(?=\nrule |\Z)", got)
        assert block is not None, f"rule {rule_name} not found in Snakefile"
        m = re.search(r"cpus_per_task=(\d+)", block.group(0))
        assert m is not None, f"rule {rule_name} does not declare `cpus_per_task=`; cannot verify group-resource sum."
        per_rule_cpus.append(int(m.group(1)))

    assert len(per_rule_cpus) == _EXPECTED_PROCESS_RULES_PER_EVENT
    aggregate = sum(per_rule_cpus)
    assert aggregate == _EXPECTED_GROUP_CPUS_PER_TASK, (
        f"Per-event group cpus_per_task aggregate is {aggregate}; expected "
        f"{_EXPECTED_GROUP_CPUS_PER_TASK} "
        f"({_EXPECTED_CPUS_PER_TASK_PER_PROCESS_RULE} × "
        f"{_EXPECTED_PROCESS_RULES_PER_EVENT}). If a per-rule cpus_per_task "
        f"changed deliberately, update the constants and audit "
        f"--group-components scheduling."
    )
    assert aggregate <= _MAX_GROUP_CPUS_PER_TASK, (
        f"Per-event group cpus_per_task aggregate {aggregate} exceeds the "
        f"ceiling {_MAX_GROUP_CPUS_PER_TASK}; risk of over-subscribing CI "
        f"runners under default --group-components scheduling."
    )


# ===== [Q8] Defect 2: report-tail plot-partition execution-locus predicate =====
#
# The report-tail plot rules (system_overview, disk_utilization,
# per_analysis_summary, errors_and_warnings, metadata, scenario_status_appendix,
# and the sensitivity per_sim / benchmarking / eda rules) are CPU-only. Under
# `--executor slurm` they MUST submit to the CPU processing partition; without it
# they inherit default-resources' GPU ensemble partition and are QOS-rejected
# (QOSMinGRES). The partition is keyed on the resolved execution LOCUS
# (`_resolved_execution_locus`), not the dispatch-family label — so a `local`
# dispatch-family run under execution_mode="slurm" (the doi_emitter emit path)
# still gets the CPU partition, while a genuine local `--cores` run gets none.

_Q8_CPU_PARTITION = "cpu-parallel-q8"


def _report_tail_partition(builder: SnakemakeWorkflowBuilder) -> str:
    """The CPU partition the report-tail plot rules would emit, read straight off
    the RuleEmissionContext (no full Snakefile generation needed)."""
    return builder._make_rule_emission_context(static_backend="plotly").report_tail_slurm_partition


def test_report_tail_partition_local_dispatch_slurm_locus_emits_cpu_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) local dispatch-family + resolved slurm locus => CPU partition emitted.

    The [Q8] Defect-2 fix: multi_sim_run_method='local' but the emitted Snakefile
    runs under `--executor slurm` (execution_mode='slurm')."""
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    monkeypatch.setattr(builder.cfg_analysis, "multi_sim_run_method", "local")
    monkeypatch.setattr(
        builder.cfg_analysis,
        "hpc_setup_and_analysis_processing_partition",
        _Q8_CPU_PARTITION,
    )
    builder._resolved_execution_locus = "slurm"  # the emit-path locus
    assert _report_tail_partition(builder) == _Q8_CPU_PARTITION


def test_report_tail_partition_genuine_local_cores_emits_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) local dispatch-family + local/unset locus => NO partition (byte-identity
    guard). A genuine `--cores` run must emit "" so the generated Snakefile stays
    byte-identical to the pre-fix form. This is why the committed multi_sim /
    master goldens need no regeneration: those tests call generate_* directly on a
    fresh builder, so _resolved_execution_locus stays None."""
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    monkeypatch.setattr(builder.cfg_analysis, "multi_sim_run_method", "local")
    monkeypatch.setattr(
        builder.cfg_analysis,
        "hpc_setup_and_analysis_processing_partition",
        _Q8_CPU_PARTITION,
    )
    # Default (None) — as on any directly-constructed builder.
    assert builder._resolved_execution_locus is None
    assert _report_tail_partition(builder) == ""
    # An explicit local locus is likewise partition-free.
    builder._resolved_execution_locus = "local"
    assert _report_tail_partition(builder) == ""


def test_report_tail_partition_native_dispatch_invariant_to_locus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) NATIVE dispatch-family (multi_sim_run_method != "local") emits the CPU
    partition regardless of the locus field — a byte-diff regression guard proving
    the new OR-clause does not perturb the ADR-19 native path (1_job / batch_job),
    whose partition is driven entirely by the pre-fix `!= "local"` clause."""
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    monkeypatch.setattr(builder.cfg_analysis, "multi_sim_run_method", "1_job_many_srun_tasks")
    monkeypatch.setattr(
        builder.cfg_analysis,
        "hpc_setup_and_analysis_processing_partition",
        _Q8_CPU_PARTITION,
    )
    for locus in (None, "local", "slurm"):
        builder._resolved_execution_locus = locus  # type: ignore[assignment]
        assert _report_tail_partition(builder) == _Q8_CPU_PARTITION, (
            f"native dispatch emitted a different partition for locus={locus!r}; "
            f"the OR-clause must leave the native path byte-identical (ADR-19)."
        )


_GOLDEN_CASES = [
    ("multi_sim.Snakefile.golden", "synth_multi_sim"),
    ("sensitivity_master.Snakefile.golden", "synth_sensitivity"),
]
_EMITTED_MODEL_KEY = "b495da2fed88fb9a"


def _emitted_from_golden(golden: str, case: str, runs_dir: str) -> tuple[str, Path]:
    """Rebuild the text the generator would emit for `case` under `runs_dir` from its golden, with its analysis dir.

    `runs_dir` is the directory that holds the case directories: ``{override}/synthetic_test_runs`` under an exported
    override, ``{cache}/synthetic_test_runs/{slug}`` otherwise. Every relative model path is re-rendered with
    `os.path.relpath`, the way the generator renders it, so the ``../`` run has the length a real emission has."""
    cache = str(cache_paths.cache_root())
    analysis_dir = Path(runs_dir) / case / case
    models = f"{cache}/synthetic_test_models/{_EMITTED_MODEL_KEY}"
    text = golden.replace("{HHEMT_CACHE}/synthetic_test_runs/{SLUG}", runs_dir)
    text = re.sub(
        r"(?:\.\./)+synthetic_test_models/\{MODEL_KEY\}",
        lambda _m: os.path.relpath(models, analysis_dir),
        text,
    )
    for placeholder, value in (
        ("{HHEMT_CACHE}", cache),
        ("{MODEL_KEY}", _EMITTED_MODEL_KEY),
        ("{PYTHON}", sys.executable),
        ("{REPO_ROOT}", str(Path(__file__).resolve().parents[1])),
    ):
        text = text.replace(placeholder, value)
    return text, analysis_dir


@pytest.fixture(params=["exported", "unset"])
def runs_dir_for_emission(request, tmp_path, monkeypatch) -> str:
    """The directory a reconstructed emission puts its case directories in, with the override variable to match.

    "exported" is a harness chunk's isolation root; "unset" is a laptop session, where a slug directory under the
    cache root holds them."""
    if request.param == "exported":
        monkeypatch.setenv(_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
        return f"{tmp_path}/synthetic_test_runs"
    monkeypatch.delenv(_RUNS_ROOT_OVERRIDE_ENV, raising=False)
    return f"{cache_paths.cache_root()}/synthetic_test_runs/slug-under-test"


@pytest.mark.parametrize(("golden_name", "case"), _GOLDEN_CASES)
def test_a_golden_compares_equal_to_its_re_emission_under_any_runs_root(golden_name, case, runs_dir_for_emission):
    """Property: the golden, re-rendered the way the generator renders it under an exported override root or under
    the slug root, compares equal after the canonical forms are built.

    Class: NEW CAPABILITY, over both goldens. The wrong implementation compares the raw masked text, which the
    override root breaks in two places: the runs root has no slug segment, and the relative model path climbs a
    different number of hops. That is the red the two byte-identity tests show when the variable is exported.

    Why it is required, checkable against the code: `_canonical_forms` is the only place the override root and the
    hop count are reconciled, and the byte-identity tests reach it only when the variable is exported, which no
    laptop run does.

    What kills it: dropping the runs-root mask, or resolving the golden's path against the wrong directory.

    A second correct implementation under which it passes: one that compares fully resolved absolute paths on both
    sides instead of placeholders."""
    golden = (GOLDENS_DIR / golden_name).read_text()
    got, analysis_dir = _emitted_from_golden(golden, case, runs_dir_for_emission)
    emitted, want = _canonical_forms(got, golden, analysis_dir)
    assert emitted == want, _unified_diff_excerpt(want, emitted)


#: The goldens carry no relative path (the figure rules read their source paths from the render manifest), so the three
#: layout mutations below would match nothing in them. The same three kinds of path are planted after the golden text
#: instead: a non-model relative path, a model path whose `../` run climbs to the cache root, and a model file path.
#: The control assertion then holds for the planted text too, and each mutation changes exactly one planted token.
_PLANTED_RELATIVE_PATHS = """
# planted by test_workflow_snakefile_byte_identity: layout-sensitive relative paths (see the comment above)
planted_dem_rel_path = '../elevation_10.00m.dem'
planted_watershed_rel_path = '../../../../synthetic_test_models/{MODEL_KEY}/watershed.geojson'
planted_boundary_rel_path = '../../../../synthetic_test_models/{MODEL_KEY}/boundary.geojson'
"""


def _drop_a_model_hop(text: str) -> str:
    """One ``../`` fewer on the first relative model path."""
    return _MODEL_PATH_RUN.sub(lambda m: m.group(1)[3:] + m.group(2), text, count=1)


@pytest.mark.parametrize(
    ("mutant", "mutate"),
    [
        ("extra hop on a non-model path", lambda t: t.replace("'../elevation_", "'../../elevation_", 1)),
        ("one model path component changed", lambda t: t.replace("/boundary.geojson", "/boundary.json", 1)),
        ("model path one hop shallower", _drop_a_model_hop),
    ],
)
@pytest.mark.parametrize(("golden_name", "case"), _GOLDEN_CASES)
def test_a_layout_regression_still_reddens_the_golden_under_an_exported_override(
    golden_name, case, mutant, mutate, tmp_path, monkeypatch
):
    """Property: each layout regression changes the emitted text so that it no longer equals the golden, with the
    override exported.

    Class: NEW CAPABILITY, must-fire. Three regressions: one extra ``../`` on a non-model path, one model-path
    component changed, and a model path one hop shallower. The last is the one a mask of the ``../`` run would
    accept, which is why the normaliser resolves paths and does not mask them. Control: the unmutated text equals the
    golden (the test above, and the first assertion here).

    Why it is required, checkable against the code: `_normalize_volatile` forbids a depth mask in its own comment,
    and a resolving normaliser is only as strict as the resolution; this pins that it did not become a mask.

    What kills it: masking the model-path ``../`` run, or ignoring the hop count when resolving.

    A second correct implementation under which it passes: one that resolves with `os.path.realpath` on a scratch
    tree instead of string arithmetic.

    Each arm also asserts that its mutation changed the text, so an arm that matches nothing cannot pass."""
    monkeypatch.setenv(_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
    golden = (GOLDENS_DIR / golden_name).read_text() + _PLANTED_RELATIVE_PATHS
    got, analysis_dir = _emitted_from_golden(golden, case, f"{tmp_path}/synthetic_test_runs")
    emitted, want = _canonical_forms(got, golden, analysis_dir)
    assert emitted == want, "control: the unmutated re-emission must equal the golden"
    mutated = mutate(got)
    assert mutated != got, f"{mutant}: the mutation changed nothing, so this arm checked nothing"
    emitted, want = _canonical_forms(mutated, golden, analysis_dir)
    assert emitted != want, f"{mutant}: the regression went unnoticed"


def test_home_data_dir_mask_survives_a_symlinked_home() -> None:
    """The ``{HOME_REL}`` mask must fire whether or not ``$HOME`` is a symlink.

    Two arms, because one arm alone cannot distinguish a working mask from a mask that
    happens to match this machine. Where ``$HOME`` resolves to itself (a laptop),
    ``os.path.relpath`` climbs only to the shared prefix and the ``../`` run is followed
    directly by ``.local/share/``. Where ``$HOME`` is a symlink onto another filesystem
    (every Lustre/GPFS cluster home), relpath climbs to ``/`` and descends through the
    RESOLVED path first, so the run is followed by the real home segments.

    Measured 2026-08-24: the pre-fix pattern masked the first arm and not the second, so
    5 byte-identity goldens failed on Rivanna while passing locally — a red that looked
    like golden staleness and was environment sensitivity. This test is the falsifiable
    guard, and it needs no cluster to run.
    """
    laptop = "'../../../.local/share/hhemt/examples/norfolk/x.shp'"
    cluster = "'../../../../../../../../../gpfs/tardis/home/u/.local/share/hhemt/examples/norfolk/x.shp'"

    for arm, text in (("laptop", laptop), ("symlinked-home", cluster)):
        got = _normalize_volatile(text)
        assert "{HOME_REL}/.local/share/" in got, (
            f"{arm} arm: the home-data-dir mask did not fire, so a machine-specific path "
            f"survives into the byte-identity comparison. got={got!r}"
        )
        assert "gpfs" not in got and "/home/u/" not in got, (
            f"{arm} arm: a machine-specific segment leaked past the mask. got={got!r}"
        )


def test_synth_root_mask_collapses_cache_root_and_worktree_slug() -> None:
    """The synth tier lives outside the repo, so {REPO_ROOT} cannot mask it.

    The cache root is host-specific and each sibling root beneath it carries ONE
    varying segment: a worktree slug under ``synthetic_test_runs``
    (``worktree_slug()`` returns ``"main"`` off-worktree and the branch slug inside
    one) and a SHA-1 cache key under ``synthetic_test_models``, the latter churning
    on every toolkit version bump. The arms are per-VARYING-SEGMENT-SHAPE rather
    than per-environment. The third arm exists because the first two cannot see a
    sibling-root gap, and the FOURTH is a negative control: it has no varying segment,
    so it must survive UNCHANGED apart from the host prefix. Under-reach and over-reach
    are different defects and a suite whose arms all have exactly one varying segment
    can only detect the first.
    """
    cache = str(cache_paths.cache_root())
    for arm, text, must_go, must_stay in (
        (
            "runs/off-worktree",
            f"'{cache}/synthetic_test_runs/main/synth_multi_sim/cfg.yaml'",
            "main",
            "{HHEMT_CACHE}/synthetic_test_runs/{SLUG}/synth_multi_sim/cfg.yaml",
        ),
        (
            "runs/in-worktree",
            f"'{cache}/synthetic_test_runs/sidequest-clearing-08-30/synth_multi_sim/cfg.yaml'",
            "sidequest-clearing-08-30",
            "{HHEMT_CACHE}/synthetic_test_runs/{SLUG}/synth_multi_sim/cfg.yaml",
        ),
        (
            "models/cache-key",
            f"'{cache}/synthetic_test_models/b495da2fed88fb9a/dem.tif'",
            "b495da2fed88fb9a",
            "{HHEMT_CACHE}/synthetic_test_models/{MODEL_KEY}/dem.tif",
        ),
        # NEGATIVE CONTROL. Nothing under this root varies by host or checkout, so the
        # SIF FILENAME must survive: run_simulation.py emits it into a rule shell, and
        # a mask that eats it deletes a generation-logic token from the golden. Without
        # this arm the test cannot fail on an over-broad pattern, because every other
        # arm has exactly one varying segment and any greedy rule satisfies them all.
        (
            "sif_cache/fixed-filename",
            f"'{cache}/sif_cache/hhemt-0.1.0-cuda.sif'",
            cache,
            "{HHEMT_CACHE}/sif_cache/hhemt-0.1.0-cuda.sif",
        ),
        # RELATIVE rendering of the models root. `os.path.relpath` strips the cache
        # prefix, so the absolute-anchored rule cannot fire and the digest survived into
        # the comparison. `must_stay` carries the `../` run, so this arm ALSO fails if a
        # later edit masks the depth -- the tripwire and the mask are asserted together.
        (
            "models/relative-rendering",
            "'../../../../synthetic_test_models/b495da2fed88fb9a/watershed.geojson'",
            "b495da2fed88fb9a",
            "../../../../synthetic_test_models/{MODEL_KEY}/watershed.geojson",
        ),
        # NEGATIVE CONTROL for the relative rule. A relative path into a FIXED-segment
        # sibling root must pass through untouched: `must_go` is the placeholder itself,
        # so the arm fails precisely when the new rule over-reaches from "one segment
        # under the models root" to "one segment under any relative root" and eats a SIF
        # filename that run_simulation.py emits into a rule shell. Without this arm the
        # fifth arm is satisfied by any greedy relative pattern.
        (
            "sif_cache/relative-fixed-filename",
            "'../../../../sif_cache/hhemt-0.1.0-cuda.sif'",
            "{MODEL_KEY}",
            "../../../../sif_cache/hhemt-0.1.0-cuda.sif",
        ),
    ):
        got = _normalize_volatile(text)
        assert must_stay in got, (
            f"{arm} arm: normalization did not produce the expected stable form — either "
            f"the mask did not fire, or it consumed a token it must preserve. got={got!r}"
        )
        assert must_go not in got, (
            f"{arm} arm: a host- or checkout-specific token leaked past the mask, so the "
            f"golden would only reproduce where it was captured. got={got!r}"
        )


@pytest.mark.parametrize("symlinked_home", [True, False])
def test_the_cache_root_mask_covers_the_resolved_spelling_on_a_symlinked_home(tmp_path, monkeypatch, symlinked_home):
    """Property: a model path rendered under the RESOLVED cache root normalises to the same placeholder form as one
    rendered under the unresolved root, with no host prefix left in front of the token.

    Class: NEW CAPABILITY. On a host whose home is a symlink (`home -> real/home`), the platform cache root is
    spelled through the link while the generator renders some paths from the resolved analysis root, so the real
    prefix survives a mask of the unresolved spelling alone: the Rivanna diff read
    `/sfs/gpfs/tardis{HHEMT_CACHE}/synthetic_test_models/...`. Arm `symlinked_home=False` is the control: a real cache
    root, where the added mask must be a no-op, so both spellings are the same string and the result is the old one.

    Why it is required, checkable against the code: `_normalize_volatile` masks `cache_root().resolve()` and
    `cache_root()` in that order; only the first catches the resolved spelling, and the host this suite was written on
    has a real home, so no other test here would see the difference.

    What kills it: removing the `.resolve()` mask line (the symlinked arm then keeps `real` in the output).

    A second correct implementation under which it passes: one that resolves every absolute path in the text before
    masking."""
    real_home = tmp_path / "real" / "home"
    real_home.mkdir(parents=True)
    if symlinked_home:
        (tmp_path / "home").symlink_to(real_home)
        home = tmp_path / "home"
    else:
        home = real_home
    unresolved_root = home / ".cache" / "hhemt"
    unresolved_root.mkdir(parents=True)
    monkeypatch.setattr(cache_paths, "cache_root", lambda: unresolved_root)
    resolved_root = unresolved_root.resolve()
    assert (resolved_root != unresolved_root) is symlinked_home, "control: the layout must have the symlink asked for"
    want = "'{HHEMT_CACHE}/synthetic_test_models/{MODEL_KEY}/boundary.geojson'"
    for spelling in (resolved_root, unresolved_root):
        got = _normalize_volatile(f"'{spelling}/synthetic_test_models/b495da2fed88fb9a/boundary.geojson'")
        assert got == want, f"{spelling}: {got!r}"


def test_no_shell_prefix_or_executable_disables_pipefail() -> None:
    """`| tee {log}` relies on Snakemake's default `set -euo pipefail` prefix.

    snakemake/shell.py::_get_process_prefix emits it under a CONJUNCTION:
    `shell_exec == "bash"` AND `cls._process_prefix is None`. Either half can be
    removed independently -- `shell.prefix(...)` defeats the second,
    `shell.executable("/bin/sh")` defeats the first -- and losing pipefail makes
    every teed rule report tee's exit status instead of its payload's, turning a
    truncated log into an INVISIBLE failure, which is strictly worse than the
    truncation this redirect form removes.

    Two assertions, because the toolkit is a Snakefile GENERATOR: a call in our
    own source and a call EMITTED INTO generated Snakefile text are different
    populations, and a guard that reads only the first is blind to the shape
    upstream's own documentation recommends.

    RESIDUAL, stated rather than implied away: an aliased import
    (`from snakemake.shell import shell as sh; sh.prefix(...)`) defeats both
    assertions. Closing it needs import resolution rather than a regex, which is
    a worse trade than the residual. This guard closes the direct-call and
    emitted-text routes for the two conjuncts NAMED ABOVE; it does not close
    aliasing.

    AND IT CANNOT CLOSE A THIRD CONJUNCT, which is environmental rather than
    lexical and is therefore outside any static check. snakemake/shell.py runs
    `if not shutil.which("bash"): ... shell.executable("sh")` AT IMPORT TIME
    (module scope, posix branch). So the precondition really reads: bash must be
    ON PATH in the process that RUNS THE RULE. Under `--executor slurm` that is
    not this machine and not the driver -- it is the innermost snakemake on the
    compute node, and for a containerized rule it is the interpreter inside the
    image. A minimal image without bash silently drops pipefail there. Snakemake
    does emit `Cannot set bash as default shell ... Falling back to sh.`, but it
    lands in the per-job SLURM log, which nothing routinely reads.

    A test running on a dev machine structurally cannot verify a property of a
    remote execution environment, so this is stated rather than guarded. If the
    toolkit ever adopts snakemake's native `container:` directive, note that
    deployment/singularity.py defaults `shell_executable` to "sh" -- which would
    defeat the first conjunct for every containerized rule by construction.
    """
    pat = re.compile(r"^\s*shell\.(prefix|executable)\s*\(")
    src = Path(__file__).resolve().parents[1] / "src" / "hhemt"
    offenders = [
        f"{p.relative_to(src)}:{i}"
        for p in src.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if pat.match(line)
    ]
    assert not offenders, (
        "shell.prefix()/shell.executable() disable Snakemake's default "
        f"`set -euo pipefail`, which `| tee {{log}}` depends on: {offenders}"
    )

    # Second population: the GENERATED Snakefile text. The byte-identity suite
    # already holds this string in hand, so this costs one generation and no fixture.
    tc = Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    generated = SnakemakeWorkflowBuilder(tc.analysis).generate_snakefile_content()
    emitted = [tok for tok in ("shell.prefix", "shell.executable") if tok in generated]
    assert not emitted, (
        "generated Snakefile emits "
        f"{emitted}, which replaces Snakemake's default `set -euo pipefail` and "
        "silently breaks exit-status propagation through `| tee {log}`"
    )
