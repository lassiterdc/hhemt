"""Byte-identity regression gate for the named-reporting-sets data-drive (R6/OE-1).

The P1b dispatcher refactor (`_emit_active_set_plot_rules`) replaces the hardcoded
`_build_plot_rule_block_*` call lists — duplicated across the multisim, sensitivity-
master, and reprocess-master generators — with one registry-driven dispatcher that
iterates the active reporting set's `renderer_selection`. The refactor must be
behavior-preserving for the SHIPPED sets: the generated Snakefile for `default`
(multisim) and `benchmarking` (sensitivity master + reprocess master) must be
byte-identical to the pre-refactor Snakefile. Snakemake keys reruns on rule
input/output/code; byte-identity ⇒ no rerun cascade for existing analyses on
landing.

This test pins the generated Snakefile text against committed golden fixtures
captured from the PRE-refactor generators (capture-then-refactor-then-assert-equal).

Golden capture (one-time, run against the PRE-refactor source before P1b lands):

    CAPTURE_REPORTING_SET_GOLDENS=1 python -m pytest \
        tests/test_synth_reporting_sets_byte_identity.py

In capture mode each test writes its golden and skips. In normal mode (the env var
absent) each test asserts byte-identity against the committed golden.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import platformdirs
import pytest

from tests.fixtures._triton_source_cache import synthetic_runs_root

_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "reporting_sets_byte_identity"
_CAPTURE = os.environ.get("CAPTURE_REPORTING_SET_GOLDENS") == "1"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SYNTH_RUNS_ROOT = synthetic_runs_root()
_SYNTH_MODELS_ROOT = Path(platformdirs.user_cache_dir("hhemt")) / "synthetic_test_models"
# The synth fixtures' runs_root is nested under pytest's tmp base (the fixtures set
# `HHEMT_TEST_RUNS_ROOT_OVERRIDE` from `tmp_path` / `tmp_path_factory.mktemp`), so the
# goldens bake `{basetemp}/{nodeid-slug}/...`. Only the basetemp PREFIX is volatile; the
# nodeid-derived slug is deterministic and is left intact as real signal.
#
# ASK PYTEST FOR THE BASETEMP, NEVER MATCH ITS SHAPE. The basetemp is CHOSEN by the
# invoker, not discoverable by pattern: the Rivanna suite harness passes an explicit
# `--basetemp=$TMPDIR/hhemt-suite-{run_id}-{cid}`, which no `pytest-of-*/pytest-N`
# pattern can ever match, so a shape-matching mask leaves the whole basetemp unmasked
# off-laptop. A shape mask is stale-by-construction for a value another program picks:
# the harness may rename that directory at any time. `getbasetemp()` is the authority
# pytest itself uses to build every `tmp_path`, so masking its literal string tracks
# whatever the invoker chose. It is also always RESOLVED, which a mask built from
# `tempfile.gettempdir()` is not -- so this additionally covers a symlinked `$TMPDIR`,
# where the two disagree and the shape mask silently matches nothing.
_PYTEST_BASETEMP: str | None = None


@pytest.fixture(autouse=True, scope="session")
def _capture_pytest_basetemp(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Stash this invocation's effective pytest basetemp for `_normalize_volatile`."""
    global _PYTEST_BASETEMP
    _PYTEST_BASETEMP = str(tmp_path_factory.getbasetemp())


# Model-cache source-path attributions appear as variable-depth ``../``-relative paths
# (``os.path.relpath`` from the deep analysis dir climbs to ``/`` then descends through
# the absolute home dir into the out-of-repo model cache). The ``../`` depth varies with
# tree nesting and the descended segment bakes the machine home — mask both, mirroring
# suite-1's ``{HOME_REL}`` pattern, while preserving the FILENAME. The content-hash dir
# is masked separately below (it is a fixture-generator content-address, not dispatch
# signal — see the ``{HASH}`` mask in ``_normalize_volatile``).
# ANCHOR ON THE SUBSTITUTED TOKEN, NOT ON THE ROOT TEXT. A ``../``-relative path
# CONTAINS the absolute root as a substring — the root's leading ``/`` is supplied by
# the final ``../`` of the climb — so the absolute replace in ``_normalize_volatile``
# fires INSIDE the relative path, and by the time any root-anchored relative regex runs
# its anchor text is already gone. Anchoring on ``{SYNTH_MODELS}`` and running AFTER the
# absolute replace strips whatever ``../`` climb and resolved-home descent precedes the
# token, which is the only part that differs between a symlinked-$HOME cluster
# (``../../..{RESOLVED_HOME_SEGMENTS}{SYNTH_MODELS}``) and a non-symlinked laptop
# (``../../..{SYNTH_MODELS}``). The FILENAME and the content-hash dir are preserved, and
# a ``../`` path carrying no ``{SYNTH_MODELS}`` token is left untouched.
_SYNTH_MODELS_PREFIX_RE = re.compile(r"(?:\.\./)+[^'\"\s]*?(?=\{SYNTH_MODELS\})")


def _normalize_volatile(text: str) -> str:
    """Mask checkout-location-, interpreter-, and synth-cache-root-specific tokens
    so the byte-identity assertion is robust to where the repo is checked out, which
    interpreter runs it, and which worktree's out-of-repo synthetic caches the goldens
    were captured against. The synth caches live under ``platformdirs`` user-cache
    (outside ``_REPO_ROOT``), so each needs its own mask beyond suite-1's ``{REPO_ROOT}``.
    Genuine generation-logic tokens (rule names, resources, command shape, source-path
    FILE IDENTITY and path STRUCTURE) are left intact so real drift still fails the
    assertion; only the environment-derived cache-key hash WITHIN a source path is masked.
    """
    text = text.replace(sys.executable, "{PYTHON}")
    text = text.replace(str(_REPO_ROOT), "{REPO_ROOT}")
    if _PYTEST_BASETEMP:  # this invocation's basetemp, whatever the invoker chose
        text = text.replace(_PYTEST_BASETEMP, "{PYTEST_TMP}")
    text = text.replace(str(_SYNTH_RUNS_ROOT), "{SYNTH_RUNS}")
    text = re.sub(r"\{SYNTH_RUNS\}/[^/\"' ]+", "{SYNTH_RUNS}/{WT}", text)  # mask worktree slug
    text = text.replace(str(_SYNTH_MODELS_ROOT), "{SYNTH_MODELS}")  # absolute form (if any)
    text = _SYNTH_MODELS_PREFIX_RE.sub("", text)  # ../-climb + resolved-home prefix left by the replace above
    # The synth-model cache-dir NAME is a 16-hex `_cache_key` over
    # SyntheticModelParams + toolkit version + SHA-1 of every
    # src/hhemt/synthetic_model/*.py (cache.py). Any generator-source edit or
    # version bump rotates it, so it is volatile w.r.t. this suite (which pins
    # generation logic, not the synth-model identity). Mask it exactly like the
    # {SYNTH_MODELS} root so a cache-key rotation cannot stale the goldens.
    text = re.sub(r"(\{SYNTH_MODELS\})/[0-9a-f]{16}/", r"\1/{MODEL_KEY}/", text)
    return text


def _check(generated: str, golden_name: str) -> None:
    """Capture mode: write the golden and skip. Normal mode: assert byte-identity."""
    golden_path = _GOLDEN_DIR / golden_name
    if _CAPTURE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(_normalize_volatile(generated))
        pytest.skip(f"captured golden {golden_name} ({len(generated)} bytes)")
    golden = golden_path.read_text()
    assert _normalize_volatile(generated) == _normalize_volatile(golden), (
        f"Generated Snakefile diverged from {golden_name} — the registry-driven "
        f"data-drive is NOT behavior-preserving. Diff the generated text against "
        f"{golden_path} to locate the drifted rule."
    )


def test_multisim_default_byte_identical(synth_multi_sim_analysis):
    """`default` set (multisim) — dispatcher must reproduce the 6-renderer call
    list + trailing export rule byte-for-byte (CHANGE 2)."""
    builder = synth_multi_sim_analysis._workflow_builder
    generated = builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )
    _check(generated, "default_multisim.Snakefile")


def test_sensitivity_master_byte_identical(synth_sensitivity_analysis):
    """`benchmarking` set (sensitivity master) — dispatcher must reproduce the
    5 common renderers + interleaved export + 2 conditional renderers byte-for-byte
    (CHANGE 3 + B-i interleave hook)."""
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder
    generated = builder.generate_master_snakefile_content(which="both", compression_level=5)
    _check(generated, "benchmarking_master.Snakefile")


def test_reprocess_master_byte_identical(synth_sensitivity_analysis):
    """`benchmarking` set (reprocess master) — identical dispatcher path as the
    production master, same set + interleave hook (CHANGE 4)."""
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder
    generated = builder.generate_reprocess_master_snakefile_content(which="both", start_with="render")
    _check(generated, "benchmarking_reprocess_master.Snakefile")


def test_synth_models_mask_converges_across_a_symlinked_home() -> None:
    """``_normalize_volatile`` must converge across a symlinked and a non-symlinked ``$HOME``.

    Asserted at the NORMALIZER, not at one mask: the defect this guards lives in the
    ORDER of the masks, not in either mask. The absolute-root replace fires inside a
    ``../``-relative path and consumes the anchor a root-anchored relative regex would
    need, so asserting on that regex alone cannot observe the failure — it substitutes
    correctly on a raw string and never reaches the ordering. Reverting the mask fix
    makes this test red; the previous regex-level form stayed green.

    Same defect class, and the same two arms, as suite-1's
    ``test_home_data_dir_mask_survives_a_symlinked_home``: a symlinked home makes
    ``os.path.relpath`` descend through the RESOLVED path, so the ``../`` run is followed
    by the real home segments before this cache root rather than by the root directly.
    """
    root_rel = str(_SYNTH_MODELS_ROOT).lstrip("/")
    key = "abc0123456789def"
    laptop = f"watershed_rel_path='../../../../../../../{root_rel}/{key}/watershed.geojson',"
    cluster = f"watershed_rel_path='../../../../../../../sfs/gpfs/tardis/{root_rel}/{key}/watershed.geojson',"

    got_laptop = _normalize_volatile(laptop)
    got_cluster = _normalize_volatile(cluster)
    assert got_laptop == got_cluster, (
        "the normalizer did not converge across homes, so the byte-identity comparison "
        f"is machine-bound. laptop={got_laptop!r} cluster={got_cluster!r}"
    )
    assert "{SYNTH_MODELS}/{MODEL_KEY}/watershed.geojson" in got_laptop, (
        f"the synth-models mask did not fire. got={got_laptop!r}"
    )
    assert "gpfs" not in got_cluster, f"a machine-specific segment leaked. got={got_cluster!r}"
    assert "../" not in got_cluster, f"a relative-climb residual survived. got={got_cluster!r}"

    # differently-positioned satisfying input: a ../-relative path carrying no
    # model-cache token is real signal and must be returned byte-unchanged.
    unrelated = "source_paths = [{'path': '../elevation_10.00m.dem', 'variables': []}]"
    assert _normalize_volatile(unrelated) == unrelated, (
        f"an unrelated ../-relative path was mangled. got={_normalize_volatile(unrelated)!r}"
    )


def test_pytest_tmp_mask_survives_an_explicitly_chosen_basetemp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_normalize_volatile`` must converge across every basetemp an invoker may choose.

    Asserted at the NORMALIZER over three REAL basetemp shapes, all measured from
    ``tmp_path_factory.getbasetemp()``: pytest's default, the Rivanna suite harness's
    explicit ``--basetemp=$TMPDIR/hhemt-suite-{run_id}-{cid}``, and a default basetemp
    under a symlinked ``$TMPDIR`` (where ``getbasetemp()`` is resolved and
    ``tempfile.gettempdir()`` is not). A shape-matching mask converges on the first arm
    only, so reverting this fix makes this test red ON CONVERGENCE -- a behavioural
    failure rather than a missing-attribute error, because ``raising=False`` lets the
    arms run against code that never reads the attribute.
    """
    arms = {
        "default": "/tmp/pytest-of-someuser/pytest-177",
        "harness-explicit": "/tmp/hhemt-suite-20260824T000000Z-00",
        "symlinked-tmpdir": "/sfs/gpfs/tardis/scratch/u/pytest-of-someuser/pytest-0",
    }
    template = "analysis_dir = '{BT}/synth_multi_sim_builder0/synthetic_test_runs/multisim',"

    got = {}
    for arm, basetemp in arms.items():
        monkeypatch.setattr(sys.modules[__name__], "_PYTEST_BASETEMP", basetemp, raising=False)
        got[arm] = _normalize_volatile(template.replace("{BT}", basetemp))

    assert len(set(got.values())) == 1, (
        "the pytest-tmp mask did not converge across basetemp shapes, so the "
        f"byte-identity comparison is invoker-bound. {got}"
    )
    masked = next(iter(got.values()))
    assert "{PYTEST_TMP}/synth_multi_sim_builder0/" in masked, f"the basetemp mask did not fire. got={masked!r}"
    for token in ("hhemt-suite", "pytest-of", "/sfs/"):
        assert token not in masked, f"an invoker-specific segment {token!r} leaked. got={masked!r}"

    # differently-positioned satisfying input: a line carrying no basetemp is real
    # signal and must be returned byte-unchanged.
    unrelated = "shell: 'python -m hhemt.setup_workflow --analysis-config cfg.yaml'"
    assert _normalize_volatile(unrelated) == unrelated, (
        f"an unrelated line was mangled. got={_normalize_volatile(unrelated)!r}"
    )
