"""V0018 behavioural tests — regeneration, both families, dry-run safety, and stamping.

The `regenerate_*` fixtures seed a PRE-FIX perf summary zarr (Total 300.5) beside the real
raw corpus captured from sa_serial_6_r1 on the synth_cc_resume_triton campaign. Post
migration every column must equal the ledger-corrected value -- the sum of the
segment-final rows at `resume_reporting_tsteps = [36, 72, 108]` plus the final checkpoint.

Both the PRE and POST numbers are measured, not chosen: PRE comes from the campaign's own
scenario_status.csv, POST from running the corrected aggregator over the captured corpus.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import xarray as xr

from hhemt.version_migration import runner

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts" / "v0018_unit_test"

#: Pre-fix, from synth_cc_resume_triton scenario_status.csv (sa_serial_6_r1).
PRE_FIX_TOTAL = 300.5
PRE_FIX_INIT = 0.14237

#: Post-fix, from the corrected aggregator over the captured corpus.
CORRECTED = {
    "Total": 409.30,
    "Compute": 405.46,
    "SWMM": 0.0,
    "MPI": 0.0,
    "Simulation": 409.09,
    "IO": 2.0076,
    "Resize": 0.0,
    "Other": 1.6524,
    "Init": 0.19931,
}


def _copy_variant(name: str, tmp_path: Path) -> Path:
    work = tmp_path / name
    shutil.copytree(FIXTURE_ROOT / name, work)
    return work


def _summary(work: Path) -> xr.Dataset:
    return xr.open_zarr(next(work.rglob("*_perf_summary.zarr")), consolidated=False)


@pytest.mark.parametrize(
    "variant", ["regenerate_triton_only", "regenerate_coupled"]
)
def test_v0018_regenerates_both_model_families(variant, tmp_path):
    """BOTH-FAMILY COVERAGE, made falsifiable.

    V0008 globbed only TRITONSWMM_perf_tseries.zarr and hardcoded out_tritonswmm/, so it
    was a silent no-op on every pure-TRITON sim -- and the campaign V0018 exists to repair
    is pure-TRITON. If V0018 ever regresses to that two-glob shape, the
    `regenerate_triton_only` parameter fails here. A prose claim of coverage is not
    falsifiable; this is.

    Every tracked metric is asserted, not just Total: a correction that fixed the headline
    wallclock while leaving a category column wrong would otherwise pass.
    """
    work = _copy_variant(variant, tmp_path)
    runner.run_migration(work, target=18, apply=True)

    summary = _summary(work)
    mismatches = []
    for col, want in CORRECTED.items():
        got = float(summary[col].item())
        if got != pytest.approx(want, rel=1e-3, abs=1e-6):
            mismatches.append(f"{col}: got {got!r}, expected {want!r}")
    assert not mismatches, (
        f"{variant}: {len(mismatches)} of {len(CORRECTED)} tracked metrics wrong after "
        f"migration:\n  " + "\n  ".join(mismatches) + "\n"
        "A Total near 300.5 means the migration never reached this model family -- check "
        "the _FAMILIES globs."
    )
    assert "V0018-regenerated" in str(summary.attrs.get("notes", ""))


@pytest.mark.parametrize(
    "variant", ["regenerate_triton_only", "regenerate_coupled"]
)
def test_pre_fix_zarr_is_actually_wrong(variant, tmp_path):
    """The differential's PRE arm. Without this the test above could pass vacuously.

    If the fixture is ever rebuilt with the corrected aggregator, the seeded zarr would
    already hold 409.30, the post-migration assertion would pass without the migration
    doing anything, and the module would become a tautology. This fails loudly instead.
    """
    work = _copy_variant(variant, tmp_path)
    ds = _summary(work)
    assert float(ds["Total"].item()) == pytest.approx(PRE_FIX_TOTAL, rel=1e-3), (
        "the fixture must be seeded with the PRE-FIX value; regenerating it with the "
        "corrected aggregator makes the regeneration test vacuous"
    )
    assert float(ds["Init"].item()) == pytest.approx(PRE_FIX_INIT, rel=1e-3), (
        "Init is the column whose INCREASE at a boundary defeated the retired predicate; "
        "its pre-fix value is what makes this differential specific rather than generic"
    )


def test_v0018_is_a_noop_on_dry_run(tmp_path):
    """DRY-RUN SAFETY, deliberately not inherited from V0008.

    runner.run_migration calls upgrade() on BOTH paths and gates only ctx.execute(), so a
    migration that writes inside upgrade() mutates on a dry run. V0008 does exactly that.
    This asserts V0018 does not: the seeded PRE-FIX value must survive untouched.
    """
    work = _copy_variant("regenerate_triton_only", tmp_path)
    runner.run_migration(work, target=18, apply=False)

    assert float(_summary(work)["Total"].item()) == pytest.approx(PRE_FIX_TOTAL, rel=1e-3), (
        "a dry run must not regenerate zarrs -- upgrade() runs on the dry-run path, so "
        "every write must be gated on ctx.dry_run"
    )


def test_v0018_skips_never_resumed_sims_byte_unchanged(tmp_path):
    """Empty ledger -> old and new agree exactly -> do not rewrite, do not touch mtimes."""
    work = _copy_variant("regenerate_triton_only", tmp_path)
    log = next(work.rglob("log_triton.json"))
    payload = json.loads(log.read_text())
    payload["resume_reporting_tsteps"] = []
    log.write_text(json.dumps(payload))

    seeded = next(work.rglob("TRITON_only_perf_summary.zarr"))
    before = sorted((str(p.relative_to(seeded)), p.stat().st_mtime_ns) for p in seeded.rglob("*") if p.is_file())
    runner.run_migration(work, target=18, apply=True)
    after = sorted((str(p.relative_to(seeded)), p.stat().st_mtime_ns) for p in seeded.rglob("*") if p.is_file())

    assert before == after, (
        "a never-resumed sim must be left byte- and mtime-identical: rewriting it would "
        "fire Snakemake's mtime triggers across the consolidate/report cascade for "
        "numerically identical output"
    )


def test_v0018_stamps_uncorrectable_when_raw_perf_is_cleared(tmp_path):
    """Resumed + raw cleared -> stamp, never silently pass as corrected."""
    work = _copy_variant("stamp_stale", tmp_path)
    runner.run_migration(work, target=18, apply=True)

    marker = next(work.rglob("_V0018_resume_reset_uncorrectable.json"), None)
    assert marker is not None, (
        "a resumed sim whose raw performance{N}.txt inputs were cleared is permanently "
        "uncorrectable and MUST be stamped -- merging it into the skip branch would "
        "launder a known-wrong value as a clean pass"
    )
    assert json.loads(marker.read_text())["resume_reporting_tsteps"] == [36, 72, 108]
    # The un-regenerable values are left ALONE, not overwritten with a guess.
    assert float(_summary(work)["Total"].item()) == pytest.approx(PRE_FIX_TOTAL, rel=1e-3)


def test_v0018_invalidates_consolidation_signals(tmp_path):
    """Regeneration must make the next run REBUILD, not skip.

    The consolidate gate skips when the tree exists, the log says complete, and the
    stamped fingerprint matches -- and that fingerprint covers tree SHAPE only, so it is
    invariant across this migration. Both signals must therefore be cleared explicitly:
    the flags decide whether Snakemake re-fires the rule, the log field decides whether
    the fired rule rebuilds or prints "Not overwriting" and exits.
    """
    work = _copy_variant("regenerate_triton_only", tmp_path)
    status = work / "_status"
    status.mkdir(exist_ok=True)
    (status / "e_consolidate_sa-0_complete.flag").touch()
    (status / "f_consolidate_master_complete.flag").touch()
    (status / "d_process_evt0_complete.flag").touch()  # unrelated family: must SURVIVE
    (work / "log.json").write_text(json.dumps({"datatree_consolidation_complete": True}))

    runner.run_migration(work, target=18, apply=True)

    assert not (status / "e_consolidate_sa-0_complete.flag").exists()
    assert not (status / "f_consolidate_master_complete.flag").exists()
    assert (status / "d_process_evt0_complete.flag").exists(), (
        "only the consolidate flag families may be cleared; touching d_process_* would "
        "re-arm clear-raw on the next run"
    )
    assert json.loads((work / "log.json").read_text())["datatree_consolidation_complete"] is None, (
        "clearing the flags alone only makes Snakemake re-fire the rule; the rule would "
        "then print 'Not overwriting' and exit unless this log signal is also cleared"
    )
