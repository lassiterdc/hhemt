"""The whole shipped migration ladder, walked from v0 under the DEFAULT target.

WHY THIS FILE EXISTS, and why the property no longer lives where it used to.
`tests/test_version_migration_V0022.py::test_the_ladder_actually_reaches_V0022_on_a_third_state_tree`
owned two properties at once: a V0022-LOCAL one (V0022 is reachable from a v21 tree and
transforms it correctly) and a MOVING one (a walk under the default target ends exactly
where `LAYOUT_VERSION` says it ends). The moving property made that arm go red at every
bump -- measured at the V0023 bump: `AssertionError: Left contains one more item:
'V0023__column_set_heterogeneity_coords'` -- because a V0022-scoped literal cannot name a
rung that did not exist when it was written. The moving property is REAL and is KEPT; it is
relocated here, where the whole maintenance cost is one appended line per bump. That arm now
passes `target=22` and asserts only what is V0022-specific, which is sound ONLY while this
module exists and runs.

WHY `EXPECTED_LADDER` IS HAND-MAINTAINED, stated at length because the next author at the
next bump WILL reach for the alternative this repo blesses on the neighbouring test.
`tests/test_version_migration_golden.py::_discover_fixture_pairs` derives its ENTIRE
parametrize set from a glob over `tests/fixtures/legacy_layouts/v*`, under a docstring
commitment to "detection via fixture glob, not hand-maintained parametrize list". That form
is LEGITIMATE THERE: a fixture-directory set is not the registry, so expectation and
observation remain two independent declarations at zero per-bump cost. Automatic tracking
per se is not the defect.

It is refused HERE for one narrow reason. A fixture directory is named `v23`; the module is
named `V0023__column_set_heterogeneity_coords`. The module NAME is not recoverable from the
directory name, so a glob-derived expectation must either re-read `versions/` -- the same
directory `registry.plan` globs through `discover_migrations`, which collapses expectation
and observation into a single read of a single directory, leaving a missing rung absent from
both sides -- or drop the name, and with it FILENAME coverage. FILENAME is one of the three
defects the relocated arm's own docstring claims for itself (`version_from`, `version_to`,
and the filename pattern). The hand-maintained list buys the filename and nothing else buys
it. Do not replace it with a glob.

THE AUTOUSE PATCH IS NOT A CONVENIENCE AND IS NOT THIS MODULE'S INVENTION. `V0001.upgrade`
calls `MigrationContext.build_expected_slugs_for_current_version`, which constructs a
`TRITONSWMM_system` from the fixture's `cached_configs/system.yaml`. That file is a SCHEMA
STUB, not a loadable config, and says so in its own header comment -- measured on an
unpatched v0 walk: `ValidationError: constant_mannings is required when
toggle_use_constant_mannings is True`, surfacing as `MigrationConflictError: migration V0001
failed at operation 0`. `test_version_migration_golden.py::_patch_build_expected_slugs`
therefore fakes that one method for every non-slow arm, and that fake is the ONLY reason its
already-green `test_pair_round_trip[v0-v23]` demonstrates the v0 traversal works at all.
This module replicates the fixture rather than importing it across test modules, because a
cross-module test import binds this file's COLLECTION to the other file's import success;
the fake is short and its rationale now sits above it in both places. Patching it costs this
module nothing it asserts -- V0001's slug derivation is owned by the golden arms, and no
assertion here reads a slug.

NOT SIMULATION-BEARING. No conftest fixture is requested; nothing reaches `compile_*`,
`TRITONSWMM_run`, `analysis.run()` or `analysis.test()`; and the autouse patch above is
precisely what keeps `TRITONSWMM_system` out of the walk.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

from hhemt.scenario import compute_event_id_slug
from hhemt.version_migration import registry, runner
from hhemt.version_migration.constants import LAYOUT_VERSION
from hhemt.version_migration.context import MigrationContext

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"

_STEM_PATTERN = re.compile(r"^V(?P<n>\d{4})__")

# HAND-MAINTAINED. One appended line per LAYOUT_VERSION bump; see the module docstring for
# why this is derived from nothing -- and in particular not from `versions/`, the directory
# the walk itself reads.
EXPECTED_LADDER: list[str] = [
    "V0001__rename_scenario_dirs",
    "V0002__snakemake_flag_rename",
    "V0003__datatree_consolidation",
    "V0004__cf_conventions_backfill",
    "V0005__inline_report_config",
    "V0006__fingerprint_schema_v3",
    "V0007__setup_target_flag_rename",
    "V0008__fix_per_rank_diff_aggregation",
    "V0009__crs_config_composite_submodel",
    "V0010__canonical_plot_id",
    "V0011__eda_subpackage",
    "V0012__eda_report_dir",
    "V0013__partition_as_sensitivity_axis",
    "V0014__static_plots_dir",
    "V0015__eda_public_notebook",
    "V0016__datatree_provenance_core",
    "V0017__version_provenance_stamp",
    "V0018__ledger_driven_resume_reset",
    "V0019__member_vocabulary",
    "V0020__master_vocabulary",
    "V0021__experiment_tree_unification",
    "V0022__promote_producer_written_experiment_tree",
    "V0023__column_set_heterogeneity_coords",
    "V0024__rank_coherent_perf_reductions",
]


def _version_implied_by(stem: str) -> int:
    """The `version_to` a Flyway-style stem declares in its own name.

    Parsed from the NAME rather than read off the imported module, deliberately: reading
    `version_to` would make assertion (ii) a second read of the registry rather than a
    comparison against the hand-maintained declaration.
    """
    match = _STEM_PATTERN.match(stem)
    assert match is not None, f"EXPECTED_LADDER entry {stem!r} is not a V{{NNNN}}__slug stem"
    return int(match.group("n"))


def _fixture_expected_slugs(fixture_dir: Path) -> set[str]:
    """Read expected slugs from the fixture's analysis.yaml without instantiating
    TRITONSWMM_analysis. Avoids PySwmm MultiSimulationError (CLAUDE.md Gotcha #3)."""
    data = yaml.safe_load((fixture_dir / "cached_configs" / "analysis.yaml").read_text())
    return {compute_event_id_slug(ix) for ix in data.get("expected_weather_indexers", [])}


@pytest.fixture(autouse=True)
def _patch_build_expected_slugs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror of `test_version_migration_golden.py::_patch_build_expected_slugs`.

    See the module docstring: without it, V0001 hard-fails on the v0 fixture's schema-stub
    system.yaml and no v0-rooted walk is possible from any module.
    """

    def fake_build(self: MigrationContext) -> set[str]:
        if self.cfg_paths is None:
            return set()
        return _fixture_expected_slugs(self.cfg_paths["analysis"].parent.parent)

    monkeypatch.setattr(MigrationContext, "build_expected_slugs_for_current_version", fake_build)


def test_the_shipped_ladder_walks_v0_to_layout_version_under_the_default_target(tmp_path: Path) -> None:
    """The three moving properties of the ladder, in one arm over the SHIPPED registry.

    `tests/test_version_migration_registry.py` cannot host this: every one of its arms
    monkeypatches `registry._versions_dir` onto a tmp_path, so it validates the registry
    ALGORITHM and never the shipped registry CONTENT.

    (i)   the walk applies exactly EXPECTED_LADDER -- membership, order, count, FILENAME;
    (ii)  the version EXPECTED_LADDER's last stem names equals LAYOUT_VERSION;
    (iii) the highest `version_to` any SHIPPED module declares equals LAYOUT_VERSION.

    (iii) is not redundant with (i) or (ii), and the case it alone reaches is the one every
    other guard misses: a `V{N+1}__*.py` shipped while the constant stays put and the list
    stays unappended. `registry.plan`'s `while cur < target:` loop never reaches an
    above-ceiling module, so (i) stays green; `EXPECTED_LADDER[-1]` still implies the
    unmoved constant, so (ii) stays green; and `registry.validate_registry` stays green
    because its completeness set is `set(range(MINIMUM_SUPPORTED_VERSION, LAYOUT_VERSION))`
    tested only as `expected - declared_from`, which is structurally blind above the
    ceiling. The migration then ships and never runs for any user. (ii) catches a bump
    ritual that is HALF done -- exactly one of {constant, list} moved; (iii) catches one
    that has not started on either.

    CI VENUE, which is why these live in pytest rather than being left to the CLI checker.
    `grep -rn "check_layout_version" .github/` exits 1 with zero matches: check-a and
    check-b are wired only in `.pre-commit-config.yaml` (pre-commit and pre-push stages),
    while the GitHub workflow runs a bare pytest. This arm is the only CI-venue guard on
    layout-version discipline.
    """
    work = tmp_path / "v0"
    shutil.copytree(FIXTURE_ROOT / "v0", work)
    cfg_paths = {
        "system": work / "cached_configs" / "system.yaml",
        "analysis": work / "cached_configs" / "analysis.yaml",
    }

    # NO explicit `target=`. `run_migration` resolves `target = LAYOUT_VERSION if target is
    # None else target`, so passing the number by hand bypasses the constant entirely and
    # every property below becomes a tautology over the literal that was passed in. The
    # default form is what a production `hhemt migrate` runs.
    result = runner.run_migration(work, apply=True, cfg_paths=cfg_paths)
    assert result.target_version == LAYOUT_VERSION, (
        "the default target did not resolve to LAYOUT_VERSION; the assertions below are no longer about the constant"
    )

    # (i) membership, order, count and FILENAME, against a declaration derived from nothing.
    assert result.migrations_applied == EXPECTED_LADDER

    # (ii) the bump ritual moved BOTH the constant and the list, or neither.
    assert _version_implied_by(EXPECTED_LADDER[-1]) == LAYOUT_VERSION, (
        f"EXPECTED_LADDER ends at {EXPECTED_LADDER[-1]!r} but LAYOUT_VERSION is "
        f"{LAYOUT_VERSION}: exactly one of the two moved at the last bump"
    )

    # (iii) a DIRECTORY OF FILES compared against an INT, neither derived from the other.
    shipped_ceiling = max(m.version_to for m in registry.discover_migrations())
    assert shipped_ceiling == LAYOUT_VERSION, (
        f"a migration module declares version_to={shipped_ceiling} but LAYOUT_VERSION is "
        f"{LAYOUT_VERSION}: a shipped migration above the ceiling never runs for any user"
    )
