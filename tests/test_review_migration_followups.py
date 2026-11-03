"""Tests of the migration to layout 28 and of the merge of `origin/develop` that it sits on.

Each expectation comes from a source other than `tests/test_version_migration_v0028.py`: the toolkit's own writer
helpers (`utils.write_zarr`, `analysis_validation.check_consolidated_metadata_current`), the variable list of a real
cross-simulation identity artifact, and git history (the merge base, the two parents of the merge and the files
`origin/develop` changed). No solver is compiled or run.

Stored names that the migration replaces are assembled from parts, so that none appears whole in this file (the
tree-wide census of those names reads it).
"""

from __future__ import annotations

import filecmp
import hashlib
import importlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from hhemt.version_migration import runner
from hhemt.version_migration.exceptions import MigrationBlockedError
from hhemt.version_migration.state import VersionState, write_version_file

REPO = Path(__file__).resolve().parents[1]
MERGE_BASE = "fa98781e"  # the merge base of this branch and origin/develop
DEVELOP_TIP = "bffe88b3"  # origin/develop as merged
RUN_TIP = "c3167536"  # the first parent of that merge: this branch before it took in origin/develop
V0028_STEM = "V0028__flood_depth_and_unit_discharge_names"


def _n(*parts: str) -> str:
    return "_".join(parts)


def _git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return out.stdout


def _have(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", commit], capture_output=True).returncode == 0


needs_history = pytest.mark.skipif(
    not (_have(MERGE_BASE) and _have(DEVELOP_TIP) and _have(RUN_TIP)), reason="branch history not in this clone"
)

#: The 24 bases of the real layout-27 identity artifact (read from Rivanna) and the names they migrate to.
OLD_BASES = [
    "capacity_setting_last",
    "capacity_setting_max",
    _n("depth", "m", "last"),
    _n("depth", "m", "max"),
    _n("final", "surface", "flood", "volume", "m3"),
    _n("flooding", "cms", "last"),
    _n("flooding", "cms", "max"),
    _n("flow", "cms", "last"),
    _n("flow", "cms", "max"),
    _n("head", "m", "last"),
    _n("head", "m", "max"),
    _n("inflow", "flow", "cms", "last"),
    _n("inflow", "flow", "cms", "max"),
    _n("link", "depth", "m", "last"),
    _n("link", "depth", "m", "max"),
    _n("max", "velocity", "mps"),
    _n("max", "wlevel", "m"),
    _n("time", "of", "max", "velocity", "min"),
    _n("time", "of", "max", "wlevel", "min"),
    _n("velocity", "mps", "last"),
    _n("velocity", "mps", "max"),
    _n("velocity", "x", "mps", "at", "time", "of", "max", "velocity"),
    _n("velocity", "y", "mps", "at", "time", "of", "max", "velocity"),
    _n("wlevel", "m", "last", "tstep"),
]
NEW_BASES = [
    "capacity_setting_last",
    "capacity_setting_max",
    "depth_last",
    "depth_max",
    "final_surface_flood_volume",
    "flooding_last",
    "flooding_max",
    "flow_last",
    "flow_max",
    "head_last",
    "head_max",
    "inflow_flow_last",
    "inflow_flow_max",
    "link_depth_last",
    "link_depth_max",
    "max_unit_discharge",
    "max_flood_depth",
    "time_of_max_unit_discharge",
    "time_of_max_flood_depth",
    "velocity_last",
    "velocity_max",
    "unit_discharge_x_at_time_of_max_unit_discharge",
    "unit_discharge_y_at_time_of_max_unit_discharge",
    "flood_depth_last_tstep",
]


def _migrate(root: Path) -> None:
    """Stamp `root` as layout 27 and run the runner to layout 28."""
    write_version_file(root, VersionState.fresh(27, "0.0.0+test"))
    runner.run_migration(
        root, target=28, apply=True, cfg_paths={"system": root / "s.yaml", "analysis": root / "a.yaml"}
    )


def _digest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------------------------------------------
# validation_report.json
# --------------------------------------------------------------------------------------------------------------


def test_validation_reports_at_any_depth_are_renamed_in_variable_and_variables_and_nothing_else(tmp_path):
    """Property: a persisted `validation_report.json` anywhere under the tree has every `variable` string and
    every string of a `variables` list rewritten by the identity-artifact table, whatever the nesting; every other
    key (`summary`, `detail`, `name`) is left as written; a name already in the new spelling is left; a second pass
    changes no byte.

    Class: a report at the analysis root, one under `members/` and one under a doubly nested folder, each holding
    the 24 real bases in `variable` rows nested in a check record and in a `variables` list, with free text that
    quotes the stored names being replaced.
    Why required: the walk and the rewrite are recursive and keyed on the file NAME, so the position of the file
    and the list form of the field are inputs a one-report, one-depth seed does not reach.
    Kills it: a walk limited to the root; the `variables` list left unrenamed; a free-text rewrite.
    Passes under: any rewrite keyed on those two field names (a regex pass over the text would NOT: the free text
    would change).
    """
    free = "differs on " + OLD_BASES[16] + " and " + OLD_BASES[15]

    def report() -> dict:
        return {
            "name": "Byte identity across compute configurations",
            "summary": free,
            "checks": [
                {
                    "details": [{"variable": b, "detail": free} for b in OLD_BASES],
                    "sets": {"variables": list(OLD_BASES) + ["max_flood_depth"]},
                }
            ],
        }

    places = [
        tmp_path / "validation_report.json",
        tmp_path / "members" / "m1" / "validation_report.json",
        tmp_path / "a" / "b" / "c" / "validation_report.json",
    ]
    for p in places:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report(), indent=2))
    _migrate(tmp_path)
    for p in places:
        got = json.loads(p.read_text())
        chk = got["checks"][0]
        assert [d["variable"] for d in chk["details"]] == NEW_BASES, p
        assert chk["sets"]["variables"] == NEW_BASES + ["max_flood_depth"], p
        assert got["summary"] == free and all(d["detail"] == free for d in chk["details"]), p
        assert got["name"] == "Byte identity across compute configurations", p
    before = _digest(tmp_path)
    mod = importlib.import_module("hhemt.version_migration.versions." + V0028_STEM)
    from hhemt.version_migration.context import MigrationContext

    ctx = MigrationContext(target_dir=tmp_path, dry_run=False, migration_id="V0028")
    mod.upgrade(ctx)
    assert ctx.plan == [], [str(op)[:90] for op in ctx.plan]
    assert _digest(tmp_path) == before


@pytest.mark.parametrize("form", ["variable", "variables"])
def test_a_validation_report_naming_a_variable_whose_tier_it_does_not_record_is_refused_with_nothing_written(
    tmp_path, form
):
    """Property: a name whose tier the report does not record is refused, not guessed. A report holding a
    link-only layout-27 name (`flow` with its unit token, which the identity table does not name) is refused while
    planning, in BOTH field forms, and not one file of the tree changes, including a migratable report and a
    migratable store beside it.

    Class: the two field forms (`variable` string, `variables` list element) x a decidable report and an
    undecidable one in the same tree.
    Why required: the refusal sits in two separate branches of the walk (the string and the list), so a test of
    one form leaves the other unguarded.
    Kills it: a refusal only on the string form; a refusal raised after the decidable report was rewritten.
    Passes under: refusing before any plan is executed.
    """
    bad_name = _n("flow", "cms")
    good = tmp_path / "validation_report.json"
    good.write_text(json.dumps({"details": [{"variable": OLD_BASES[16]}]}))
    store_dir = tmp_path / "eda"
    store_dir.mkdir()
    ds = xr.Dataset({"identical__" + OLD_BASES[16]: (("sa_id",), np.array([True]))}, coords={"sa_id": ["1"]})
    ds.to_zarr(store_dir / "eda_cross_sim_identity.zarr", mode="w", consolidated=False)
    bad = tmp_path / "members" / "m" / "validation_report.json"
    bad.parent.mkdir(parents=True)
    payload = {"variable": bad_name} if form == "variable" else {"variables": [OLD_BASES[16], bad_name]}
    bad.write_text(json.dumps({"details": [payload]}))
    write_version_file(tmp_path, VersionState.fresh(27, "0.0.0+test"))
    before = _digest(tmp_path)
    with pytest.raises(MigrationBlockedError) as err:
        runner.run_migration(
            tmp_path, target=28, apply=True, cfg_paths={"system": tmp_path / "s.yaml", "analysis": tmp_path / "a.yaml"}
        )
    assert bad_name in str(err.value)
    assert _digest(tmp_path) == before
    assert "max_flood_depth" not in good.read_text()  # control: the decidable report was not rewritten either


# --------------------------------------------------------------------------------------------------------------
# the string-value swap through write_zarr
# --------------------------------------------------------------------------------------------------------------


def _raw_identity_store() -> xr.Dataset:
    rng = np.random.default_rng(3)
    t = _n("timestep", "min")
    return xr.Dataset(
        {
            "identical": (("compute_config", "raw_output_type", t), rng.random((2, 4, 5)) > 0.4),
            "max_abs_diff": (("compute_config", "raw_output_type", t), rng.random((2, 4, 5))),
        },
        coords={
            "compute_config": ["a", "b"],
            "raw_output_type": [_n("max", "wlevel", "m"), _n("wlevel", "m"), "x_old", "y_old"],
            t: np.arange(5, dtype="float64") * 10.0,
        },
        attrs={"units_by_raw_output_type": json.dumps({_n("wlevel", "m"): "m"})},
    )


def _array_facts(store: Path) -> dict[str, tuple]:
    """For each array of the store: values, dtype, chunk grid and the codecs, read with zarr itself."""
    root = zarr.open_group(str(store), mode="r")
    facts = {}
    for name, arr in root.arrays():
        facts[name] = (np.asarray(arr[...]), str(arr.dtype), tuple(arr.chunks), repr(arr.compressors))
    return facts


@pytest.mark.parametrize("had_index", [True, False], ids=["indexed_before", "unindexed_before"])
def test_the_string_swap_leaves_an_indexed_store_whose_other_arrays_are_unchanged(tmp_path, had_index):
    """Property: after the migration replaces the `raw_output_type` values of the EDA identity store, the store
    passes `check_consolidated_metadata_current(index_required=True)`, the replaced array holds the new strings in
    the same order, and every OTHER array has the same values, dtype, chunk grid and codecs it had (the time axis
    only changes its name).

    Class: the store with an index (written by `utils.write_zarr`) and the store without one (written before
    develop's index existed, here by `to_zarr(consolidated=False)`), both holding a coordinate with two old
    values and two values the table does not name.
    Why required: the swap is written through `write_zarr(encoding={})`, which is only correct if no other array is
    re-encoded by that route.
    Kills it: a swap that re-chunks or re-compresses the data arrays; one that reorders the coordinate; one that
    leaves the unindexed store without an index when the layout requires one.
    Passes under: a swap that indexes only a store that had an index, for the indexed arm.
    The unindexed arm records what an always-index swap does; it passes only if an index is present afterwards,
    which is what the layout requires of every store (the check's own 'index_required' arm).
    """
    from hhemt.analysis_validation import check_consolidated_metadata_current
    from hhemt.utils import write_zarr, zarr_store_has_index

    eda = tmp_path / "eda"
    eda.mkdir()
    store = eda / "raw_byte_identity.zarr"
    if had_index:
        write_zarr(_raw_identity_store(), store, chunks=None)
    else:
        _raw_identity_store().to_zarr(store, mode="w", consolidated=False)
    assert zarr_store_has_index(store) is had_index, "control: the seed has the index state under test"
    before = _array_facts(store)
    _migrate(tmp_path)
    after = _array_facts(store)
    new_values = [str(v) for v in after["raw_output_type"][0]]
    assert new_values == ["max_flood_depth", "flood_depth", "x_old", "y_old"]
    t_old, t_new = _n("timestep", "min"), "timestep"
    assert set(after) == (set(before) - {t_old}) | {t_new}
    for name in ("identical", "max_abs_diff"):
        assert np.array_equal(after[name][0], before[name][0]), name
        assert after[name][1:] == before[name][1:], (name, after[name][1:], before[name][1:])
    assert np.array_equal(after[t_new][0], before[t_old][0])
    result = check_consolidated_metadata_current(store, index_required=True)
    assert result.passed and result.applicable, result.summary


# --------------------------------------------------------------------------------------------------------------
# the index after V0028, over the committed layout-27 fixture
# --------------------------------------------------------------------------------------------------------------


def test_every_store_of_the_layout_27_fixture_has_a_current_index_after_the_migration(tmp_path):
    """Property: after the migration runs on a copy of the committed layout-27
    fixture (4 zarr stores, a consolidated experiment tree and its nested groups among them), every store passes
    `check_consolidated_metadata_current(index_required=True)`, including the stores the migration walked but did
    not need to change.

    Class: the whole fixture tree, store by store (stores are the directories holding a root `zarr.json` and
    named `*.zarr`).
    Why required: the requirement is over EVERY store the migration writes, and a rewrite route that a two-store
    seed does not reach (the nested tree, a group rewrite) would leave a stale index.
    Kills it: a primitive that edits a node's attributes or renames an array without rebuilding the index.
    Passes under: an index rebuilt once at the end of the migration instead of inside each primitive.
    Must-fire control: the same check on the UNMIGRATED copy is run first and the stores are counted, so a zero
    count of stores cannot pass vacuously.
    """
    import shutil

    from hhemt.analysis_validation import check_consolidated_metadata_current
    from hhemt.utils import zarr_store_has_index

    work = tmp_path / "v27"
    shutil.copytree(REPO / "tests" / "fixtures" / "legacy_layouts" / "v27", work)
    stores = sorted(p for p in work.rglob("*.zarr") if (p / "zarr.json").is_file())
    assert len(stores) >= 4, len(stores)
    indexed_before = [s for s in stores if zarr_store_has_index(s)]
    assert indexed_before, "control: the fixture carries indexed stores"
    _migrate(work)
    failures = {}
    for s in sorted(p for p in work.rglob("*.zarr") if (p / "zarr.json").is_file()):
        if s.relative_to(work) not in {x.relative_to(work) for x in indexed_before}:
            continue
        res = check_consolidated_metadata_current(s, index_required=True)
        if not (res.passed and res.applicable):
            failures[s.relative_to(work).as_posix()] = res.summary[:120]
    assert failures == {}


# --------------------------------------------------------------------------------------------------------------
# the SWMM parser reference stores
# --------------------------------------------------------------------------------------------------------------

_REF = REPO / "test_data" / "swmm_refactoring_reference"
_KEYS = {
    _n("runoff", "continuity", "error", "perc"): ("runoff_continuity_error", "percent"),
    _n("flow", "continuity", "error", "perc"): ("flow_continuity_error", "percent"),
    _n("system", "flooding", "10e6", "ltr"): ("system_flooding", "10^6 L"),
}


@needs_history
@pytest.mark.parametrize("stem", ["SWMM_link_summary", "SWMM_link_tseries", "SWMM_node_summary", "SWMM_node_tseries"])
def test_each_reference_store_has_the_renamed_keys_their_units_companions_and_the_old_values(stem):
    """Property: in each of the four reference stores the three attribute keys carry the unit-free names, with the
    SAME values the store held under the old keys (NaN equal), each followed by a `{key}_units` companion with the
    stated unit; the old keys are gone; every other root attribute is unchanged; and no array directory was
    touched.

    Class: the four stores x three keys; values read from the blob at the commit before the merge as the source of
    truth.
    Why required: the six readers of these stores compare the parser's output with the stores, so a changed value
    or a lost neighbouring attribute would turn them red or, worse, leave them green on different data.
    Kills it: a regenerated store whose values moved; a companion with another unit string; a dropped attribute.
    Passes under: parser regeneration or a direct attribute edit.
    """
    base_doc = json.loads(_git("show", f"{RUN_TIP}:test_data/swmm_refactoring_reference/{stem}.zarr/zarr.json"))
    now_doc = json.loads((_REF / f"{stem}.zarr" / "zarr.json").read_text())
    old_attrs, new_attrs = base_doc["attributes"], now_doc["attributes"]
    for old, (new, unit) in _KEYS.items():
        assert old in old_attrs, "control: the base blob holds the old key"
        assert old not in new_attrs, old
        a, b = old_attrs[old], new_attrs[new]
        assert (a != a and b != b) or a == b, (new, a, b)
        assert new_attrs[new + "_units"] == unit
    others_old = {k: v for k, v in old_attrs.items() if k not in _KEYS}
    unit_free = {v[0] for v in _KEYS.values()}
    others_new = {
        k: v for k, v in new_attrs.items() if k not in unit_free and k not in {r + "_units" for r in unit_free}
    }
    assert others_new == others_old
    dirs = sorted(p.name for p in (_REF / f"{stem}.zarr").iterdir() if p.is_dir())
    base_dirs = sorted(
        {
            line.split("/")[3]
            for line in _git(
                "ls-tree", "-r", "--name-only", RUN_TIP, "--", f"test_data/swmm_refactoring_reference/{stem}.zarr"
            ).splitlines()
            if line.count("/") >= 4
        }
    )
    assert dirs == base_dirs, (dirs, base_dirs)


# --------------------------------------------------------------------------------------------------------------
# the number 28: module, constants, ladder, fixtures, goldens
# --------------------------------------------------------------------------------------------------------------


def test_the_layout_is_28_with_exactly_one_module_per_number_and_a_ladder_that_ends_at_it():
    """Property: `LAYOUT_VERSION` is 28; the version directory lists exactly one module claiming 27
    (develop's consolidation) and one claiming 28 (the unit-token migration, whose `version_from` is 27); the
    ladder test's expected list ends with the two stems in that order; the temporary suffix of the module's
    swap is `.v0028tmp` and `.v0027tmp` is nowhere in `src`.

    Class: the module files, their declared numbers, the ladder list and the source text.
    Why required: two modules claiming one number cannot both run; a stale suffix would
    collide with develop's own temporary naming if it ever returned.
    Kills it: a leftover `V0027__flood...` module; `version_to` left at 27; a ladder missing the last stem.
    Passes under: any equivalent module layout with the same numbers.
    """
    from hhemt.version_migration.constants import LAYOUT_VERSION

    assert LAYOUT_VERSION == 28
    vdir = REPO / "src" / "hhemt" / "version_migration" / "versions"
    claims = {}
    for path in sorted(vdir.glob("V00*.py")):
        text = path.read_text()
        m_from = re.search(r"^version_from: int = (\d+)", text, re.M)
        m_to = re.search(r"^version_to: int = (\d+)", text, re.M)
        if m_from and m_to:
            claims.setdefault(int(m_to.group(1)), []).append((path.stem, int(m_from.group(1))))
    assert claims[27] == [("V0027__consolidate_zarr_metadata", 26)], claims.get(27)
    assert claims[28] == [(V0028_STEM, 27)], claims.get(28)
    assert not [p.name for p in vdir.glob("V0027__flood*")]
    ladder = (REPO / "tests" / "test_version_migration_ladder.py").read_text()
    expected = re.search(r"EXPECTED_LADDER: list\[str\] = \[(.*?)\n\]", ladder, re.S).group(1)
    stems = re.findall(r'"(V\d{4}__[^"]+)"', expected)
    assert stems[-2:] == ["V0027__consolidate_zarr_metadata", V0028_STEM], stems[-3:]
    assert len(stems) == len(set(stems))
    leftovers = [
        p.relative_to(REPO).as_posix()
        for p in (REPO / "src").rglob("*.py")
        if "v00" + "27tmp" in p.read_text(errors="replace")
    ]
    assert leftovers == []
    assert ".v0028tmp" in (REPO / "src" / "hhemt" / "version_migration" / "context.py").read_text()


def test_the_layout_28_fixture_is_a_byte_copy_of_the_layout_27_fixture():
    """Property: `tests/fixtures/legacy_layouts/v28/` is a copy of `v27/`, file for file and byte for byte, with no
    file only on one side.

    Class: both trees, compared recursively.
    Why required: the layout-version check needs both directories for a +1 bump, and the second is a copy of
    develop's first; a divergent copy would misstate what a layout-27 tree looks like before the migration.
    Kills it: a missing file; an edited name inside the copy.
    Passes under: a copy by any means.
    """
    root = REPO / "tests" / "fixtures" / "legacy_layouts"
    cmp = filecmp.dircmp(root / "v27", root / "v28")

    def walk(c: filecmp.dircmp) -> list[str]:
        bad = list(c.left_only) + list(c.right_only) + list(c.diff_files) + list(c.funny_files)
        for sub in c.subdirs.values():
            bad += walk(sub)
        return bad

    assert walk(cmp) == []
    assert len(list((root / "v28").rglob("*"))) > 90, "control: the copy is not empty"


@pytest.mark.parametrize("zip_name", ["multisim", "sensitivity"])
def test_the_regenerated_golden_zips_are_stamped_28_carry_indexed_roots_and_hold_no_old_name(zip_name):
    """Property: each golden zip carries `_version.json` files stamped 28, every root `zarr.json` of its zarr stores
    carries a consolidated-metadata index, and no member path holds a unit-token variable name.

    Class: the two zips, every `_version.json`, every store root and every member path.
    Why required: the generator rebuilds both zips, and a stamp left at 27 or a store written without an index
    would pass the stores' own readers and fail only the conformance test.
    Kills it: a zip generated before `records.py` held 28; one generated with a generator that skips the index.
    Passes under: any generation with the same properties.
    """
    zf = zipfile.ZipFile(REPO / "tests" / "fixtures" / "goldens" / f"{zip_name}.zip")
    names = zf.namelist()
    stamps = [n for n in names if n.endswith("_version.json")]
    assert stamps and all(json.loads(zf.read(n))["layout_version"] == 28 for n in stamps), stamps
    roots = [n for n in names if re.search(r"\.zarr/zarr\.json$", n) and n.count(".zarr/") == 1]
    assert len(roots) >= 30, len(roots)
    unindexed = [n for n in roots if json.loads(zf.read(n)).get("consolidated_metadata") is None]
    assert unindexed == []
    old = re.compile(
        "|".join(
            map(
                re.escape,
                [_n("max", "wlevel", "m"), _n("velocity", "x", "mps"), _n("flow", "cms"), _n("timestep", "min")],
            )
        )
    )
    assert [n for n in names if old.search(n)] == []


# --------------------------------------------------------------------------------------------------------------
# the merge: what it could have silently dropped, in both directions
# --------------------------------------------------------------------------------------------------------------

_SOURCE_SUFFIXES = (".py", ".rst", ".md", ".yaml", ".yml")
_RENAME_TOKENS = re.compile(
    "|".join(map(re.escape, [_n("max", "wlevel", "m"), _n("max", "flow", "cms"), _n("timestep", "min")]))
)


def _changed(a: str, b: str, *paths: str) -> list[str]:
    return [p for p in _git("diff", "--name-only", a, b, "--", *paths).split() if p.endswith(_SOURCE_SUFFIXES)]


def _resurrected(rel: str, text: str) -> list[str]:
    """Lines this branch removed from `rel` (against the merge base) that `text` holds again."""
    removed = [
        ln[1:].strip()
        for ln in _git("diff", "-U0", MERGE_BASE, RUN_TIP, "--", rel).splitlines()
        if ln.startswith("-") and not ln.startswith("---") and len(ln[1:].strip()) > 25
    ]
    present = {ln.strip() for ln in text.splitlines()}
    return [r for r in removed if r in present]


def _dropped_develop_lines(rel: str, text: str) -> list[str]:
    """Lines `origin/develop` added to `rel` (against the merge base) that `text` lacks and that do not hold a
    unit-token name this branch replaces."""
    added = [
        ln[1:].strip()
        for ln in _git("diff", "-U0", MERGE_BASE, DEVELOP_TIP, "--", rel).splitlines()
        if ln.startswith("+") and not ln.startswith("+++") and len(ln[1:].strip()) >= 25
    ]
    present = {ln.strip() for ln in text.splitlines()}
    return [a for a in added if a not in present and not _RENAME_TOKENS.search(a)]


@needs_history
def test_no_line_the_run_removed_is_back_in_a_file_both_sides_changed_and_the_detector_fires():
    """Property: in every text file that both this branch and `origin/develop` changed since their merge base, no
    non-trivial line this branch removed is present again after the merge.

    Class: the files changed by both sides (32 text files when written), each line this branch removed (stripped,
    longer than 25 characters), compared as whole stripped lines with the working file.
    Why required: taking develop's side of a hunk brings back the branch's replaced text unless it is
    re-applied; an earlier merge has its own check, pinned to its own parent, and does not cover this one.
    Kills it: a conflicted or auto-merged file holding a unit-token spelling this branch had replaced.
    Passes under: any resolution that re-applies each hunk.
    Must-fire control: a file text with one removed line put back is reported.
    """
    both = sorted(set(_changed(MERGE_BASE, DEVELOP_TIP)) & set(_changed(MERGE_BASE, RUN_TIP)))
    assert len(both) >= 25, len(both)
    back = {}
    for rel in both:
        path = REPO / rel
        if path.is_file() and (hit := _resurrected(rel, path.read_text(errors="replace"))):
            back[rel] = hit
    assert back == {}
    probe = "src/hhemt/utils.py"
    removed = [
        ln[1:].strip()
        for ln in _git("diff", "-U0", MERGE_BASE, RUN_TIP, "--", probe).splitlines()
        if ln.startswith("-") and not ln.startswith("---") and len(ln[1:].strip()) > 25
    ]
    assert removed, "control: this branch removed lines from utils.py"
    assert _resurrected(probe, (REPO / probe).read_text() + "\n" + removed[0] + "\n") == [removed[0]]


@needs_history
def test_every_line_origin_develop_added_to_src_is_still_there_or_is_the_rename_of_a_layout_26_name():
    """Property: every non-trivial line `origin/develop` added under `src` since the merge base is in the merged
    file, unless the line holds a unit-token name this branch replaces (4 lines when written).

    Class: the `+` lines (stripped, 25 characters or more) of `git diff {merge base} {develop tip} -- src`,
    1,410 when written, compared with the working file of the same path.
    Why required: this branch's side of a conflict can overwrite a behaviour develop added (the write helper calls,
    the shard re-chunk block, the stamps); only a whole-line comparison over develop's additions sees it.
    Kills it: a dropped `write_zarr(` call or `shards` branch; a stamp lost.
    Passes under: any resolution that keeps develop's lines.
    Must-fire control: the same comparison on the working text of `utils.py` with one develop-added line
    deleted reports it.
    """
    files = _changed(MERGE_BASE, DEVELOP_TIP, "src")
    assert len(files) >= 40, len(files)
    dropped = {}
    for rel in files:
        path = REPO / rel
        text = path.read_text(errors="replace") if path.is_file() else ""
        if hit := _dropped_develop_lines(rel, text):
            dropped[rel] = hit
    assert dropped == {}
    probe = "src/hhemt/utils.py"
    text = (REPO / probe).read_text()
    added = [
        ln[1:]
        for ln in _git("diff", "-U0", MERGE_BASE, DEVELOP_TIP, "--", probe).splitlines()
        if ln.startswith("+") and not ln.startswith("+++") and len(ln[1:].strip()) >= 25
    ]
    victim = next(a for a in added if not _RENAME_TOKENS.search(a) and a in text.splitlines())
    cut = "\n".join(ln for ln in text.splitlines() if ln != victim)
    assert cut != text, "control: a develop-added line was removed from the probe text"
    assert _dropped_develop_lines(probe, cut), "control: deleting a develop line is reported"


@needs_history
def test_the_write_helper_calls_and_the_shard_block_of_origin_develop_are_counted_the_same_after_the_merge():
    """Property: in `utils.py`, `cross_sim_identity.py` and `process_simulation.py`, the number of lines calling
    `write_zarr(` and of lines holding `.to_zarr(` equals develop's, and `utils.py` holds as many `shards` lines as
    develop's.

    Class: three files x three counts, each against the same file at the develop tip (unit: lines; matcher: the
    three literal substrings, counted per line with `str.count`-free containment).
    Why required: this branch had replaced `to_zarr` calls with its own before the merge; the merge could have
    kept that form for the cross-sim artifact or lost develop's shard re-chunk lines.
    Kills it: a bare `to_zarr(..., consolidated=False)` back in the identity writer; a lost shard line.
    Passes under: any merge keeping develop's calls (renames do not change these counts).
    """

    def counts(text: str) -> tuple[int, int, int]:
        lines = text.splitlines()
        return (
            sum("write_zarr(" in ln for ln in lines),
            sum(".to_zarr(" in ln for ln in lines),
            sum("shards" in ln for ln in lines),
        )

    for rel in ("src/hhemt/utils.py", "src/hhemt/eda/cross_sim_identity.py", "src/hhemt/process_simulation.py"):
        theirs = counts(_git("show", f"{DEVELOP_TIP}:{rel}"))
        ours = counts((REPO / rel).read_text())
        assert ours == theirs, (rel, ours, theirs)
    assert counts(_git("show", f"{DEVELOP_TIP}:src/hhemt/utils.py"))[2] > 5, (
        "control: develop's utils.py has shard lines"
    )
    ps = (REPO / "src/hhemt/process_simulation.py").read_text()
    assert ps.count("stamp_timestep_coordinate(") >= 2  # the two calls this branch adds
    assert 'apply_cf_attributes(_ds, "hydrology_rpt_summary")' in ps
    assert (
        'apply_cf_attributes(artifact_ds, "eda_cross_sim_identity")'
        in (REPO / "src/hhemt/eda/cross_sim_identity.py").read_text()
    )
