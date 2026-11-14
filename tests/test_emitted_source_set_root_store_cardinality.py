"""The emission boundary: one emit must declare at most ONE root consolidated store.

WHY A CARDINALITY ASSERTION AND NOT AN ORDERING. The bundle file set is EXACTLY the union
of manifest-declared `source_paths`, and two live declarers resolve the root store by two
DIFFERENT resolvers -- `eda/_config_diff.config_diff_source_paths` through
`utils.resolve_experiment_tree`, and `report_renderers/metadata` through its own
declared-attributes-first resolver. On a two-store root those two return different stores,
so the union contains two and the emitter copies two. The defect is therefore visible in
the emitted set WITHOUT resolving anything: the union's intersection with
`ROOT_TREE_NAMES` has size two. Asserting the cardinality needs no discriminator to
survive packaging and no choice of ordering.

WHY THE FIXTURE IS A HAND-WRITTEN MANIFEST. The two committed bundle fixtures cannot
testify here: both declare ZERO renderers and ZERO source paths at a layout stamp several
versions behind the live one, so they are hand-built minimal artifacts rather than emitted
bundles, and a payload that is empty by construction cannot exhibit the property. This
module therefore asserts over the MANIFEST SHAPE the emitter writes, which is the same
surface the stipulation defines the file set on, and it does so without emitting anything
-- emission is downstream of a render which is downstream of a run.

THE BOUND THIS CARRIES. The declarer set is not exhaustively enumerated; a third declarer
using a third resolution path would change the cardinality without changing the verdict.
The assertion is over the SET, so a third declarer is admitted automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.utils import ROOT_TREE_NAMES

EXPERIMENT, SENSITIVITY, REGULAR = ROOT_TREE_NAMES


def _manifest(base: Path, name: str, declarations: dict[str, list[str]]) -> Path:
    """Write a bundle manifest carrying only the key this assertion reads."""
    root = base / name
    root.mkdir(parents=True)
    path = root / "bundle_manifest.json"
    path.write_text(json.dumps({"source_paths_by_renderer": declarations}), encoding="utf-8")
    return path


def declared_root_stores(manifest_path: Path) -> list[str]:
    """Which members of `ROOT_TREE_NAMES` the emit's declared source union names.

    Matched on BASENAME rather than on the full path, because the declarers relativize
    differently and a prefix comparison would miss one of them.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    declared = manifest.get("source_paths_by_renderer") or {}
    paths = [p for value in declared.values() for p in (value if isinstance(value, list) else [value])]
    basenames = {Path(p).name for p in paths}
    return sorted(name for name in ROOT_TREE_NAMES if name in basenames)


def test_two_declarers_naming_two_root_stores_is_a_finding(tmp_path):
    """THE VIOLATING ARM. Red on the state the two live declarers produce."""
    path = _manifest(
        tmp_path,
        "two_declarers_disagree",
        {
            "config_diff_maps": [f"/a/{REGULAR}", "/a/scenario_status.csv"],
            "metadata": [f"/a/{EXPERIMENT}"],
        },
    )
    assert len(declared_root_stores(path)) == 2


@pytest.mark.parametrize(
    "case_id,declarations",
    [
        (
            "both-declarers-agree-on-the-unified-name",
            {"config_diff_maps": [f"/a/{EXPERIMENT}"], "metadata": [f"/a/{EXPERIMENT}"]},
        ),
        (
            "both-declarers-agree-on-a-RETIRED-name",
            {"config_diff_maps": [f"/a/{SENSITIVITY}"], "metadata": [f"/a/{SENSITIVITY}", "/a/eda/x.zarr"]},
        ),
        ("no-declarer-names-a-root-store", {}),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_conforming_emit_names_at_most_one_root_store(tmp_path, case_id, declarations):
    """THE SATISFYING ARMS, at three DIFFERENT correct positions.

    The second is the one that matters: a correct emit may agree on a RETIRED name rather
    than on the unified one, and an assertion written against the unified position alone
    would redden it. The third is the shape both committed fixtures actually have.
    """
    path = _manifest(tmp_path, case_id, declarations)
    assert len(declared_root_stores(path)) <= 1


def test_the_predicate_reads_basenames_not_prefixes(tmp_path):
    """THE INSTRUMENT GUARD. Declarers relativize differently; a prefix test misses one."""
    path = _manifest(
        tmp_path,
        "mixed_relativization",
        {"a": [f"/abs/path/{EXPERIMENT}"], "b": [f"members/member_0/{REGULAR}"]},
    )
    assert declared_root_stores(path) == sorted([EXPERIMENT, REGULAR])
