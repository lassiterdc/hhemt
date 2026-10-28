"""Phase-3 entry-point tests — guards + orchestration + CombinedBundle round-trip.

The Phase-4 render + merge seams are monkeypatched so these tests isolate the
orchestration guards and the deterministic emit (crate + manifest + read-model)
from the not-yet-landed ``combined`` ReportingSet render wiring.
"""

import json
from pathlib import Path

import pytest

from hhemt.bundle import _combine as CB
from hhemt.bundle._emit import _emit_bundle_zip
from hhemt.exceptions import ConfigurationError, ProcessingError
from hhemt.version_migration.constants import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
)


def test_requires_two_bundles(tmp_path):
    with pytest.raises(ConfigurationError):
        CB.combine_bundle([tmp_path / "only_one"])


def test_blocking_aborts(tmp_path, monkeypatch):
    from hhemt.bundle._compatibility import (
        CompatibilityDivergence,
        CompatibilityReport,
    )
    from hhemt.bundle._compatibility import (
        CompatibilitySeverity as S,
    )

    rep = CompatibilityReport(
        divergences=[
            CompatibilityDivergence("case_name", "experiment", S.BLOCKING, "a", "b", "norfolk", "houston"),
        ]
    )
    # **kwargs so the double tracks check_bundle_compatibility's keyword-only options
    # (declare_solver_split) without this test asserting anything about them.
    monkeypatch.setattr(CB, "check_bundle_compatibility", lambda roots, **_kw: rep)
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    with pytest.raises(ConfigurationError, match="not combine-compatible"):
        CB.combine_bundle([a, b])


def test_combinedbundle_from_directory(tmp_path):
    root = tmp_path / "combined"
    root.mkdir()
    # NOTE: the manifest MUST carry the local bundle_schema_version — CombinedBundle
    # .from_directory validates it (CR5), so an empty {} manifest raises BundleSchemaError.
    (root / BUNDLE_MANIFEST_FILENAME).write_text(f'{{"bundle_schema_version": {BUNDLE_SCHEMA_VERSION}}}')
    cb = CB.CombinedBundle.from_directory(root)
    assert cb.root == root.resolve()


def test_from_directory_rejects_wrong_schema_version(tmp_path):
    from hhemt.bundle import BundleSchemaError

    root = tmp_path / "combined_wrongver"
    root.mkdir()
    (root / BUNDLE_MANIFEST_FILENAME).write_text(f'{{"bundle_schema_version": {BUNDLE_SCHEMA_VERSION + 1}}}')
    with pytest.raises(BundleSchemaError):
        CB.CombinedBundle.from_directory(root)


def _make_child_bundle(parent: Path, name: str, payload: str) -> Path:
    """A minimal on-disk child bundle: just enough for the emit's copytree."""
    root = parent / name
    root.mkdir()
    (root / "data.txt").write_text(payload)
    (root / BUNDLE_MANIFEST_FILENAME).write_text(f'{{"bundle_schema_version": {BUNDLE_SCHEMA_VERSION}}}')
    return root


def _archive_bytes(staging: Path, zip_path: Path) -> bytes:
    # Reuses the single-analysis deterministic archive contract (_emit_bundle_zip:
    # sorted entries + fixed date_time + ZIP_STORED) to prove combined-emit determinism.
    _emit_bundle_zip(staging, zip_path)
    return zip_path.read_bytes()


def test_combined_emit_deterministic(tmp_path, monkeypatch):
    """CR4: two combines of the same N bundles are byte-identical.

    Both volatile surfaces are neutralized in the emit — the manifest OMITS
    ``created_at_utc`` and the combined crate strips ro-crate-py's per-run
    ``datePublished`` — so the archives are directly byte-identical with no
    compare-time field stripping needed.
    """
    from hhemt.bundle._compatibility import CompatibilityReport

    # Isolate the emit: real experiment ids need a full Bundle load, and the render
    # is the Phase-4 seam — monkeypatch both to focus on emit determinism.
    monkeypatch.setattr(CB, "_experiment_id", lambda r: r.name)
    monkeypatch.setattr(CB, "_render_combined_report", lambda *a, **k: None)

    a = _make_child_bundle(tmp_path, "exp_a", "A")
    b = _make_child_bundle(tmp_path, "exp_b", "B")
    roots = sorted([a, b])
    report = CompatibilityReport()

    out1 = tmp_path / "c1"
    out2 = tmp_path / "c2"
    CB._emit_combined_bundle(roots, None, report, out1)
    CB._emit_combined_bundle(roots, None, report, out2)

    # The combined crate carries no wall-clock (datePublished stripped — CR4).
    rocrate_text = (out1 / "ro-crate-metadata.json").read_text()
    assert "datePublished" not in rocrate_text
    # The combined manifest omits the volatile created_at_utc (CR4).
    manifest_text = (out1 / BUNDLE_MANIFEST_FILENAME).read_text()
    assert "created_at_utc" not in manifest_text

    # Full-archive byte identity across the two independent emits.
    z1 = _archive_bytes(out1, tmp_path / "c1.zip")
    z2 = _archive_bytes(out2, tmp_path / "c2.zip")
    assert z1 == z2


def _child_with_root_tree(parent: Path, *, shallow: list[str] | None) -> Path:
    """A child bundle root carrying a real (tiny) root tree plus a manifest.

    Every arm below builds the SAME on-disk tree and varies only the manifest's
    ``metadata_only_stores`` value -- whether the key is present, and WHICH store it names.
    Varying WHICH store is named is the load-bearing half: an arm set that only ever pairs a
    naming list against an empty one cannot tell the correct predicate apart from a bare
    ``if shallow_stores:``, because across that space the two are extensionally identical.
    """
    import xarray as xr

    from hhemt.utils import resolve_experiment_tree

    root = parent / "child"
    root.mkdir(parents=True)
    store = resolve_experiment_tree(root)
    xr.Dataset({"max_wlevel_m": ("a", [1.0, 2.0])}).to_zarr(store)
    manifest: dict = {"bundle_schema_version": BUNDLE_SCHEMA_VERSION}
    if shallow is not None:
        manifest["metadata_only_stores"] = shallow
    (root / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return root


def test_intercomparison_refuses_a_metadata_only_child_store(tmp_path):
    """A shallow-carried child store must ABORT the combine, not read as fill values.

    The bundle emit carries a store SHALLOWLY (zarr metadata documents, no chunk files) when
    every figure that declared it read only its metadata. Such a store re-opens cleanly and
    returns FILL VALUES for every data read, so an unguarded combine writes
    ``identical: True, max_abs_diff: 0.0`` for every pair with no error and no warning.
    Refusal is the only honest outcome: SKIPPING yields an empty pair list, which the roll-up
    renders as "no pairs" -- the same false-clean shape the guard exists to prevent.
    """
    root = _child_with_root_tree(tmp_path / "a", shallow=["experiment_datatree.zarr"])
    with pytest.raises(ProcessingError) as exc:
        CB._load_intercomparison_subs(root)
    assert "metadata-only" in str(exc.value)


def test_intercomparison_loads_a_fully_carried_child_store(tmp_path):
    """Satisfying inputs: the same tree loads unless the manifest names the ROOT TREE itself.

    The last two arms are what keep the predicate NARROW, and they are not interchangeable.
    ``eda/raw_byte_identity.zarr`` is the real-world case the narrow placement exists for --
    a figure qualified an EDA artifact while the root tree was carried in full -- and it is
    what a later reader will look for. ``members/member_0/experiment_datatree.zarr`` shares the
    root tree's BASENAME while differing in path, and it is the only arm that separates a
    full-relative-path match from a basename or substring match.
    """
    assert CB._load_intercomparison_subs(_child_with_root_tree(tmp_path / "b", shallow=[])) == ({}, {})
    assert CB._load_intercomparison_subs(_child_with_root_tree(tmp_path / "c", shallow=None)) == ({}, {})
    assert CB._load_intercomparison_subs(
        _child_with_root_tree(tmp_path / "d", shallow=["eda/raw_byte_identity.zarr"])
    ) == ({}, {})
    assert CB._load_intercomparison_subs(
        _child_with_root_tree(tmp_path / "e", shallow=["members/member_0/experiment_datatree.zarr"])
    ) == ({}, {})


def _walk_order(resolve, root: Path, names: tuple[str, ...]) -> list[str]:
    """The priority order a resolver exhibits on a root carrying every name in ``names``.

    Creates one empty directory per name (both resolvers decide by ``.exists()`` alone),
    then repeatedly asks the resolver which store it picks and removes that store, so the
    sequence of picks is the resolver's full ordering rather than only its first choice.
    """
    for name in names:
        (root / name).mkdir()
    order: list[str] = []
    for _ in names:
        picked = resolve(root)
        order.append(picked.name)
        picked.rmdir()
    return order


def test_combine_and_metadata_walkers_share_the_armless_root_tree_order(tmp_path):
    """Property: with no ``cfg_analysis.yaml`` and no analysis object, the combine resolver and
    the metadata resolver rank the three root-store names identically, and that ranking is
    ``hhemt.utils.ROOT_TREE_NAMES``.

    Class: REGRESSION -- red on the tree before ``_combine_merge`` imported the canonical
    tuple, where its private copy ranked ``analysis_datatree.zarr`` ahead of
    ``sensitivity_datatree.zarr`` (measured: combine picked experiment, analysis, sensitivity
    while metadata picked experiment, sensitivity, analysis).

    Why it is required: a root carrying both retired names resolved to a different store
    depending on which module asked, and ``_resolve_consolidated_tree``'s docstring claimed the
    two agreed. Asserting on the orderings' values is what keeps that claim from being text.

    What kills it: re-introducing a second priority tuple in either module, or reordering
    ``ROOT_TREE_NAMES`` in only one resolver's walk.

    A second correct implementation that still passes: either resolver reaching the canonical
    order by any route (a different loop, a helper, a sorted view of the tuple).
    """
    from hhemt.bundle._combine_merge import _resolve_root_tree
    from hhemt.report_renderers.metadata import _resolve_consolidated_tree
    from hhemt.utils import ROOT_TREE_NAMES

    combine_root = tmp_path / "combine"
    metadata_root = tmp_path / "metadata"
    combine_root.mkdir()
    metadata_root.mkdir()
    combine_order = _walk_order(_resolve_root_tree, combine_root, ROOT_TREE_NAMES)
    metadata_order = _walk_order(lambda r: _resolve_consolidated_tree(r, None), metadata_root, ROOT_TREE_NAMES)

    assert combine_order == metadata_order == list(ROOT_TREE_NAMES)


def test_sensitivity_attribute_is_bound_to_the_experiment_filename(synth_sensitivity_analysis):
    """Property: on a constructed sensitivity analysis, ``analysis_paths.sensitivity_datatree_zarr``
    is the path ``experiment_datatree.zarr``, which is the binding ``analysis.py`` makes in its
    constructor when ``toggle_sensitivity_analysis`` is true.

    I70 (named and pinned, NOT repaired). The attribute's name says "sensitivity" and the file it
    is bound to says "experiment"; the two disagree. This test records the disagreement as it
    stands, deliberately pinned and not repaired, so that nobody changes one side without
    noticing the other. IT GOES RED WHEN I70 IS REPAIRED -- by renaming the attribute or by rebinding the
    filename -- and it is then REWRITTEN to the repaired pair, never loosened (no ``in`` check, no
    suffix match, no deletion of the assertion).

    Class: NEW CAPABILITY. The plausible wrong implementation is a constructor that binds the
    attribute to ``sensitivity_datatree.zarr`` (the retired name); the assertion is red under it.

    Why the subject is a constructed analysis and not an ``AnalysisPaths`` built here: an
    ``AnalysisPaths`` the test builds would assert the value the test supplied. The subject is
    the binding the constructor makes. It also does not read source text.

    A second correct implementation that still passes: the constructor computing the same path
    through a helper, a different join, or a differently named local variable.
    """
    analysis = synth_sensitivity_analysis

    assert analysis.cfg_analysis.toggle_sensitivity_analysis is True
    assert analysis.analysis_paths.sensitivity_datatree_zarr.name == "experiment_datatree.zarr"
