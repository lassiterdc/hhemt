"""Report labels describe what the code actually did, not what it did by default.

Two labels are checked. The peak-flood-depth figure records, in its manifest, how
each map panel's data was transformed: the depth threshold below which cells were
masked as dry, and the canvas a large raster was pre-aggregated onto. Both are
configurable, so a label written as a literal goes false the moment a user changes
either one. The combined report stacks each experiment's figure under a heading
naming the model that produced it, and indexes it by a `models` facet; an
experiment whose only child ran TRITON alone must be labelled TRITON, not
TRITON-SWMM.

These tests touch only label construction and the combined-report page harvest.
They compile no solver and run no simulation.
"""

import json

from hhemt.bundle.combined_snakefile_generator import _harvest_per_experiment_rule_specs


def test_peak_depth_transform_labels_name_the_configured_threshold_and_canvas():
    """Property: the depth and WSE panels' provenance `transform` text names the dry
    threshold the mask used and the datashader canvas the raster was aggregated onto.

    Class: any threshold and any canvas size, tested at non-default values for both,
    with and without datashader pre-aggregation. This is the smallest unit that builds
    the label: the figure builder that consumes it needs a consolidated analysis tree
    and a DEM, so it is not driven here.

    Why required: the manifest is the record a reader consults to learn how a panel's
    numbers were produced, and a label naming `depth>0` and `512x512` while the code
    masked at 0.05 m and aggregated onto 1024x768 misreports both.

    What kills it: a hardcoded threshold or canvas in either label, or a datashader
    suffix that appears when pre-aggregation did not run.

    A second correct implementation that still passes: formatting the two labels inline
    at each ProvenanceRef from the same two values.
    """
    from hhemt.report_renderers.per_sim_peak_flood_depth import _plotly_map_transforms

    depth, wse = _plotly_map_transforms(0.05, (1024, 768), use_datashader=True)
    assert depth == "masked to watershed and depth>0.05 m; datashader pre-rasterized (1024x768, max)"
    assert wse == "depth, summed with DEM elevation; datashader pre-rasterized (1024x768, max)"

    depth, wse = _plotly_map_transforms(0.05, (1024, 768), use_datashader=False)
    assert depth == "masked to watershed and depth>0.05 m"
    assert wse == "depth, summed with DEM elevation"


_FIGURE = "<html><head></head><body><div>one model's figure</div></body></html>"


def _write_child(bundle_root, eid, plot_id):
    """A minimal child bundle: an empty analysis config, one figure and its manifest."""
    child = bundle_root / "child_crates" / eid
    (child / "plots" / "maps").mkdir(parents=True, exist_ok=True)
    (child / "cfg_analysis.yaml").write_text("{}\n", encoding="utf-8")
    (child / "plots" / "maps" / f"{plot_id}.html").write_text(_FIGURE, encoding="utf-8")
    (child / "plots" / "maps" / f"{plot_id}.manifest.json").write_text(
        json.dumps({"plot_id": plot_id, "source_paths_relative": [f"data/{eid}.zarr"]}), encoding="utf-8"
    )


def _single_spec(bundle_root):
    specs = [s for s in _harvest_per_experiment_rule_specs(bundle_root) if "config_diff_maps" in s.rule_name]
    assert len(specs) == 1, [s.rule_name for s in specs]
    spec = specs[0]
    page = (bundle_root / spec.output_path_template).read_text(encoding="utf-8")
    return page, json.loads(spec.report_kwargs["labels"])


def test_single_section_is_tagged_with_the_model_its_figure_came_from(tmp_path):
    """Property: a composed page with one section tags that section with the model of
    the child the figure came from, and its `models` facet says the same.

    Class: the three bases that compose one section — a TRITON-only base, a coupled-only
    base, and a base whose child carries no model token. Each is a differently-positioned
    input: the first two record a model and the third records none.

    Why required: the section heading is drawn from the section's model tag, so a
    TRITON figure tagged as TRITON-SWMM is presented to the reader as the coupled model's
    result.

    What kills it: tagging the single section TRITON-SWMM whatever its source, which reds
    the TRITON-only case; or reading "TRITON only" for a base that records no model.

    A second correct implementation that still passes: omitting the model tag for a
    model-less base and labelling it from the facet alone.
    """
    tri_root = tmp_path / "tri"
    _write_child(tri_root, "exp_triton", "config_diff_maps")
    page, labels = _single_spec(tri_root)
    assert '<div class="model-section" data-model="triton"' in page
    assert '<div class="model-section" data-model="tritonswmm"' not in page
    assert labels["models"] == "TRITON only"

    ts_root = tmp_path / "ts"
    _write_child(ts_root, "exp_tritonswmm", "config_diff_maps")
    page, labels = _single_spec(ts_root)
    assert '<div class="model-section" data-model="tritonswmm"' in page
    assert labels["models"] == "TRITON-SWMM only"

    none_root = tmp_path / "none"
    _write_child(none_root, "exp", "config_diff_maps")
    page, labels = _single_spec(none_root)
    assert '<div class="model-section" data-model="tritonswmm"' not in page
    assert '<div class="model-section" data-model="triton"' not in page
    assert labels["models"] == "model not recorded"
