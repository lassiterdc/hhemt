"""The combine-time direct render writes the cross-solver figure only for a multi-solver combine.

`_combine._render_combined_report` direct-renders every figure of the `combined` reporting set
before the report is built. The cross-solver figure is conditional on `has_multiple_solvers`,
so on a single-solver combine neither the figure nor its manifest sidecar may reach disk. This
module drives that function down both paths that call it: the emit path
(`_emit_combined_bundle`, which writes the read-model and then renders) and
`CombinedBundle.regenerate_report` (which renders against an existing bundle root).

Nothing here compiles a solver or runs a simulation. The two steps after the direct render, the
child-figure refresh and the Snakemake report, are replaced by recorders; the direct render
itself runs for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import hhemt.bundle.combined_snakefile_generator as generator
from hhemt.bundle import _combine as C

_FIGURE = "plots/cross_experiment/cross_solver_comparison.html"
_MANIFEST = "plots/cross_experiment/cross_solver_comparison.manifest.json"
#: Written by an unconditional combined renderer on every run; the positive control that the
#: direct render ran at all, so "cross-solver absent" is never "nothing rendered".
_CONTROL = "plots/cross_experiment/errors_and_warnings.html"
_PIN_A = "a38338b09e62e57c936f51516bbdbe495d89a546"
_PIN_B = "b1ff2689" + "0" * 32


def _bundle(root: Path, pin: str, arm_group: str) -> Path:
    """One single-arm bundle: a pinned `cfg_system.yaml` and a consolidated tree."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg_system.yaml").write_text(f"TRITONSWMM_branch_key: {pin}\n")
    groups = {
        "/": xr.Dataset(),
        "/member_0": xr.Dataset(attrs={"run_mode": "clean", "n_mpi_procs": 1, "n_omp_threads": 1, "n_gpus": 0}),
        f"/member_0/{arm_group}": xr.Dataset(),
        f"/member_0/{arm_group}/performance": xr.Dataset({"Total": ("Rank", np.asarray([10.0]))}),
    }
    xr.DataTree.from_dict(groups).to_zarr(root / "sensitivity_datatree.zarr", mode="w", consolidated=False)
    return root


@pytest.fixture
def quiet_render(monkeypatch):
    """Replace the two steps AFTER the direct render with recorders; return the call log."""
    calls: list[str] = []
    monkeypatch.setattr(C, "_rerender_child_report_figures", lambda root: calls.append("children"))
    monkeypatch.setattr(generator, "render_combined_report_via_snakemake", lambda root, formats: calls.append("report"))
    return calls


def _emit(tmp_path: Path, monkeypatch, roots: list[Path], *, write_read_model: bool) -> Path:
    """Drive the emit path; the unrelated writers are stubbed, the read-model writer is real unless disabled."""
    monkeypatch.setattr(C, "_combined_experiment_ids", lambda rs: [r.name for r in rs])
    for name in ("_write_combined_rocrate", "_write_combined_compatibility", "_write_combined_intercomparison"):
        monkeypatch.setattr(C, name, lambda *a, **k: None)
    monkeypatch.setattr(C, "_write_combined_bundle_manifest", lambda *a, **k: None)
    if not write_read_model:
        monkeypatch.setattr(C, "_write_cross_solver_comparison", lambda *a, **k: None)
    out = tmp_path / "out"
    C._emit_combined_bundle(roots, None, None, out)
    return out


def _regenerate(tmp_path: Path, monkeypatch, read_model: str | None) -> Path:
    """Drive `regenerate_report` against a bundle root holding only the given read-model text."""
    root = tmp_path / "bundle"
    root.mkdir()
    if read_model is not None:
        (root / "cross_solver_comparison.json").write_text(read_model, encoding="utf-8")
    monkeypatch.setattr("hhemt.provenance.assert_plots_match_running_build", lambda *a, **k: None)
    (root / "analysis_report.zip").write_text("x")  # regenerate_report requires the report to exist afterwards
    C.CombinedBundle(root).regenerate_report(format="zip")
    return root


def _two_pin_roots(tmp_path: Path) -> list[Path]:
    return [
        _bundle(tmp_path / "a_coupled", _PIN_A, "tritonswmm"),
        _bundle(tmp_path / "b_coupled", _PIN_B, "tritonswmm"),
    ]


def test_emit_path_single_solver_writes_neither_figure_nor_manifest(tmp_path, monkeypatch, quiet_render):
    """Property: a combine over one solver leaves the cross-solver figure and manifest off disk.

    Class: every emit-path combine whose bundles share one `TRITONSWMM_branch_key`. The two
    bundles here are the two model arms of one pin, the clean-vs-resume shape.

    Why required: DoD 1b says that when the predicate is false neither consumer puts the figure
    or its manifest on disk; a manifest alone would still declare a figure that does not exist.

    What kills it: iterating the raw `rset.renderer_selection` in `_render_combined_report`
    (the mutation arm below runs exactly that and must see the file). Second correct
    implementation that still passes: skipping the selection inside the renderer instead.
    """
    roots = [
        _bundle(tmp_path / "a_coupled", _PIN_A, "tritonswmm"),
        _bundle(tmp_path / "a_triton", _PIN_A, "triton_only"),
    ]
    out = _emit(tmp_path, monkeypatch, roots, write_read_model=True)

    assert json.loads((out / "cross_solver_comparison.json").read_text())["distinct_solver_count"] == 1
    assert (out / _CONTROL).exists(), "positive control: the unconditional combined figures were rendered"
    assert not (out / _FIGURE).exists()
    assert not (out / _MANIFEST).exists()
    assert quiet_render == ["children", "report"]


def test_emit_path_absent_read_model_writes_neither_figure_nor_manifest(tmp_path, monkeypatch, quiet_render):
    """Property: an emit with no cross-solver read-model leaves the figure and manifest off disk.

    Class: every emit whose read-model writer produced nothing (a damaged or older producer).
    Differently positioned from the single-solver case: no count exists to be read at all.

    Why required: the renderer would write a placeholder page for this input, which is the
    orphan file the gate exists to prevent.

    What kills it: the same ungated loop. Second correct implementation: the gate reading the
    bundle list instead of the read-model, provided an absent read-model stays inert.
    """
    roots = [_bundle(tmp_path / "a_coupled", _PIN_A, "tritonswmm")]
    out = _emit(tmp_path, monkeypatch, roots, write_read_model=False)

    assert not (out / "cross_solver_comparison.json").exists()
    assert (out / _CONTROL).exists()
    assert not (out / _FIGURE).exists()
    assert not (out / _MANIFEST).exists()


def test_emit_path_multi_solver_writes_the_figure_and_its_manifest(tmp_path, monkeypatch, quiet_render):
    """Property: a combine over two or more solvers renders the figure and its manifest.

    Class: every emit-path combine whose bundles carry two distinct pins.

    Why required: the two tests above would also pass a loop that never rendered the figure.

    What kills it: a gate that tests `> 2` or drops the selection always. Second correct
    implementation: gating per template rather than per selection (one template here).
    """
    out = _emit(tmp_path, monkeypatch, _two_pin_roots(tmp_path), write_read_model=True)

    assert json.loads((out / "cross_solver_comparison.json").read_text())["distinct_solver_count"] == 2
    assert (out / _CONTROL).exists()
    assert (out / _FIGURE).exists()
    assert (out / _MANIFEST).exists()


@pytest.mark.parametrize(
    "read_model",
    [json.dumps({"distinct_solver_count": 1}), None],
    ids=["single-solver", "absent"],
)
def test_regenerate_path_without_a_multi_solver_read_model_writes_neither_figure_nor_manifest(
    tmp_path, monkeypatch, quiet_render, read_model
):
    """Property: re-rendering a bundle whose read-model is single-solver or absent writes no figure.

    Class: `regenerate_report` against every bundle root without a multi-solver count; both
    inputs sit on the inert side but differ in whether a read-model exists.

    Why required: DoD 1b names this path beside the emit path, and a re-render over a bundle
    that earlier carried the figure must not revive it. Not established here: the bundle root
    starts empty, so removal of an already-present stale figure is not exercised.

    What kills it: the ungated loop. Second correct implementation: skipping at render time.
    """
    root = _regenerate(tmp_path, monkeypatch, read_model)

    assert (root / _CONTROL).exists()
    assert not (root / _FIGURE).exists()
    assert not (root / _MANIFEST).exists()


def test_regenerate_path_multi_solver_writes_the_figure_and_its_manifest(tmp_path, monkeypatch, quiet_render):
    """Property: re-rendering a bundle whose read-model counts two solvers writes the figure.

    Class: `regenerate_report` against a root holding a multi-solver count and no cells, which
    renders the placeholder page; the assertion is on presence, not content.

    Why required: positive arm for the regenerate path.

    What kills it: a gate that never admits the selection. Second correct implementation:
    evaluating the predicate in `regenerate_report` before calling the render.
    """
    root = _regenerate(tmp_path, monkeypatch, json.dumps({"distinct_solver_count": 2}))

    assert (root / _CONTROL).exists()
    assert (root / _FIGURE).exists()
    assert (root / _MANIFEST).exists()


def test_mutation_arm_the_ungated_loop_does_write_the_figure_on_a_single_solver_combine(
    tmp_path, monkeypatch, quiet_render
):
    """Mutation arm: with the helper replaced by the raw registry the single-solver files DO appear.

    This is the loop reverted to `rset.renderer_selection`, expressed by swapping what the loop
    iterates. It proves the absence assertions above can fail, so they are not vacuous: if this
    test stops seeing the figure, the instrument has lost its ability to detect the defect.
    """
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    monkeypatch.setattr(
        generator, "_active_combined_selections", lambda root: get_reporting_set("combined").renderer_selection
    )
    root = _regenerate(tmp_path, monkeypatch, json.dumps({"distinct_solver_count": 1}))

    assert (root / _FIGURE).exists()
    assert (root / _MANIFEST).exists()
