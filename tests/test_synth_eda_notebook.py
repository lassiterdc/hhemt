"""Phase-3 unit tests for hhemt.eda._notebook (ADR-13, D-NB1)."""

from __future__ import annotations

import nbformat
import pytest


class _FakeEdaCfg:
    enabled_plots = ["config_diff_maps"]
    plotly_js_mode = "inline"
    tabulator_js_mode = "cdn"
    eda = None  # the seed cell reads ctx.cfg_analysis.eda at EXECUTION, not at emit.


class _FakeAnalysisCfg:
    analysis_id = "synth_demo"
    eda = _FakeEdaCfg()


def test_resolve_notebook_path_normalizes_and_is_non_clobbering(tmp_path):
    from hhemt.eda._notebook import _resolve_notebook_path

    assert _resolve_notebook_path(tmp_path, None) == tmp_path / "eda.ipynb"
    assert _resolve_notebook_path(tmp_path, "myeda") == tmp_path / "myeda.ipynb"
    assert _resolve_notebook_path(tmp_path, "myeda.ipynb") == tmp_path / "myeda.ipynb"
    (tmp_path / "eda.ipynb").write_text("{}")
    assert _resolve_notebook_path(tmp_path, None) == tmp_path / "eda_1.ipynb"
    (tmp_path / "eda_1.ipynb").write_text("{}")
    assert _resolve_notebook_path(tmp_path, None) == tmp_path / "eda_2.ipynb"


def test_emit_eda_notebook_validates_and_does_not_clobber(tmp_path):
    from hhemt.eda._notebook import emit_eda_notebook

    cfg = _FakeAnalysisCfg()
    p1 = emit_eda_notebook(tmp_path, cfg_analysis=cfg, eda_cfg=cfg.eda, is_bundle=False)
    assert p1 == tmp_path / "eda.ipynb"
    nbformat.validate(nbformat.read(p1, as_version=4))  # structurally valid
    # second call MUST NOT overwrite -> numeric sibling
    p2 = emit_eda_notebook(tmp_path, cfg_analysis=cfg, eda_cfg=cfg.eda, is_bundle=False)
    assert p2 == tmp_path / "eda_1.ipynb"
    assert p1.exists() and p2.exists()


def test_is_bundle_omits_calc_cell(tmp_path):
    from hhemt.eda._notebook import emit_eda_notebook

    cfg = _FakeAnalysisCfg()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    analysis_nb = nbformat.read(
        emit_eda_notebook(tmp_path / "a", cfg_analysis=cfg, eda_cfg=cfg.eda, is_bundle=False),
        as_version=4,
    )
    bundle_nb = nbformat.read(
        emit_eda_notebook(tmp_path / "b", cfg_analysis=cfg, eda_cfg=cfg.eda, is_bundle=True),
        as_version=4,
    )
    a_src = "\n".join(c.source for c in analysis_nb.cells)
    b_src = "\n".join(c.source for c in bundle_nb.cells)
    assert "byte-identity" in a_src  # calc cell present on a source root
    assert "byte-identity" not in b_src  # omitted on a bundle root


@pytest.mark.parametrize("is_bundle", [False, True], ids=["source_root", "bundle_root"])
def test_seed_figure_cell_dispatches_on_enabled_plots(tmp_path, is_bundle):
    """The seed-figure cell CONSULTS enabled_plots instead of hardcoding one builder.

    Pre-fix RED on both assertions: the emitted cell imports
    config_diff_maps_figure_from_root and calls it unconditionally, so neither
    "enabled_plots" nor any dem-resolution builder name appears in the notebook.
    Both strings are live names today (config/eda.py and eda/_report.py), so neither
    assertion is green-by-construction in either world.

    THE TWO ARMS ARE is_bundle, NOT enabled_plots. Post-fix the cell's SOURCE is
    identical for every enabled_plots value, because the dispatch happens when the
    notebook RUNS rather than when it is emitted (see _FakeEdaCfg's own note above).
    The axis that DOES vary the emitted source is is_bundle: a bundle root yields a
    5-cell notebook and a source root a 6-cell one -- the ADR-9 calc cell is gated on
    it -- and BOTH must still carry the seed-figure cell. That is a genuinely different
    correct state, and it catches a defect the single-arm form cannot: an
    implementation that placed the _BUILDERS map inside the is_bundle-gated region, or
    a later edit that moved it there, would emit the dispatch on a source root and
    silently omit it on a bundle root while a source-root-only test stayed green.

    SCOPE, stated because it is still narrower than it looks: this witnesses that the
    cell consults the config and carries the builder map on BOTH root kinds. It does
    NOT witness which builder executes on a given root; that needs a notebook run.
    """
    from hhemt.eda._notebook import emit_eda_notebook

    cfg = _FakeAnalysisCfg()
    nb = nbformat.read(
        emit_eda_notebook(tmp_path, cfg_analysis=cfg, eda_cfg=cfg.eda, is_bundle=is_bundle),
        as_version=4,
    )
    src = "\n".join(c.source for c in nb.cells)

    assert "enabled_plots" in src, "the seed-figure cell does not consult enabled_plots"
    assert "dem_resolution_cost_error" in src, (
        "the builder map omits the dem-resolution family, so a dem-resolution root still gets the config-diff builder"
    )
