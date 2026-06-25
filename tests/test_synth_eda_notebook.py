"""Phase-3 unit tests for hhemt.eda._notebook (ADR-13, D-NB1)."""

from __future__ import annotations

import nbformat


class _FakeEdaCfg:
    enabled_plots = ["eda_cross_sim_identity"]
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
