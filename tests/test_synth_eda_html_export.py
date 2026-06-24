"""Phase-4 unit tests for hhemt.eda._html_export (ADR-14 best-effort)."""

from __future__ import annotations


def test_export_eda_html_degrades_to_none_on_bad_kernel(tmp_path, recwarn):
    from hhemt.eda._html_export import export_eda_html

    # A non-existent notebook path forces nbformat.read to raise -> best-effort None.
    out = export_eda_html(tmp_path / "missing.ipynb", root=tmp_path, kernel_name="python3")
    assert out is None
    assert any("EDA HTML export skipped" in str(w.message) for w in recwarn.list)

def test_export_eda_html_unknown_kernel_degrades(tmp_path, recwarn):
    import nbformat
    from nbformat.v4 import new_code_cell, new_notebook

    from hhemt.eda._html_export import export_eda_html

    nb = new_notebook(cells=[new_code_cell("1 + 1")])
    p = tmp_path / "eda.ipynb"
    nbformat.write(nb, p)
    out = export_eda_html(p, root=tmp_path, kernel_name="definitely-not-a-real-kernel")
    assert out is None  # NoSuchKernel degrades to warning, never raises
