"""Phase-2 unit tests for hhemt.eda._local_surface.emit_eda_local_surface (ADR-12)."""

from __future__ import annotations


def test_emit_eda_local_surface_writes_import_clean_package(tmp_path):
    from hhemt.eda._local_surface import emit_eda_local_surface

    pkg = emit_eda_local_surface(tmp_path)
    assert pkg == tmp_path / "eda_local"
    assert (pkg / "__init__.py").is_file()
    assert (pkg / "_bootstrap.py").is_file()
    # import-clean convention: the skeleton must not import a bundle-relative sibling.
    bootstrap_src = (pkg / "_bootstrap.py").read_text()
    assert "import hhemt" in bootstrap_src
    assert "from eda_local" not in bootstrap_src and "import eda_local" not in bootstrap_src

def test_emit_eda_local_surface_is_idempotent(tmp_path):
    from hhemt.eda._local_surface import emit_eda_local_surface

    emit_eda_local_surface(tmp_path)
    # A user-authored sibling module must survive a re-emit (only the two skeleton
    # files are toolkit-owned).
    (tmp_path / "eda_local" / "my_reduction.py").write_text("x = 1\n")
    emit_eda_local_surface(tmp_path)
    assert (tmp_path / "eda_local" / "my_reduction.py").read_text() == "x = 1\n"
