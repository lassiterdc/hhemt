"""Best-effort nbconvert HTML export of the emitted EDA notebook (ADR-14).

Executes the notebook in-process (``nbclient.NotebookClient``) against the env's
``python3`` kernelspec, then converts to a single self-contained HTML file
(``HTMLExporter(embed_images=True)``, Plotly bundle inlined once by nbconvert).
BEST-EFFORT: any failure (missing kernel, cold-start, dead kernel, cell error,
timeout) degrades to a warning and returns ``None`` — the EDA loop NEVER fails on
HTML-export failure. The notebook is the source of truth.
"""

from __future__ import annotations

import warnings
from pathlib import Path

#: Per-cell execution timeout (s). A backstop against a hung kernel, not a tight
#: bound — EDA cells re-derive datatrees and render figures.
_EXECUTE_TIMEOUT_S = 600
#: The kernelspec name every standard install registers; nbconvert --execute's default.
_KERNEL_NAME = "python3"


def export_eda_html(
    notebook_path: Path,
    *,
    root: Path,
    timeout: int = _EXECUTE_TIMEOUT_S,
    kernel_name: str = _KERNEL_NAME,
) -> Path | None:
    """Execute ``notebook_path`` and write ``{root}/eda_report/eda_report.html``.

    Returns the HTML path on success, ``None`` on any failure (warned, never raised).
    """
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbconvert import HTMLExporter

        nb = nbformat.read(notebook_path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()
        exporter = HTMLExporter(embed_images=True)
        body, _resources = exporter.from_notebook_node(nb)
        out_dir = root / "eda_report"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "eda_report.html"
        out_path.write_text(body, encoding="utf-8")
        return out_path
    except Exception as exc:  # noqa: BLE001 — best-effort by design (ADR-14)
        warnings.warn(
            f"EDA HTML export skipped: nbconvert execution of {notebook_path} "
            f"failed ({type(exc).__name__}: {exc}). The notebook is the source of "
            f"truth; open it in Jupyter to explore. EDA loop continues.",
            stacklevel=2,
        )
        return None
