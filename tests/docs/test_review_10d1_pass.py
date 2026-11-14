"""Reviewer tests of the exploratory-analysis tutorial: the code on the page runs against the committed example
result and draws the committed difference plots, and the import line of the seeded cell resolves.

Each page-side check is one callable that its test and its control both call.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

_REPO = Path(__file__).resolve().parents[2]
_PAGE = _REPO / "docs" / "tutorials" / "exploratory-analysis.md"
_ASSETS = _REPO / "docs" / "tutorials" / "assets"
_PLOTS = ("eda-peak-flow-difference.png", "eda-peak-flood-depth-difference.png")
_PINNED_MATPLOTLIB = "3.11.0"
PAGE_TEXT = _PAGE.read_text(encoding="utf-8")


def python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def runnable_blocks(text: str) -> list[str]:
    """The blocks that read the committed result: those that name neither the analysis directory nor a loader call."""
    return [b for b in python_blocks(text) if "{analysis_dir}" not in b and "load_eda_context" not in b]


def draw(text: str, workdir: Path) -> dict[str, bytes]:
    """Run the runnable blocks of `text` in `workdir` on the committed store and return the PNG bytes they save."""
    import matplotlib.pyplot as plt

    (workdir / "assets").mkdir(exist_ok=True)
    link = workdir / "assets" / "irene-example-result.zarr"
    if not link.exists():
        link.symlink_to(_ASSETS / "irene-example-result.zarr")
    namespace: dict = {}
    cwd = Path.cwd()
    try:
        os.chdir(workdir)
        for block in runnable_blocks(text):
            exec(compile(block, "page-block", "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(cwd)
        plt.close("all")
    return {name: (workdir / "assets" / name).read_bytes() for name in _PLOTS}


# A decoded pixel counts as different when any channel moves by more than this many of 255 levels.
_PIXEL_STEP = 32
# The share of pixels that may differ by more than the step. Measured with the page code on two library builds,
# 0.0 for both plots; the wrong implementations of the controls below differ in 0.02 or more of the pixels.
_PIXEL_SHARE = 0.002


def different_pixel_share(left: bytes, right: bytes) -> float:
    """The share of pixels whose largest channel difference exceeds `_PIXEL_STEP`, 1.0 for images of unequal size."""
    import io

    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(left)).convert("RGB"), dtype=int)
    b = np.asarray(Image.open(io.BytesIO(right)).convert("RGB"), dtype=int)
    if a.shape != b.shape:
        return 1.0
    return float((abs(a - b).max(axis=2) > _PIXEL_STEP).mean())


_PINNED = pytest.mark.skipif(
    matplotlib.__version__ != _PINNED_MATPLOTLIB,
    reason="the plot pixels are fixed only under matplotlib " + _PINNED_MATPLOTLIB,
)


@_PINNED
def test_the_committed_plots_are_the_plots_the_page_code_draws(tmp_path):
    """Class: each committed difference plot against the PNG that the page's own code saves from the committed result.

    NEW CAPABILITY. The tutorial's own tests check that an image name equals a saved name, not that the pixels come
    from the code. The comparison is on decoded pixels, because the PNG bytes depend on the encoder build (the same
    drawing encodes to different bytes in two environments with equal pixels). Plausible wrong implementation: a plot
    made with the subtraction reversed, another group, an unclipped colour range or another axis scale, committed
    under the same name and size. Red under it: the controls below draw each of those and assert a share of
    different pixels above the tolerance. Under another matplotlib the check is skipped, not failed.
    """
    drawn = draw(PAGE_TEXT, tmp_path)
    for name in _PLOTS:
        share = different_pixel_share(drawn[name], (_ASSETS / name).read_bytes())
        assert share <= _PIXEL_SHARE, f"{name}: {share:.5f} of the pixels differ"


# Each wrong implementation, the text edit that makes it, and the plot it changes.
_WRONG = {
    "reversed flow subtraction": ("flow_coupled - flow_swmm", "flow_swmm - flow_coupled", _PLOTS[0]),
    "reversed depth subtraction": ("depth_coupled - depth_triton", "depth_triton - depth_coupled", _PLOTS[1]),
    "linear flow axis": ('ax.set_yscale("symlog", linthresh=0.1)', 'ax.set_yscale("linear")', _PLOTS[0]),
    "unclipped colour range": ("    robust=True,\n", "", _PLOTS[1]),
    "wrong group": ('depth_triton = tree["triton_only/triton"]', 'depth_triton = tree["tritonswmm/triton"]', _PLOTS[1]),
}


@_PINNED
@pytest.mark.parametrize("wrong", sorted(_WRONG))
def test_each_wrong_implementation_draws_a_plot_above_the_tolerance(wrong, tmp_path):
    """Control: the page with one wrong edit draws the named plot with more than the tolerated share changed."""
    old, new, plot = _WRONG[wrong]
    edited = PAGE_TEXT.replace(old, new)
    assert edited != PAGE_TEXT
    drawn = draw(edited, tmp_path)
    share = different_pixel_share(drawn[plot], (_ASSETS / plot).read_bytes())
    assert share > _PIXEL_SHARE, f"{wrong}: only {share:.5f} of the pixels differ"


def test_the_pixel_share_reads_equal_and_different_images():
    """Control: one image against itself reads 0.0 and against another size reads 1.0."""
    data = (_ASSETS / _PLOTS[0]).read_bytes()
    assert different_pixel_share(data, data) == 0.0
    assert different_pixel_share(data, (_ASSETS / _PLOTS[1]).read_bytes()) == 1.0


def import_lines(text: str) -> list[str]:
    """The `from hhemt... import ...` lines of the python blocks."""
    return [ln for block in python_blocks(text) for ln in block.splitlines() if ln.startswith("from hhemt")]


def unresolved_imports(lines: list[str]) -> list[str]:
    bad = []
    for ln in lines:
        try:
            exec(ln, {})  # noqa: S102
        except ImportError:
            bad.append(ln)
    return bad


def test_every_toolkit_import_on_the_page_resolves():
    """Class: every `from hhemt... import ...` line of every python block of the page.

    NEW CAPABILITY. Plausible wrong implementation: a rename of `load_eda_context` or `render_eda_plots` in the
    toolkit with the page and the seeded notebook left as they are. Red under it: the control feeds a name that
    the package does not define.
    """
    lines = import_lines(PAGE_TEXT)
    assert lines
    assert unresolved_imports(lines) == []


def test_the_import_check_reports_a_missing_name():
    """Control: an import of a name that the package lacks is reported, a real one is not."""
    assert unresolved_imports(["from hhemt.eda import a_name_the_package_lacks"]) != []
    assert unresolved_imports(["from hhemt.eda import load_eda_context"]) == []
