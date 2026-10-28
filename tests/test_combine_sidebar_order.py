"""The combined report's sidebar order, as the render function actually emits it.

`render_combined_report_via_snakemake` builds the sidebar category order inline and hands it to
the report surgery that bakes it into the rendered page. The function otherwise shells out to
Snakemake twice, so the other combine tests never call it: they check the child-category helper
and a test-side copy of the expression, and the expression itself was guarded by nothing that
runs locally. This module runs the real function with exactly two things replaced -- the
Snakemake subprocess runner and the surgery call -- and reads the order at the hand-off.

Nothing here starts a subprocess or a solver. The subprocess seam is replaced at both places the
function could reach it: the name the generator module binds, and the defining module, whose
replacement raises, so a future import change that bypasses the first fake fails loudly instead
of launching Snakemake.
"""

from __future__ import annotations

import json
from pathlib import Path

import hhemt.bundle.combined_snakefile_generator as generator
import hhemt.report_renderers._react_surgery as react_surgery
import hhemt.subprocess_utils as subprocess_utils
from hhemt.bundle.combined_snakefile_generator import _distinct_child_categories
from hhemt.report_renderers._reporting_sets import get_reporting_set

_FIGURE_HTML = "<html><head></head><body><div>figure</div></body></html>"


def _write_child(bundle_root: Path, eid: str, plot_id: str) -> None:
    """One child_crates/{eid} bundle: an empty analysis config (so the default reporting set
    applies) and one figure with its manifest, which is all the harvest reads."""
    child = bundle_root / "child_crates" / eid
    (child / "plots" / "maps").mkdir(parents=True, exist_ok=True)
    (child / "cfg_analysis.yaml").write_text("{}\n", encoding="utf-8")
    (child / "plots" / "maps" / f"{plot_id}.html").write_text(_FIGURE_HTML, encoding="utf-8")
    (child / "plots" / "maps" / f"{plot_id}.manifest.json").write_text(
        json.dumps({"plot_id": plot_id, "source_paths_relative": [f"data/{eid}.zarr"]}), encoding="utf-8"
    )


class _Completed:
    returncode = 0


def test_combined_sidebar_order_is_fixed_categories_then_child_figure_categories(tmp_path, monkeypatch) -> None:
    """Property: the category order the combined render passes to report surgery is the combined
    set's fixed categories, in their declared order, followed by exactly the child figure
    categories the composed pages carry, and nothing else.

    Class: the input class is any combined bundle whose children carry at least one harvested
    figure; this test uses two children of one base experiment.

    Why it is required: the order is what the rendered sidebar is sorted by. A dropped child
    category sorts last alphabetically instead of by its set's order, a reordered list puts the
    experiments' sections above the cross-experiment ones, and an added name writes a category
    no page carries into the page's ordering literal. None of these raises, so without this test
    a wrong order ships with the suite green.

    What kills it: building the order as the fixed categories alone, as the child categories
    before the fixed ones, or with any extra category appended.

    A second correct implementation still passes: one that builds the same sequence another way,
    for example by chaining the two parts or by merging them into a tuple, because only the
    sequence handed to surgery is compared.

    Not established: the order surgery then writes into the HTML. That step is replaced here, and
    its own tests cover how it uses the order it receives. Only the html format is rendered; the
    order is computed once, before the per-format loop, so the zip format receives the same list.
    """
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "expA_tritonswmm", "maps_fig")
    _write_child(bundle_root, "expA_triton", "maps_fig")
    # The render refuses to start when pre-rendered figures are missing and the bundle carries no
    # cfg_system.yaml, because a missing figure would make Snakemake re-run a plot rule. Nothing
    # is re-run here, since the subprocess is replaced, so an empty file stands in for it.
    (bundle_root / "cfg_system.yaml").write_text("{}\n", encoding="utf-8")

    subprocess_calls: list[list[str]] = []

    def _fake_run(cmd, logfile=None, cwd=None, echo_to_stdout=False):
        subprocess_calls.append(list(cmd))
        if "--report" in cmd:
            Path(cmd[cmd.index("--report") + 1]).write_text("<html><body></body></html>", encoding="utf-8")
        return _Completed()

    def _refuse_real_subprocess(*args, **kwargs):
        raise AssertionError("the render reached the real subprocess runner; this test must start no process")

    received_orders: list[list[str]] = []

    def _record_surgery(html_text, **kwargs):
        received_orders.append(list(kwargs["category_order"]))
        return html_text

    monkeypatch.setattr(generator, "run_subprocess_with_tee", _fake_run)
    monkeypatch.setattr(subprocess_utils, "run_subprocess_with_tee", _refuse_real_subprocess)
    monkeypatch.setattr(react_surgery, "apply_post_process_surgery", _record_surgery)

    generator.render_combined_report_via_snakemake(bundle_root, formats=("html",))

    # Both Snakemake steps went through the replacement, so the call above exercised the real
    # function's order computation and no process was started.
    assert [("--touch" in c, "--report" in c) for c in subprocess_calls] == [(True, False), (False, True)]
    assert len(received_orders) == 1, f"surgery was called {len(received_orders)} times for one format"

    fixed = list(get_reporting_set("combined").category_order)
    child_categories = _distinct_child_categories(bundle_root)
    # State the denominator: with no child categories the tail comparison below could not fail.
    assert child_categories, "the fixture produced no child figure categories"
    assert not set(child_categories) & set(fixed), "child and fixed categories overlap in the fixture"

    order = received_orders[0]
    assert order[: len(fixed)] == fixed, f"the sidebar order does not open with the fixed categories: {order}"
    assert order[len(fixed) :] == child_categories, (
        f"after the fixed categories the sidebar order is {order[len(fixed) :]}, "
        f"not the child figure categories {child_categories}"
    )
