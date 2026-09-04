"""Two-arm differential for the combine path's composed-reporting-set refusal (S19).

Needs no synthetic model, no golden and no registry mutation: both call sites are
inside top-level functions taking only a bundle_root, so a tmp_path tree carrying
child_crates/{eid}/cfg_analysis.yaml is the whole fixture.
"""

from __future__ import annotations

import pytest
import yaml


def _child_crate(bundle_root, eid, reporting_set):
    child = bundle_root / "child_crates" / eid
    child.mkdir(parents=True, exist_ok=True)
    (child / "cfg_analysis.yaml").write_text(
        yaml.safe_dump(
            {
                "analysis_id": eid,
                "toggle_sensitivity_analysis": True,
                "report": {"reporting_set": reporting_set, "disabled_renderers": []},
            }
        )
    )
    return child


def test_single_set_child_is_harvested(tmp_path):
    """ARM B -- the differently-positioned satisfying arm, and the load-bearing half.

    A refusal written as `if _names:` rather than `if len(_names) > 1:` fires on EVERY
    child and turns the whole combine path into a crash. Only this arm catches that,
    and no violating arm can.

    Asserted on CONTENT, not on type: `isinstance(categories, list)` passes on `[]`, so
    it catches neither an empty harvest nor any other degeneration -- and it cannot
    catch the mis-written refusal either, because that RAISES. Derived from the set's
    own templates rather than from a literal list, so it does not decay when a set
    gains a renderer.
    """
    from hhemt.bundle.combined_snakefile_generator import _distinct_child_categories
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    _child_crate(tmp_path, "exp_a", ["benchmarking"])
    categories = _distinct_child_categories(tmp_path)

    assert categories, "a single-set child harvested no categories at all"
    # category lives in report_kwargs, NOT as a RuleSpecTemplate attribute -- there is
    # no `t.category`, and reaching for one raises AttributeError on the first template.
    declared = {
        (t.report_kwargs or {}).get("category")
        for sel in get_reporting_set("benchmarking").renderer_selection
        for t in sel.rule_spec_template
    }
    assert set(categories) <= declared | set(get_reporting_set("combined").category_order)


def test_composed_child_raises_naming_both_sets(tmp_path):
    """ARM A -- a composed child must REFUSE, naming the experiment and both sets.

    Taking the first name instead would harvest a subset and the combined report would
    look complete while omitting a member set's figures -- the silent failure the RAISE
    was chosen over.
    """
    from hhemt.bundle.combined_snakefile_generator import _distinct_child_categories
    from hhemt.exceptions import ConfigurationError

    _child_crate(tmp_path, "exp_b", ["benchmarking", "dem-resolution"])
    with pytest.raises(ConfigurationError) as exc:
        _distinct_child_categories(tmp_path)
    msg = str(exc.value)
    assert "exp_b" in msg
    assert "benchmarking" in msg and "dem-resolution" in msg


def test_composed_child_raises_in_the_figure_harvest_too(tmp_path):
    """The twin call site. Both sites carry the same refusal and a repair applied to
    one only is the byte-identical-twin failure this cluster has now hit three times."""
    from hhemt.bundle.combined_snakefile_generator import _harvest_per_experiment_rule_specs
    from hhemt.exceptions import ConfigurationError

    _child_crate(tmp_path, "exp_c", ["benchmarking", "dem-resolution"])
    with pytest.raises(ConfigurationError) as exc:
        _harvest_per_experiment_rule_specs(tmp_path)
    assert "exp_c" in str(exc.value)
