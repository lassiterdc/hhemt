"""Two-arm differential for the benchmarking `models` label (`__MODEL_ARM_LABEL__`).

Pairs the Spec 5-10 change that threads a model-arm label through the SHARED plot-rule
emitter. The arms are deliberately asymmetric in what they can catch:

  (i)   a set arm MUST produce the `models` key -- FAILS against pre-fix source, because
        the registry carries no sentinel before Spec 10 and no substitution can invent one.
  (ii)  a rule whose labels carry NO sentinel MUST emit byte-identically under a set and an
        empty arm -- the over-firing guard, which stays meaningful while (i) is green.
  (iii) an empty arm MUST drop the key entirely rather than emit `"models": ""`, which
        `combined_snakefile_generator._plot_id_facets` refuses ("the report renders a blank
        index column").

The labels template is read FROM the registry, never restated here, so this test cannot
drift from `_reporting_sets.py`.
"""

from __future__ import annotations

import pytest

from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.workflow import RuleEmissionContext, RuleSpec, _emit_plot_rule

_SENTINEL = "__MODEL_ARM_LABEL__"


def _benchmarking_labels_template() -> str:
    """The live registry's benchmarking labels string. Sourced, not restated."""
    for sel in get_reporting_set("benchmarking").renderer_selection:
        for tpl in sel.rule_spec_template:
            if tpl.renderer_module == "sensitivity_benchmarking":
                return tpl.report_kwargs["labels"]
    raise AssertionError("no sensitivity_benchmarking template in the benchmarking set")


def _ctx(model_arm: str) -> RuleEmissionContext:
    return RuleEmissionContext(
        python_executable="python",
        log_dir_rel="_logs",
        conda_env_path="",
        config_args_str="--system-config cfg_system.yaml --analysis-config cfg_analysis.yaml",
        is_sensitivity=True,
        static_backend="plotly",
        model_arm=model_arm,
    )


def _spec(labels: str) -> RuleSpec:
    return RuleSpec(
        rule_name="plot_sensitivity_benchmarking",
        renderer_module="sensitivity_benchmarking",
        input_flags=("_status/f_consolidate_experiment_complete.flag",),
        output_path_template="plots/sensitivity/benchmarking/benchmarking__{independent_var}.vs.total__OUTPUT_EXT__",
        source_paths=("sensitivity_datatree.zarr",),
        wildcards=("independent_var",),
        extra_cli_flags=(),
        extra_params=(),
        report_kwargs={
            "caption": "report/captions/sensitivity_benchmarking.rst",
            "category": "Key Results",
            "subcategory": "Benchmarking",
            "labels": labels,
        },
        resources_yaml="mem_mb=4000, time_min=10",
        log_path_template="_logs/plots/sensitivity_benchmarking_{independent_var}.log",
    )


def test_registry_carries_the_arm_sentinel():
    """Spec 10's precondition. Without it arms (i) and (iii) are vacuous."""
    assert _SENTINEL in _benchmarking_labels_template()


def test_arm_i_set_arm_emits_the_models_key():
    """FAILS pre-fix: no sentinel in the registry => no key can be produced."""
    emitted = _emit_plot_rule(_spec(_benchmarking_labels_template()), _ctx("TRITON-SWMM"))
    assert '"models": "TRITON-SWMM"' in emitted
    assert _SENTINEL not in emitted, "the sentinel must be consumed, never emitted"


def test_arm_iii_empty_arm_drops_the_key_rather_than_emitting_an_empty_value():
    emitted = _emit_plot_rule(_spec(_benchmarking_labels_template()), _ctx(""))
    assert '"models"' not in emitted
    assert '""' not in emitted.split("labels=")[1].split("\n")[0]
    assert _SENTINEL not in emitted


def test_arm_ii_a_rule_without_the_sentinel_is_byte_identical_under_both_arms():
    """The over-firing guard. A sentinel-free labels string must be untouched."""
    plain = '{"figure": "Summary table"}'
    set_arm = _emit_plot_rule(_spec(plain), _ctx("TRITON-SWMM"))
    empty_arm = _emit_plot_rule(_spec(plain), _ctx(""))
    assert set_arm == empty_arm
    assert plain in set_arm


# --- Arm (iv): the one that catches INERTNESS ------------------------------------
# Arms (i)-(iii) construct the context directly, so they verify the emitter and the
# registry and say NOTHING about whether a real builder RESOLVES an arm. Round 8
# measured exactly that gap: all three passed while the feature was inert on the
# sensitivity-master source path. This arm drives a REAL builder and must FAIL before
# Spec 8c and PASS after.
@pytest.mark.slow
def test_arm_iv_a_real_sensitivity_master_builder_resolves_a_nonempty_arm(
    synth_sensitivity_analysis,
):
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder._base_builder
    ctx = builder._make_rule_emission_context(static_backend=builder._get_report_cfg_static_backend())
    assert ctx.model_arm, (
        "a sensitivity master must resolve a non-empty model arm; an empty arm makes the "
        "whole labels channel silently inert (Round 8: 3 benchmarking goldens, 0 models keys)"
    )
    generated = synth_sensitivity_analysis.sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both", compression_level=5
    )
    assert f'"models": "{ctx.model_arm}"' in generated
