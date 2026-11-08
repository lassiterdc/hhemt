"""The settings that the developer's rulings built or removed, each with a default arm and a moved-value arm.

For every setting built here, one arm shows that the
default gives the output the code gave before the setting was read, and one arm shows that a
moved value changes it. A must-fire control runs the moved-value check against a reader that
ignores the setting and requires the check to fail, so a check that passes for any reader is
caught. The tests that take a `*_completed_isolated` fixture compile and run the solver and belong on a
cluster; the others compile and run nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from hhemt.config.hpc_system import PartitionSpec, hpc_system_config
from hhemt.config.static_plots import (
    PeakFloodDepthStaticConfig,
    SensitivityBenchmarkingStaticConfig,
)
from hhemt.config.synthetic_experiment import synthetic_experiment_config
from hhemt.exceptions import ConfigurationError
from hhemt.validation import (
    ValidationResult,
    _check_partition_spec_request,
    _validate_partition_spec_caps_for_analysis,
    _validate_per_member_row_caps,
)


def _must_fire(check, ignoring_reader):
    """Run `check` against a reader that ignores the setting; the check must raise AssertionError."""
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        check(ignoring_reader)


# ---------------------------------------------------------------- depth_clip_quantile_lower


def test_depth_clip_lower_default_leaves_vmin_and_reads_no_data(monkeypatch):
    import hhemt.report_renderers.per_sim_peak_flood_depth as pfd

    def boom(*a, **k):
        raise AssertionError("the default must not read any depth data")

    monkeypatch.setattr(pfd, "_shared_depth_range", boom)
    map_cfg = SimpleNamespace(depth_clip_quantile_lower=None)
    assert pfd._apply_depth_clip_quantile_lower(None, None, map_cfg, 0.01, 3.0) == 0.01


def _clip(pfd, monkeypatch, lower_setting, lower_value, vmin=0.01, vmax=3.0):
    monkeypatch.setattr(pfd, "_shared_depth_range", lambda *a, **k: (lower_value, vmax))
    map_cfg = SimpleNamespace(depth_clip_quantile_lower=lower_setting)
    return pfd._apply_depth_clip_quantile_lower(None, None, map_cfg, vmin, vmax)


def test_depth_clip_lower_moved_value_raises_vmin_never_lowers_it(monkeypatch):
    import hhemt.report_renderers.per_sim_peak_flood_depth as pfd

    assert _clip(pfd, monkeypatch, 0.05, 0.4) == pytest.approx(0.4)
    assert _clip(pfd, monkeypatch, 0.05, 0.001) == 0.01  # never below depth_vmin
    assert _clip(pfd, monkeypatch, 0.05, 5.0) == 0.01  # at or above vmax would empty the bar


def test_depth_clip_lower_must_fire_control(monkeypatch):
    import hhemt.report_renderers.per_sim_peak_flood_depth as pfd

    def check(reader):
        assert reader(0.05) != reader(None), "a moved value must change the colour bar's lower end"

    def reader_that_reads(setting):
        return _clip(pfd, monkeypatch, setting, 0.4)

    check(reader_that_reads)
    _must_fire(check, lambda setting: 0.01)  # a reader that ignores the setting


# ---------------------------------------------------------------- aggregation


def _replicate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(sa_id="s0", group_value="serial", n_devices=1, config_id="serial-1", wallclock_s=100.0),
            dict(sa_id="s1", group_value="serial", n_devices=1, config_id="serial-1", wallclock_s=110.0),
            dict(sa_id="s2", group_value="serial", n_devices=1, config_id="serial-1", wallclock_s=150.0),
        ]
    )


def _serial_line(aggregation: str | None):
    from hhemt.report_renderers.sensitivity_benchmarking import _compute_scaling_series

    kwargs = {} if aggregation is None else {"aggregation": aggregation}
    speedup_pg, _eff, _all, _eff_all = _compute_scaling_series(_replicate_frame(), **kwargs)
    return speedup_pg["serial"][0][1]


def test_aggregation_default_is_the_mean():
    assert _serial_line(None) == _serial_line("mean") == pytest.approx(100.0 / 120.0)


def test_aggregation_moved_value_changes_the_statistic_the_line_divides():
    assert _serial_line("median") == pytest.approx(100.0 / 110.0)
    assert _serial_line("min") == pytest.approx(100.0 / 100.0)
    assert _serial_line("max") == pytest.approx(100.0 / 150.0)


def test_aggregation_must_fire_control():
    def check(reader):
        assert reader("max") != reader("mean")

    check(_serial_line)
    _must_fire(check, lambda aggregation: _serial_line("mean"))


# ---------------------------------------------------------------- PartitionSpec caps


def _errors(spec, **request):
    result = ValidationResult(context="t")
    base = dict(n_omp_threads=None, mem_gb_per_cpu=None, run_mode="serial")
    base.update(request)
    _check_partition_spec_request(
        result, field_prefix="analysis", who="analysis", partition_name="p", spec=spec, **base
    )
    return result.errors


def test_partition_caps_default_spec_refuses_nothing_including_mpi():
    spec = PartitionSpec()
    assert spec.supports_mpi is True  # omitting the key accepts multi-rank jobs
    assert _errors(spec, n_omp_threads=64, mem_gb_per_cpu=64, run_mode="mpi") == []
    assert _errors(spec, run_mode="hybrid") == []


def test_partition_caps_moved_values_refuse():
    assert any("max_cpus_per_task" in e.message for e in _errors(PartitionSpec(max_cpus_per_task=4), n_omp_threads=8))
    # memory: the cap is in MB, the request in GB, so 2 GB per CPU is 2000 MB
    assert _errors(PartitionSpec(max_mem_mb_per_cpu=2000), mem_gb_per_cpu=2) == []
    assert any(
        "max_mem_mb_per_cpu" in e.message for e in _errors(PartitionSpec(max_mem_mb_per_cpu=1999), mem_gb_per_cpu=2)
    )
    assert any("supports_mpi" in e.message for e in _errors(PartitionSpec(supports_mpi=False), run_mode="mpi"))
    assert any("supports_mpi" in e.message for e in _errors(PartitionSpec(supports_mpi=False), run_mode="hybrid"))
    assert _errors(PartitionSpec(supports_mpi=False), run_mode="openmp") == []
    spec = PartitionSpec(available_gpu_models=["a100"], gpu_hardware="a6000")
    assert any("available_gpu_models" in e.message for e in _errors(spec))
    assert _errors(PartitionSpec(available_gpu_models=["a100"], gpu_hardware="a100")) == []


def test_partition_caps_must_fire_control():
    def check(reader):
        assert reader(PartitionSpec(max_cpus_per_task=4), 8), "a request over the cap must be refused"

    check(lambda spec, n: _errors(spec, n_omp_threads=n))
    _must_fire(check, lambda spec, n: [])  # a preflight that ignores the cap


def _analysis(**kw):
    base = dict(hpc_ensemble_partition="p", n_omp_threads=2, mem_gb_per_cpu=2, run_mode="openmp")
    base.update(kw)
    return SimpleNamespace(**base)


def test_plain_analysis_is_checked_against_its_ensemble_partition():
    hpc = SimpleNamespace(partitions={"p": PartitionSpec(max_cpus_per_task=4, supports_mpi=False)})
    ok = ValidationResult(context="t")
    _validate_partition_spec_caps_for_analysis(_analysis(), hpc, ok)
    assert ok.errors == []
    over = ValidationResult(context="t")
    _validate_partition_spec_caps_for_analysis(_analysis(n_omp_threads=8, run_mode="mpi"), hpc, over)
    assert {e.field for e in over.errors} == {"analysis.n_omp_threads", "analysis.run_mode"}
    # no hpc_system_config, or no ensemble partition: no-op
    none = ValidationResult(context="t")
    _validate_partition_spec_caps_for_analysis(_analysis(), None, none)
    _validate_partition_spec_caps_for_analysis(_analysis(hpc_ensemble_partition=None), hpc, none)
    assert none.errors == []


def test_sensitivity_rows_are_checked_and_inherit_the_analysis_value(tmp_path):
    csv = tmp_path / "rows.csv"
    pd.DataFrame(
        {
            "analysis.n_omp_threads": [2, 16, np.nan],
            "analysis.run_mode": ["openmp", "openmp", "mpi"],
        }
    ).to_csv(csv, index=False)
    cfg = SimpleNamespace(
        toggle_sensitivity_analysis=True,
        sensitivity_analysis=csv,
        hpc_ensemble_partition="p",
        n_omp_threads=2,
        mem_gb_per_cpu=2,
        run_mode="openmp",
    )
    hpc = SimpleNamespace(partitions={"p": PartitionSpec(max_cpus_per_task=4, supports_mpi=False)})
    result = ValidationResult(context="t")
    _validate_per_member_row_caps(cfg, hpc, result)
    fields = sorted(e.field for e in result.errors)
    assert fields == [
        "sensitivity_analysis.row[1].n_omp_threads",
        "sensitivity_analysis.row[2].run_mode",
    ]


# ---------------------------------------------------------------- executor_profile_extras / local_gpus_for_workflow


def _builder_ns(extras=None, local_gpus=None):
    from hhemt.workflow import SnakemakeWorkflowBuilder

    ns = SimpleNamespace(
        cfg_hpc_system=SimpleNamespace(executor_profile_extras=extras if extras is not None else {}),
        cfg_analysis=SimpleNamespace(local_cpu_cores_for_workflow=4, local_gpus_for_workflow=local_gpus),
    )
    ns._merge_executor_profile_extras = lambda config: SnakemakeWorkflowBuilder._merge_executor_profile_extras(
        ns, config
    )
    return ns


def _local_profile_text(**kw) -> str:
    from hhemt.workflow import SnakemakeWorkflowBuilder

    config = SnakemakeWorkflowBuilder.generate_snakemake_config(_builder_ns(**kw), "local")
    return yaml.dump(config, default_flow_style=False, sort_keys=False, width=float("inf"))


def test_executor_profile_extras_empty_writes_todays_profile_byte_for_byte():
    baseline = _local_profile_text()
    assert _local_profile_text(extras={}) == baseline
    assert "slurm-requeue" not in baseline


def test_executor_profile_extras_moved_value_reaches_the_profile():
    text = _local_profile_text(extras={"slurm-requeue": True})
    assert "slurm-requeue: true" in text


def test_executor_profile_extras_merges_lists_and_refuses_a_key_the_toolkit_owns():
    from hhemt.workflow import SnakemakeWorkflowBuilder

    config = {"default-resources": ["nodes=1"], "executor": "slurm"}
    SnakemakeWorkflowBuilder._merge_executor_profile_extras(
        _builder_ns(extras={"default-resources": ["slurm_qos=long"]}), config
    )
    assert config["default-resources"] == ["nodes=1", "slurm_qos=long"]
    with pytest.raises(ConfigurationError, match="executor"):
        SnakemakeWorkflowBuilder._merge_executor_profile_extras(_builder_ns(extras={"executor": "local"}), config)


def test_local_gpus_default_passes_nothing_and_a_number_sets_the_gpu_resource():
    baseline = _local_profile_text()
    assert _local_profile_text(local_gpus=None) == baseline
    assert "gpu=" not in baseline
    assert "gpu=2" in _local_profile_text(local_gpus=2)


def test_profile_settings_must_fire_control():
    def check(reader):
        assert reader(2) != reader(None)

    check(lambda gpus: _local_profile_text(local_gpus=gpus))
    _must_fire(check, lambda gpus: _local_profile_text(local_gpus=None))  # a profile that ignores the setting


# ---------------------------------------------------------------- reprex scratch_dir


def test_reprex_scratch_dir_redirects_only_analysis_dir(tmp_path):
    from hhemt.bundle._reprex import _redirect_analysis_dir

    cfg_path = tmp_path / "analysis_config.yaml"
    cfg_path.write_text(yaml.safe_dump({"analysis_dir": "/bundle", "n_gpus": 1}))
    before = cfg_path.read_text()
    _redirect_analysis_dir(cfg_path, Path("/srv/data") / "bundle")
    after = yaml.safe_load(cfg_path.read_text())
    assert after == {"analysis_dir": "/srv/data/bundle", "n_gpus": 1}
    assert yaml.safe_load(before)["analysis_dir"] == "/bundle"


def test_reprex_reads_scratch_dir_only_when_set():
    import inspect

    from hhemt.bundle import _reprex

    src = inspect.getsource(_reprex.reprex)
    assert "if reprex_cfg.scratch_dir is not None:" in src  # None keeps today's location


# ---------------------------------------------------------------- the four static-plot policies


def _static(**kw):
    return PeakFloodDepthStaticConfig(
        plot_id="per_sim_peak_flood_depth", renderer_kind="per_sim_peak_flood_depth", **kw
    )


def test_subplot_label_default_draws_nothing_and_a_label_draws_text():
    from hhemt.report_renderers.system_overview import _draw_static_subplot_label

    fig, ax = plt.subplots()
    _draw_static_subplot_label(ax, _static())
    _draw_static_subplot_label(ax, None)
    assert [t.get_text() for t in ax.texts] == []
    _draw_static_subplot_label(ax, _static(subplot_label="(a)"))
    assert [t.get_text() for t in ax.texts] == ["(a)"]
    plt.close(fig)


def test_vmin_vmax_strategy_quantile_clips_the_depth_upper_bound_and_absolute_does_not():
    from hhemt.config.viz_vocabulary import VminVmaxStrategy
    from hhemt.report_renderers.per_sim_peak_flood_depth import _static_quantile_vmax

    depths = xr.DataArray(np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 100.0]]))
    assert _static_quantile_vmax(_static(), depths) is None  # absolute: today's range
    assert _static_quantile_vmax(None, depths) is None
    quantile_cfg = _static(vmin_vmax_strategy=VminVmaxStrategy.quantile, vmax_quantile=0.5)
    assert _static_quantile_vmax(quantile_cfg, depths) == pytest.approx(3.0)
    no_q = _static(vmin_vmax_strategy=VminVmaxStrategy.quantile)
    assert _static_quantile_vmax(no_q, depths) is None


def test_panel_scale_policy_independent_drops_the_shared_depth_range():
    from hhemt.config.viz_vocabulary import PanelScalePolicy
    from hhemt.report_renderers.per_sim_peak_flood_depth import _uses_shared_depth_scale

    assert _uses_shared_depth_scale(None) is True
    assert _uses_shared_depth_scale(_static()) is True  # default shared
    assert _uses_shared_depth_scale(_static(panel_scale_policy=PanelScalePolicy.independent)) is False


def test_value_encoding_policy_default_keeps_color_and_marker():
    from hhemt.config.viz_vocabulary import ValueEncodingPolicy
    from hhemt.report_renderers.sensitivity_benchmarking import _apply_value_encoding

    palette = ["#111111", "#222222", "#333333"]
    cfg = SensitivityBenchmarkingStaticConfig(plot_id="sens", renderer_kind="sensitivity_benchmarking")
    assert _apply_value_encoding(cfg, 2, "red", "o", palette) == ("red", "o")
    assert _apply_value_encoding(None, 2, "red", "o", palette) == ("red", "o")
    one = SensitivityBenchmarkingStaticConfig(
        plot_id="sens",
        renderer_kind="sensitivity_benchmarking",
        value_encoding_policy=ValueEncodingPolicy.one_to_one_dict,
    )
    assert _apply_value_encoding(one, 1, "red", "o", palette) == ("#222222", "o")
    shape = SensitivityBenchmarkingStaticConfig(
        plot_id="sens",
        renderer_kind="sensitivity_benchmarking",
        value_encoding_policy=ValueEncodingPolicy.value_to_shape,
    )
    color, marker = _apply_value_encoding(shape, 1, "red", "o", palette)
    assert color == "red" and marker != "o"


# ---------------------------------------------------------------- ToolkitPin.sdist_sha256


def _pin(sha=None, version="0.1.0"):
    return SimpleNamespace(version=version, sdist_sha256=sha)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_sdist_pin_unset_fetches_nothing(monkeypatch):
    import hhemt.experiment_definition as ed

    def boom(*a, **k):
        raise AssertionError("an unset digest must not touch the network")

    monkeypatch.setattr(ed, "_fetch_sdist_chunks", boom)
    ed._verify_toolkit_pin_sdist(_pin(None))


def test_sdist_pin_match_passes_and_mismatch_names_both_digests(monkeypatch):
    import hhemt.experiment_definition as ed

    payload = [b"abc", b"def"]
    monkeypatch.setattr(ed, "_fetch_sdist_chunks", lambda version: iter(payload))
    ed._verify_toolkit_pin_sdist(_pin(_sha(b"abcdef")))
    wrong = "0" * 64
    with pytest.raises(ConfigurationError) as exc:
        ed._verify_toolkit_pin_sdist(_pin(wrong))
    assert wrong in str(exc.value) and _sha(b"abcdef") in str(exc.value)


def test_sdist_pin_unreachable_index_refuses_naming_the_pin(monkeypatch):
    import hhemt.experiment_definition as ed

    def down(version):
        raise OSError("network down")
        yield  # pragma: no cover - makes this a generator like the real fetch

    monkeypatch.setattr(ed, "_fetch_sdist_chunks", down)
    with pytest.raises(ConfigurationError) as exc:
        ed._verify_toolkit_pin_sdist(_pin("a" * 64, version="0.1.0"))
    assert "toolkit_pin" in str(exc.value) and "0.1.0" in str(exc.value) and "a" * 64 in str(exc.value)


def test_sdist_pin_must_fire_control(monkeypatch):
    import hhemt.experiment_definition as ed

    monkeypatch.setattr(ed, "_fetch_sdist_chunks", lambda version: iter([b"abcdef"]))

    def check(verify):
        with pytest.raises(ConfigurationError):
            verify(_pin("0" * 64))

    check(ed._verify_toolkit_pin_sdist)
    _must_fire(check, lambda pin: None)  # a verifier that never compares


# ---------------------------------------------------------------- the removed settings


def test_removed_settings_are_gone_from_their_models():
    assert "default_execution_mode" not in hpc_system_config.model_fields
    assert not [
        n for n in SensitivityBenchmarkingStaticConfig.model_fields if n.startswith("gpu") and n.endswith("_marker")
    ]
    assert "include_resume" not in synthetic_experiment_config.model_fields


def test_removed_hpc_key_is_refused_in_a_profile():
    with pytest.raises(Exception, match="default_execution_mode"):
        hpc_system_config.model_validate(
            {"hpc_name": "x", "default_execution_mode": "batch_job", "partitions": {"p": {}}}
        )


# ---------------------------------------------------------------- merge and index-failure guards
#
# The tests below fail against the code as built: each names an input that the profile merge or the
# digest check handles wrongly. A test that turns green only when the source changes is the
# instrument; loosening it to turn it green defeats it.


def _profile_extras_hpc(extras: dict):
    """Build the real `hpc_system_config`, so the load-time guard has already accepted `extras`."""
    return hpc_system_config.model_validate(
        {
            "hpc_name": "x",
            "default_account": "acct",
            "max_concurrent_jobs": 10,
            "max_concurrent_cpus": 100,
            "partitions": {"standard": {"max_runtime": 60}},
            "executor_profile_extras": extras,
        }
    )


def _extras_builder_ns(extras: dict, *, local_gpus=None):
    """A builder stand-in whose `generate_snakemake_config` reads a real `hpc_system_config`."""
    from hhemt.workflow import SnakemakeWorkflowBuilder

    ns = SimpleNamespace(
        cfg_hpc_system=_profile_extras_hpc(extras),
        cfg_analysis=SimpleNamespace(
            local_cpu_cores_for_workflow=4,
            local_gpus_for_workflow=local_gpus,
            hpc_ensemble_partition="standard",
            hpc_restart_times_other=1,
        ),
        _override_hpc_restart_times_other=None,
        _resolve_account=lambda: "acct",
    )
    ns._merge_executor_profile_extras = lambda config: SnakemakeWorkflowBuilder._merge_executor_profile_extras(
        ns, config
    )
    return ns


def _resource_names(entries) -> list[str]:
    return [str(e).split("=", 1)[0].strip() for e in entries]


def _profile_or_none(mode: str, extras: dict, **kw):
    """The generated profile, or None when the builder refuses `extras` with a ConfigurationError."""
    from hhemt.workflow import SnakemakeWorkflowBuilder

    try:
        return SnakemakeWorkflowBuilder.generate_snakemake_config(_extras_builder_ns(extras, **kw), mode)
    except ConfigurationError:
        return None


def test_extras_cannot_give_the_in_flight_cpu_bag_a_second_value():
    """Property: a `resources` entry in `executor_profile_extras` never leaves two `hhemt_cpus` entries.

    Class: slurm-mode profiles built with `max_concurrent_cpus` set and an extras `resources` list
    that names `hhemt_cpus` (the load-time guard scans only `default-resources` and `set-resources`,
    so this input loads). Why it is required: Snakemake keeps the LAST of two same-named resources
    (measured: `parse_resources(["gpu=2", "gpu=8"])` returns `{'gpu': 8}`), so concatenating the lists
    silently replaces the in-flight CPU cap the toolkit emits. What kills it: the merge appending
    the extras list after the toolkit's. A second correct implementation under which it still passes:
    refusing the key with a ConfigurationError, or dropping the extras entry and keeping the cap.
    """
    config = _profile_or_none("slurm", {"resources": ["hhemt_cpus=1"]})
    if config is None:
        return
    names = _resource_names(config["resources"])
    assert names.count("hhemt_cpus") == 1, f"two entries name hhemt_cpus: {config['resources']}"


def test_extras_cannot_give_a_toolkit_default_resource_a_second_value():
    """Property: a `default-resources` extras entry never leaves two `slurm_account` entries.

    Class: slurm-mode profiles with an extras `default-resources` entry naming a resource the
    toolkit emits in the same list but the load-time name guard omits (`slurm_account`). Why it is
    required: the later entry wins in Snakemake (measured: `DefaultResources(["slurm_account=a",
    "slurm_account=b"])` keeps `b`), so a job is billed to an account the toolkit did not choose.
    What kills it: the merge concatenating the two lists. A second correct implementation under
    which it still passes: refusing the key, or adding `slurm_account` to the guard's name set so
    the extras never load.
    """
    config = _profile_or_none("slurm", {"default-resources": ["slurm_account=other"]})
    if config is None:
        return
    names = _resource_names(config["default-resources"])
    assert names.count("slurm_account") == 1, f"two entries name slurm_account: {config['default-resources']}"


def test_extras_cannot_give_the_local_gpu_limit_a_second_value():
    """Property: an extras `resources` entry never leaves two `gpu` entries when `local_gpus_for_workflow` is set.

    Class: local-mode profiles with `local_gpus_for_workflow` set and an extras `resources` list
    naming `gpu`. Why it is required: the chunk builds `local_gpus_for_workflow` to bound GPU
    scheduling, and the same last-wins rule lets an extras entry replace it with no error. What
    kills it: the merge concatenating the two lists. A second correct implementation under which it
    still passes: refusing the key, or keeping the toolkit's entry and dropping the extras one.
    """
    config = _profile_or_none("local", {"resources": ["gpu=8"]}, local_gpus=2)
    if config is None:
        return
    names = _resource_names(config["resources"])
    assert names.count("gpu") == 1, f"two entries name gpu: {config['resources']}"


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(lambda: __import__("http.client").client.IncompleteRead(b"abc", 10), id="connection-dropped"),
        pytest.param(lambda: __import__("json").JSONDecodeError("Expecting value", "<html>", 0), id="html-answer"),
    ],
)
def test_sdist_pin_index_failure_that_is_not_an_oserror_still_refuses_naming_the_pin(monkeypatch, failure):
    """Property: every way the index read fails refuses the run with a ConfigurationError that names the pin.

    Class: failures of the fetch seat that are not OSError: `http.client.IncompleteRead` (a
    connection that drops mid-body; an HTTPException) and `json.JSONDecodeError` (an index that
    answers with a page; a ValueError). Why it is required: the spec refuses an index that cannot be
    reached, naming the pin, and a raw exception exits with a traceback that names neither the pin nor
    the declared digest. What kills it: `_verify_toolkit_pin_sdist` catching only OSError. A second
    correct implementation under which it still passes: catching these exception types in the
    verifier, or converting them to OSError inside `_fetch_sdist_chunks`.
    """
    import hhemt.experiment_definition as ed

    def _broken(version):
        yield b"abc"
        raise failure()

    monkeypatch.setattr(ed, "_fetch_sdist_chunks", _broken)
    with pytest.raises(ConfigurationError) as exc:
        ed._verify_toolkit_pin_sdist(_pin("a" * 64, version="0.1.0"))
    assert "toolkit_pin" in str(exc.value) and "0.1.0" in str(exc.value)


# ------------------------------------------------ wiring: each built read, reached through its entry point
#
# The tests above call the helpers directly, so a deleted call from `render()`, `preflight_validate()`,
# `run_experiment()` or `reprex()` would leave them green. The tests below reach each read site THROUGH the
# entry point and go red when that one call is deleted. The first three groups run on the laptop; the
# render groups use a synthetic completed fixture, so they are compile-tier and run on Rivanna.


def _spy(monkeypatch, module, name, calls):
    """Wrap `module.name` so each call is recorded as `(caller function name, args, kwargs)` and passes through."""
    import sys

    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        calls.setdefault(name, []).append((sys._getframe(1).f_code.co_name, args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)


def test_reprex_redirects_the_run_outputs_into_scratch_dir_only_when_it_is_set(monkeypatch, tmp_path):
    """Property: `reprex()` points the runnable analysis config's `analysis_dir` under `scratch_dir`, and only then.

    Class: a bundle directory with `reprex_config.scratch_dir` set and with it None, the bundle readers
    replaced by stand-ins that leave the redirect as the only writer of `analysis_dir`. Why it is required:
    the setting is delivered by that one call inside `reprex()`. What kills it: deleting the call.
    """
    import hhemt.bundle._emit as emit
    import hhemt.bundle._reprex as reprex_mod
    import hhemt.config.loaders as loaders
    import hhemt.validation as validation

    bundle = tmp_path / "bundle_x"
    bundle.mkdir()
    seen = {}

    def fake_analysis_config(bundle_root, **kwargs):
        path = Path(bundle_root) / "analysis_config.yaml"
        path.write_text(yaml.safe_dump({"analysis_dir": str(bundle_root), "n_gpus": 1}))
        return path

    class _Cfg(SimpleNamespace):
        def model_copy(self, update=None):
            return _Cfg(**{**self.__dict__, **(update or {})})

    def fake_yaml_to_model(path, model_cls):
        data = yaml.safe_load(Path(path).read_text()) if str(path).endswith("analysis_config.yaml") else {}
        seen["analysis_dir"] = data.get("analysis_dir")
        return _Cfg(**data)

    monkeypatch.setattr(emit, "reconstitute_runnable_config", lambda bundle_root, **kw: bundle_root / "system.yaml")
    monkeypatch.setattr(emit, "reconstitute_runnable_analysis_config", fake_analysis_config)
    monkeypatch.setattr(loaders, "yaml_to_model", fake_yaml_to_model)
    monkeypatch.setattr(
        validation, "preflight_validate", lambda *a, **k: validation.ValidationResult(context="reprex-wiring")
    )
    monkeypatch.setattr(reprex_mod, "_find_sif_entity", lambda root: None)
    monkeypatch.setattr(reprex_mod, "_bundle_is_container_mode", lambda root: False)
    monkeypatch.setattr(reprex_mod, "_scan_zero_user_info", lambda root: [])
    monkeypatch.setattr(reprex_mod, "_emit_amendments", lambda *a, **k: [])

    def run(scratch):
        cfg = SimpleNamespace(
            scratch_dir=scratch,
            sif_root=tmp_path / "sifs",
            target_ensemble_partition="p",
            target_setup_and_analysis_processing_partition=None,
        )
        reprex_mod.reprex(bundle, cfg, SimpleNamespace(partitions={"p": PartitionSpec()}))
        return seen["analysis_dir"]

    assert run(None) == str(bundle.resolve())
    assert run(tmp_path / "scratch_me") == str(tmp_path / "scratch_me" / bundle.resolve().name)


def test_preflight_validate_checks_the_analysis_request_against_its_partition_caps(monkeypatch):
    """Property: `preflight_validate` refuses an analysis whose request is over its ensemble partition's caps.

    Class: a real `preflight_validate` call with every other validator replaced by a no-op, an analysis that
    asks for 8 OpenMP threads on a partition capped at 4, and an HPC profile that is None. Why it is required:
    the analysis-level cap check is called from this one place. What kills it: deleting the call.
    """
    import hhemt.validation as validation

    empty = lambda *a, **k: validation.ValidationResult(context="stub")  # noqa: E731
    for name in ("validate_system_config", "validate_analysis_config", "validate_data_consistency"):
        monkeypatch.setattr(validation, name, empty)
    for name in (
        "_validate_per_member_system_configs",
        "_validate_selected_event_forcing_extent",
        "_validate_event_window_columns",
        "_validate_per_member_row_caps",
        "_validate_per_row_partition_requires_batch_job",
        "_validate_setup_mem_sizing",
        "_validate_container_config",
        "_validate_resume_interruption_schedule",
        "_validate_coupling_exchange_flow",
        "_validate_clear_scenario_timeseries",
        "_validate_timeseries_keep",
    ):
        monkeypatch.setattr(validation, name, lambda *a, **k: None)
    hpc = SimpleNamespace(partitions={"p": PartitionSpec(max_cpus_per_task=4)})
    over = validation.preflight_validate(None, _analysis(n_omp_threads=8), cfg_hpc_system=hpc)
    assert [e.field for e in over.errors] == ["analysis.n_omp_threads"]
    within = validation.preflight_validate(None, _analysis(n_omp_threads=2), cfg_hpc_system=hpc)
    assert within.errors == []
    assert validation.preflight_validate(None, _analysis(n_omp_threads=8), cfg_hpc_system=None).errors == []


def test_run_experiment_verifies_the_pin_digest_before_anything_else(monkeypatch, tmp_path):
    """Property: `run_experiment` refuses a contradicted pin digest before it resolves overrides or builds.

    Class: a definition whose `toolkit_pin.sdist_sha256` is wrong, with the index replaced by a stub, and
    every later step left real so that a skipped check surfaces as a different exception. Why it is
    required: the digest check has this one caller. What kills it: deleting the call.
    """
    import hhemt.experiment_definition as ed

    definition = SimpleNamespace(toolkit_pin=_pin("0" * 64, version="0.1.0"))
    monkeypatch.setattr(ed, "load_experiment_definition", lambda definition_dir: definition)
    monkeypatch.setattr(ed, "_fetch_sdist_chunks", lambda version: iter([b"abcdef"]))
    with pytest.raises(ConfigurationError) as exc:
        ed.run_experiment(tmp_path, "uva", dry_run=True)
    assert "toolkit_pin" in str(exc.value) and _sha(b"abcdef") in str(exc.value)


# Render groups (compile-tier: Rivanna).


def _report_cfg_with_backend(analysis, backend, **sensitivity_update):
    report = analysis.cfg_analysis.report
    update = {"interactive": report.interactive.model_copy(update={"static_backend": backend})}
    if sensitivity_update:
        update["sensitivity"] = report.sensitivity.model_copy(update=sensitivity_update)
    return report.model_copy(update=update)


def _depth_static(**kw):
    return PeakFloodDepthStaticConfig(
        plot_id="per_sim_peak_flood_depth__evt.wiring", renderer_kind="per_sim_peak_flood_depth", **kw
    )


@pytest.mark.slow
def test_depth_render_reaches_its_clip_label_and_shared_range_reads(
    synthetic_multisim_completed_isolated, tmp_path, monkeypatch
):
    """Property: a static depth render reaches the lower-clip, label and shared-range reads of the shared policy.

    Class: `per_sim_peak_flood_depth.render` with a static config (`panel_scale_policy` shared, `subplot_label`
    set) on a completed synthetic analysis. What kills it: deleting the clip call (M3a), the label call (M6a)
    or the shared-range reads.
    """
    import hhemt.report_renderers.per_sim_peak_flood_depth as pfd

    analysis = synthetic_multisim_completed_isolated
    calls = {}
    for name in (
        "_apply_depth_clip_quantile_lower",
        "_draw_static_subplot_label",
        "_shared_depth_max",
        "_shared_wse_range",
        "_uses_shared_depth_scale",
    ):
        _spy(monkeypatch, pfd, name, calls)
    static = _depth_static(subplot_label="(a)")
    pfd.render(analysis, analysis.cfg_analysis.report, tmp_path / "depth.pdf", event_iloc=0, static_cfg=static)
    assert any(c[0] == "render" for c in calls["_apply_depth_clip_quantile_lower"])
    assert calls["_draw_static_subplot_label"][0][1][1].subplot_label == "(a)"
    assert calls["_shared_depth_max"] and calls["_shared_wse_range"]


@pytest.mark.slow
def test_depth_render_independent_policy_drops_both_cross_event_ranges(
    synthetic_multisim_completed_isolated, tmp_path, monkeypatch
):
    """Property: `panel_scale_policy="independent"` makes the depth render skip the depth and WSE cross-event scans.

    Class: the same render with the independent policy. What kills it: deleting the policy read at either
    range site (M4): the scan is then called.
    """
    import hhemt.report_renderers.per_sim_peak_flood_depth as pfd
    from hhemt.config.viz_vocabulary import PanelScalePolicy

    analysis = synthetic_multisim_completed_isolated
    calls = {}
    _spy(monkeypatch, pfd, "_shared_depth_max", calls)
    _spy(monkeypatch, pfd, "_shared_wse_range", calls)
    static = _depth_static(panel_scale_policy=PanelScalePolicy.independent)
    pfd.render(analysis, analysis.cfg_analysis.report, tmp_path / "depth.pdf", event_iloc=0, static_cfg=static)
    assert "_shared_depth_max" not in calls and "_shared_wse_range" not in calls


@pytest.mark.slow
def test_depth_render_quantile_strategy_takes_the_quantile_across_events_when_shared(
    synthetic_multisim_completed_isolated, tmp_path, monkeypatch
):
    """Property: `vmin_vmax_strategy="quantile"` sets the depth upper bound across events (shared) or per figure.

    Class: two renders of `per_sim_peak_flood_depth.render` with a quantile strategy. What kills it: deleting
    the strategy read (M5a), or computing the quantile per figure under `shared`.
    """
    import hhemt.report_renderers.per_sim_peak_flood_depth as pfd
    from hhemt.config.viz_vocabulary import PanelScalePolicy, VminVmaxStrategy

    analysis = synthetic_multisim_completed_isolated
    shared_calls = {}
    _spy(monkeypatch, pfd, "_static_quantile_depth_vmax", shared_calls)
    _spy(monkeypatch, pfd, "_shared_depth_range", shared_calls)
    shared = _depth_static(vmin_vmax_strategy=VminVmaxStrategy.quantile, vmax_quantile=0.5)
    pfd.render(analysis, analysis.cfg_analysis.report, tmp_path / "shared.pdf", event_iloc=0, static_cfg=shared)
    assert shared_calls["_static_quantile_depth_vmax"]
    assert any(c[2].get("upper_quantile") == 0.5 for c in shared_calls["_shared_depth_range"])
    independent_calls = {}
    _spy(monkeypatch, pfd, "_shared_depth_range", independent_calls)
    independent = _depth_static(
        vmin_vmax_strategy=VminVmaxStrategy.quantile, vmax_quantile=0.5, panel_scale_policy=PanelScalePolicy.independent
    )
    pfd.render(
        analysis, analysis.cfg_analysis.report, tmp_path / "independent.pdf", event_iloc=0, static_cfg=independent
    )
    assert "_shared_depth_range" not in independent_calls


@pytest.mark.slow
def test_depth_plotly_render_reaches_the_lower_clip(synthetic_multisim_completed_isolated, tmp_path, monkeypatch):
    """Property: the interactive depth figure builder applies the lower clip.

    Class: `per_sim_peak_flood_depth.render` with the plotly backend and no static config. What kills it:
    deleting the clip call in `_build_peak_flood_depth_figure` (M3b).
    """
    import hhemt.report_renderers.per_sim_peak_flood_depth as pfd

    analysis = synthetic_multisim_completed_isolated
    calls = {}
    _spy(monkeypatch, pfd, "_apply_depth_clip_quantile_lower", calls)
    pfd.render(analysis, _report_cfg_with_backend(analysis, "plotly"), tmp_path / "depth.html", event_iloc=0)
    assert any(c[0] == "_build_peak_flood_depth_figure" for c in calls["_apply_depth_clip_quantile_lower"])


@pytest.mark.slow
def test_conduit_render_reaches_its_label_and_quantile_reads(
    synthetic_multisim_completed_isolated, tmp_path, monkeypatch
):
    """Property: a static conduit-flow render draws the label and takes the peak-flow quantile.

    Class: `per_sim_conduit_flow.render` with `subplot_label` and a quantile strategy. What kills it:
    deleting the label call (M6b) or the strategy read (M5b).
    """
    import hhemt.report_renderers.per_sim_conduit_flow as cf
    from hhemt.config.static_plots import ConduitFlowStaticConfig
    from hhemt.config.viz_vocabulary import VminVmaxStrategy

    analysis = synthetic_multisim_completed_isolated
    calls = {}
    _spy(monkeypatch, cf, "_draw_static_subplot_label", calls)
    _spy(monkeypatch, cf.np, "nanquantile", calls)
    static = ConduitFlowStaticConfig(
        plot_id="per_sim_conduit_flow__evt.wiring",
        renderer_kind="per_sim_conduit_flow",
        subplot_label="(b)",
        vmin_vmax_strategy=VminVmaxStrategy.quantile,
        vmax_quantile=0.37,
    )
    cf.render(analysis, analysis.cfg_analysis.report, tmp_path / "conduit.pdf", event_iloc=0, static_cfg=static)
    assert calls["_draw_static_subplot_label"][0][1][1].subplot_label == "(b)"
    assert any(len(c[1]) > 1 and c[1][1] == 0.37 for c in calls["nanquantile"])


@pytest.mark.slow
def test_system_overview_render_reaches_its_label(synthetic_multisim_completed_isolated, tmp_path, monkeypatch):
    """Property: a static system-overview render draws the label.

    Class: `system_overview.render` with `subplot_label`. What kills it: deleting the label call (M6c).
    """
    import hhemt.report_renderers.system_overview as so
    from hhemt.config.static_plots import SystemOverviewStaticConfig

    analysis = synthetic_multisim_completed_isolated
    calls = {}
    _spy(monkeypatch, so, "_draw_static_subplot_label", calls)
    static = SystemOverviewStaticConfig(plot_id="system_overview", renderer_kind="system_overview", subplot_label="(c)")
    so.render(analysis, analysis.cfg_analysis.report, tmp_path / "overview.pdf", static_cfg=static)
    assert calls["_draw_static_subplot_label"][0][1][1].subplot_label == "(c)"


@pytest.mark.slow
def test_benchmarking_static_render_reaches_the_label_and_both_encoding_reads(
    synthetic_sensitivity_completed_isolated, tmp_path, monkeypatch
):
    """Property: a static benchmarking render draws the label and applies the value encoding in both panel functions.

    Class: `sensitivity_benchmarking.render` with a static config. What kills it: deleting the label call
    (M6d) or the encoding call in `_draw_panel` or `_draw_metric_panel` (M7a, M7b).
    """
    import hhemt.report_renderers.sensitivity_benchmarking as sb
    import hhemt.report_renderers.system_overview as so
    from hhemt.config.viz_vocabulary import ValueEncodingPolicy

    analysis = synthetic_sensitivity_completed_isolated.experiment
    calls = {}
    _spy(monkeypatch, so, "_draw_static_subplot_label", calls)
    _spy(monkeypatch, sb, "_apply_value_encoding", calls)
    static = SensitivityBenchmarkingStaticConfig(
        plot_id="sensitivity_benchmarking__var.n_devices",
        renderer_kind="sensitivity_benchmarking",
        subplot_label="(d)",
        value_encoding_policy=ValueEncodingPolicy.one_to_one_dict,
    )
    sb.render(
        analysis, analysis.cfg_analysis.report, tmp_path / "bench.pdf", independent_var="n_devices", static_cfg=static
    )
    assert calls["_draw_static_subplot_label"][0][1][1].subplot_label == "(d)"
    callers = {c[0] for c in calls["_apply_value_encoding"]}
    assert {"_draw_panel", "_draw_metric_panel"} <= callers


@pytest.mark.slow
def test_benchmarking_plotly_render_reads_the_aggregation_at_the_scaling_series(
    synthetic_sensitivity_completed_isolated, tmp_path, monkeypatch
):
    """Property: the interactive benchmarking render passes `aggregation` to the scaling series.

    Class: `sensitivity_benchmarking.render` with the plotly backend and `aggregation="max"`. What kills it:
    deleting the read at the `_compute_scaling_series` call (M1). The replicate average inside
    `_plotly_metric_panel` (M2) is not reached by this fixture's groups (measured: no string-argument
    `agg` call in a render), so it is guarded by the test below.
    """
    import hhemt.report_renderers.sensitivity_benchmarking as sb

    analysis = synthetic_sensitivity_completed_isolated.experiment
    calls = {}
    _spy(monkeypatch, sb, "_compute_scaling_series", calls)
    report = _report_cfg_with_backend(analysis, "plotly", aggregation="max")
    sb.render(analysis, report, tmp_path / "bench.html", independent_var="n_devices")
    assert calls["_compute_scaling_series"][0][2].get("aggregation") == "max"


def test_plotly_metric_panel_collapses_replicates_with_the_configured_statistic():
    """Property: the line through a multi-point group takes the configured replicate statistic before the per-x minimum.

    Class: `_plotly_metric_panel` on a frame whose configuration `cfg-a` has replicates 10, 20 and 60 at x=2
    and `cfg-b` has 25 at x=2, so the per-x minimum over mean (30, 25) is 25, over max (60, 25) is 25 and over
    min (10, 25) is 10, and over median (20, 25) is 20. What kills it: reverting the replicate average to a
    fixed mean (M2): `max` would then still give 25 but `min` and `median` would not.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import hhemt.report_renderers.sensitivity_benchmarking as sb
    from hhemt.report_renderers._provenance import ProvenanceLog

    df = pd.DataFrame(
        {
            "group_value": ["openmp"] * 6,
            "indep_value": [1, 2, 2, 2, 2, 4],
            "config_id": ["cfg-a", "cfg-a", "cfg-a", "cfg-a", "cfg-b", "cfg-a"],
            "wallclock_disp": [5.0, 10.0, 20.0, 60.0, 25.0, 3.0],
        }
    )

    def line_y_at_two(statistic):
        fig = make_subplots(rows=1, cols=1)
        sens_cfg = SimpleNamespace(aggregation=statistic, palette=["#000000"] * 8, line_width=2, point_size=36)
        sb._plotly_metric_panel(
            fig,
            df,
            y_col="wallclock_disp",
            row=1,
            panel_id="p",
            group_by_var=None,
            sens_cfg=sens_cfg,
            prov=ProvenanceLog(),
            show_in_legend=False,
        )
        line = next(t for t in fig.data if isinstance(t, go.Scatter) and t.mode == "lines")
        return dict(zip(list(line.x), list(line.y), strict=True))[2]

    assert line_y_at_two("mean") == pytest.approx(25.0)
    assert line_y_at_two("min") == pytest.approx(10.0)
    assert line_y_at_two("median") == pytest.approx(20.0)
    assert line_y_at_two("max") == pytest.approx(25.0)
