"""Tests for the ``pmix_gds`` profile field and the two instructive messages that guard it.

Each test checks behaviour at the entry point a user meets, which is the YAML file loader: the field exports
``PMIX_MCA_gds`` only when set, and the two refusals name the file, the field, the value and the literal fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hhemt.config.loaders import load_analysis_config, load_hpc_system_config
from hhemt.exceptions import ConfigurationError
from hhemt.run_simulation import triton_launch_env

_REPO = Path(__file__).resolve().parent.parent


def _profile(tmp_path, **extra):
    path = tmp_path / "my_cluster_profile.yaml"
    path.write_text(yaml.safe_dump({"hpc_name": "t", "partitions": {"standard": {"max_runtime": 60}}, **extra}))
    return path


def _analysis_yaml(tmp_path, **over):
    from tests.test_config_analysis_report_field import _minimum_valid_cfg_analysis_dict

    (tmp_path / "weather.nc").touch()
    (tmp_path / "events.csv").touch()
    base = _minimum_valid_cfg_analysis_dict(tmp_path)
    base["report"] = {"interactive": {"static_backend": "matplotlib"}}
    path = tmp_path / "my_prior_experiment_analysis.yaml"
    path.write_text(yaml.safe_dump({**base, **over}))
    return path


@pytest.mark.parametrize("bad", ["shmem2", "Hash", "hash ", "", True, False, 1, ["hash"], {"hash": 1}])
def test_every_value_other_than_hash_is_refused_at_load_with_the_file_the_field_and_the_literal_fix(tmp_path, bad):
    path = _profile(tmp_path, pmix_gds=bad)
    with pytest.raises(ConfigurationError) as raised:
        load_hpc_system_config(path)
    text = str(raised.value)
    for needle in (path.name, "pmix_gds", "hash", "gds_shmem2", "pmix_gds: hash", "/tmp"):
        assert needle in text, (needle, text)


def test_the_message_for_a_bad_value_echoes_the_offending_value_so_the_edit_is_findable(tmp_path):
    path = _profile(tmp_path, pmix_gds="shmem2")
    with pytest.raises(ConfigurationError) as raised:
        load_hpc_system_config(path)
    assert "shmem2" in str(raised.value)


def test_hash_and_null_and_absent_all_load(tmp_path):
    assert load_hpc_system_config(_profile(tmp_path, pmix_gds="hash")).pmix_gds == "hash"
    assert load_hpc_system_config(_profile(tmp_path, pmix_gds=None)).pmix_gds is None
    assert load_hpc_system_config(_profile(tmp_path)).pmix_gds is None


def test_extra_env_refusal_loaded_from_the_analysis_file_names_the_file_the_key_the_field_and_the_literal_fix(tmp_path):
    path = _analysis_yaml(tmp_path, TRITON_extra_env={"PMIX_MCA_gds": "hash"})
    with pytest.raises(ConfigurationError) as raised:
        load_analysis_config(path)
    text = str(raised.value)
    for needle in (path.name, "PMIX_MCA_gds", "TRITON_extra_env", "pmix_gds: hash", "HPC system config"):
        assert needle in text, (needle, text)


def test_extra_env_refusal_fires_for_any_value_of_the_variable(tmp_path):
    for value in ("hash", "shmem2", ""):
        path = _analysis_yaml(tmp_path, TRITON_extra_env={"PMIX_MCA_gds": value})
        with pytest.raises(ConfigurationError) as raised:
            load_analysis_config(path)
        assert "pmix_gds: hash" in str(raised.value)


def test_a_neighbouring_pmix_variable_is_not_swept_into_the_refusal(tmp_path):
    path = _analysis_yaml(tmp_path, TRITON_extra_env={"PMIX_MCA_psec": "native"})
    assert load_analysis_config(path).TRITON_extra_env == {"PMIX_MCA_psec": "native"}


from tests.test_triton_launch_env_wiring import _FULL_ANALYSIS as _ANALYSIS  # noqa: E402


@pytest.mark.parametrize("run_mode", ["serial", "openmp", "mpi", "hybrid", "gpu"])
def test_the_export_follows_the_field_in_every_run_mode_and_only_the_field(run_mode):
    from tests.test_triton_launch_env_wiring import _full_hpc

    on, _ = triton_launch_env(_ANALYSIS, _full_hpc(pmix_gds="hash"), run_mode)
    off, _ = triton_launch_env(_ANALYSIS, _full_hpc(pmix_gds=None), run_mode)
    assert on["PMIX_MCA_gds"] == "hash"
    assert "PMIX_MCA_gds" not in off
    assert {k: v for k, v in on.items() if k != "PMIX_MCA_gds"} == off


def test_a_real_profile_with_the_field_set_reaches_the_launch_environment(tmp_path):
    cfg = load_hpc_system_config(_profile(tmp_path, pmix_gds="hash"))
    env, _ = triton_launch_env(_ANALYSIS, cfg, "mpi")
    assert env["PMIX_MCA_gds"] == "hash"


def test_the_uva_example_profile_sets_the_field_and_still_loads():
    path = _REPO / "test_data" / "norfolk_coastal_flooding" / "hpc_system_config_uva.yaml"
    assert yaml.safe_load(path.read_text())["pmix_gds"] == "hash"
    assert load_hpc_system_config(path).pmix_gds == "hash"


def test_the_how_to_documents_the_field_in_the_example_and_the_export_list():
    text = (_REPO / "docs" / "how-to" / "hpc-profile-setup.md").read_text()
    assert "pmix_gds:" in text
    assert "`pmix_gds` exports `PMIX_MCA_gds`" in text
    assert "gds_shmem2" in text
