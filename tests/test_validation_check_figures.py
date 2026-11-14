"""Reviewer tests for the provenance-completeness count (build iteration 2a.2)."""

from __future__ import annotations

import pytest

from hhemt import analysis_validation as av

_UNCOUNTED = ("plots", "report")
_BUILD_A = {"hhemt_sha": "a" * 40, "hhemt_version": "v0.1.0+A", "hhemt_dirty": "false"}
_BUILD_B = {"hhemt_sha": "b" * 40, "hhemt_version": "v0.1.0+B", "hhemt_dirty": "false"}


def _stamps(**overrides):
    stamps = {s: dict(_BUILD_A) for s in av._PROVENANCE_STAGES}
    stamps.update(overrides)
    return stamps


@pytest.mark.parametrize("stage", _UNCOUNTED)
@pytest.mark.parametrize(
    "value", [_BUILD_B, None, {**_BUILD_A, "hhemt_dirty": "true"}], ids=["other_build", "absent", "dirty"]
)
def test_a_stage_stamped_after_the_report_is_written_does_not_change_the_verdict(monkeypatch, stage, value) -> None:
    """Class: the stages that run after the validation report is written, in each state a stamp can take.

    Plots and report cannot carry a stamp when this check runs, so no state of theirs
    may fail or reshape the count; the uniform four-stage analysis passes.
    """
    monkeypatch.setattr(av, "_collect_stage_stamps", lambda _a: _stamps(**{stage: value}))
    res = av.check_provenance_completeness(object())
    assert res.passed is True
    assert "all 4 stages stamped" in res.summary


# ---- Log recoveries: distinct documents, once each -----------------------------------------


def _log_sweep(tmp_path, *, n_members: int, system_doc: str = "{}"):
    """A sweep master whose members share one system log; one event and one model per member."""
    import json
    from types import SimpleNamespace

    import pandas as pd

    system_log = tmp_path / "system" / "system.log.json"
    system_log.parent.mkdir(parents=True)
    system_log.write_text(system_doc)
    master_log = tmp_path / "master" / "analysis.log.json"
    master_log.parent.mkdir(parents=True)
    master_log.write_text(json.dumps({}))
    members = {}
    for i in range(n_members):
        mdir = tmp_path / f"member_{i}"
        (mdir / "sims" / "e1").mkdir(parents=True)
        (mdir / "analysis.log.json").write_text("{}")
        (mdir / "sims" / "e1" / "log_tritonswmm.json").write_text("{}")
        members[str(i)] = SimpleNamespace(
            _get_enabled_model_types=lambda: ["tritonswmm"],
            analysis_paths=SimpleNamespace(simulation_directory=mdir / "sims", f_log=mdir / "analysis.log.json"),
            _system=SimpleNamespace(log=SimpleNamespace(logfile=system_log)),
            df_sims=pd.DataFrame({"event": [0]}),
            _retrieve_weather_indexer_using_integer_index=lambda i: {"event_id": "e1"},
        )
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
        sensitivity=SimpleNamespace(members=members),
        analysis_paths=SimpleNamespace(f_log=master_log),
    )


def test_members_sharing_one_system_log_examine_each_document_once(tmp_path, monkeypatch) -> None:
    """15 members each own an analysis log and a model log and share one system log.

    That is 15 + 15 + 1 = 31 distinct documents, plus the master's own analysis log: 32.
    Counting the shared system log once per member gave 45 and left the master log out.
    """
    import hhemt.scenario as sc

    monkeypatch.setattr(sc, "compute_event_id_slug", lambda ev: "e1")
    res = av.check_log_recoveries(_log_sweep(tmp_path, n_members=15))
    assert "Examined 32 log documents" in res.summary
    assert "0 could not be read" in res.summary
    assert res.passed is True


def test_a_recovery_recorded_in_the_shared_system_log_is_reported_once(tmp_path, monkeypatch) -> None:
    import json

    import hhemt.scenario as sc

    monkeypatch.setattr(sc, "compute_event_id_slug", lambda ev: "e1")
    record = {"error": "bad json", "quarantine": "system.log.json.unreadable", "at": "t", "writer": "w"}
    sweep = _log_sweep(tmp_path, n_members=15, system_doc=json.dumps({"recovered_from_unparseable": [record]}))
    res = av.check_log_recoveries(sweep)
    assert res.passed is False
    assert len(res.details) == 1
    assert "sa_id" not in res.details[0]
    assert res.summary.startswith("1 log document was reconstructed")


def test_a_plain_analysis_counts_its_own_log_once(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pandas as pd

    import hhemt.scenario as sc

    monkeypatch.setattr(sc, "compute_event_id_slug", lambda ev: "e1")
    (tmp_path / "sims" / "e1").mkdir(parents=True)
    for name in ("analysis.log.json", "system.log.json", "sims/e1/log_tritonswmm.json"):
        (tmp_path / name).write_text("{}")
    plain = SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
        sensitivity=None,
        _get_enabled_model_types=lambda: ["tritonswmm"],
        analysis_paths=SimpleNamespace(simulation_directory=tmp_path / "sims", f_log=tmp_path / "analysis.log.json"),
        _system=SimpleNamespace(log=SimpleNamespace(logfile=tmp_path / "system.log.json")),
        df_sims=pd.DataFrame({"event": [0]}),
        _retrieve_weather_indexer_using_integer_index=lambda i: {"event_id": "e1"},
    )
    assert "Examined 3 log documents" in av.check_log_recoveries(plain).summary


def test_one_system_log_reached_through_two_spellings_is_examined_once(tmp_path, monkeypatch) -> None:
    """Class: members whose system-log path differs as text but names one file (a symlink, a relative form).

    The count is of documents, so two spellings of one file are one document; keying on
    the path as written would count the shared log once per spelling.
    """
    import hhemt.scenario as sc

    monkeypatch.setattr(sc, "compute_event_id_slug", lambda ev: "e1")
    sweep = _log_sweep(tmp_path, n_members=2)
    real = tmp_path / "system" / "system.log.json"
    link = tmp_path / "system_link.json"
    link.symlink_to(real)
    first, second = sweep.sensitivity.members.values()
    first._system.log.logfile = real
    second._system.log.logfile = link
    # 2 member analysis logs + 2 model logs + 1 system log + the master log
    assert "Examined 6 log documents" in av.check_log_recoveries(sweep).summary


# ---- Resource usage: a clean GPU run and a CPU-only run both return no issue rows ---------------


def _status_frame(run_mode, n_gpus, backend, verdict, distinct):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "scenario_directory": "/x/e1",
                "run_completed": True,
                "actual_nTasks": 1,
                "actual_omp_threads": 1,
                "actual_gpu_backend": backend,
                "gpu_binding_verdict": verdict,
                "actual_distinct_gpus": distinct,
            }
        ]
    )


def _resource_summary(run_mode, n_gpus, backend, verdict, distinct):
    from types import SimpleNamespace

    cfg = SimpleNamespace(run_mode=run_mode, n_mpi_procs=1, n_omp_threads=1, n_gpus=n_gpus)
    analysis = SimpleNamespace(cfg_analysis=cfg)
    return av.check_resource_usage(analysis, df_status=_status_frame(run_mode, n_gpus, backend, verdict, distinct))


def test_a_cpu_only_analysis_says_it_has_no_gpu_model_runs() -> None:
    import numpy as np

    res = _resource_summary("cpu", 0, "none", np.nan, np.nan)
    assert res.passed is True
    assert "no GPU model runs in this analysis" in res.summary
    assert "GPU binding" not in res.summary


@pytest.mark.parametrize("n_gpus", [1, 2], ids=["one_gpu", "two_gpus"])
def test_a_clean_gpu_analysis_does_not_claim_it_has_no_gpu_model_runs(n_gpus) -> None:
    res = _resource_summary("gpu", n_gpus, "HIP", "pass", n_gpus)
    assert res.passed is True
    assert "no GPU model runs" not in res.summary
    assert "GPU binding: 0 rows not measured" in res.summary


def test_an_analysis_with_no_log_out_does_not_claim_it_has_no_gpu_model_runs() -> None:
    import numpy as np
    import pandas as pd

    frame = _status_frame("gpu", 1, "HIP", "pass", 1).assign(actual_nTasks=np.nan)
    from types import SimpleNamespace

    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(run_mode="gpu", n_mpi_procs=1, n_omp_threads=1, n_gpus=1))
    res = av.check_resource_usage(analysis, df_status=frame)
    assert "no GPU model runs" not in res.summary
    assert isinstance(frame, pd.DataFrame)


def test_the_gpu_row_count_agrees_with_the_validators_own_count_on_a_mixed_sweep(capsys) -> None:
    """Class: sweep tables mixing GPU and CPU rows, completed and not.

    The check's count and the validator's must be one figure; the validator prints its
    own ("GPU rows examined: N"), so the two are compared on the same frame.
    """
    import re
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from hhemt.consolidate_workflow import count_examined_gpu_rows, validate_resource_usage

    rows = []
    for mode, done in [("gpu", True), ("cpu", True), ("gpu", False), ("gpu", True), ("cpu", False)]:
        rows.append(
            {
                "scenario_directory": f"/x/{len(rows)}",
                "run_completed": done,
                "run_mode": mode,
                "n_mpi_procs": 1,
                "n_omp_threads": 1,
                "n_gpus": 1 if mode == "gpu" else 0,
                "actual_nTasks": 1,
                "actual_omp_threads": 1,
                "actual_gpu_backend": "HIP" if mode == "gpu" else "none",
                "gpu_binding_verdict": "pass" if mode == "gpu" else np.nan,
                "actual_distinct_gpus": 1 if mode == "gpu" else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(run_mode="cpu", n_mpi_procs=1, n_omp_threads=1, n_gpus=0))
    validate_resource_usage(analysis, df_status=frame)
    printed = int(re.search(r"GPU rows examined: (\d+)", capsys.readouterr().out).group(1))
    assert count_examined_gpu_rows(analysis, df_status=frame) == printed == 2
