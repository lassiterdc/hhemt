"""The tutorial's job-count bullet is true of the Workflow Performance page.

The bullet says the page's job count leaves out the workflow run that rendered the report, and
that the page names that workflow run when the report is rendered inside one. The tests render
the page from log fixtures with `SLURM_JOB_NAME` set and unset and assert the behaviour the
bullet states, so a change to the exclusion fails here and sends the maintainer to the bullet.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from hhemt.config.report import report_config
from hhemt.report_renderers import workflow_performance

TUTORIAL = Path(__file__).resolve().parent.parent / "docs" / "tutorials" / "norfolk-compute-config-sweep.md"
EARLIER_RUN = "9e62392e-a7d2-476a-a066-cc6fb19708f1"
RENDERING_RUN = "0c1d2e3f-4a5b-4c6d-8e7f-90a1b2c3d4e5"
EARLIER_LABEL = "2026-11-01T015232.143811"
RENDERING_LABEL = "2026-11-01T024124.511220"


def _log(analysis_dir: Path, label: str, run_id: str, job_ids: list[str]) -> None:
    log_dir = analysis_dir / ".snakemake" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"SLURM run ID: {run_id}"] + [
        f"Job {n} has been submitted with SLURM jobid {job_id} (log: /x/{job_id}.log)."
        for n, job_id in enumerate(job_ids)
    ]
    (log_dir / f"{label}.snakemake.log").write_text("\n".join(lines) + "\n")


def _render(analysis_dir: Path) -> tuple[str, dict]:
    analysis = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=types.SimpleNamespace(analysis_id="A1"),
    )
    output = analysis_dir / "plots" / "workflow_performance.html"
    workflow_performance.render(analysis, report_config(), output)
    manifest = json.loads((analysis_dir / "plots" / "workflow_performance.manifest.json").read_text())
    return output.read_text(), manifest


@pytest.fixture
def two_workflow_runs(tmp_path: Path) -> Path:
    _log(tmp_path, EARLIER_LABEL, EARLIER_RUN, ["101", "102"])
    _log(tmp_path, RENDERING_LABEL, RENDERING_RUN, ["201"])
    return tmp_path


def _has_job(html: str, job_id: str) -> bool:
    return f'"Job ID (job record)": "{job_id}"' in html


def test_the_job_count_leaves_out_the_rendering_workflow_run_and_names_it(
    two_workflow_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `SLURM_JOB_NAME` equal to the rendering workflow run's id, its job is absent, the
    earlier workflow run's jobs are present, and the page names the workflow run left out."""
    monkeypatch.setenv("SLURM_JOB_NAME", RENDERING_RUN)
    html, manifest = _render(two_workflow_runs)

    assert not _has_job(html, "201")
    assert _has_job(html, "101") and _has_job(html, "102")
    assert f"left out: <code>{RENDERING_LABEL}</code>" in html
    assert manifest["renderer_data"]["excluded_job_count"] == 1


def test_the_job_count_leaves_nothing_out_when_not_rendered_inside_a_workflow_run(
    two_workflow_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUST-FIRE CONTROL: with the variable unset every job is counted and the page says no
    workflow run is left out. A change that always excluded, or never excluded, fails one of
    the two tests."""
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    html, manifest = _render(two_workflow_runs)

    assert all(_has_job(html, job_id) for job_id in ("101", "102", "201"))
    assert "No workflow run is left out." in html
    assert manifest["renderer_data"]["excluded_job_count"] == 0


def test_the_tutorial_bullet_states_what_the_page_does() -> None:
    """The bullet is present and names the facts the tests above pin: a report rendered inside a
    workflow run leaves it out and names it, and a later render counts every job."""
    text = " ".join(TUTORIAL.read_text().split())

    assert "rendered inside a workflow run leaves that workflow run out" in text
    assert "the page names the omitted workflow run" in text
    assert "rendered later, from the analysis directory, counts every job" in text
