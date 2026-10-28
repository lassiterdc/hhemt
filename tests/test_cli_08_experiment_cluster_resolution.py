"""How `hhemt reset` and `hhemt run-experiment` decide which cluster an experiment targets.

The problem this file guards: both verbs find an experiment from its definition directory, and an
experiment's analysis tree lives on one cluster. When the user omits `--cluster`, the verbs
must pick the experiment definition's only declared cluster, and must refuse rather than guess when the
definition declares none or several, because `reset` deletes whatever tree it is pointed at.
Both verbs share one resolver, so every case runs against both.

Nothing here builds an analysis, reaches Snakemake, compiles or simulates: `run-experiment`'s
runner and `reset`'s case builder are replaced at their module seams
(`hhemt.experiment_definition.run_experiment` / `.build_case_from_definition`, both imported inside
the verb bodies) by recorders. The experiment definition's `experiment.yaml` is real and schema-valid,
because the resolver reads its declared clusters.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import hhemt.experiment_definition as eb
from hhemt.cli import app

runner = CliRunner()


class _Stop(Exception):
    """Raised by the run-experiment recorder so the verb ends once the cluster is recorded."""


def _write_definition(root: Path, clusters: list[str]) -> Path:
    """A schema-valid experiment definition declaring exactly `clusters` under hpc_system_config."""
    root.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "description": "cluster-resolution fixture",
        "system_config": "configs/system.yaml",
        "analysis_config": "configs/analysis.yaml",
        "hpc_system_config": {name: f"hpc/hpc_system_config_{name}.yaml" for name in clusters},
        "toolkit_pin": {"version": "0.1.0"},
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    return root


def _patch_run_experiment(monkeypatch, recorded: list[str]) -> None:
    def _fake(definition_dir, cluster, **kwargs):
        recorded.append(cluster)
        raise _Stop

    monkeypatch.setattr(eb, "run_experiment", _fake)


def _patch_reset(monkeypatch, recorded: list[str], tmp_path: Path) -> None:
    gone = tmp_path / "analysis_dir_never_created"

    def _fake(definition, definition_dir, cluster, **kwargs):
        recorded.append(cluster)
        analysis = SimpleNamespace(
            analysis_paths=SimpleNamespace(analysis_dir=gone),
            delete=lambda **kw: None,
        )
        return SimpleNamespace(analysis=analysis)

    monkeypatch.setattr(eb, "build_case_from_definition", _fake)


_VERBS = {
    "reset": lambda d: ["reset", "--experiment", str(d), "--yes"],
    "run-experiment": lambda d: ["run-experiment", "--experiment", str(d)],
}


def _invoke(verb: str, definition: Path, extra: list[str], monkeypatch, tmp_path: Path):
    recorded: list[str] = []
    if verb == "reset":
        _patch_reset(monkeypatch, recorded, tmp_path)
    else:
        _patch_run_experiment(monkeypatch, recorded)
    result = runner.invoke(app, [*_VERBS[verb](definition), *extra])
    return result, recorded


# (declared clusters, extra argv, expected cluster or None for "refused with exit 2")
_STATES = {
    "one-declared-no-flag": (["uva"], [], "uva"),
    "two-declared-no-flag": (["frontier", "uva"], [], None),
    "zero-declared-no-flag": ([], [], None),
    "two-declared-flag-names-one": (["frontier", "uva"], ["--cluster", "frontier"], "frontier"),
    "control-one-declared-matching-flag": (["uva"], ["--cluster", "uva"], "uva"),
}


@pytest.mark.parametrize("state", list(_STATES))
@pytest.mark.parametrize("verb", list(_VERBS))
def test_experiment_verbs_resolve_the_cluster_one_way(verb, state, monkeypatch, tmp_path):
    """Property: both verbs resolve the target cluster by the same rule.

    Class: every `reset` and `run-experiment` invocation, across the four definition states the
    rule distinguishes (one declared cluster and no `--cluster`; two declared and none given;
    zero declared and none given; two declared with `--cluster` naming one) plus a control
    whose explicit, matching `--cluster` must resolve identically to the first state. Why it is
    required: `reset` deletes the tree on the resolved cluster, so a wrong default or a silent
    guess between several clusters destroys the wrong experiment's results; a refusal must be
    exit 2 and, where several are declared, must name each so the user can choose. What kills
    it: `reset` absent, `--cluster` still required, a default that picks the first of several,
    or a refusal that names no cluster. A second correct implementation under which it still
    passes: one that reads the declared clusters from a different definition field or resolves them
    inside the runner rather than in the CLI, since only the cluster handed to the seam, the
    exit code and the named clusters are asserted.
    """
    declared, extra, expected = _STATES[state]
    definition = _write_definition(tmp_path / "definition", declared)

    result, recorded = _invoke(verb, definition, extra, monkeypatch, tmp_path)

    if expected is None:
        assert result.exit_code == 2, f"exit {result.exit_code}: {result.output}"
        assert recorded == [], "a refused invocation must not reach the case builder or runner"
        assert "--cluster" in result.output, f"refusal does not point at --cluster: {result.output}"
        for name in declared:
            assert name in result.output, f"refusal does not name {name!r}: {result.output}"
    else:
        assert recorded == [expected], f"exit {result.exit_code}: {result.output}"
