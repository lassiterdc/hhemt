"""Submit-path driver claim + generated-Snakefile write safety.

Two mechanisms, both introduced after the 2026-08-31 `norfolk_stochastic`
double-launch in which three drivers stacked on one analysis tree:

1. ``SnakemakeWorkflowBuilder._acquire_submit_driver_claim`` -- the submit-path
   sibling of ``_acquire_reprocess_driver_claim``. The submit path WROTE an
   orchestrator sentinel and never READ the gate; writing is not gating.
2. ``_write_snakefile_atomic`` / ``_resolve_snakefile_path`` -- the write is made
   atomic and one-deep archiving, and a DRY RUN never displaces an existing
   production Snakefile.

The discriminating tests are ``test_unmatched_override_raises_and_unlinks_nothing``
and ``test_override_is_scoped_to_the_named_driver``: an implementation that treats
the override as a blanket skip passes every happy path and fails only those two.
"""

from __future__ import annotations

import os
import socket
import types

import pytest

from hhemt import orchestrator_sentinels as osent
from hhemt.exceptions import WorkflowError
from hhemt.workflow import (
    SnakemakeWorkflowBuilder,
    _resolve_snakefile_path,
    _write_snakefile_atomic,
)


class _StubBuilder:
    """Minimal stand-in exposing only what the claim helper touches."""

    def __init__(self, analysis_dir):
        self.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
        # _max_plausible_job_lifetime_min(cfg) reads hpc_total_job_duration_min.
        self.cfg_analysis = types.SimpleNamespace(hpc_total_job_duration_min=60)

    _acquire_submit_driver_claim = SnakemakeWorkflowBuilder._acquire_submit_driver_claim
    _orchestrator_liveness_gate = SnakemakeWorkflowBuilder._orchestrator_liveness_gate
    _tmux_session_is_live = SnakemakeWorkflowBuilder._tmux_session_is_live
    _get_module_load_prefix = lambda self: ""  # noqa: E731


@pytest.fixture()
def builder(tmp_path):
    return _StubBuilder(tmp_path)


def _write_live_local_sentinel(analysis_dir, driver_id):
    """A same-host `local` sentinel whose pid is alive (ours) -> gate says ALIVE."""
    return osent.write_orchestrator_sentinel(
        analysis_dir,
        driver_id=driver_id,
        workflow_submission_mode="local",
        pid=os.getpid(),
    )


# --------------------------------------------------------------------------- #
# claim: gate behaviour
# --------------------------------------------------------------------------- #


def test_dry_run_claims_nothing_and_writes_no_sentinel(builder, tmp_path):
    assert builder._acquire_submit_driver_claim(tmp_path, workflow_submission_mode="batch_job", dry_run=True) is None
    assert not osent.orchestrator_dir(tmp_path).exists()


def test_clean_tree_claims_and_records_the_real_submission_mode(builder, tmp_path):
    did = builder._acquire_submit_driver_claim(tmp_path, workflow_submission_mode="batch_job", dry_run=False)
    assert did
    payloads = osent.read_orchestrator_sentinels(tmp_path)
    assert len(payloads) == 1
    # The submit path records its REAL mode, not a synthetic "local".
    assert payloads[0]["workflow_submission_mode"] == "batch_job"
    assert payloads[0]["hostname"] == socket.gethostname()


def test_live_same_host_driver_refuses_with_an_actionable_message(builder, tmp_path):
    _write_live_local_sentinel(tmp_path, f"{os.getpid()}-{socket.gethostname()}-deadbeef")
    with pytest.raises(WorkflowError) as excinfo:
        builder._acquire_submit_driver_claim(tmp_path, workflow_submission_mode="local", dry_run=False)
    msg = str(excinfo.value)
    assert "--override-live-driver" in msg
    assert "ORIGIN HOST" in msg
    # No second claim was written.
    assert len(osent.read_orchestrator_sentinels(tmp_path)) == 1


# --------------------------------------------------------------------------- #
# claim: the override is an ASSERTION, not a force flag
# --------------------------------------------------------------------------- #


def test_named_override_reclaims_that_sentinel_and_proceeds(builder, tmp_path):
    victim = f"{os.getpid()}-{socket.gethostname()}-deadbeef"
    _write_live_local_sentinel(tmp_path, victim)

    did = builder._acquire_submit_driver_claim(
        tmp_path,
        workflow_submission_mode="local",
        dry_run=False,
        override_live_driver=victim,
    )
    assert did and did != victim
    ids = {p["driver_id"] for p in osent.read_orchestrator_sentinels(tmp_path)}
    # The asserted-dead sentinel is GONE from disk: the assertion is recorded as
    # an act, not merely skipped for this invocation.
    assert victim not in ids
    assert ids == {did}


def test_unmatched_override_raises_and_unlinks_nothing(builder, tmp_path):
    """DISCRIMINATING. A blanket-skip implementation passes every happy path and
    fails here: a typo'd id that silently no-opped would read to the operator as
    an override that was honoured."""
    victim = f"{os.getpid()}-{socket.gethostname()}-deadbeef"
    _write_live_local_sentinel(tmp_path, victim)

    with pytest.raises(WorkflowError) as excinfo:
        builder._acquire_submit_driver_claim(
            tmp_path,
            workflow_submission_mode="local",
            dry_run=False,
            override_live_driver="no-such-driver-id",
        )
    assert "names no sentinel" in str(excinfo.value)
    ids = {p["driver_id"] for p in osent.read_orchestrator_sentinels(tmp_path)}
    assert ids == {victim}, "an unmatched assertion must not reclaim anything"


def test_override_is_scoped_to_the_named_driver(builder, tmp_path):
    """DISCRIMINATING. A boolean force flag would proceed here; a named
    assertion must not outrun its own evidence."""
    host = socket.gethostname()
    asserted = f"{os.getpid()}-{host}-aaaaaaaa"
    other = f"{os.getpid()}-{host}-bbbbbbbb"
    _write_live_local_sentinel(tmp_path, asserted)
    _write_live_local_sentinel(tmp_path, other)

    with pytest.raises(WorkflowError) as excinfo:
        builder._acquire_submit_driver_claim(
            tmp_path,
            workflow_submission_mode="local",
            dry_run=False,
            override_live_driver=asserted,
        )
    assert other in str(excinfo.value)
    ids = {p["driver_id"] for p in osent.read_orchestrator_sentinels(tmp_path)}
    assert ids == {other}, "only the named sentinel is reclaimed"


# --------------------------------------------------------------------------- #
# generated-Snakefile write safety
# --------------------------------------------------------------------------- #


def test_write_archives_one_deep_and_keeps_only_the_previous_generation(tmp_path):
    sf = tmp_path / "Snakefile"
    _write_snakefile_atomic(sf, "gen1")
    assert sf.read_text() == "gen1"
    assert not (tmp_path / "Snakefile.prev").exists()

    _write_snakefile_atomic(sf, "gen2")
    assert sf.read_text() == "gen2"
    assert (tmp_path / "Snakefile.prev").read_text() == "gen1"

    _write_snakefile_atomic(sf, "gen3")
    assert sf.read_text() == "gen3"
    # One deep: .prev holds the SECOND-oldest, never the oldest.
    assert (tmp_path / "Snakefile.prev").read_text() == "gen2"


def test_write_leaves_no_temp_files_behind(tmp_path):
    sf = tmp_path / "Snakefile"
    _write_snakefile_atomic(sf, "x")
    _write_snakefile_atomic(sf, "y")
    assert not list(tmp_path.glob("*.tmp"))


def test_dry_run_never_displaces_an_existing_snakefile(tmp_path):
    """The measured harm: the production Snakefile's content is NOT invariant
    (it carries `alive_by_token`), so a dry run against a live arm generates the
    wait-rule form and would install it over the good one."""
    good = tmp_path / "Snakefile"
    good.write_text("the live driver's DAG")

    target = _resolve_snakefile_path(tmp_path, dry_run=True)
    assert target.name == "Snakefile.dryrun"
    _write_snakefile_atomic(target, "11394 wait rules")
    assert good.read_text() == "the live driver's DAG"


def test_dry_run_on_a_fresh_tree_still_creates_the_production_snakefile(tmp_path):
    """A dry run never DISPLACES; it may CREATE. Where there is nothing to
    displace the legacy behaviour is strictly better -- df_status and
    df_snakemake_allocations parse `analysis_dir/"Snakefile"` by literal name,
    and tests/test_workflow_1job_dry_run.py asserts it exists after a dry run."""
    assert _resolve_snakefile_path(tmp_path, dry_run=True).name == "Snakefile"


def test_real_submit_always_targets_the_production_snakefile(tmp_path):
    (tmp_path / "Snakefile").write_text("old")
    assert _resolve_snakefile_path(tmp_path, dry_run=False).name == "Snakefile"
