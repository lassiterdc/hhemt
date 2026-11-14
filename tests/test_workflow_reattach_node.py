"""The tmux reattach host must be the node that actually holds the session."""

from hhemt.workflow import SnakemakeWorkflowBuilder


class TestResolveReattachNode:
    """A session is created by a local subprocess, so only the submitting host can hold it."""

    def test_in_slurm_prefers_the_true_host_over_the_configured_login_node(self):
        # The always-wrong case before this fix: inside an allocation the two differ by
        # construction, and the configured hint won.
        assert (
            SnakemakeWorkflowBuilder._resolve_reattach_node(
                submission_node="udc-aw33-2c1",
                login_node="login1.hpc.virginia.edu",
                in_slurm=True,
            )
            == "udc-aw33-2c1"
        )

    def test_outside_slurm_the_configured_login_node_still_wins(self):
        # The round-robin-alias case the override exists for is preserved unchanged.
        assert (
            SnakemakeWorkflowBuilder._resolve_reattach_node(
                submission_node="udc-ba37-32c0",
                login_node="login1.hpc.virginia.edu",
                in_slurm=False,
            )
            == "login1.hpc.virginia.edu"
        )

    def test_outside_slurm_with_no_config_falls_back_to_the_detected_host(self):
        assert (
            SnakemakeWorkflowBuilder._resolve_reattach_node(
                submission_node="udc-ba37-32c0",
                login_node=None,
                in_slurm=False,
            )
            == "udc-ba37-32c0"
        )

    def test_in_slurm_with_no_config_still_returns_the_true_host(self):
        assert (
            SnakemakeWorkflowBuilder._resolve_reattach_node(
                submission_node="udc-aw33-2c1",
                login_node=None,
                in_slurm=True,
            )
            == "udc-aw33-2c1"
        )
