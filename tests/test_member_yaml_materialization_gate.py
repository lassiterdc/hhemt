"""Member YAML materialization is a DRIVER-only side effect.

Regression guard for the report-tail workflow failure in which ~63 renderer
subprocesses each rewrote all N member config YAMLs, putting every
destination name under concurrent-rename churn on the shared filesystem. Two
renderers failed with FileNotFoundError reading the very target they had just
renamed into place, taking the whole workflow down at the report tail before
`render_report` ran.

The predicate is extracted as a free function so this test exercises EXACTLY the
gate without standing up a full sensitivity-analysis construction (same pattern
as `_unlink_dprocess_flags_for_regenerate`).
"""

from hhemt.sensitivity_analysis import _should_materialize_analysis_yaml


def test_orchestrator_writes_when_target_absent(tmp_path):
    assert _should_materialize_analysis_yaml(tmp_path / "member_0.yaml", True) is True


def test_orchestrator_rewrites_when_target_present(tmp_path):
    """The driver stays authoritative: it rewrites even an existing target."""
    target = tmp_path / "member_0.yaml"
    target.write_text("stale: true\n")
    assert _should_materialize_analysis_yaml(target, True) is True


def test_non_orchestrator_does_not_rewrite_existing_target(tmp_path):
    """The load-bearing assertion: renderers are pure readers.

    This is the case that removes N-1 writers from the report tail.
    """
    target = tmp_path / "member_0.yaml"
    target.write_text("present: true\n")
    assert _should_materialize_analysis_yaml(target, False) is False


def test_non_orchestrator_still_writes_absent_target(tmp_path):
    """Bootstrap fallback: a missing YAML is always written, whoever constructs.

    Keeps the gate safe on any first-to-materialize path, so no consumer can
    encounter a missing member config.
    """
    assert _should_materialize_analysis_yaml(tmp_path / "member_0.yaml", False) is True
