"""A sensitivity experiment consolidates each member once per summary pass, and takes no model selector.

``create_analysis_summaries`` builds every member's own consolidated outputs. It used to
take a ``which`` argument that nothing read and that, at its default, ran each member's
full consolidation twice. These tests pin the call counts and the signatures. They build
the experiment with ``object.__new__`` and give it stand-in members, so they construct no
analysis, compile nothing and run no solver.
"""

import inspect

import pytest

from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis


class _CountingMember:
    """A member that only counts how often its consolidation is asked for."""

    def __init__(self) -> None:
        self.calls = 0

    def _consolidate_analysis_outputs(self, **kwargs) -> None:
        self.calls += 1


def _experiment_with_members(n_members: int = 3):
    """An experiment stand-in holding ``n_members`` counting members and a stubbed log update."""
    experiment = object.__new__(TRITONSWMM_sensitivity_analysis)
    experiment.members = {str(i): _CountingMember() for i in range(n_members)}
    experiment._update_experiment_log = lambda: None
    return experiment


def test_create_analysis_summaries_consolidates_each_member_once():
    """One ``create_analysis_summaries()`` call consolidates every member exactly once.

    Class: every member of the experiment, whatever their number; three are used.
    Why it is required: a second pass re-reads every member's summaries for no change in
    the output, and before this was fixed the default argument caused exactly that.
    What kills it: calling the per-member helper twice, or once per model type.
    A second correct implementation under which it still passes: one that loops over the
    members directly instead of through the private helper.
    """
    experiment = _experiment_with_members()
    experiment.create_analysis_summaries()
    assert [m.calls for m in experiment.members.values()] == [1, 1, 1]


def test_consolidate_outputs_consolidates_each_member_once_through_create_analysis_summaries():
    """One ``consolidate_outputs()`` call consolidates every member once via the summary pass.

    ``consolidate_analysis_outputs`` is stubbed, so this excludes the second consolidation
    that the experiment-tree assembly performs on its own. The assertion is therefore
    about the ``create_analysis_summaries`` path, not about ``consolidate_outputs`` end to end.

    Class: every member; three are used.
    Why it is required: ``consolidate_outputs`` is the public entry that runs both passes,
    so a doubled summary pass inside it is paid on every consolidation.
    What kills it: a ``consolidate_outputs`` that calls the summary pass twice, or a summary
    pass that consolidates each member twice.
    A second correct implementation under which it still passes: one that calls the
    per-member helper directly instead of ``create_analysis_summaries``.
    """
    experiment = _experiment_with_members()
    tree_calls = []
    experiment.consolidate_analysis_outputs = lambda **kwargs: tree_calls.append(kwargs)
    experiment.consolidate_outputs()
    assert len(tree_calls) == 1
    assert [m.calls for m in experiment.members.values()] == [1, 1, 1]


@pytest.mark.parametrize(
    "method_name",
    ["create_analysis_summaries", "consolidate_outputs", "consolidate_analysis_outputs"],
)
def test_consolidation_methods_take_no_model_selector(method_name):
    """None of the three consolidation methods accepts a ``which`` argument.

    Class: the three public consolidation methods of the sensitivity experiment.
    Why it is required: an accepted argument that changes nothing tells a caller they can
    restrict consolidation to one model, and they cannot.
    What kills it: re-adding ``which`` to any of the three signatures.
    A second correct implementation under which it still passes: one that renames the
    keyword-only ``verbose`` or ``compression_level`` parameters, since only ``which`` is
    asserted absent.
    """
    method = getattr(TRITONSWMM_sensitivity_analysis, method_name)
    assert "which" not in inspect.signature(method).parameters
