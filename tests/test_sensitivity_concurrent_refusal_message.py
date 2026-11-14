"""The refusal to run sensitivity members concurrently reads as words, not glued fragments.

`TRITONSWMM_sensitivity_analysis.run_all_sims(concurrent=True)` refuses at once,
because concurrent members need resource tracking the toolkit does not have. Its
message is assembled from adjacent string literals, and literals with no space at a
join print as one word ("requiresmore"), which a user reads as a typo in the one
sentence explaining why their call failed.

The refusal is the method's first statement, so it is reached on an instance built
without a constructor; nothing here compiles a solver or runs a simulation.
"""

import pytest

from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis


def test_concurrent_refusal_message_has_no_glued_words():
    """Property: no join between the message's string literals glues two words.

    Class: the three joins that printed glued, each a differently-positioned break in
    the sentence; any one of them glued fails the test.

    Why required: the message is the only explanation a caller gets for the refusal.

    What kills it: dropping the space at any of the three joins.

    A second correct implementation that still passes: any rewording of the message
    that keeps its words separated and still names `run_simulations_concurrently`.
    """
    sens = object.__new__(TRITONSWMM_sensitivity_analysis)
    with pytest.raises(RuntimeError) as exc:
        sens.run_all_sims(pickup_where_leftoff=False, concurrent=True)
    msg = str(exc.value)
    glued = [w for w in ("requiresmore", "availabilitytracking", "functionin") if w in msg]
    assert glued == [], msg
    assert "run_simulations_concurrently" in msg
