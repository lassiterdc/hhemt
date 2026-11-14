"""The two shared-tier messages state why the compiled tier is special, in the message itself.

The problem this file guards: when a chunk writes the shared compiled tier (`_software`), the
aggregator prints either a `[shared-by-design-write]` advisory or a `[concurrent-build]`
problem, and both used to justify themselves by citing "properties 1 and 2" of a harness README
that lives in a private repository. A reader of the public summary cannot open that README, so
the message pointed at evidence the reader does not have. The fact itself is short: the compiled
tier is written once, by the warm build step that runs before the chunk array.

Nothing here runs a suite: each case builds a synthetic run directory and calls `aggregate()`,
using the same fixture builder as `tests/test_verdict_partition.py`.
"""

import pytest

from hhemt.suite import aggregate as A
from tests.test_verdict_partition import SOFTWARE, _build, _entry


@pytest.mark.parametrize(
    ("samples", "bucket", "tag"),
    [
        pytest.param(
            ["triton/build_tritonswmm_cpu/compilation.log"], "problems", "[concurrent-build]", id="concurrent-build"
        ),
        pytest.param([".triton.provision.lock"], "advisories", "[shared-by-design-write]", id="shared-by-design-write"),
    ],
)
def test_shared_tier_messages_state_the_fact_not_a_private_readme(tmp_path, samples, bucket, tag):
    """Property: both shared-tier messages state the warm-build fact and cite no README.

    Class: the `[concurrent-build]` problem (two chunks write the same compiled-tier path) and
    the `[shared-by-design-write]` advisory (chunks take the tier's locks), each as `aggregate()`
    prints it. Why it is required: the summary is published, and a justification that names
    "properties 1 and 2" of a README the reader cannot open is an unreadable warrant for a
    verdict-bearing line. What kills it: either message still naming "properties 1 and 2" or a
    README, or dropping the fact that the tier is written by the warm build step before the chunk
    array. A second correct implementation under which it still passes: wording the fact
    differently around the same two phrases, since only those phrases and the absence of the two
    citations are asserted.
    """
    _build(
        tmp_path,
        chunk_entries=[{SOFTWARE: _entry("_software", written=True, samples=samples)} for _ in range(2)],
    )
    result = A.aggregate(tmp_path, scope="array")

    messages = [m for m in result[bucket] if m.startswith(tag)]
    assert messages, f"no {tag} message emitted: {result[bucket]}"
    for message in messages:
        assert "properties 1 and 2" not in message, message
        assert "readme" not in message.lower(), message
        assert "warm build step" in message, message
        assert "before the chunk array" in message, message
