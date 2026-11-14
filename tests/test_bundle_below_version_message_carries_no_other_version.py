"""The below-version bundle refusal mentions no schema version but the two it compares.

`Bundle.from_directory` refuses a bundle whose `bundle_schema_version` is below the
toolkit's. The message states the bundle's version and the toolkit's, explains the two
ways an old bundle fails, and tells the reader to re-emit it. Any other version number
in it is a fragment of a per-version history, which nothing keeps current and which
reads as complete when it is not. A check for one spelling of a transition ("v3->v4")
misses the same history written another way ("schema 3 -> 4", "6 to 7"), so this
module checks the numbers the message carries rather than how they are written.

It builds only a bundle manifest in a temporary directory. It compiles no solver and
runs no simulation.
"""

import json
import re

import pytest

from hhemt.bundle import BUNDLE_SCHEMA_VERSION, Bundle, BundleSchemaError

# A transition between two numbers in any of the spellings a list would use, with or
# without a leading "v": "v3->v4", "3 -> 4", "6 to 7", "v6→v7".
_ANY_TRANSITION = re.compile(r"\bv?\d+\s*(?:->|→|to)\s*v?\d+\b", re.IGNORECASE)


def _below_version_message(tmp_path, bundle_version):
    """Refuse a bundle at `bundle_version` and return the message with the bundle path removed."""
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "bundle_manifest.json").write_text(
        json.dumps({"bundle_schema_version": bundle_version, "bundle_root_invariants": {}})
    )
    with pytest.raises(BundleSchemaError) as excinfo:
        Bundle.from_directory(root)
    return str(excinfo.value).replace(str(root.resolve()), "{root}").replace(str(root), "{root}")


@pytest.mark.parametrize("bundle_version", [1, BUNDLE_SCHEMA_VERSION - 1], ids=["oldest", "one-below"])
def test_below_version_message_carries_only_the_two_compared_versions(tmp_path, bundle_version):
    """Property: the below-version refusal carries no version number other than the
    bundle's and the toolkit's, and no transition between two numbers appears in any spelling.

    Class: every bundle version below the toolkit's, positioned at the oldest and at the
    one directly below the current. At the second position a transition between exactly
    the two compared versions ("6 to 7") carries no foreign number, which is why the
    transition check is kept beside the number check.

    Why required: a version history written into this message goes stale at the next
    schema bump, and a reader cannot tell a stale list from a complete one.

    What kills it: restoring the per-version list in any spelling, with or without a
    "v" prefix, or naming any schema version other than the two being compared.

    A second correct implementation that still passes: a message that states only the
    two compared versions, worded differently, with any explanation of the failure
    modes that carries no further version number.
    """
    msg = _below_version_message(tmp_path, bundle_version)
    assert _ANY_TRANSITION.findall(msg) == [], msg
    numbers = {int(n) for n in re.findall(r"\d+", msg)}
    assert numbers <= {bundle_version, BUNDLE_SCHEMA_VERSION}, (numbers, msg)
