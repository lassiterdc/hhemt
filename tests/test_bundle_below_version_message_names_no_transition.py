"""The below-version bundle refusal explains how an old bundle fails, without listing versions.

`Bundle.from_directory` refuses a bundle whose `bundle_schema_version` is below the
toolkit's `BUNDLE_SCHEMA_VERSION`. Its message explains the two ways an old bundle
fails (a key the model has retired, or a key that has since become required) and
tells the reader to re-emit the bundle. It used to also carry a per-version list of
what each schema bump changed. That list was maintained by hand, so it stopped
several versions short of the current schema and told a reader with a recent bundle
nothing about the bump that actually broke it.

These tests build only a bundle manifest in a temporary directory. They compile no
solver and run no simulation.
"""

import json
import re

import pytest

from hhemt.bundle import BUNDLE_SCHEMA_VERSION, Bundle, BundleSchemaError

# A version transition written the way the retired list wrote it ("v3->v4"), plus
# the arrow and prose spellings a rewrite would plausibly reach for.
_TRANSITION = re.compile(r"\bv\d+\s*(?:->|→|to)\s*v\d+\b", re.IGNORECASE)


def _below_version_message(tmp_path, bundle_version):
    (tmp_path / "bundle_manifest.json").write_text(
        json.dumps({"bundle_schema_version": bundle_version, "bundle_root_invariants": {}})
    )
    with pytest.raises(BundleSchemaError) as excinfo:
        Bundle.from_directory(tmp_path)
    return str(excinfo.value)


@pytest.mark.parametrize("bundle_version", [1, BUNDLE_SCHEMA_VERSION - 1], ids=["oldest", "one-below"])
def test_below_version_message_names_no_version_transition(tmp_path, bundle_version):
    """Property: the below-version refusal names no specific schema-version transition,
    and still explains the two failure modes and the re-emit remedy.

    Class: every bundle version below the toolkit's, positioned here at the oldest
    version and at the one directly below the current one -- the position a
    hand-kept list is most likely to have stopped short of.

    Why required: a per-version list in this message is a second copy of the schema
    history that nothing keeps current, so it goes stale at every bump and reads as
    complete when it is not. The two failure modes and the re-emit instruction hold
    for every bump, so they are what the message must keep.

    What kills it: restoring any "vN->vM" transition to the message, or dropping the
    re-emit instruction or either failure mode.

    A second correct implementation that still passes: a message that words the two
    failure modes differently (for example "a field the toolkit no longer accepts" /
    "a field it now requires") while keeping the words "retired", "required" and
    "Re-emit", or one that names the current version only.
    """
    msg = _below_version_message(tmp_path, bundle_version)
    assert not _TRANSITION.search(msg), f"the message names a version transition: {_TRANSITION.search(msg).group(0)!r}"
    assert "Re-emit" in msg, "the message lost its re-emit instruction"
    assert "retired" in msg and "required" in msg, "the message lost its two-failure-mode explanation"
