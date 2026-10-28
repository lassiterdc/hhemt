"""Bundle-loading errors carry no internal planning label, however they are worded.

`Bundle.from_directory` refuses a manifest with no schema version and a bundle with no
analysis config. Both messages reach someone holding a bundle they did not build, who
has no way to resolve a project-internal plan phase or requirement id. These tests
check the two messages against a pattern for such labels and against the facts each
must still state, without fixing the wording, so a correct rewording passes and a
rewording that keeps a label in a different spelling does not.

These tests build minimal bundle directories in a temporary path. They compile no
solver and run no simulation.
"""

import json
import re

import pytest

from hhemt.bundle import BUNDLE_MANIFEST_FILENAME, BUNDLE_SCHEMA_VERSION, Bundle

# Plan-phase names in any spelling, and short ids made of a letter or acronym and a
# number, the form a requirement or decision id takes. Matched after the bundle's own
# paths are removed, so a temporary directory name cannot produce a match.
_INTERNAL_LABEL = re.compile(r"Plan[- ]?Phase|\bPhase[- ]?\d+\b|\b(?:ADR|PIP)-?\d+\b|\b[RDF]-?\d+\b", re.IGNORECASE)


def _message(tmp_path, manifest, exc_type):
    """Build a bundle with `manifest`, load it, and return the raised message with paths removed."""
    root = tmp_path / "bundle"
    root.mkdir()
    (root / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(exc_type) as exc:
        Bundle.from_directory(root)
    return str(exc.value).replace(str(root.resolve()), "{root}").replace(str(root), "{root}")


@pytest.mark.parametrize(
    ("manifest", "exc_type", "facts"),
    [
        pytest.param(
            {}, ValueError, (r"bundle_schema_version", r"bundle_root_invariants"), id="missing-schema-version"
        ),
        pytest.param(
            {"bundle_schema_version": BUNDLE_SCHEMA_VERSION},
            FileNotFoundError,
            (r"cfg_analysis\.yaml", r"\breport\b", r"analysis[ _]config"),
            id="missing-analysis-config",
        ),
    ],
)
def test_bundle_load_error_states_its_facts_without_an_internal_label(tmp_path, manifest, exc_type, facts):
    """Property: each of the two bundle-load errors names the facts a reader needs and
    no internal plan phase or requirement id.

    Class: the two messages `Bundle.from_directory` raises for a manifest with no
    `bundle_schema_version` and for a bundle with no `cfg_analysis.yaml`, each reached
    by its minimal input. The assertion reads the raised message, which both the
    labelled and the plain wording produce, and checks a label pattern and a set of
    fact tokens rather than any one sentence.

    Why required: a plan-phase name or requirement id points a reader at a document
    they cannot see, so the message explains nothing to the person who receives it.

    What kills it: restoring a plan-phase name or a requirement id in any spelling the
    pattern covers, or dropping the fact the message rests on (the older-bundle key, or
    `report` being required on the analysis config).

    A second correct implementation that still passes: any wording that states those
    facts in different words, for example "written before manifests carried a
    bundle_root_invariants key" or "the analysis config's report field is required".
    """
    msg = _message(tmp_path, manifest, exc_type)
    assert _INTERNAL_LABEL.findall(msg) == [], msg
    missing = [f for f in facts if not re.search(f, msg, re.IGNORECASE)]
    assert missing == [], (missing, msg)
