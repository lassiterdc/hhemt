"""Check that the census's emitted rung-1 block is the ladder's rung 1 and not a wider population.

The problem: the approved shape's call-site ladder defines rung 1 as "any actuator attribute, any
receiver", and R1's Definition of Done requires the census to emit "the rung-1 count and site
list". The census walks two kinds of site -- an attribute call (``.run(``, ``.submit_workflow(``,
``.test(``, ``.reprocess(``, ``.submit_reprocess_workflow(``) and a command-line invocation that
NAMES a command whose callback reaches one of those -- because run-reachability needs both. Folding
the second kind into the rung-1 block inflates it by every CLI invocation in the tree, so the
emitted rung 1 is then a population the ladder never defined and no seat ever measured, and the
figure cannot be compared with the ladder row it is published beside.

How this module checks it: it runs the census in a child pytest with the artifact redirected to a
temporary path, then reads the emitted ``rung_1`` block and asserts every site it counts is an
attribute site. Only collection happens inside the census child; nothing compiles and no planted
node is executed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CENSUS_REL = "tests/test_build_request_census.py"
_ARTIFACT_ENV = "HHEMT_BUILD_REQUEST_CENSUS_ARTIFACT"


def _run_census(artifact: Path) -> dict:
    """Run the census with its artifact redirected, and return the parsed artifact."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST")}
    env["HHEMT_FORBID_COMPILE"] = "1"
    env[_ARTIFACT_ENV] = str(artifact)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", _CENSUS_REL, "-p", "no:cacheprovider", "-q"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if not artifact.is_file():
        raise AssertionError(
            f"the census wrote no artifact at {artifact}, so it did not run:\n" + (proc.stdout + proc.stderr)[-4000:]
        )
    return json.loads(artifact.read_text(encoding="utf-8"))


def test_rung_1_counts_attribute_sites_only(tmp_path):
    """Property: the emitted rung-1 block counts attribute call sites, not command-line invocations.

    Class: every collected tree in which some run-reaching node reaches its run only through a
    command-line invocation -- ``invoke(app, [...])`` in process, or a child ``hhemt`` /
    ``python -m hhemt`` call -- whose named command's callback reaches a run. Each such node
    contributes a "cli" site to the artifact, and a rung-1 block that counts it is counting a
    population the ladder's rung 1 does not define. Why this is required, as a consequence
    checkable in the code alone: ``_rung_artifact`` takes ``_unique_sites(nodes)``, which carries
    both the "attr" and the "cli" kind, so the emitted count is the union of two site kinds while
    the ladder row published beside it is one kind; a reader comparing the two is comparing a
    mixed figure against an attribute-only one and has no key in the artifact that separates them.
    What kills it: restricting the rung-1 population to sites whose kind is "attr" -- either by
    filtering ``_unique_sites`` inside ``_rung_artifact`` or by building rung 1 from the walk's own
    attribute calls before ``finalize`` appends the command-line sites. A second correct
    implementation under which this still passes: one that selects the rung-1 population by
    ``_RUN_ATTRS`` membership of each site's ``attr`` field rather than by the site's ``kind``,
    since a command-line site carries no attribute name -- the assertion turns on which sites are
    COUNTED, not on the field that excluded them.

    Interface premise, enforced: every rung-1 site carries its ``kind``, because the ladder
    predicate is defined over site KINDS and an artifact that carries the kind nowhere cannot be
    checked against the ladder at all. The first assertion enforces it, so an artifact that drops
    the field reds here by construction rather than passing this check vacuously.
    """
    artifact = _run_census(tmp_path / "census_artifact.json")
    rung_1 = artifact["rung_1"]
    kinds = [site.get("kind") for site in rung_1["sites"]]
    cli_sites = [site for site in rung_1["sites"] if site.get("kind") == "cli"]
    assert kinds and all(kind is not None for kind in kinds), (
        "the rung-1 sites carry no kind field, so the ladder's rung-1 predicate cannot be checked "
        f"against the emitted block: count={rung_1['count']}, unit={rung_1['unit']}"
    )
    assert not cli_sites, (
        f"the ladder's rung 1 is 'any actuator attribute, any receiver', but the emitted rung_1 "
        f"block counts {rung_1['count']} sites of which {len(cli_sites)} are command-line "
        f"invocations ({len({s['where'].rsplit(':', 1)[0] for s in cli_sites})} files). Attribute "
        f"sites in the same block: {len(kinds) - len(cli_sites)}. Either the rung-1 block must "
        f"count attribute sites only, or the ladder row it is published against must be restated "
        f"to name the wider population -- and no key in the artifact currently separates the two."
    )
