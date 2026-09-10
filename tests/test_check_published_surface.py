"""Regression tests for scripts/check_published_surface.py.

The leaf's four guards are covered here as PURE functions plus two CLI arms.
Nothing here builds a site: `fidelity_findings` and `baseline_findings` take the
anchor set and the population as arguments, which is why the leaf was written
with them separated from `main()` in the first place.

The fixtures are SYNTHETIC rather than the live 183-name population. A fixture
keyed on the live surface would go red on any legitimate API change, which would
make it a measurement of the surface rather than of the check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "check_published_surface.py"


def _load():
    sys.path.insert(0, str(_REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("check_published_surface", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_published_surface"] = mod
    spec.loader.exec_module(mod)
    return mod


cps = _load()

MANIFESTED = ("pkg", "pkg.quiet", "pkg.loud")
POPULATION = {"pkg", "pkg.loud", "pkg.loud.Thing", "pkg.loud.Thing.method"}


def test_the_explained_exemption_is_derived_not_pinned():
    """MIRROR B's non-risk class must retire itself.

    `pkg.quiet` is manifested and absent from the population, so it is explained.
    Give it a docstring -- modelled here by putting it IN the population -- and
    the exemption must disappear on its own. A name-pinned exemption passes the
    first arm and fails the second, which is the whole reason this is computed.
    """
    assert cps._explained_module_anchors(POPULATION, MANIFESTED) == {"pkg.quiet"}
    assert cps._explained_module_anchors(POPULATION | {"pkg.quiet"}, MANIFESTED) == set()


def test_a_derived_name_the_site_does_not_render_is_a_finding():
    """MIRROR A. The gate would be scanning prose the page no longer has, so its
    own "scanned N rendered docstring(s)" line becomes a false statement.
    """
    anchors = {"pkg", "pkg.quiet", "pkg.loud", "pkg.loud.Thing"}
    out = cps.fidelity_findings(POPULATION, anchors, MANIFESTED)
    assert out == ["derived but NOT rendered: pkg.loud.Thing.method"]


def test_a_rendered_anchor_outside_the_explained_class_is_a_finding():
    """MIRROR B, and its complement in one test.

    `pkg.quiet` is a manifested module with no docstring and must stay silent;
    `pkg.loud.Thing.extra` is a rendered member the derivation never reached and
    must fire. Asserting only the first would pass on a check that reports
    nothing at all.
    """
    anchors = {"pkg", "pkg.quiet", "pkg.loud", "pkg.loud.Thing", "pkg.loud.Thing.method", "pkg.loud.Thing.extra"}
    out = cps.fidelity_findings(POPULATION, anchors, MANIFESTED)
    assert out == ["rendered but NOT derived: pkg.loud.Thing.extra"]


def test_the_reconciled_state_produces_no_finding():
    """The differently-positioned satisfying arm: a site carrying every derived
    name PLUS the explained module anchor is a correct state, and it is not the
    state either finding arm was written against.
    """
    anchors = POPULATION | {"pkg.quiet"}
    assert cps.fidelity_findings(POPULATION, anchors, MANIFESTED) == []


def test_a_missing_baseline_is_loud_rather_than_skipped(tmp_path):
    """An absent baseline must not read as a clean surface. This is the same
    argument as `--site-dir required=True`: a skipped check is byte-identical to
    a passing one, so the skip has to be spelled.
    """
    out = cps.baseline_findings(POPULATION, tmp_path / "nope.txt")
    assert len(out) == 1 and "no baseline" in out[0]


def test_a_widened_surface_names_the_added_symbol(tmp_path):
    """Both directions, because a count-only comparison passes whenever one name
    is added and another removed.
    """
    b = tmp_path / "baseline.txt"
    b.write_text("\n".join(sorted(POPULATION - {"pkg.loud.Thing.method"})) + "\n", encoding="utf-8")
    out = cps.baseline_findings(POPULATION, b)
    assert out == ["ADDED to the published surface, not in the baseline: pkg.loud.Thing.method"]

    b.write_text("\n".join(sorted(POPULATION | {"pkg.loud.Thing.gone"})) + "\n", encoding="utf-8")
    out = cps.baseline_findings(POPULATION, b)
    assert out == ["REMOVED from the published surface, still in the baseline: pkg.loud.Thing.gone"]

    b.write_text("\n".join(sorted(POPULATION)) + "\n", encoding="utf-8")
    assert cps.baseline_findings(POPULATION, b) == []


def test_a_blank_line_in_the_baseline_is_not_a_missing_symbol(tmp_path):
    """The baseline is written with a trailing newline, so a naive `splitlines`
    reader would carry an empty string and report it REMOVED on every run.
    """
    b = tmp_path / "baseline.txt"
    b.write_text("\n".join(sorted(POPULATION)) + "\n\n  \n", encoding="utf-8")
    assert cps.baseline_findings(POPULATION, b) == []


def test_the_site_dir_is_required_rather_than_optional():
    """A skipped fidelity check is byte-identical to a clean one, so the argument
    parser refuses the invocation rather than letting the check no-op.
    """
    with pytest.raises(SystemExit) as exc:
        cps.main([])
    assert exc.value.code == 2


def test_an_absent_site_dir_is_a_usage_error_not_a_finding(tmp_path):
    """Exit 2 rather than 1: a site that was never built is an environment
    problem and must not be reported as a surface change.
    """
    assert cps.main(["--site-dir", str(tmp_path / "no_such_site")]) == 2
