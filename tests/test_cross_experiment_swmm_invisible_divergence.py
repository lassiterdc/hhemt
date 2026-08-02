"""VMS-S4: the TRITON-visible / SWMM-invisible clean-vs-resume signature.

The finding fires on a PATTERN, not on a mere non-identity: every coupled config's
`max_wlevel_m` differs clean-vs-resume while every config's `max_flow_cms` is identical.
That asymmetry is the fingerprint of the SWMM node-depth scatter defect on the resume path,
and it is invisible to every SWMM-side artifact.

The negative controls are the load-bearing half. An assertion that only checked "the finding
appears on the campaign payload" would pass for an implementation that fires on ANY
non-identity — which would relabel ordinary clean-vs-resume differences as a solver defect.
Each control below occupies a DIFFERENT correct state, so together they pin the pattern
rather than one satisfying position.
"""

from __future__ import annotations

import json

from hhemt.report_renderers.cross_experiment_errors_and_warnings import (
    _swmm_invisible_divergence_finding,
)


def _write(root, pairs):
    (root / "combined_intercomparison.json").write_text(
        json.dumps({"pairs": pairs}), encoding="utf-8"
    )
    return root


def _pair(model, variable, identical):
    return {"model": model, "variable": variable, "identical": identical}


def test_fires_on_the_depth_differs_flow_identical_signature(tmp_path):
    """VIOLATING input: 14/14 depth differ, 0/14 flow differ -> the finding."""
    pairs = [_pair("TRITON-SWMM", "max_wlevel_m", False) for _ in range(14)]
    pairs += [_pair("TRITON-SWMM", "max_flow_cms", True) for _ in range(14)]
    finding = _swmm_invisible_divergence_finding(_write(tmp_path, pairs))

    assert finding is not None
    assert finding["passed"] is False
    assert finding["level"] == "aggregate"
    # The denominators must be NAMED, per the disclosed-denominator rule.
    assert "14/14" in finding["summary"]
    assert "0/14 differ" in finding["summary"]


def test_silent_when_both_variables_differ(tmp_path):
    """Differently-positioned satisfying input: no ASYMMETRY, so no signature.

    Both variables differing is an ordinary clean-vs-resume divergence. An implementation
    keyed on 'depth differs' alone would fire here and misattribute it to the scatter defect.
    """
    pairs = [_pair("TRITON-SWMM", "max_wlevel_m", False) for _ in range(14)]
    pairs += [_pair("TRITON-SWMM", "max_flow_cms", False) for _ in range(14)]
    assert _swmm_invisible_divergence_finding(_write(tmp_path, pairs)) is None


def test_silent_when_both_variables_are_identical(tmp_path):
    """Differently-positioned satisfying input: a clean reproduction owes no finding."""
    pairs = [_pair("TRITON-SWMM", "max_wlevel_m", True) for _ in range(14)]
    pairs += [_pair("TRITON-SWMM", "max_flow_cms", True) for _ in range(14)]
    assert _swmm_invisible_divergence_finding(_write(tmp_path, pairs)) is None


def test_silent_on_a_pure_triton_only_payload(tmp_path):
    """Differently-positioned satisfying input: no coupled flow variable exists at all.

    A pure-TRITON run has no SWMM conduits, so max_flow_cms is absent by construction and
    the asymmetry is not even expressible — the finding must not fire on half a pattern.
    """
    pairs = [_pair("TRITON", "max_wlevel_m", False) for _ in range(14)]
    assert _swmm_invisible_divergence_finding(_write(tmp_path, pairs)) is None


def test_silent_when_the_read_model_is_absent_or_malformed(tmp_path):
    """Graceful-absent: a missing or unparseable read-model yields None, never a raise.

    The renderer runs inside a Snakemake rule; raising here would kill the report over a
    derived annotation.
    """
    assert _swmm_invisible_divergence_finding(tmp_path) is None
    (tmp_path / "combined_intercomparison.json").write_text("{not json", encoding="utf-8")
    assert _swmm_invisible_divergence_finding(tmp_path) is None
