"""Reviewer pins for the stale-figure check: classes of tree the coder's own tests leave to one instance.

Every test writes ``tmp_path`` files only and calls ``_check_stale_figures`` directly; none starts a
solver or requests a ``synth_*`` fixture. All of them are GREEN on the tree they were written
against and exist to stay green: each names the single change to ``_stale_member_of`` or
``_check_stale_figures`` that turns it red, and that change was applied to a copy of the source and
run before the test was kept.

A STALE FIGURE here means a non-sidecar file at any depth under
``plots/sensitivity/per_sim/member-{N}/`` whose ``N`` is not among the ids of the live member set.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from hhemt.bundle._emit import _check_stale_figures
from hhemt.exceptions import ConfigurationError

PER_SIM = "sensitivity/per_sim"


def _figure(plots: Path, rel: str, *, sidecar: bool = True) -> Path:
    """Write a figure file, and by default its stem-named ``.manifest.json`` sidecar."""
    figure = plots / rel
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_text(f"figure {rel}")
    if sidecar:
        (figure.parent / f"{figure.stem}.manifest.json").write_text(json.dumps({"plot_id": figure.stem}))
    return figure


def _master(live_ids: list[int], setup_file: Path) -> SimpleNamespace:
    """A master-shaped stand-in: the live ids at ``.sensitivity.df_setup`` plus the setup-file path."""
    df_setup = pd.DataFrame({"v": [1] * len(live_ids)}, index=pd.Index(live_ids, name="member_id"))
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(sensitivity_analysis=setup_file),
        sensitivity=SimpleNamespace(df_setup=df_setup),
    )


def _run(tmp_path: Path, build, live_ids: list[int]):
    """Build a tree with ``build(plots)``, run the check, return ``(analysis_dir, analysis)``."""
    analysis_dir = tmp_path / "analysis"
    plots = analysis_dir / "plots"
    plots.mkdir(parents=True)
    build(plots)
    setup_file = tmp_path / "setup.csv"
    setup_file.write_text("member_id\n")
    return analysis_dir, _master(live_ids, setup_file)


def _check(analysis, analysis_dir: Path, allow: bool):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _check_stale_figures(analysis, analysis_dir, analysis_dir / "plots", allow)


@pytest.mark.parametrize(
    "not_a_member_figure",
    [
        "other/member-99/fig.html",
        "per_sim/member-99/fig.html",
        "sensitivity/summary/member-99/fig.html",
        f"{PER_SIM}/notmember-99/fig.html",
    ],
)
def test_a_figure_outside_the_sensitivity_per_sim_member_directories_is_never_stale(tmp_path, not_a_member_figure):
    """Property: only a file under ``plots/sensitivity/per_sim/member-{N}/`` can be stale.

    Class: every figure path that carries a ``member-99`` segment, or a segment ending in it, but
    is not below ``plots/sensitivity/per_sim/member-{N}/``. Id 99 is not live.
    Why: a stale figure is defined by the per-member directory the sensitivity builders write;
    a refusal on any other ``member-*`` path would stop an emit over a figure of no removed
    member. Kills: dropping the leading ``sensitivity/per_sim`` prefix test in
    ``_stale_member_of`` and matching any ``member-*`` path part, or testing ``"member-" in`` the
    directory name instead of ``startswith`` (the ``notmember-99`` case). A second correct implementation
    walks only ``plots/sensitivity/per_sim`` and tests its immediate children; it passes this test.
    """

    def build(plots: Path) -> None:
        _figure(plots, f"{PER_SIM}/member-22/live.html")
        _figure(plots, not_a_member_figure)

    analysis_dir, analysis = _run(tmp_path, build, [22])
    assert _check(analysis, analysis_dir, False) == []


def test_the_member_directory_position_is_what_makes_a_figure_stale_control(tmp_path):
    """Must-fire control for the class above: the SAME file name under the member directory refuses.

    Property: moving ``fig.html`` from outside to inside ``plots/sensitivity/per_sim/member-99/``
    flips the verdict from no refusal to a refusal. Class: one figure, two positions. Why: a
    control that varies the path position (not the token ``member-99``) shows the matcher above
    can fire at all. Kills: a ``_stale_member_of`` that never returns a member id. A second
    correct implementation is any detector keyed on the same directory; it passes.
    """

    def build(plots: Path) -> None:
        _figure(plots, f"{PER_SIM}/member-22/live.html")
        _figure(plots, f"{PER_SIM}/member-99/fig.html")

    analysis_dir, analysis = _run(tmp_path, build, [22])
    with pytest.raises(ConfigurationError) as raised:
        _check(analysis, analysis_dir, False)
    assert raised.value.field == "bundle_stale_figures"


def test_a_stale_figure_is_found_at_every_depth_below_its_member_directory(tmp_path):
    """Property: depth below ``member-{N}/`` does not change whether a figure is stale.

    Class: figures directly in the member directory, one directory down, and three directories
    down. Why: the builders write ``member-{N}/{event}/...`` today, and a figure written
    directly in the member directory, or under a deeper event grouping, is still a figure of that
    removed member. Kills: requiring an exact path depth in ``_stale_member_of`` (the coder's
    fixtures all sit at one depth, so they cannot tell). A second correct implementation measures
    ``Path.parts`` only for the ``member-`` prefix and counts every descendant file; it passes.
    """
    rels = [
        f"{PER_SIM}/member-16/flat.html",
        f"{PER_SIM}/member-16/e0/one_down.html",
        f"{PER_SIM}/member-16/e0/a/b/three_down.html",
    ]

    def build(plots: Path) -> None:
        _figure(plots, f"{PER_SIM}/member-22/live.html")
        for rel in rels:
            _figure(plots, rel)

    analysis_dir, analysis = _run(tmp_path, build, [22])
    assert sorted(_check(analysis, analysis_dir, True)) == sorted(f"plots/{r}" for r in rels)


def test_a_stale_figure_with_no_sidecar_is_stale_and_a_lone_sidecar_is_not(tmp_path):
    """Property: staleness is a property of the figure file; a sidecar never makes a directory stale.

    Class: a stale figure whose sidecar is missing, and a stale member directory holding only
    a ``.manifest.json``. Why: a regenerable figure is a non-sidecar file; a sidecar travels with
    its figure and an orphan sidecar is not a figure. Kills: detecting staleness by walking
    ``*.manifest.json`` files (the first half goes red) or by counting any file (the second half
    goes red). A second correct implementation walks figures and looks sidecars up by stem;
    it passes.
    """

    def build(plots: Path) -> None:
        _figure(plots, f"{PER_SIM}/member-22/live.html")
        _figure(plots, f"{PER_SIM}/member-16/no_sidecar.html", sidecar=False)
        orphan = plots / f"{PER_SIM}/member-7/gone.manifest.json"
        orphan.parent.mkdir(parents=True)
        orphan.write_text("{}")

    analysis_dir, analysis = _run(tmp_path, build, [22])
    assert _check(analysis, analysis_dir, True) == [f"plots/{PER_SIM}/member-16/no_sidecar.html"]


def test_every_stale_member_is_named_with_its_own_figure_count(tmp_path):
    """Property: the refusal reports every stale member, each with its own count, and the total.

    Class: two stale members holding different numbers of figures (member-16 holds 2, member-3
    holds 1). Why: the message is the user's only account of what is stale; naming only one
    member, or one count for all, sends the user to fix part of it. Kills: reporting only the
    first stale member, or giving every member the total. A second correct implementation lists
    members in any order; this test does not pin the order.
    """

    def build(plots: Path) -> None:
        _figure(plots, f"{PER_SIM}/member-22/live.html")
        _figure(plots, f"{PER_SIM}/member-16/a.html")
        _figure(plots, f"{PER_SIM}/member-16/b.html")
        _figure(plots, f"{PER_SIM}/member-3/c.html")

    analysis_dir, analysis = _run(tmp_path, build, [22])
    with pytest.raises(ConfigurationError) as raised:
        _check(analysis, analysis_dir, False)
    text = str(raised.value)
    assert "3 stale figure file(s)" in text
    assert "2 member(s)" in text
    counts = dict(re.findall(r"plots/sensitivity/per_sim/member-(\d+)/\s+\((\d+) figure files?\)", text))
    assert counts == {"16": "2", "3": "1"}


_STALE_NAMES = [
    "peak_flood_depth__member.16.html",
    "retired_family__member.16.html",
    "a.b.c.html",
    "plain_no_double_underscore.html",
    "Peak_Flood_Depth__member.16.html",
    "wild__member.16__evt.e0.svg",
]

_SNAKEFILES = {
    "one-literal": 'rule a:\n    output:\n        "plots/peak_flood_depth__x.html",\n',
    "wildcard-family": 'rule a:\n    output:\n        "plots/{kind}__member.{m}.html",\n',
    "dotted-stem": 'rule a:\n    output:\n        "plots/a.b.c.html",\n        "plots/wild__member.{m}.svg",\n',
    "no-figure-rules": "rule all:\n    input:\n        []\n",
    "absent": None,
}


@pytest.mark.parametrize("variant", list(_SNAKEFILES))
def test_the_check_and_the_prune_partition_the_stale_figures_of_any_snakefile(tmp_path, variant):
    """Property: every stale figure is listed by the check XOR deleted by the prune, never both, never neither.

    Class: Snakefiles whose declared families are a literal, a wildcard, a dotted stem, none at all,
    or which are absent (the prune then skips), over stale-member figures of declared, undeclared,
    differently-cased, dotted and double-underscore-free names. Why: the override's warning and the
    manifest key must name files that reach the bundle, and the prune's `pruned_orphan_figures` must
    not repeat them. Killed by: a check that spares by a differently spelled family test (case folded,
    prefix match), that ignores a missing Snakefile, or that spares a declared family. Second correct
    implementation: the check calls `_prune_undeclared_figures` on a scratch copy and subtracts; it passes.
    """
    from hhemt.bundle._emit import _prune_undeclared_figures

    analysis_dir = tmp_path / "analysis"
    plots = analysis_dir / "plots"
    plots.mkdir(parents=True)
    text = _SNAKEFILES[variant]
    if text is not None:
        (analysis_dir / "Snakefile").write_text(text)
    _figure(plots, f"{PER_SIM}/member-22/live.html")
    stale_all = {f"plots/{PER_SIM}/member-16/e0/{n}" for n in _STALE_NAMES}
    for n in _STALE_NAMES:
        _figure(plots, f"{PER_SIM}/member-16/e0/{n}")
    setup = tmp_path / "setup.csv"
    setup.write_text("member_id\n")
    analysis = _master([22], setup)
    listed = set(_check(analysis, analysis_dir, True))
    pruned = {r for r in _prune_undeclared_figures(analysis_dir, plots) if r in stale_all}
    assert listed & pruned == set(), "a listed file was deleted by the prune"
    assert listed | pruned == stale_all, "a stale figure is neither listed nor pruned"
    for rel in listed:
        assert (analysis_dir / rel).is_file(), "a listed file does not survive to the bundle"
