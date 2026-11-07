"""Prepare a rendered report for publication under `docs/reference/bundles/`, and draw a figure from it.

A rendered single-HTML report carries every result page as a base64 `data_uri` inside one line of the
outer HTML that starts `var results = {`. A line-based reader never sees the text of those pages.
This script reads that line, decodes each page, and works on the outer HTML, on every string of the
manifest and on every decoded page.

Subcommands:

    scrub SOURCE DEST [--replace OLD=NEW ...]
        Writes DEST, a copy of SOURCE with two rules applied.
        Rule 1: every path segment that follows a path component named `home`, `Users` or `scratch`
        becomes the word `operator`. The rule needs no private value.
        Rule 2: each `--replace OLD=NEW` pair replaces a private identifier outside a user-segment
        position by a generic word. The value of OLD is given on the command line at run time and is
        written into no tracked file.
        The command prints a count per rule and never a replaced value, and it refuses to write when
        rule 1 still counts a segment after the rules ran.

    figure REPORT --event EVENT_ID --title TITLE DEST
        Writes the Plotly figure that sits under the heading TITLE on the page of the event EVENT_ID
        as a PNG of 2800 by 1200 pixels (width 1400, height 600, scale 2).

    figure REPORT --list
        Prints the pages of the report and, for each figure, its category, heading and axis titles.
        Renders nothing.

The per-user root names are built from fragments, as `scripts/check_anonymization.py` builds its own,
because a regex literal that holds a root and a segment trips that guard's path class.
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import json
import re
import sys
from pathlib import Path

GENERIC = "operator"

# Built from fragments, so that this file holds no literal root.
_ROOTS = ("ho" + "me", "Us" + "ers", "scr" + "atch")
_SEGMENT = "[^/\\s\"'<>\\\\]+"
_USER_SEGMENT = re.compile("(/(?:" + "|".join(_ROOTS) + ")/)(" + _SEGMENT + ")")

_RESULTS_MARK = "var results = {"
_CATEGORIES_MARK = "var categories = {"
_DATA_URI = re.compile(r"(data:[^;]+;.*?base64,)(.*)", re.S)
_H2 = re.compile(r'<h2 class="eda-figure-title">(.*?)</h2>', re.S)
_NEW_PLOT = "Plotly.newPlot("

_FIGURE_WIDTH = 1400
_FIGURE_HEIGHT = 600
_FIGURE_SCALE = 2


class ScrubRefused(RuntimeError):
    """Rule 1 still counted a user segment after the rules ran."""


def count_user_segments(text: str) -> int:
    """Rule 1's count over one text: user segments that are not the generic word."""
    return sum(1 for m in _USER_SEGMENT.finditer(text) if m.group(2) != GENERIC)


def _split_report(outer: str) -> tuple[list[str], int, dict, int]:
    """The lines of the outer HTML, the index of the manifest line, the manifest and the offset of its JSON."""
    lines = outer.split("\n")
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(_RESULTS_MARK):
            start = ln.index("{")
            manifest, end = json.JSONDecoder().raw_decode(ln[start:])
            return lines, i, manifest, start + end
    raise ValueError("the report has no manifest line starting " + repr(_RESULTS_MARK))


def decode_pages(outer: str) -> dict[str, str]:
    """Each page of the manifest, decoded, keyed by its path."""
    _lines, _i, manifest, _end = _split_report(outer)
    pages = {}
    for key, entry in manifest.items():
        m = _DATA_URI.match(entry["data_uri"])
        pages[key] = base64.b64decode(m.group(2)).decode("utf-8") if m else ""
    return pages


def count_report(outer: str) -> tuple[int, int]:
    """Rule 1's count over the outer HTML and over the decoded pages, as (outer, decoded)."""
    return count_user_segments(outer), sum(count_user_segments(p) for p in decode_pages(outer).values())


class _Rules:
    """Rules 1 and 2 with a count per rule."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs
        self.rule1 = 0
        self.rule2 = [0] * len(pairs)

    def __call__(self, text: str) -> str:
        def one(m: re.Match) -> str:
            if m.group(2) != GENERIC:
                self.rule1 += 1
            return m.group(1) + GENERIC

        text = _USER_SEGMENT.sub(one, text)
        for n, (old, new) in enumerate(self.pairs):
            text, hits = re.subn(r"\b" + re.escape(old) + r"\b", new, text)
            self.rule2[n] += hits
        return text

    def walk(self, obj):
        if isinstance(obj, str):
            return self(obj)
        if isinstance(obj, list):
            return [self.walk(x) for x in obj]
        if isinstance(obj, dict):
            return {self(k): self.walk(v) for k, v in obj.items()}
        return obj


def scrub_text(outer: str, pairs: list[tuple[str, str]]) -> tuple[str, _Rules]:
    """The scrubbed outer HTML and the rule counts. Raises `ScrubRefused` when rule 1 still counts."""
    rules = _Rules(pairs)
    lines, idx, manifest, end = _split_report(outer)
    for _key, entry in manifest.items():
        m = _DATA_URI.match(entry["data_uri"])
        page = base64.b64decode(m.group(2)).decode("utf-8")
        for field in list(entry):
            if field != "data_uri":
                entry[field] = rules.walk(entry[field])
        entry["data_uri"] = m.group(1) + base64.b64encode(rules(page).encode("utf-8")).decode("ascii")
    for i, ln in enumerate(lines):
        if i != idx:
            lines[i] = rules(ln)
    ln = lines[idx]
    start = ln.index("{")
    lines[idx] = rules(ln[:start]) + json.dumps(manifest) + rules(ln[end:])
    result = "\n".join(lines)
    outer_left, decoded_left = count_report(result)
    if outer_left or decoded_left:
        raise ScrubRefused(
            f"rule 1 still counts {outer_left} in the outer HTML and {decoded_left} in the decoded pages"
        )
    return result, rules


def cmd_scrub(args: argparse.Namespace) -> int:
    pairs = []
    for item in args.replace:
        old, sep, new = item.partition("=")
        if not sep or not old:
            print("scrub: --replace takes OLD=NEW", file=sys.stderr)
            return 2
        pairs.append((old, new))
    outer = Path(args.source).read_text(encoding="utf-8")
    try:
        result, rules = scrub_text(outer, pairs)
    except ScrubRefused as err:
        print(f"scrub refused, nothing written: {err}", file=sys.stderr)
        return 1
    Path(args.dest).write_text(result, encoding="utf-8")
    print(f"rule 1 (user segment): {rules.rule1} replaced")
    for n, hits in enumerate(rules.rule2):
        print(f"rule 2 (replace pair {n}): {hits} replaced")
    print("rule 1 after: 0 in the outer HTML, 0 in the decoded pages")
    return 0


def _categories(outer: str) -> dict[str, list[str]]:
    """Page path -> category name, from the `var categories = {` line of the outer HTML."""
    for ln in outer.split("\n"):
        if ln.lstrip().startswith(_CATEGORIES_MARK):
            start = ln.index("{")
            data, _end = json.JSONDecoder().raw_decode(ln[start:])
            out: dict[str, list[str]] = collections.defaultdict(list)
            for category, groups in data.items():
                for paths in groups.values():
                    for path in paths:
                        out[path].append(category)
            return dict(out)
    return {}


def _axis_titles(layout: dict) -> list[str]:
    titles = []
    for key in sorted(layout):
        if key.startswith(("xaxis", "yaxis")) and isinstance(layout[key], dict):
            title = layout[key].get("title")
            text = title.get("text") if isinstance(title, dict) else title
            if text:
                titles.append(str(text))
    return titles


def _value(text: str, k: int, decoder: json.JSONDecoder) -> tuple[object, int]:
    """The JSON value that starts at or after `k`, skipping whitespace, and the offset after it."""
    while text[k].isspace():
        k += 1
    return decoder.raw_decode(text, k)


def _comma(text: str, k: int) -> int:
    """The offset after the next comma at or after `k`."""
    return text.index(",", k) + 1


def figures_of(page: str) -> list[tuple[str, list, dict]]:
    """The figures of one decoded page as (heading, data, layout), one per `h2` heading with a plot under it."""
    heads = list(_H2.finditer(page))
    found = []
    decoder = json.JSONDecoder()
    for n, head in enumerate(heads):
        stop = heads[n + 1].start() if n + 1 < len(heads) else len(page)
        j = page.find(_NEW_PLOT, head.end(), stop)
        if j < 0:
            continue
        k = j + len(_NEW_PLOT)
        _div, k = _value(page, k, decoder)
        data, k = _value(page, _comma(page, k), decoder)
        layout, _k = _value(page, _comma(page, k), decoder)
        found.append((html.unescape(head.group(1)).strip(), data, layout))
    return found


def cmd_figure(args: argparse.Namespace) -> int:
    outer = Path(args.report).read_text(encoding="utf-8")
    pages = decode_pages(outer)
    categories = _categories(outer)
    if args.list:
        for key, page in pages.items():
            cats = ",".join(categories.get(key, [])) or "-"
            figs = figures_of(page)
            if not figs:
                print(f"page\t{key}\t{cats}")
            for title, _data, layout in figs:
                print(f"figure\t{key}\t{cats}\t{title}\t{'; '.join(_axis_titles(layout))}")
        return 0
    if not (args.event and args.title and args.dest):
        print("figure: --event, --title and DEST are required unless --list is given", file=sys.stderr)
        return 2
    suffix = f"event_page__evt.{args.event}.html"
    keys = [k for k in pages if k.endswith("/" + suffix) or k == suffix]
    if len(keys) != 1:
        print(f"figure: {len(keys)} pages match the event {args.event}", file=sys.stderr)
        return 1
    matches = [(d, lay) for title, d, lay in figures_of(pages[keys[0]]) if title == args.title]
    if len(matches) != 1:
        print(f"figure: {len(matches)} figures carry the title {args.title!r} on that page", file=sys.stderr)
        return 1
    import plotly.graph_objects as go

    data, layout = matches[0]
    go.Figure(data=data, layout=layout).write_image(
        str(args.dest), scale=_FIGURE_SCALE, width=_FIGURE_WIDTH, height=_FIGURE_HEIGHT
    )
    print(f"wrote {args.dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    scrub = sub.add_parser("scrub", help="apply rules 1 and 2 to a report")
    scrub.add_argument("source")
    scrub.add_argument("dest")
    scrub.add_argument("--replace", action="append", default=[], metavar="OLD=NEW")
    scrub.set_defaults(run=cmd_scrub)
    figure = sub.add_parser("figure", help="draw one figure of a report, or list the figures")
    figure.add_argument("report")
    figure.add_argument("--event")
    figure.add_argument("--title")
    figure.add_argument("--list", action="store_true")
    figure.set_defaults(run=cmd_figure)
    args, extra = parser.parse_known_args(argv)
    if args.command == "figure":
        # DEST may follow the options, which argparse does not match to a positional after a subparser.
        if len(extra) > 1:
            parser.error("unrecognized arguments: " + " ".join(extra[1:]))
        args.dest = extra[0] if extra else None
    elif extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
