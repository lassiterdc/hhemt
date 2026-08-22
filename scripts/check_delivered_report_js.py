"""Every `<script>` in a delivered report must COMPILE. Byte probes cannot see this.

WHY THIS EXISTS, stated as the failure it catches rather than as the rule it asserts.

A rendered hhemt report embeds each result page as a base64 `data_uri`, and the combined
report nests arm pages one level deeper inside `<iframe srcdoc>`. Every acceptance check
this campaign had read the TEXT of that emitted code -- is the bundle present, is the
control label there, are there data rows. A PARSE error is a property of the code AS A
PROGRAM, so none of those checks can distinguish a script that runs from one the browser
cannot compile.

Measured 2026-08-21: each Tabulator table fragment opened with a top-level
`const tableOptions`, and a multi-table page concatenates N fragments into ONE `<script>`.
N>=2 is a redeclaration -- a parse-time SyntaxError -- and a classic script that fails to
parse executes NONE of its statements, including the mount code that clones the
`<template>` markup into the DOM. Two whole report sections rendered EMPTY on all five
reports, twice, while three separate byte-level instruments returned exit 0.

This needs NO browser and NO install: node's `new Function(src)` compiles without
executing. Run it against a delivered set before putting that set in front of a reviewer.

Usage:
    python scripts/check_delivered_report_js.py [--dest DIR] [--quiet]

    --dest  directory holding the delivered *.html reports
            (default: $HHEMT_QA_DEST, else ~/Downloads/cc_resume_report_qa)

Exit 0 = every script compiles. Exit 1 = at least one does not. Exit 2 = cannot run
(no node, no reports) -- deliberately DISTINCT from 1, because "the check could not run"
must never be readable as "the check passed" or as "the artifact is broken".
"""

from __future__ import annotations

import argparse
import base64
import html as _html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: A result object's own declared path. Keying on the PATH rather than the basename is
#: load-bearing for the combined report, where ~66 basenames recur across experiments and a
#: basename-keyed decoder silently keeps whichever payload it reached last.
_RESULT_KEY = re.compile(r'"([^"]+\.html)":\s*\{')
_URI_ANY = re.compile(r'"data_uri":\s*"data:text/html;charset=utf8;filename=([^;]+);base64,([A-Za-z0-9+/=]+)"')
_SCRIPT = re.compile(r"<script\b[^>]*>(.*?)</script>", re.S)
_SRCDOC = re.compile(r'srcdoc="(.*?)"></iframe>', re.S)

_REPORTS = (
    "clean_triton_report.html",
    "clean_tritonswmm_report.html",
    "resume_triton_report.html",
    "resume_tritonswmm_report.html",
    "combined_BOTH_MODELS_report.html",
)

_PROBE = (
    "const fs=require('fs');\n"
    # argv[2], NOT argv[1]. Under `node probe.js target.js` argv[1] is probe.js itself,
    # which parses fine -- so an argv[1] probe reads ITSELF and reports success for any
    # target. That silent-pass bug was caught by a control test, not by review.
    "try { new Function(fs.readFileSync(process.argv[2],'utf8')); }\n"
    "catch (e) { console.error(e.message); process.exit(1); }\n"
)


def decoded_by_path(text: str) -> dict[str, str | None]:
    """Return {declared_path: decoded_html}, collision-free, absences visible as None."""
    spans = [(m.group(1), m.start()) for m in _RESULT_KEY.finditer(text)]
    out: dict[str, str | None] = {}
    for i, (path, start) in enumerate(spans):
        end = spans[i + 1][1] if i + 1 < len(spans) else len(text)
        m = _URI_ANY.search(text, start, end)
        if not m:
            out[path] = None
            continue
        try:
            out[path] = base64.b64decode(m.group(2)).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - a decode failure is data, not a crash
            out[path] = f"<<DECODE-FAILED {exc}>>"
    return out


def child_docs(page_html: str) -> list[str]:
    """A paired page nests each arm in <iframe srcdoc>; unescape ONLY that payload.

    NEVER unescape the page itself. A page that is not srcdoc-wrapped is already what the
    browser parses, and unescaping it CORRUPTS the scripts: the vendored Tabulator bundle
    contains a literal `&quot;` and the toolkit's own table script contains dozens, all
    inside JS string literals. Unescaping turns them into bare `"` and yields a spurious
    `SyntaxError: Unexpected string` against code the browser compiles fine -- a FALSE
    DEFECT reported against correct work, which is the direction that accuses.
    """
    kids = [_html.unescape(m) for m in _SRCDOC.findall(page_html)]
    return kids or [page_html]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default=os.environ.get("HHEMT_QA_DEST", ""))
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser() if args.dest else Path.home() / "Downloads" / "cc_resume_report_qa"
    if shutil.which("node") is None:
        print("CANNOT RUN: node is not on PATH (needed to compile JS)", file=sys.stderr)
        return 2

    present = [dest / n for n in _REPORTS if (dest / n).exists()]
    if not present:
        print(f"CANNOT RUN: no delivered reports under {dest}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        probe = work / "probe.js"
        probe.write_text(_PROBE)
        manifest = []
        for report in present:
            text = report.read_text(encoding="utf-8", errors="replace")
            for path, page in sorted(decoded_by_path(text).items()):
                if not page or page.startswith("<<DECODE"):
                    continue
                for ki, kid in enumerate(child_docs(page)):
                    for si, src in enumerate(_SCRIPT.findall(kid)):
                        if not src.strip():
                            continue
                        f = work / f"{report.stem}__{path.replace('/', '__')}.{ki}.{si}.js"
                        f.write_text(src)
                        manifest.append((report.name, path, f"{ki}.{si}", f))

        failed = 0
        for report_name, path, sid, f in manifest:
            r = subprocess.run(["node", str(probe), str(f)], capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                failed += 1
                if not args.quiet:
                    print(f"FAIL {report_name} {path} script{sid} :: {r.stderr.strip()}")

    print(f"checked={len(manifest)} failed={failed} reports={len(present)} dest={dest}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
