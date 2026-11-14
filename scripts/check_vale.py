#!/usr/bin/env python3
"""Fail when Vale reports an alert on a published page.

Every rule of `.vale/styles/` has warning level, and Vale exits 0 on a warning, so its own exit code cannot
gate a build. This script runs Vale over the tracked pages, prints each alert with its page, line, and rule,
and exits 1 when any alert exists.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

PAGES = ("docs", "README.md")


def tracked_pages() -> list[str]:
    """The Markdown pages under `PAGES` that git tracks.

    The documentation build writes generated pages into `docs/` (they are gitignored), so Vale is given the tracked
    files and not the directory.
    """
    listed = subprocess.run(["git", "ls-files", "-z", *PAGES], capture_output=True, text=True, check=True).stdout
    return sorted(path for path in listed.split("\0") if path.endswith(".md"))


def main() -> int:
    vale = shutil.which("vale")
    if vale is None:
        print("check_vale.py: `vale` is not on PATH; run `pip install vale`.", file=sys.stderr)
        return 2
    run = subprocess.run(
        [vale, "--config=.vale.ini", "--output=JSON", *tracked_pages()],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode not in (0, 1):
        print(run.stderr or run.stdout, file=sys.stderr)
        return 2
    alerts = json.loads(run.stdout or "{}")
    count = 0
    for page, found in sorted(alerts.items()):
        for alert in found:
            count += 1
            print(f"{page}:{alert['Line']}: {alert['Check']}: {alert['Message']}")
    print(f"check_vale.py: {count} alert(s) over {', '.join(PAGES)}")
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main())
