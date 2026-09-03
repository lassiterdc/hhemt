#!/usr/bin/env python3
"""CI check enforcing the caption/renderer paired-edit discipline.

Usage:
    python scripts/check_caption_pairing.py [base_ref=HEAD~1]

Exit 0 = pass; exit 1 = enforcement failure with structured message.

WHAT A GREEN RESULT MEANS
    No captioned renderer changed without EVERY caption it owns being touched in
    the same change.

WHAT A GREEN RESULT DOES NOT MEAN
    It does NOT mean any caption is accurate, that its column names match the
    table, that its units match, or that its prose describes the current table.
    A caption edited to fix a typo satisfies this check completely. This guard
    forces a human to LOOK at the caption; it cannot make them look correctly.
    It also does not see output-shape changes that live in a shared helper
    (_tabulator_defaults, _figure_emission) rather than in the renderer module.

THE UNIT IS THE (renderer, caption) PAIR, NEVER THE MODULE
    One renderer module can own several captions -- eda_compute_sensitivity owns
    SIX. An earlier draft of this check keyed a dict on the module and assigned
    rather than accumulated, so five of twenty caption files were silently
    dropped while the summary line still read like full coverage. That is the
    false-clear class this guard exists to prevent, so the pair is the unit at
    every layer: parsing, violation reporting, and the allowlist key.

Motivating measurement (83181c3..11af9ea): FIVE captioned renderers changed with
no caption edit, and FOUR of those five were narrating a table their renderer no
longer emitted -- a phantom `Severity` column, `bytes` where the renderer emits
MiB (twice), and an `each cell is ...` claim that stopped being true when derived
rows arrived. None was reported by any existing guard; a human found them by
reading a rendered report, and the correction round-trip is a full re-render plus
combine plus pull.

The renderer <-> caption pairing is read from the EXISTING registry: every
`RuleSpecTemplate` in report_renderers/_reporting_sets.py carries both
`renderer_module=` and `report_kwargs={"caption": "report/captions/<name>.rst"}`.
No second registry is introduced. The registry is parsed with `ast` rather than
imported so the check has no runtime dependency on the hhemt package.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "src" / "hhemt" / "report_renderers" / "_reporting_sets.py"
RENDERER_DIR = Path("src/hhemt/report_renderers")
CAPTION_DIR = Path("src/hhemt/report_templates/captions")
ALLOWLIST_PATH = REPO_ROOT / "_caption_pairing_allowlist.yaml"


def _changed_files(base_ref: str) -> set[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def _pairs_from_registry() -> list[tuple[str, str]]:
    """Sorted unique `(renderer_module, caption_repo_path)` pairs.

    ACCUMULATES into a set: a module owning several captions contributes several
    pairs. Keying a dict on the module here is the defect described in the module
    docstring -- it drops every caption but the last for any multi-caption module.

    AST-parsed, never imported: a CI guard that imports the package it guards
    fails for reasons unrelated to the invariant it tests.
    """
    tree = ast.parse(REGISTRY_PATH.read_text())
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "RuleSpecTemplate"):
            continue
        module = None
        caption = None
        for kw in node.keywords:
            if kw.arg == "renderer_module" and isinstance(kw.value, ast.Constant):
                module = kw.value.value
            elif kw.arg == "report_kwargs" and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values, strict=False):
                    if isinstance(k, ast.Constant) and k.value == "caption" and isinstance(v, ast.Constant):
                        caption = v.value
        if module and caption and caption.startswith("report/captions/"):
            stem = caption[len("report/captions/") :]
            pairs.add((module, (CAPTION_DIR / stem).as_posix()))
    return sorted(pairs)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _allowlist() -> dict[tuple[str, str], dict]:
    """`{(renderer, caption): entry}`. Absent file = empty allowlist, never an error.

    Keyed by the PAIR: a module-keyed exemption would let one justification cover
    every caption that module owns -- six, for eda_compute_sensitivity -- which is
    a blanket rather than an exemption. One justification covers one re-read.
    """
    if not ALLOWLIST_PATH.exists():
        return {}
    raw = yaml.safe_load(ALLOWLIST_PATH.read_text()) or {}
    return {(e["renderer"], e["caption"]): e for e in (raw.get("unpaired_allowlist") or [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_ref", nargs="?", default="HEAD~1")
    args = ap.parse_args()

    changed = _changed_files(args.base_ref)
    pairs = _pairs_from_registry()
    allow = _allowlist()
    n_renderers = len({m for m, _ in pairs})

    violations: list[str] = []
    stale_exemptions: list[str] = []
    for module, caption_rel in pairs:
        renderer_rel = (RENDERER_DIR / f"{module}.py").as_posix()
        if renderer_rel not in changed or caption_rel in changed:
            continue
        entry = allow.get((module, caption_rel))
        if entry is None:
            violations.append(
                f"  {caption_rel}\n"
                f"      not edited, but its renderer {renderer_rel} changed.\n"
                f"      Re-read this caption against the table that renderer now emits."
            )
            continue
        # An exemption is scoped to the renderer CONTENT it was written for. Once
        # the renderer changes again the recorded digest no longer matches and the
        # guard fires again -- a justification string alone becomes a permanent
        # hole, which is measurable in this repo's own sibling allowlist.
        live = _sha256(REPO_ROOT / renderer_rel)
        if entry.get("renderer_sha256") != live:
            stale_exemptions.append(
                f"  {caption_rel}\n"
                f"      allowlisted against a DIFFERENT version of {renderer_rel}.\n"
                f"      recorded: {entry.get('renderer_sha256')}\n"
                f"      current:  {live}\n"
                f"      Re-read the caption; then either update it or re-record the digest."
            )

    if not violations and not stale_exemptions:
        print(
            f"OK: caption pairing clean over {args.base_ref} "
            f"({len(pairs)} caption(s) across {n_renderers} renderer(s) checked). "
            "NOTE: this means no renderer changed without its caption(s) being "
            "touched. It does NOT mean any caption is accurate."
        )
        return 0

    print("FAIL: caption(s) whose renderer changed without a paired caption edit.\n")
    if violations:
        print("Unpaired captions:")
        print("\n".join(violations))
    if stale_exemptions:
        print("\nStale allowlist exemptions:")
        print("\n".join(stale_exemptions))
    print(
        f"\nChecked {len(pairs)} caption(s) across {n_renderers} renderer(s).\n"
        "This guard checks PAIRING, not agreement: it cannot tell whether a caption\n"
        "is accurate, only whether anyone looked at it since the renderer moved.\n"
        f"To exempt one caption, add an entry to {ALLOWLIST_PATH.name} naming BOTH\n"
        "the renderer and the caption, with the renderer's current sha256."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
