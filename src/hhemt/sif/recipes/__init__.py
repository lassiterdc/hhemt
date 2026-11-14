"""family_recipe: ``{mpi}-{accel}`` -> (recipe path, recipe text, digest-pinned base ref)."""

from __future__ import annotations

import re
from pathlib import Path

_FROM = re.compile(r"^From:\s*(\S+@sha256:[0-9a-f]{64})\s*$", re.M)


def family_recipe(family: str, recipes_dir: Path | None = None) -> tuple[Path, str, str]:
    d = Path(recipes_dir) if recipes_dir else Path(__file__).parent
    p = d / f"{family}.def"
    if not p.is_file():
        raise FileNotFoundError(
            f"no recipe for family {family!r} under {d} (have: {sorted(x.name for x in d.glob('*.def'))})"
        )
    text = p.read_text()
    m = _FROM.search(text)
    if not m:
        raise ValueError(f"{p}: From: must be a digest-pinned reference (repo@sha256:...), never a tag")
    return p, text, m.group(1)
