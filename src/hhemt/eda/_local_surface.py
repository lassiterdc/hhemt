"""eda_local/ source-independent surface emitter (ADR-12).

emit_eda_local_surface(root) writes a bundle-adjacent {root}/eda_local/ Python
package — the source-code-independent EDA authoring surface (NEVER git-tracked
src/hhemt/). General users author reductions + plot functions here; it imports the
INSTALLED hhemt via plain `import hhemt` (no sys.path surgery, no editable install,
no namespace merge, no MetaPathFinder — source-independence by LOCATION).

The package MUST stay import-clean (import only installed hhemt + third-party libs,
never a bundle-relative sibling) so a renderer copies cleanly into src/hhemt/eda/ on
the developer-only promotion port (ADR-11 amendment / OD-15).
"""

from __future__ import annotations

from pathlib import Path

_INIT_BODY = '''\
"""eda_local — your source-code-independent EDA surface (ADR-12).

Author EDA reductions + plot functions here. This package is OUTSIDE git-tracked
hhemt source, so your exploration never modifies `src/hhemt/`. Import the installed
toolkit with a plain `import hhemt`. Keep modules import-clean (installed hhemt +
third-party only) so a function ports cleanly into `src/hhemt/eda/` if a developer
later promotes it into the report.
"""
'''

_BOOTSTRAP_BODY = '''\
"""eda_local/_bootstrap.py — wrong-kernel diagnostic (ADR-12).

Run this first if `import hhemt` fails in your notebook: it confirms the kernel
resolves the installed `hhemt` and prints the resolving prefix so a wrong-env
kernel is an obvious, labeled failure rather than a cryptic ImportError.
"""

from __future__ import annotations

def check_hhemt_import() -> str:
    import hhemt

    print(f"hhemt resolves to: {hhemt.__file__}")
    return hhemt.__file__

if __name__ == "__main__":
    check_hhemt_import()
'''


def emit_eda_local_surface(root: Path) -> Path:
    """Emit the {root}/eda_local/ source-independent package skeleton. Returns the dir.

    Idempotent: re-writes the two toolkit-owned skeleton files on every call (they are
    provenance, not user content — user-authored modules live alongside as new files
    and are never overwritten).
    """
    pkg = Path(root) / "eda_local"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(_INIT_BODY, encoding="utf-8")
    (pkg / "_bootstrap.py").write_text(_BOOTSTRAP_BODY, encoding="utf-8")
    return pkg
