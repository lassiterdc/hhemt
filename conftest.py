"""Worktree-aware import guard. See worktree-aware project testing protocol."""
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
_src_str = str(_SRC)

# Layer 1: sys.path prepend — wins over site-packages .pth finder for path imports (compat-mode editables only).
if _src_str in sys.path:
    sys.path.remove(_src_str)
sys.path.insert(0, _src_str)

# Layer 2: PYTHONPATH export — transitive to every subprocess. Always move _src_str to the front.
_existing = os.environ.get("PYTHONPATH", "")
_parts = [p for p in _existing.split(os.pathsep) if p and p != _src_str]
os.environ["PYTHONPATH"] = os.pathsep.join([_src_str, *_parts])

# Testing-time seams (see protocol doc).
_DISABLE = os.environ.get("TRITON_SWMM_TOOLKIT_DISABLE_WORKTREE_GUARD") == "1"
_ALLOW = os.environ.get("TRITON_SWMM_TOOLKIT_ALLOW_INSTALLED") == "1"
_FORCE_WRONG_SRC = os.environ.get("TRITON_SWMM_TOOLKIT_FORCE_WRONG_SRC")

if not _DISABLE:
    try:
        import TRITON_SWMM_toolkit  # noqa: E402
        _resolved = Path(TRITON_SWMM_toolkit.__file__).resolve() if _FORCE_WRONG_SRC is None else Path(_FORCE_WRONG_SRC).resolve()
    except ImportError as _exc:
        # Shared conda env's editable install may point at a removed worktree — surface the friendly message instead of a raw ModuleNotFoundError.
        msg = (
            f"worktree-test-guard: import TRITON_SWMM_toolkit failed ({_exc}).\n"
            f"  expected prefix: {_SRC}\n"
            f"  shared conda env's editable install may point at a removed or stale path.\n"
            f"  Set TRITON_SWMM_TOOLKIT_ALLOW_INSTALLED=1 to bypass for installed-package testing."
        )
        if _ALLOW:
            sys.__stderr__.write(f"[worktree-test-guard] WARNING: {msg}\n")
            sys.__stderr__.flush()
        else:
            sys.__stderr__.write(f"[worktree-test-guard] {msg}\n")
            sys.__stderr__.flush()
            sys.exit(99)
    else:
        try:
            _resolved.relative_to(_SRC)
        except ValueError:
            msg = (
                f"worktree-test-guard: TRITON_SWMM_toolkit.__file__ = {_resolved}\n"
                f"  expected prefix: {_SRC}\n"
                f"  shared conda env's editable install points elsewhere.\n"
                f"  Set TRITON_SWMM_TOOLKIT_ALLOW_INSTALLED=1 to bypass for installed-package testing."
            )
            if _ALLOW:
                sys.__stderr__.write(f"[worktree-test-guard] WARNING: {msg}\n")
                sys.__stderr__.flush()
            else:
                sys.__stderr__.write(f"[worktree-test-guard] {msg}\n")
                sys.__stderr__.flush()
                sys.exit(99)
        else:
            sys.__stdout__.write(f"[worktree-test-guard] sys.path prefix: {_SRC}\n")
            sys.__stdout__.flush()
