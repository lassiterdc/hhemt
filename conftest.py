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
_DISABLE = os.environ.get("HHEMT_DISABLE_WORKTREE_GUARD") == "1"
_ALLOW = os.environ.get("HHEMT_ALLOW_INSTALLED") == "1"
_FORCE_WRONG_SRC = os.environ.get("HHEMT_FORCE_WRONG_SRC")

if not _DISABLE:
    # Layer 3 is DELEGATED to hhemt._worktree_guard.worktree_mismatch_message so the
    # assertion and its message body exist ONCE. What stays here is pytest-specific and
    # the shared module must not own it: the HHEMT_FORCE_WRONG_SRC pytester seam, the
    # sys.__stderr__/__stdout__ writes that bypass pytest's already-installed capture
    # middleware, and the success-path marker the protocol's Smoke Validation
    # Requirement greps for. The `label` argument pins this tier's wording, which
    # tests/test_worktree_guard.py asserts verbatim.
    from hhemt._worktree_guard import worktree_mismatch_message  # noqa: E402

    msg = worktree_mismatch_message(
        expected_src=_SRC,
        force_wrong_src=_FORCE_WRONG_SRC,
        label="worktree-test-guard",
    )
    if msg is None:
        sys.__stdout__.write(f"[worktree-test-guard] sys.path prefix: {_SRC}\n")
        sys.__stdout__.flush()
    elif _ALLOW:
        sys.__stderr__.write(f"[worktree-test-guard] WARNING: {msg}\n")
        sys.__stderr__.flush()
    else:
        sys.__stderr__.write(f"[worktree-test-guard] {msg}\n")
        sys.__stderr__.flush()
        sys.exit(99)
