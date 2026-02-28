# Tech Debt: Pre-existing ruff violations in workflow.py

**Noted**: 2026-02-28

`ruff check src/TRITON_SWMM_toolkit/workflow.py` reports 24 violations — all pre-existing,
none introduced by the lock-detection feature (2026-02-28). The convention requires ruff to
pass before submitting code; this debt should be cleared in a dedicated cleanup pass.

## Violation Summary (as of 2026-02-28)

| Code | Count | Description |
|------|-------|-------------|
| F401 | 5 | Unused imports: `psutil`, `shutil`, `signal`, `os`, `typing.Any` |
| F541 | 8 | f-string literals with no placeholders (e.g., `f"nodes=1"`) |
| I001 | 1 | Import block unsorted |
| E501 | 4 | Lines too long (in bash heredocs, hard to wrap) |
| W291/W293 | 2 | Trailing/blank-line whitespace (inside heredoc strings) |
| UP024 | 1 | `EnvironmentError` should be `OSError` |
| B005 | 1 | `.strip()` with multi-character string (misleading) |

## Fix Approach

Most are auto-fixable:
```bash
ruff check --fix src/TRITON_SWMM_toolkit/workflow.py
```

Remaining E501 violations are in bash heredocs embedded as Python strings — line-wrapping
them would break the generated scripts. Suppress with `# noqa: E501` on those lines.

The B005 `.rstrip(" && ")` issue in `_submit_tmux_workflow` is a real logic concern:
`str.strip(chars)` treats its argument as a *character set*, not a substring. The intent
is to strip the trailing ` && ` suffix — use `removesuffix(" && ")` instead.

## Priority

Low — violations are cosmetic or minor. Address in a dedicated linting pass, not mixed
into feature work.
