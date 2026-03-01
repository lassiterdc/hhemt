# Contributing

## Development setup

1. Fork and clone the repository
2. Create a conda environment or virtual environment
3. Install in development mode: `pip install -e ".[docs]"`
4. Install pre-commit hooks: `pre-commit install`

## Workflow

- Create a feature branch from `main`
- Make changes with tests
- Run `ruff check .` and `ruff format .`
- Run `pytest`
- Submit a pull request

## Documentation

Build docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

## Development Principles

### Raise questions rather than make assumptions
When you encounter uncertainty or discrepancies — especially when implementing a pre-written plan that may have stale components — err on the side of caution and ask the developer how to proceed.

### Plan, then implement
Follow a plan-then-implement strategy. If implementing a plan uncovers a need to change it or its success criteria — including deviations from the planned approach, scope changes, or new risks — raise the discrepancy before continuing rather than adapting silently.

### Let's do things right, even if it takes more effort
- Always be on the lookout for better ways of achieving development goals and raise these ideas
- Raise concerns when you suspect the developer is making design decisions that diverge from best practices
- Look for opportunities to make code more efficient (vectorize operations, avoid loops with pandas, etc.)

### Backward compatibility is NOT a priority

**Rationale**: Single developer codebase. Clean code matters more than preserved APIs. Git history is the safety net.

When refactoring:
- ❌ Don't add deprecation warnings
- ❌ Don't keep old APIs "for compatibility"
- ❌ Don't create compatibility shims or aliases
- ✅ Do update all usage sites immediately
- ✅ Do delete obsolete code completely

### Most function arguments should not have defaults

Default function arguments can lead to difficult-to-debug unexpected behavior. Avoid default values unless a default is almost always the correct choice (e.g., `verbose=True`). This is especially true for configuration fields that users populate — the user should make an intentional choice about every input.

### Avoid aliases

Do not create aliases for functions, classes, or variables. An alias is a second name for the same thing — it creates confusion about which name is authoritative and is a form of backward-compatibility shim. If something needs renaming, rename it and update all call sites.

### No cruft/all variables, imports, and function arguments must be used

Unused elements are a signal that implementation may be incomplete. Treat them as an investigation trigger, not just lint to suppress.

If you come across an unused variable, import, or function argument, investigate before removing:
1. Check whether the surrounding implementation is incomplete
2. Find planning documents that touched that function and determine whether implementation is planned
3. If still uncertain, raise the concern with the developer with hypotheses about why it exists
4. The only exception: elements included for a currently-planned implementation, marked with a comment referencing the planning document

Report your observations, hypotheses, and recommendations to the developer.

After investigation and with approval from the developer, remove unused code, dead branches, commented-out blocks, and stale imports.

### Functions have docstrings, type hints, and type checking

Apply this standard to code you write or modify. For existing code in touched scripts, apply organically — accumulate adherence naturally as scripts are touched rather than doing a global retrofit pass.

### Fail-fast

Critical paths must raise exceptions; never silently return `False` or `None` on failure.

### Preserve context in exceptions

Exceptions should include file paths, return codes, and log locations for actionable debugging.

### Prefer log-based completion checks over file existence checks

A file may exist but be corrupt, incomplete, or from a previous failed run. File existence checks can mask errors when log checks are available.

- **Exception**: File existence is appropriate for verifying *input* files before reading them.

### Keep system-agnostic software

System-specific information belongs in user-defined configuration files. Avoid hardcoded paths or machine-specific constants in core code.

### Track project-agnostic utility candidates

When writing utility functions that could plausibly belong in a shared library (e.g., general-purpose file I/O helpers, generic array operations), note them in a dedicated tracking document. Do not extract them immediately — track them so they can be evaluated together.

---

## AI Workflow

This project uses Claude Code with structured workflow skills. When working with AI assistance:

- `CONTRIBUTING.md` — development principles and working norms (this file)
- `CLAUDE.md` — AI-specific working norms and project context (auto-loaded by Claude Code)
- `architecture.md` — project structure and key modules

Workflow skills (available globally, invoke by name):
- `/implementation-plan` — design a complete plan before coding
- `/proceed-with-implementation` — preflight check before implementing a plan
- `/qaqc-and-commit` — post-implementation QA review and commit
