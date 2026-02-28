# Planning Documentation Structure

Planning documents are organized by type. Each type directory has a `completed/` subdirectory for finished work, and a `shelved/` subdirectory created as needed for deprioritized or blocked work.

## Directory Structure

```
planning/
├── bugs/               # Bug fix plans
│   └── completed/      # Implemented bug fixes
├── features/           # New functionality plans
│   └── completed/      # Implemented features
├── refactors/          # Code improvement plans
│   └── completed/      # Implemented refactors
└── reference/          # Vision docs, specs, roadmaps (not actionable tasks)
```

## Moving Documents

When a document's status changes:

- **Task completed** → Move to `completed/` within the same type directory
- **Task blocked/deprioritized** → Move to `shelved/` within the same type directory (create if needed)
- **Task no longer relevant** → Delete

## Naming Conventions

- No strict prefix required; use descriptive snake_case filenames
- Reference docs: `*_vision.md`, `*_spec.md`, `*_roadmap.md`
