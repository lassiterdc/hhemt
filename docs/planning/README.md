# Planning Documentation Structure

Planning documents are organized by type. Each type directory has a `completed/` subdirectory for finished work, and a `shelved/` subdirectory created as needed for deprioritized or blocked work.

## Directory Structure

```
planning/
├── bugs/               # Bug fix plans
│   └── completed/      # Implemented bug fixes
├── features/           # New functionality plans
│   └── completed/      # Implemented features
└── refactors/          # Code improvement plans
    └── completed/      # Implemented refactors
```

## Moving Documents

When a document's status changes:

- **Task completed** → Move to `completed/` within the same type directory
- **Task blocked/deprioritized** → Move to `shelved/` within the same type directory (create if needed)
- **Task no longer relevant** → Delete

## Naming Conventions

- All planning docs use a `YYYY-MM-DD_descriptive_snake_case_name.md` filename where the date is the creation date
- Exceptions: `README.md` and other persistent tracking documents with no creation lifecycle
