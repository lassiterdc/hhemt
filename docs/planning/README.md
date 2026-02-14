# Planning Documentation Structure

This directory contains planning documents organized by status and category.

## Directory Structure

```
planning/
├── active/              # Currently planned or in-progress work
│   ├── bugs/           # Bug fixes (fixing_*.md)
│   ├── features/       # New functionality (feat_*.md, *_plan.md)
│   └── refactors/      # Code improvements (refac_*.md, *_design.md)
├── completed/          # Finished work (for reference/history)
├── shelved/            # Deprioritized or blocked work
├── debugging/          # Active debugging sessions
└── reference/          # Vision docs, specs, roadmaps (not actionable tasks)
```

## Document Counts

- **Active/Bugs**: 3 documents
- **Active/Features**: 7 documents
- **Active/Refactors**: 13 documents
- **Completed**: 2 documents
- **Debugging**: 3 documents
- **Reference**: 6 documents

## Moving Documents

When a document's status changes:

- **Task started** → Keep in `active/` (already there)
- **Task completed** → Move to `completed/`
- **Task blocked/deprioritized** → Move to `shelved/`
- **Task no longer relevant** → Delete or move to `completed/` with note

## Naming Conventions

- Bug fixes: `fixing_*.md`
- Features: `feat_*.md` or `*_plan.md`
- Refactors: `refac_*.md` or `*_design.md`
- Reference: `*_vision.md`, `*_spec.md`, `*_roadmap.md`
