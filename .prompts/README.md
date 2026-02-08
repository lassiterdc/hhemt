# Prompt Templates

Reusable prompts for common workflow tasks. Reference with `@.prompts/<filename>` in chat.

## Available Prompts

### Workflow Orchestration

- **`validate_and_proceed.md`** — Check priorities/trackers, fix inconsistencies, proceed with next work
- **`next_priority.md`** — Identify what to work on next from priorities.md

### Testing & Validation

- **`smoke_tests.md`** — Run all 4 smoke tests in correct order (PC_01-05)
- **`import_audit.md`** — Verify all imports after module refactor

### Version Control

- **`commit_phase.md`** — Commit current phase with proper message format
- **`update_tracker.md`** — Update tracker docs after completing a phase

### Documentation

- **`doc_freshness_check.md`** — Check if CLAUDE.md or agent docs need updates

### Planning

- **`plan_implementation.md`** — Use plan mode to design approach before coding

## Usage

Simply reference a prompt in chat:

```
@.prompts/validate_and_proceed.md
```

Claude will execute the prompt's instructions.

## Philosophy

These prompts align with `CLAUDE.md` development philosophy:
- No backward compatibility shims in code
- Update all import sites immediately
- Run smoke tests after significant changes
- Keep documentation fresh
- Commit with descriptive messages
