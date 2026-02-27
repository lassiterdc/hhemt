# Commit Current Phase

Create a well-structured commit for the current phase of work.

## Commit Message Format

```
<type>: <short summary> (<reference to plan/phase>)

<detailed description of what changed, 2-4 paragraphs>

<specific technical details or file changes>

<test results summary>

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## Commit Types

- `feat:` — new feature
- `refactor:` — code restructuring without behavior change
- `fix:` — bug fix
- `test:` — test additions/modifications
- `docs:` — documentation only
- `chore:` — tooling, dependencies, etc.

## Guidelines

- Reference the plan document and phase (e.g., "cruft cleanup plan Phase 2")
- Include file statistics if significant (e.g., "305 lines deleted, 13 inserted")
- List key files changed
- State that smoke tests pass
- Use HEREDOC format for multi-line messages

## Expected Actions

1. Review `git status` and ensure only intended changes staged
2. Run smoke tests if not already run
3. Craft descriptive commit message
4. Execute commit with Co-Authored-By line
