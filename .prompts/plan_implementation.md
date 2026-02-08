# Plan Implementation Approach

Use plan mode to design an implementation approach before writing code.

## When to Use

Per `CLAUDE.md`:
- New feature implementation with multiple valid approaches
- Code modifications affecting existing behavior
- Architectural decisions (patterns, technologies)
- Multi-file changes (3+ files)
- Unclear requirements needing exploration first
- Any task where user preferences matter

## Plan Mode Process

1. **Explore codebase:**
   - Use Glob, Grep, Read to understand existing patterns
   - Identify affected files and dependencies
   - Look for similar existing implementations

2. **Design approach:**
   - Outline file structure and module organization
   - Identify import sites that need updating
   - Plan for no compatibility shims (update all sites immediately)
   - Consider test coverage

3. **Present plan:**
   - Write structured implementation plan
   - List all affected files
   - Document decisions and trade-offs
   - Note any risks or gotchas

4. **Get approval:**
   - Use ExitPlanMode to request user review
   - Wait for approval before implementing

## Expected Output

- Detailed implementation plan written to appropriate location
- List of files to create/modify/delete
- Migration strategy (especially import updates)
- Test strategy
