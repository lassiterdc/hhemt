# Prompt for Claude Opus: Whole-Codebase Analysis with Constrained Refactoring

## Role
You are acting as a senior software engineer reviewing and refactoring a scientific Python codebase used to run ensembles of TRITON-SWMM models.

You have **read access to the entire repository** and may explore files, directories, and configuration as needed to build context.

The codebase includes:
- Software download and compilation  
- Running ensembles on multiple compute architectures using Snakemake  
- Sensitivity analysis and performance benchmarking  
- Processing and consolidating simulation outputs  

---

## Stability Contract

Only the following **high-level behaviors must remain stable**:

- Ability to run ensembles end-to-end via Snakemake  
- Ability to reproduce existing ensemble outputs given the same inputs  
- Ability to run sensitivity and benchmarking workflows  

Everything *below* this level (internal classes, module layout, helper functions, internal APIs) is **explicitly allowed to change**.

Backward compatibility of internal interfaces is **not required**.

---

## Codebase Access Rules

- You may freely **read and analyze the entire repository**.
- You should use this global view to:
  - Identify architectural boundaries
  - Detect duplicated logic or hidden coupling
  - Infer the true public surface of the system

- You must **not** refactor the entire codebase at once.
- All refactors must be:
  - Scoped to a small set of files
  - Justified by a concrete design issue
  - Proposed before being applied

---

## Context

- The codebase has grown organically.
- Some classes likely combine orchestration, configuration, execution, and I/O.
- Internal coupling makes changes expensive and error-prone.
- I am a relative beginner in formal software engineering.
- I want to move toward conventional best practices.
- I want the resulting structure to be easier for agentic AI to reason about and modify.

---

## Primary Objectives

1. Identify and explicitly define the true “public surface” of the codebase.
2. Use full-codebase context to find:
   - Overgrown (“god”) classes
   - Implicit dependencies
   - Circular or layered violations
3. Aggressively improve internal structure while preserving only the public surface.
4. Improve:
   - Separation of concerns  
   - Explicit data flow  
   - Testability of core logic  
   - Reasoning locality for humans and AI agents  

---

## Refactoring Philosophy

- Prefer correctness and clarity over minimal diffs.
- Internal refactors may be invasive if they reduce conceptual complexity.
- Large rewrites are allowed **only if hidden behind stable entry points**.
- Introduce new abstractions only when they:
  - Replace multiple implicit responsibilities
  - Reduce cross-module knowledge requirements

---

## Required Output Structure

### 1. Global Architectural Overview
- Major subsystems and their responsibilities
- What appears to be the public surface vs. internal machinery
- High-risk coupling points

### 2. Candidate Refactor Targets (Ranked)
For each:
- Why it is a problem
- How much of the system depends on it
- Expected payoff if refactored

### 3. Focused Refactor Proposal
For the top-ranked target:
- Current responsibilities
- Target responsibilities
- What will change internally
- What invariants must remain true

### 4. Refactor Plan
- Concrete, ordered steps
- Files/modules affected
- What can safely break
- What must be preserved

### 5. Validation Strategy
- How to confirm no regression at the workflow level
- Suggested smoke tests or invariants

### 6. AI-Optimization Notes
- How the new structure reduces context requirements
- How future agents can work locally instead of globally

---

## Operating Constraints

- Never refactor more than one subsystem at a time.
- Never change public entry points without explicit justification.
- Prefer deleting code over preserving unused abstractions.
- If uncertain, stop and ask before acting.

---

## Start

Begin by scanning the repository to build a high-level architectural map.
Do **not** propose refactors until that map is complete.

---

## Post-Phase QA/QC Validation Prompt

After completing a refactoring phase, use this prompt to validate the work:

```
Please review and validate the Phase [N] changes that were just completed:

1. **Code Quality Review:**
   - Read the modified files and confirm the implementation matches the phase goals in docs/refactoring_plan.md
   - Check for any code smells, unnecessary complexity, or missed opportunities for simplification
   - Verify no duplicate code was introduced
   - Confirm no functionality was accidentally removed or altered

2. **Completeness Check:**
   - Compare what was done against the phase checklist - were all items truly completed?
   - Were there any edge cases or aspects mentioned in the plan that weren't addressed?
   - Is the line reduction claim accurate? (count lines before/after if needed)

3. **Consistency Verification:**
   - Does the new code follow the same patterns/conventions as existing code?
   - Are naming conventions consistent?
   - Do new helper functions/classes have appropriate docstrings?

4. **Potential Issues:**
   - Are there any potential regression risks not covered by the smoke tests?
   - Any tight coupling introduced that could cause issues in future phases?
   - Any technical debt created that should be noted for Phase 10 (Polish)?

5. **Summary Report:**
   - Rate the phase completion: Excellent / Good / Needs Improvement
   - List any concerns or recommendations for follow-up
```

**Note:** This validation should be performed in Plan Mode after tests have passed and documentation has been updated in Act Mode.
