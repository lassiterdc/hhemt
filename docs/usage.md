# Usage

## Getting started

Import the package and call the sample function:

```python
import TRITON_SWMM_toolkit

result = TRITON_SWMM_toolkit.hello("world")
print(result)  # Hello, world!
```

## Development workflow

This project follows a plan-then-implement workflow:

```mermaid
sequenceDiagram
    participant D as Developer
    participant C as Claude Code
    participant P as Planning Doc

    D->>C: Describe task
    C->>P: Write implementation plan
    P-->>D: Review and approve
    D->>C: /proceed-with-implementation
    C->>C: Preflight check
    D->>C: Approve
    C->>C: Implement
    C->>C: /qaqc-and-commit
    C-->>D: QA report + commit
```
