---
type: concept
title: Configuration loading and preflight validation
description: Explains configuration ownership, strict input-path loading, conditional requirements, aggregated diagnostics, and launch-time guards.
tags: [configuration, validation, preflight]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-7c7ce9fe78fdbc933b114210
    resource: repo://hooks/config_reference.py
  - id: openwiki-source-21b9d8cf4c93042742d6c138
    resource: repo://src/hhemt/analysis.py
  - id: openwiki-source-15bc8d0cbe6b4bd1d7c8f314
    resource: repo://src/hhemt/cli.py
  - id: openwiki-source-84f2133a4654ea7ea78b4b50
    resource: repo://src/hhemt/config/analysis.py
  - id: openwiki-source-a2676b551372a0e1c0280717
    resource: repo://src/hhemt/config/base.py
  - id: openwiki-source-8c5027a2ab5d04823b61b767
    resource: repo://src/hhemt/config/hpc_system.py
  - id: openwiki-source-f8786678943e3866638cadbf
    resource: repo://src/hhemt/config/loaders.py
  - id: openwiki-source-5fee4361564019699ffd6a76
    resource: repo://src/hhemt/validation.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Configuration loading and preflight validation

## Ownership and precedence

- **System config:** study-area inputs, model selection, terrain processing, and software locations owned by `TRITONSWMM_system`.
- **Analysis config:** selected weather/events, run resources and method, retention, sensitivity, toolkit identity, and rendering settings.
- **HPC system config:** reusable cluster partition capabilities, GPU backend information, modules, and executor-profile extras. Its models use plain Pydantic `BaseModel`, not the local-input-path base class: remote cluster descriptions are not local file-existence assertions.
- **Report config:** a required inline `report:` block in the analysis YAML. An explicit `analysis.run(report_config=...)` path overrides that block.

Use the generated configuration reference in `docs/reference/config-schema.md` through the documentation build rather than maintaining a second field inventory. `hooks/config_reference.py` derives field records and conditional requiredness from live model metadata.

## Load-time contract

The common loader reads YAML with `safe_load`, rejects an empty/top-level-null document, and invokes `model_validate` with a document-level existence context. Pydantic schema failures become `ConfigurationError` with the originating YAML path. Missing files remain `FileNotFoundError`; empty documents remain `ValueError`.

For models inheriting `cfgBaseModel`, extra fields are forbidden. Conditional `required_when` declarations are enforced before field conversion; omitted trigger fields use their declared defaults, so omission cannot evade a conditional requirement.

Existence modes describe the current read, not a promise of later validation:

| Mode | Meaning |
| --- | --- |
| `runnable` (default) | Declared input paths must exist now. |
| `template` | Placeholder path strings are not yet runnable paths. |
| `metadata` | Read final metadata without asserting local input availability. |

The wildcard path validator runs **after** Pydantic conversion, so YAML strings typed as `Path` receive the check. It expands `~` and skips toolkit-owned output fields; output/build directories need not already exist. Lists of paths are outside this scalar validator and need their owning field's validation. These functions do not rebase relative paths against the YAML file's directory: the path is checked as supplied after user expansion. Template mode is not an environment-variable substitution mechanism.

`assert_both_configs_load()` attempts both system and analysis loaders and collects their `ConfigurationError`s into one diagnostic. It deliberately does not aggregate missing-file or empty-document errors into schema errors.

## Preflight is a second layer

`preflight_validate()` returns a combined `ValidationResult` rather than stopping at the first issue. It merges system, analysis, and data-consistency checks, then checks sensitivity definitions/member configs, selected forcing extents and event windows, partition caps, report runtime dependencies, container configuration, retention settings, and model/boundary compatibility. Some checks warn rather than refuse; for example setup memory sizing is advisory, whereas an unavailable Plotly static-export backend is an error.

The CLI run path schema-checks both files, constructs and links the objects, and calls `analysis.validate(build_sifs=...)` before dispatch. A normal workflow preview is:

```bash
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml --dry-run
```

A dry run is not a solver correctness test, nor a blanket guarantee of no filesystem effects: object construction and workflow preparation have their own persistence contracts. See [ownership](../architecture/ownership.md) and [ensemble lifecycle](../workflows/ensemble-lifecycle.md).

## Launch-time guards and execution axes

Passing schema validation does not establish binary compatibility or cross-node visibility. Analysis construction independently enforces the configured toolkit SHA and clean running identity. Container-mode preflight resolves a container specification and image identity; native runs instead depend on host build state. See [solver builds and containers](../integrations/solver-builds-and-containers.md).

For SLURM-locus submission, `assert_configs_visible_cross_node()` checks YAML locations and declared system/analysis paths against the resolved system temporary directory. It refuses node-local staging before compute is spent. `HHEMT_ALLOW_NODE_LOCAL_CONFIGS=1` bypasses that guard at the operator's risk; it does not make the filesystem shared. Explicit local execution avoids dispatching rules to other nodes. The guard is bounded to the roots it checks, not proof that every arbitrary mount is shared.

Cluster profile extras cannot override toolkit-emitted per-rule resources through `set-resources` or `default-resources`; the HPC model raises rather than allowing a profile to silently replace correctness-critical scheduling values. See [HPC execution](../operations/hpc-execution.md).

No configuration command, solver, scheduler query, or test suite was executed for this page.
