---
type: integration
title: Scenario inputs and SWMM model preparation
description: Event forcing, runnable SWMM role derivation, runoff generation, and explicit surface coupling and boundary files.
tags: [scenario, swmm, forcing, coupling, preparation]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-a4d8f719ac710d6d204d0826
    resource: repo://src/hhemt/prepare_scenario_runner.py
  - id: openwiki-source-51208d52fef3a5c875db333f
    resource: repo://src/hhemt/scenario_inputs.py
  - id: openwiki-source-92b126da4e4928a37957abfc
    resource: repo://src/hhemt/scenario.py
  - id: openwiki-source-04443c3466e25262a54b89fb
    resource: repo://src/hhemt/swmm_runnable.py
  - id: openwiki-source-ca40d3b1e19cec7df122de70
    resource: repo://src/hhemt/swmm_runoff_modeling.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Scenario inputs and SWMM model preparation

## Preparation boundary

A scenario joins an indexed weather event to the system's terrain and runnable SWMM model. `prepare_scenario` skips a completed scenario unless overwrite is requested, checks native build availability (skipping that native-build gate for container execution), writes event weather and forcing files, derives hydraulics and optional standalone models, optionally executes hydrology, and creates external-boundary inputs. Container execution therefore does not imply an on-cluster native compile. See [solver builds and containers](solver-builds-and-containers.md).

The preparation entrypoint is `python -m hhemt.prepare_scenario_runner`. Its subprocess boundary isolates PySWMM preparation to avoid concurrent-simulation conflicts; it can be launched with or without `srun`. The runner loads the HPC partition's GPU target, creates observer system/analysis objects with `skip_log_update=True`, prepares one integer event index, and only returns success after refreshing the scenario log and observing `scenario_creation_complete`. Missing config paths return 2; exceptions or incomplete preparation return 1. Execution locus is a launcher concern, not a different input model.

## Weather event identity and extent

`compute_event_id_slug` joins event indexer names and values in mapping order. `resolve_event_window` uses exactly one matching row of the declared event-window CSV, or the weather coordinate's full endpoints when no CSV is supplied. A malformed declared CSV raises instead of falling back to inferred missingness.

Writing scenario weather selects the event and clips to that window. Zero selections and CSV endpoints that snap inward off the weather axis are refused. Reading `ds_event_ts` also checks zero-length and missing forcing values, including when an old scenario weather file already exists. The toolkit refuses incomplete forcing rather than padding or inferring a shorter valid interval; the forcing-agreement helper also requires a uniform time step.

## One source model, three roles

`SWMM_model` is a runnable source, not a placeholder template. The operator guide requires SWMM 5.2 CMS inputs and a rain-gauge mapping. `derive_role_model` uses an explicit section/role table:

- `full`: retains the standalone model without converting outfalls.
- `hydraulics`: supplies the drainage network to TRITON-SWMM.
- `hydrology`: supplies runoff generation, setting routing to `KINWAVE`.

Requested outfalls become junctions in hydraulics and hydrology, with supplied DEM-derived depths; an unknown outfall name raises `ConfigurationError`. If no outfall remains, a reserved `hhemt_dummy_*` chain is inserted so SWMM has an outlet. Series pointers and outfall stage-series retargeting are handled during derivation; event timing and forcing rewriting are a separate responsibility.

Preparation refuses non-CMS hydraulics and standalone inputs, and US-unit hydrology inputs. Gauge rainfall files come from the event dataset using unique mapped gauge IDs. Optional hydrology execution uses `Simulation.execute()`, resets completion before running, and sets it only afterwards; the validated SWMM-stack and declared-test-venue checks run before solver execution.

## Runoff versus exchange declarations

Hydrology produces runoff forcing; the coupling declaration selects where network/surface exchange is allowed. They are not interchangeable. `ScenarioInputGenerator` writes `coupling_nodes.txt` alongside `hydraulics.inp` without adding or deleting its `[INFLOWS]` rows. It selects from configured candidates against the processed DEM and link table, writes the final list, and optionally writes a manhole table. Ineligible dropped nodes are warned about; when one-per-cell reduction is enabled, `_status/inflow_node_selection.json` records the selection, including an empty record when nothing was reduced.

The method's historical name mentions one node per grid cell, but the actual selection is controlled by configuration. Do not infer that every scenario collapses all junctions to one per cell. Solver compatibility with the explicit list is a separate preflight contract; see [coupling diagnostics](../concepts/coupling-diagnostics.md).

## Boundary files and terrain relationship

External-boundary records generate `extbc/loc.extbc` in record order. Supported types are water level, free outflow, normal slope, and Froude number. Water-level records additionally write a series in meters with hours relative to the first level timestamp. Slopes and Froude numbers write scalar values. With the toggle off, only the location header is written.

Generation snaps records against the processed DEM and checks matched outfalls, refusing collected placement/distance problems before writing. Scenario logs retain record types, transformed endpoints, snapped cells, and outfall distances. Under weather-sourced boundaries, SWMM water-level `.dat` files and TRITON level files call the same `boundary_level_series` function, so they share the converted values. Under model-sourced boundaries, the SWMM model keeps its own series rather than generating weather-source `.dat` files.

The [runnable SWMM guide](../../docs/how-to/using-a-runnable-swmm-model.md) explains outfall keep/convert decisions, unit/datum mappings, DEM rim resolution, and validation remedies. Run `analysis.validate()` before submission; see [configuration and validation](../concepts/configuration-and-validation.md) and [ensemble lifecycle](../workflows/ensemble-lifecycle.md).

## Verification boundary

This page is source-grounded; no SWMM/TRITON simulation or scheduler launch was performed. The runner's completion log is the operational success check, not evidence that every existing prepared directory is current after an input change.
