# Limitations and constraints

What hhemt does not do, and what it constrains about how you work. Read this
alongside [Capabilities](capabilities.md). That page is what the toolkit is for,
this one is where it stops.

This page covers **architectural and environmental** limits, which are properties
of the toolkit and are stated here because a reader can act on them. It is
deliberately not a validation envelope for the underlying solvers; see
[Scientific validity](#scientific-validity) below for why that distinction
matters and where to look instead.

## hhemt does not contain the science

The toolkit is an **orchestration framework**. The physics is TRITON's (2D
hydrodynamics) and SWMM's (stormwater network); hhemt compiles them, feeds them,
runs them at scale, and consolidates what they emit.

Three consequences follow, and they are the ones most likely to matter:

- **A solver's limitations are hhemt's limitations.** Anything TRITON cannot
  represent, an hhemt ensemble cannot represent either.
- **Solver version is part of your result.** The toolkit clones and builds
  TRITON at a pinned commit; two analyses built from different pins are not
  necessarily comparable. The pin is recorded in your consolidated output's
  provenance, and a set pin is verified fail-loud at setup rather than silently
  reused from a stale clone.
- **Coupled and standalone SWMM are not the same build.** The coupled model uses
  a SWMM vendored inside TRITON, which travels with the TRITON pin. The
  standalone model is built from its own tag. They can differ.

## Environment constraints

**Python 3.11–3.12 only** (`requires-python = ">= 3.11, < 3.13"`). The upper
bound is not a policy choice: it is forced by a transitive dependency that pins
an older pydantic with no wheel for 3.13. It lifts when that dependency relaxes.

**The SWMM engine is version-gated at runtime.** Before executing SWMM the toolkit
checks the installed `pyswmm` and `swmm-toolkit` against the pairing it certifies,
and refuses rather than run against an engine build it cannot vouch for. Silent
execution against an uncertified engine is the outcome that guard exists to
prevent. `pyproject.toml` pins that pairing directly, so a pip install resolves a
stack the guard accepts. There is an override, and using it means accepting a
result from an unchecked engine. Conda remains the recommended path for a
different reason: `environment.yaml` pins the whole HPC stack, including the
Snakemake SLURM executor plugins. See [Installation](../how-to/installation.md).

## Scale and hardware

**More GPUs is not always faster.** Domain decomposition adds communication, and
below some domain size that cost dominates the compute it saves. Benchmark your
own domain rather than assuming the largest allocation you can obtain is the
fastest; the sensitivity and benchmarking machinery exists precisely so this is
measurable rather than guessed.

**Per-row partition variation requires `batch_job` dispatch.** One SLURM
allocation cannot span partitions, so a sensitivity sweep that varies hardware
across rows cannot use a single-allocation dispatch method. Preflight rejects
the combination rather than failing mid-run.

**Walltime is a real boundary.** A simulation killed at walltime resumes from its
most recent checkpoint, but resumption is a property of the solver's checkpoint
cadence, not something the toolkit can add. Size allocations with that in mind.

## Reproducibility boundaries

**Bit-identical results are not guaranteed across hardware.** The toolkit's own
consistency checking treats within-hardware-family reproducibility and
cross-hardware comparison as different questions, and reports them separately,
because a GPU result and a serial-CPU result of the same scenario are expected to
differ at some magnitude. Cross-hardware agreement is a characterized divergence,
not an assertion of equality.

**A render bundle's schema version must match.** A bundle emitted by a different
toolkit version is refused rather than read on a best-effort basis. Re-emit it
from source instead.

## Scientific validity

**This page does not tell you whether hhemt's results are valid for your study.**
That question is about the coupled model's behaviour against observations in a
domain like yours, and it is answered by validation work and its publications,
not by the orchestration layer's documentation.

If you are deciding whether to base published work on this toolkit, the citation surface in the
[README](https://github.com/lassiterdc/hhemt#readme) and the
[FAIR scope table](../reference/fair-scope-table.md) are the right starting
points, and the underlying solvers' own literature is the authority on their
physics.

## See also

- [Capabilities](capabilities.md): what the toolkit is for.
- [When and why re-runs happen](rerun-faq.md): why a run you expected to be a
  no-op re-executed.
- [Installation](../how-to/installation.md): the environment contract and its
  failure modes.
