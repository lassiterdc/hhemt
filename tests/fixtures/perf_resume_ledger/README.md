# Raw performance capture — clean and resumed, `sa_serial_6_r1`

Real solver output, captured from Rivanna, for testing that **every tracked
performance metric** aggregates correctly under **both clean and resume**
conditions. It replaces a hand-built expectation with ground truth: the prior
fixture wrote `Init = 0` on every row, degenerate in exactly the column that
defeats the reset detector, so it passed while the defect it was meant to catch
shipped.

## What is here

```
clean_triton/
  performance/performance{1..144}.txt   144 files, 26,806 bytes
  walltime_model_triton_sa_serial_6_r1_evt0.jsonl      113 bytes
resume_triton/
  performance/performance{1..144}.txt   144 files, 27,300 bytes
  walltime_model_triton_sa_serial_6_r1_evt0.jsonl      454 bytes
```

Serial holds the rank count at 1, which is why the full 144-step set is small
enough to track.

## Provenance

Captured 2026-08-15 from `/scratch/$USER/hhemt_experiments/` on Rivanna:

| Arm | Source |
|---|---|
| clean | `synth_cc_clean_triton/synth_cc_clean_triton/subanalyses/sa_serial_6_r1/sims/event_index.0/out_triton/performance` |
| resume | `synth_cc_resume_triton/synth_cc_resume_triton/subanalyses/sa_serial_6_r1/sims/event_index.0/out_triton/performance` |

Oracles come from each arm's `logs/sims/_walltime/`.

**The experiment root carries a doubled path segment** (`synth_cc_clean_triton/synth_cc_clean_triton/`),
and the arm names prefix-collide: `synth_cc_resume_triton` is a strict prefix of
`synth_cc_resume_tritonswmm`. A glob written as `*synth_cc_resume_triton*`
silently resolves to the **coupled** arm. Both traps fired while locating these
files and were caught only because the resulting file counts came back `0`
instead of something plausible. Anchor arm paths exactly; never glob them.

Raw performance files are **not** carried in render bundles — they are outside a
bundle's manifest-harvest scope. A bundle is the wrong tree to look in.

## Why both arms

The clean arm is the calibration that separates *the aggregator is wrong* from
*the reset predicate is wrong*. Measured agreement against the ledger is
0.996 / 0.960 on clean, against 0.689 / 0.480 / 0.609 on resume — so a
resume-only fixture cannot show the aggregator is sound, and a clean-only
fixture never exercises the arm the defect reaches.

## What the oracles carry, and what they do not

Each line is one solver attempt:

```json
{"attempt": 0, "wall_s": 117.61507987976074, "completed": false, "slurm_jobid": "18396225", "slurm_step_id": "0"}
```

- **clean** — one attempt, `completed: true`, 418.09 s.
- **resume** — four attempts; the first three `completed: false`, the last `true`.
  Sum 436.12 s.

Four attempts means **three interruptions**, which agrees independently with the
resume ledger's `resume_reporting_tsteps = [36, 72, 108]` — also three. Two
artifacts, produced by different mechanisms, reporting the same reset count.

**These files carry durations and attempt boundaries, not wall-clock timestamps.**
That distinction is load-bearing: the governing constraint requires resume
identification to read a permanent artifact rather than infer resets from the
performance text, and it names *the timestamp of the interruptions/resumes*.
The reset **indices** live in the ledger; the per-attempt **walls** live here.
Whether that pair satisfies the constraint on its own, or a wall-clock source
must be joined, is an open question — do not treat index and timestamp as
interchangeable.

## The schema under test

The header names every tracked metric, and coverage means all of them, not the
one the benchmarking figure happens to plot:

```
%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total
```

`Init` is the column that matters most: it **increases** across a resume
boundary while others decrease, which is what defeats an all-columns-decreased
reset predicate.
