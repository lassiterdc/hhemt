# Testing the toolkit

Why the test suite is shaped the way it is. The commands live in
[Contributing](../contributing.md#workflow) and in the
[CLI reference](../reference/cli.md#hhemt-test-toolkit); this page is the reasoning
behind them. It names the one recipe that defines the laptop tier and otherwise
carries no commands of its own.

## Two tiers, and why the split does not depend on your machine

A test that compiles the coupled solver or executes a simulation costs minutes. A
test that decides something before a simulation starts costs milliseconds. Running
both sets on every change would make the fast feedback loop useless, so the suite is
split into a fast tier and a compile tier. The fast tier holds everything the toolkit
decides before a simulation starts, plus the on-disk layout contract. The compile tier
holds every test that compiles TRITON-SWMM or asserts on a tree a simulation produced.

The split is derived rather than declared. Each test's tier, carried by the
`compile_tier` marker, follows from the fixtures it requests, resolved when pytest
collects the suite, so the same test lands in the same tier on every machine. An
earlier arrangement decided the tier from whether `cmake` happened to be on your
PATH, which meant two developers running the same
recipe could gate on different sets of tests without either of them knowing.

## The laptop tier and the Rivanna tier

The two tiers run in two places, and the venue is a rule rather than a convenience.

**The laptop tier** is everything that is neither `compile_tier` nor `requires_snakemake_subprocess`. The
`test-laptop` recipe in the repository's `justfile` runs it as one pytest invocation with `HHEMT_FORBID_COMPILE=1`, so
that a test which reaches for the solver is refused instead of building it. The invocation selects
`not compile_tier and not requires_snakemake_subprocess` and spreads the selected tests over every core with xdist
(`-n` equal to the core count, `--dist loadgroup`). `loadgroup` keeps each `xdist_group` on one worker: the tests
derived as sharing the free runs root run together on one worker, and the rest are handed out one at a time, which
leaves no tail of consecutive batches. The recipe runs under the conda environment with the `docs` extra installed and
with `VIRTUAL_ENV` removed, so the interpreter that runs the documentation-content tests is the one that has the
documentation handlers.

The 96 tests that carry `requires_snakemake_subprocess` (they start Snakemake as a child process) are not in the laptop
tier: run serially they cost far more than the rest of the tier together (the measured figures are below). They run in every whole-suite
Rivanna array, and a `--select-tier compile_tier` Rivanna run leaves them out because they are not `compile_tier`. A
second recipe in the `justfile`, `test-snakemake-subprocess`, runs them locally and serially when you want them. `just`
is not installed on every machine; each recipe is one shell command, and the `justfile` holds its exact text, so it can
be run directly.

The tier needs no solver: tests that need a finished analysis tree start from a committed
golden tree (a small, generator-built, finished analysis) and reprocess it, so the laptop never
compiles TRITON-SWMM. Which tests build a private tree of their own is a committed, reviewed
list, compared against the build-request census by a test, so a new test of that kind cannot
appear unseen.

**What has been measured on the laptop.** Four runs exist, each named by its run id. The first used the
selection the recipe now has, run by hand from the recipe's body, and the fourth used that selection plus
the 14 tests of `tests/test_check_docs_content_memo.py`; no run through the recipe itself has been
recorded. The first and second were made as separate selections, so their sum is not a run of the recipe.
The pool runs of the first and third had failures, so this page does not call the tier green; the fourth
passed all of its tests but did not meet the target.

- `laptop-20261104T172458Z-11ac40f5`: the pool with the 96 `requires_snakemake_subprocess`
  tests deselected, 12 workers, 354.57 s, `3 failed, 5916 passed, 12 skipped, 1 xfailed`. That
  is 1.18 times the 300 s target. The three failures are stray-process assertions in
  `tests/test_suite_isolation_root_node_local.py`. State at the start: HEAD `11ac40f5` with a
  clean tree, MemAvailable 21,279 MiB, swap 4,777 of 8,191 MiB used, load average 4.14, and one
  core held for the whole run by a stuck `git grep` process of another session's worktree (a
  confound: 99.4 to 99.5 % CPU in every state snapshot). Sampled every second: peak session PSS
  10.47 GiB, minimum MemAvailable 11.08 GiB, peak swap used 4.93 GiB of 8.00 GiB, headroom
  against MemTotal 30.72 - 10.47 = 20.25 GiB, memory-pressure stall 2.19 s (the difference of
  the `some` total between the first and last sample). Control: a known allocation of 2 GiB held
  for 20 s under the same sampler was read as a peak of 2.004 GiB against 2.000 GiB.
- `laptop-20261104T173100Z-11ac40f5`: the 96 `requires_snakemake_subprocess` tests run
  serially, 1,498.28 s, `95 passed, 1 skipped`. State at the start: HEAD `11ac40f5`,
  MemAvailable 22,026 MiB, swap 5,037 of 8,191 MiB used, load average 18.68 (the previous
  run had just ended), the same stuck process. Peak session PSS 2.00 GiB, peak swap used
  4.92 GiB. Together with the pool above (`laptop-20261104T172458Z-11ac40f5`) the two take
  1,852.85 s.
- `laptop-20261104T064202Z-47e1074a`: the pool with the 96 inside, 12 workers, 843.49 s,
  `4 failed, 6008 passed, 13 skipped, 1 xfailed, 2 errors`. State at the start: HEAD `47e1074a`,
  MemAvailable 15,803 MiB, swap 5,630 of 8,191 MiB used, load average 2.88 at the first sample;
  whether the stuck process was present was not recorded.

- `laptop-20261105T033340Z-cfd68f2a`: the pool again, 12 workers, 428.16 s, `5959 passed, 12 skipped, 1 xfailed`,
  no failure, 1.43 times the 300 s target: the target was NOT met. State at the start: HEAD `cfd68f2a` with a clean
  tree, MemAvailable 17,920 MiB, swap 5,512 of 8,191 MiB used, load average 3.98, and the stuck `git grep` process of
  the earlier runs was in the snapshot (99.5 % CPU). Sampled every second: peak session PSS 8.11 GiB, minimum
  MemAvailable 10.96 GiB, peak swap used 7.99 GiB of 8.00 GiB, memory-pressure stall 13.88 s, and a load average
  rising to a maximum of 24.99, with one to three other pytest or snakemake processes outside the session in every
  five-second sample (the sampler counts them and does not identify them). Control: a known allocation of 2 GiB held for 20 s
  under the same sampler was read as a peak of 2.004 GiB against 2.000 GiB. The run was memory-pressured, with swap
  nearly full and a stall several times the first run's. Observed, with no cause claimed: it took 73.59 s (20.8 %)
  longer, its summed call time was 3,508.2 s against 3,026.1 s, its peak session PSS 8.11 GiB against 10.47 GiB, and
  its swap use peaked at 7.99 GiB against 4.93 GiB, all against the pool of `laptop-20261104T172458Z-11ac40f5`
  (354.57 s).

**How many local sessions fit.** The estimate is floor((MemAvailable at start - R) / peak use),
with R the larger of 2 GiB and 10% of MemTotal. From `laptop-20261104T172458Z-11ac40f5`:
MemAvailable at start 20.78 GiB, MemTotal 30.72 GiB, R 3.07 GiB, peak use 10.47 GiB, so
floor((20.78 - 3.07) / 10.47) = 1: serialised, at most one local hhemt pytest session at a
time. This is an estimate from one run. It does not cover the two limits memory does not
cover: two sessions never share one worktree's build cache, and every session takes one
worker per core.

**The Rivanna tier** is the `compile_tier` tests, run through the pytest harness kept with
the private companion repository. Solver compiles and solver runs belong to Rivanna scratch and never to
a laptop. The harness runs the tests as a SLURM array, and its `scope` token states what a
run covered:

- `scope=array` is the array alone. Inside a SLURM allocation a set of tests structurally
  cannot evaluate, so an array run covers that complement not at all.
- `scope=union` is the array plus its complement, and only it supports a claim about the
  suite.

A Rivanna run reports at `scope=array` unless its complement also ran, and no suite-level
green is claimed from an array-scoped result. This page gives no Rivanna timing because
none has been recorded yet. The harness
README describes a complement run, but no runner for it exists at present, so a
`scope=union` result cannot currently be produced.

## The fixture cache, and why a rebuild is silent

The expensive fixtures build an analysis tree once and reuse it. Reuse is what makes
the compile tier bearable, and it is governed by a marker file the builder writes
into the tree when it finishes.

The marker records the *provenance* of the code that built the tree: a digest over
the three builder sources in `tests/`, plus the commit the toolkit was at. Any
difference makes the marker stale, and **a stale marker rebuilds the tree exactly as
a missing one does.** Of the marker's four states, only one, a tree that is present
but incomplete, prints anything. Stale and missing both rebuild in silence.

Two consequences follow, and the second is the one that costs time.

**The shared cache is invalidated by every commit.** Provenance includes the toolkit's
current commit, so committing anything at all, including a documentation change,
makes every cached tree stale. This is not a rare condition; it is the normal state
of any run that follows a commit.

**A first run at any new provenance is not comparable to an earlier one.** A test that
goes red on such a run may be a regression, or it may be an artefact of the rebuild
that ran underneath it. The run cannot tell you which, because the rebuild is silent.
Before treating a new red as a regression, run it a second time at the same
provenance. The triggers are: any commit, any edit to one of the builder sources, and
anything that relocates the analysis tree, which the array suite's component split
requires; see the
[CLI reference](../reference/cli.md#hhemt-test-toolkit) for that last case.

Relocating the tree carries a further cost worth pricing before you ask for it: one
shared fixture build becomes one build per relocated part, paid in the same wall clock
the split is trying to save.

## The compile venue, and why the token relaxes rather than arms

Both tiers run under a guard that is armed by default: a pytest session refuses to
build the solver and refuses to execute a simulation unless the session declares a
permitted *venue*.

The guard exists because those two operations are the ones that reach outside the
process, and a machine that cannot support them fails in ways that look like test
failures rather than like environment problems. Declaring a venue makes the
requirement explicit at the point it binds.

The token relaxes and never arms. Absent, misspelt, or unrecognised, it lands on the
safe side at both places that read it, so a typo tightens the guard instead of
disabling it. That direction is deliberate: a guard whose failure mode is silent
permission is worse than no guard, because it reports success from a machine that
never ran the thing.

What the token names is a *capability* rather than a machine, and which machines this
project actually uses for those runs is a project rule rather than a property of the
code. Both are stated in
[Contributing](../contributing.md#workflow).

## Why the long suite is an array

The compile tier takes hours. Run end to end on one machine it is long enough that
nobody runs it, which turns a gate into a formality. Chunking it across a scheduler
array turns wall clock into a scheduling problem instead of a waiting problem.

Chunking has a consequence for how results are read. Each chunk reports its own
outcome, so a verdict for the suite is a computation over chunks rather than a single
exit code, and it has to account for chunks that never reported at all. That is why
aggregation is a separate step with its own output rather than something the last
chunk does on its way out.

## What a green claim means

Two results look alike and are not.

A run whose scope covers the array plus its complement supports a claim about the
suite. A run whose scope covers only the array does not, because the complement's
tests were never executed under it.

A triage run is narrower still: it re-runs the previous run's failing and unevaluated
set, so a green triage means those specific tests now pass, and it says nothing about
the tests triage did not select. The toolkit enforces the distinction in the
filenames it writes rather than trusting a reader to remember it. The
[CLI reference](../reference/cli.md#hhemt-test-toolkit) states both rules precisely.
