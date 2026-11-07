# Testing the toolkit

The commands live in [Contributing](../contributing.md#workflow) and in the
[CLI reference](../reference/cli.md#hhemt-test-toolkit).

## Tiers and machine independence

A test that compiles the coupled solver or executes a simulation costs minutes. A
test that decides something before a simulation starts costs milliseconds. Running
both sets on every change would make the feedback loop useless, so the suite splits into a
fast tier and a compile tier. The fast tier holds everything the toolkit decides before
a simulation starts, plus the on-disk layout contract. The compile tier holds every test
that compiles TRITON-SWMM or asserts on a tree a simulation produced.

The split is derived and never declared. The `compile_tier` marker follows from the fixtures a
test requests, resolved when pytest collects the suite, so the same test lands in the same
tier on every machine. Deciding the tier from the contents of a contributor's `PATH`
would let contributors running a single recipe gate on different sets of tests.

## Markers and the fast tier

The markers below classify a test:

- `compile_tier` marks a test that compiles the solver or runs a simulation. Collection
  derives the marker from the fixture closure. A test that compiles directly in the body
  of the test carries the marker by hand.
- `requires_snakemake_subprocess` marks a test that starts Snakemake as a child process.
  A child scheduler under xdist is nested parallelism, so the fast-tier recipe leaves the
  tests out, and a second recipe runs them serially.
- `docs` marks a test that compares a published page with the code the page describes. Such
  a test runs no solver, and `tests/docs/conftest.py` applies the marker to every test under
  `tests/docs/`.

The fast tier is every test outside `compile_tier` and `requires_snakemake_subprocess`. The
`test-no-compile-no-subprocess` recipe of the `justfile` runs that tier with xdist over every core and sets
`HHEMT_FORBID_COMPILE=1`, so a test that reaches for the solver is refused and never
builds the solver. `just --list` prints the recipe names. Each recipe is a single shell
command, and the `justfile` holds the exact text, so a machine without `just` runs the
command directly.

The tier needs no solver. A test that needs a finished analysis tree starts from a
committed golden tree and reprocesses that tree. The golden tree is a small finished
analysis that a generator builds. The tests that build a private tree of their own form a
committed, reviewed list, and a test compares the list with the census of build requests,
so a new test of that class fails the comparison until the list names the test.

## The compile tier and the venue

Solver compiles and solver runs need a cluster that provides the toolchain and a scratch
filesystem for the build trees. The fast tier runs on any machine. The compile tier
runs as a scheduler array through `hhemt test toolkit`. The
[CLI reference](../reference/cli.md#hhemt-test-toolkit) describes the command.

## The fixture cache, and why a rebuild is silent

The expensive fixtures build an analysis tree once and reuse the tree. Reuse makes
the compile tier bearable, and a marker file governs the reuse. The builder writes
the marker into the tree when the build finishes.

The marker records the provenance of the code that built the tree. The provenance holds a digest over
the builder sources in `tests/`, a digest over the toolkit's Python sources under
`src/hhemt/`, and the terms that identify the case, the pinned solver source, and the
environment. Any difference makes the marker stale, and **a stale marker rebuilds the
tree as a missing marker does.** Only a tree that is present but incomplete prints
anything. Stale and missing trees both rebuild in silence.

**Every edit to the toolkit's Python sources invalidates the shared cache.** Provenance
includes a digest of every `*.py` file under `src/hhemt/`, read from the working tree, so
editing any of the files makes every cached tree stale. Committing the edit changes no digest term, because the digest reads
the working tree. The digest carries no
commit identity, so a commit that edits none of the files leaves the term unchanged.

**A first run at a new provenance is not comparable to an earlier run.** A test that
goes red on such a run may show a regression, or the red may come from the rebuild that ran
underneath the test, and the run separates neither case because the rebuild is silent.
Run the test a second time at the same provenance before treating a new red as a
regression. The triggers are any edit to a Python file under `src/hhemt/`, any edit to a
builder source, and anything that relocates the analysis tree, such as the component split
of the array suite. The [CLI reference](../reference/cli.md#hhemt-test-toolkit) covers
the component split. Relocation turns a single shared fixture build into one build per relocated part.

Pytest sessions against one checkout share one build cache, and the failures that
result look like real ones. Run one session per checkout.

## The compile venue token

A guard runs under both tiers and is armed by default. A pytest session refuses to
build the solver and refuses to execute a simulation unless the session declares a
permitted *venue*.

The guard exists because building the solver and executing a simulation reach
outside the process, and a machine that is unable to support them fails in ways that read
as test failures and hide the environment problem. Declaring a venue makes the
requirement explicit at the point of binding.

The token relaxes and never arms. Absent, misspelt, or unrecognised, the token lands on the
safe side at both places that read the token, so a typo tightens the guard and never
disables the guard. A guard whose failure mode is silent
permission is worse than no guard, because the guard reports success from a machine that
never ran the tests.

The token names a capability and no machine. `PERMITTED_VENUES` in
`src/hhemt/_compile_venue.py` holds `toolchain` alone. The project rule places the compile tier on a cluster,
and [Contributing](../contributing.md#workflow) states the rule.

## Array execution of the long suite

The compile tier takes hours. Run end to end on one machine, the tier is long enough that
nobody runs the tier, and a gate that nobody runs becomes a formality. Chunking the tier
across a scheduler array converts waiting time into a scheduling problem.

Chunking changes how results are read. Each chunk reports a separate outcome, so a verdict for
the suite is a computation over chunks and no longer a single exit code. The computation
accounts for chunks that never reported. Aggregation is therefore a separate step with a
separate output, and the last chunk skips the aggregation on the way out.

## Scope of a green claim

The harness records the coverage of a run in a `scope` token, and results from different scopes
support different claims.

- `scope=array` is the array alone. A set of tests is structurally unable to evaluate inside a
  scheduler allocation, so an array run covers that complement not at all.
- `scope=union` is the array plus the complement, and only that scope supports a claim
  about the suite.

A triage run is narrower than both scopes. The triage re-runs the failing and unevaluated
set of the previous run, so a green triage covers only the selected tests and the tests the
triage left out stay unevaluated. The toolkit enforces the distinction in the
filenames the toolkit writes and trusts no reader to remember the distinction. The
[CLI reference](../reference/cli.md#hhemt-test-toolkit) states both rules precisely.
