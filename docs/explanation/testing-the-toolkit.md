# Testing the toolkit

Why the test suite is shaped the way it is. The commands live in
[Contributing](../contributing.md#workflow) and in the
[CLI reference](../reference/cli.md#hhemt-test-toolkit); this page is the reasoning
behind them, and it deliberately carries no commands of its own.

## Two tiers, and why the split does not depend on your machine

A test that compiles the coupled solver or executes a simulation costs minutes. A
test that decides something before a simulation starts costs milliseconds. Running
both sets on every change would make the fast feedback loop useless, so the suite is
split into a fast tier and a compile tier.

The split is derived rather than declared. Each test's tier follows from the fixtures
it requests, resolved when pytest collects the suite, so the same test lands in the
same tier on every machine. An earlier arrangement decided the tier from whether a
compiler happened to be on your PATH, which meant two developers running the same
recipe could gate on different sets of tests without either of them knowing.

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
