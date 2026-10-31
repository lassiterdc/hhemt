# Tutorials Table of Contents

Learning-oriented, end-to-end walkthroughs that take you from zero to a working
coupled TRITON-SWMM run.

Most tutorials are a fixed sequence with no decision points. Follow one top to
bottom and it is expected to succeed on your own machine. The two worked examples
at the end of this page are read rather than run: each embeds a finished report. If you want to
accomplish one specific task instead, start from the
[How-To Guides](../how-to/index.md).

## The headline: a benchmarking study

[Run a benchmarking study](benchmarking-study.md) is the study the toolkit was
designed to run: one model, many compute configurations, and one report that says
which configuration to use on your hardware. It carries four example setups,
from two members on your own machine to the same study repeated on a second
cluster, and each setup is reproducible on its own.

Start there if you already have a working install and want to know what your
cluster can do with the model you care about.

## The walkthroughs

- [Quickstart](quickstart.md): clone, create the environment, and run a
  coupled simulation from an interactive Python session, serial and dispatched
  locally on your own machine.
- [Norfolk end-to-end](norfolk-end-to-end.md): the full path on the Norfolk,
  VA coastal-flooding case study: setup, smoke test, a local serial run,
  varying the compute configuration, and a rendered report.

If you have never run the toolkit, do the Quickstart first. It takes a fresh
clone to a finished report and gives you the environment every benchmarking
setup assumes. The Norfolk walkthrough assumes the Quickstart is complete and
varies the same run one axis at a time.

## Worked examples with a finished report

These two pages show what an experiment produces, and each embeds the report it
produced, so you can read the result before you run anything.

- [Observed events](norfolk-observed-events.md): an ensemble over six storms that
  really happened on the Norfolk case study, with all three model families, and how
  its events are selected by key from the observed-event index.
- [Compute configurations](norfolk-compute-config-sweep.md): one storm run under 15
  compute configurations, the decision about how many cores a simulation should get,
  and the evidence and limits behind it.
