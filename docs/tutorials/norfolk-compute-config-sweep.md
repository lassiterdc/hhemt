# Compute configurations: how many cores should one simulation get?

A **compute configuration** is the answer to "how does one simulation use the machine": how
many MPI ranks, how many OpenMP threads per rank, how many nodes. This page is about the
decision that follows from comparing them. It states the decision the data supports, shows
the evidence, and says how far that evidence can be trusted. The report that holds the
evidence is embedded at the end.

The experiment runs one storm, Hurricane Irene (2011-08-27 00:44 to 2011-08-28 08:58), on
the Norfolk, VA case study at a 3.5 m grid, with the coupled TRITON-SWMM model only. It runs
that one storm under 15 compute configurations on a CPU partition of a university cluster,
natively and with every simulation on one node. The compute configuration is the only
designed difference between the 15 runs; the limits section below says what else differs.

## The decision

**Give each simulation 8 cores.** Within the 8-core options this sweep tried, eight OpenMP
threads on one rank finished fastest, and the 4 rank by 2 thread hybrid finished slowest.
Past 8 cores the sweep shows sharply diminishing returns.

The evidence, all from the table in the next section:

- **Each doubling of cores up to 8 cut the run time by 32 to 38 percent.** Taking the fastest
  configuration at each core count, the speedup per doubling is 1.61 (1 to 2 cores), 1.55
  (2 to 4) and 1.47 (4 to 8).
- **The doubling from 8 to 16 cores cut it by 18.7 percent.** The fastest 8-core run took
  0.999 hours and the fastest 16-core run took 0.812 hours, a speedup of 1.23 for 62.5
  percent more core-hours (8.00 to 13.00).
- **Efficiency falls from 0.62 at 4 cores to 0.46 at 8 and 0.28 at 16.** Efficiency is the
  speedup divided by the core count, so 0.46 means eight cores did the work of 3.7.
- **At 8 cores one split is an outlier.** Three of the four 8-core splits finished between
  0.999 and 1.084 hours (OpenMP 0.999, 2 by 4 hybrid 1.077, MPI 1.084). The 4 rank by 2 thread
  hybrid took 1.331 hours, 33 percent longer than OpenMP, from one run. The same two-thread
  hybrid shape is the fastest 16-core row (8 by 2, 0.812 hours), so the pattern does not
  continue across core counts and the table supports no general rule about splits. The
  [Observed events](norfolk-observed-events.md) ensemble used this 4 by 2 split.
- **At 4 cores the split does not matter much.** The three 4-core splits finished within about
  3 percent of each other (1.472 to 1.516 hours).

Two things follow from that, and they have different standing. The first is a reading of the
table: the knee of the curve, where the return on added cores drops fastest, falls at 8
cores. The second is a judgment: if your ensemble has more simulations than your cores can
run at once, total core-hours set your total time, and at 8 cores each simulation costs 8.00
core-hours against 13.00 at 16. If you have few simulations and need each to finish as soon
as possible, 16 cores is the faster choice and the table prices it.

## The sweep

The sweep is a [sensitivity analysis](../reference/glossary.md#sensitivity-analysis) whose
axis is the compute configuration, the study the [Benchmarking study](benchmarking-study.md)
teaches in full. Its 15 rows are every combination of MPI ranks and OpenMP threads, on one
node, whose product is at most 16 and that the full CPU matrix contains. Each row is one
[member](../reference/glossary.md#member).

```text
member_id,analysis.run_mode,analysis.n_nodes,analysis.n_mpi_procs,analysis.n_omp_threads,analysis.n_gpus,analysis.mem_gb_per_cpu,analysis.hpc_time_min_per_sim
32,serial,1,1,1,0,2,720
26,openmp,1,1,2,0,2,360
27,openmp,1,1,4,0,2,360
28,openmp,1,1,8,0,2,360
29,openmp,1,1,16,0,2,360
22,mpi,1,2,1,0,2,360
23,mpi,1,4,1,0,2,360
24,mpi,1,8,1,0,2,360
25,mpi,1,16,1,0,2,360
4,hybrid,1,2,2,0,2,360
5,hybrid,1,2,4,0,2,360
6,hybrid,1,4,2,0,2,360
7,hybrid,1,2,8,0,2,360
8,hybrid,1,4,4,0,2,360
9,hybrid,1,8,2,0,2,360
```

The analysis config turns the sweep on, points at that sheet, runs the single Irene event,
and selects the `benchmarking` reporting set:

```yaml
toggle_sensitivity_analysis: true
sensitivity_analysis: compute_config_sweep.csv
weather_events_to_simulate: irene_event_index.csv
run_mode: serial
multi_sim_run_method: batch_job
execution_environment: native
report:
  reporting_set: benchmarking
```

Run it with the config-driven command. The dry run validates all 15 members before anything
is submitted.

```bash
hhemt run --system-config my_system.yaml --analysis-config compute_config_sweep.yaml --dry-run
hhemt run --system-config my_system.yaml --analysis-config compute_config_sweep.yaml
```

Add your HPC profile on a cluster, as the Benchmarking study shows. A sensitivity analysis
runs the coupled TRITON-SWMM model alone, so the system config enables that model family only.

## The evidence

The wall-clock figure is the solver's reported total run time for the simulation
(`performance.Total`, taken as the slowest rank), in hours. Core-hours are the wall-clock
hours times the cores. Speedup is the serial wall-clock divided by the configuration's, and
efficiency is the speedup divided by the cores. Each row is one run.

| Cores | Member | Split (ranks x threads) | Mode | Wall-clock (h) | Core-hours | Speedup | Efficiency | Source run |
|---:|---:|---|---|---:|---:|---:|---:|---|
| 1 | 32 | 1 x 1 | serial | 3.672 | 3.67 | 1.00 | 1.00 | reused |
| 2 | 26 | 1 x 2 | openmp | 2.277 | 4.55 | 1.61 | 0.81 | final job |
| 2 | 22 | 2 x 1 | mpi | 2.424 | 4.85 | 1.51 | 0.76 | final job |
| 4 | 27 | 1 x 4 | openmp | 1.472 | 5.89 | 2.50 | 0.62 | final job |
| 4 | 4 | 2 x 2 | hybrid | 1.492 | 5.97 | 2.46 | 0.62 | final job |
| 4 | 23 | 4 x 1 | mpi | 1.516 | 6.07 | 2.42 | 0.61 | final job |
| 8 | 28 | 1 x 8 | openmp | 0.999 | 8.00 | 3.67 | 0.46 | reused |
| 8 | 5 | 2 x 4 | hybrid | 1.077 | 8.62 | 3.41 | 0.43 | reused |
| 8 | 24 | 8 x 1 | mpi | 1.084 | 8.67 | 3.39 | 0.42 | reused |
| 8 | 6 | 4 x 2 | hybrid | 1.331 | 10.65 | 2.76 | 0.34 | reused |
| 16 | 9 | 8 x 2 | hybrid | 0.812 | 13.00 | 4.52 | 0.28 | final job |
| 16 | 25 | 16 x 1 | mpi | 0.832 | 13.32 | 4.41 | 0.28 | final job |
| 16 | 8 | 4 x 4 | hybrid | 0.837 | 13.39 | 4.39 | 0.27 | final job |
| 16 | 29 | 1 x 16 | openmp | 0.863 | 13.80 | 4.26 | 0.27 | final job |
| 16 | 7 | 2 x 8 | hybrid | 0.888 | 14.21 | 4.14 | 0.26 | final job |

Rows are ordered by cores, then by wall-clock. The report draws the same numbers as four
panels (wall-clock, compute cost, speedup and efficiency) against the number of devices, MPI
ranks and OpenMP threads.

## How far to trust it

Read the decision with these limits, which the table makes visible and the report does not
state.

- **Five of the 15 members were not re-simulated in the job that produced the report.**
  Members 5, 6, 24, 28 and 32 come from an earlier run of the same suite, which the toolkit
  reused because their inputs had not changed. The report presents all 15 as one sweep. These
  five include the serial baseline that every speedup divides by, and all four 8-core rows.
  The 8-core rows therefore agree with each other, but the comparison between 8 cores and
  either 4 or 16 crosses two jobs.
- **Each configuration ran once.** The table has one run per row, so it cannot say how much
  run time varies from run to run on the same configuration. The 4-core rows differ by 3
  percent, and a difference that small is not evidence of a better split.
- **One storm, one grid, one machine.** The decision is for this model at 3.5 m on this
  partition. The sweep stops at 16 cores per simulation, so it says nothing about returns
  beyond 16.
- **The report's Metadata page carries older wording.** Its analysis description says six
  configurations on Hurricane Irene, written before the lattice grew to 15. The tables and
  figures embed all 15.
- **It is documentation, not a benchmark.** A study meant to support a published claim needs
  replicates and the full range of core counts.

## The embedded report

The report is one file of about 98 MiB that needs no sibling files, so the frame below takes a
moment to load. Its figure pages load the Plotly library from a CDN (34 of its 40 pages do), so
the figures stay blank without a network connection. The same file opens on its own through
[this link](../reference/bundles/norfolk-compute-config-sweep-report.html).

<iframe
  src="../../reference/bundles/norfolk-compute-config-sweep-report.html"
  width="100%"
  height="600"
  loading="lazy"
  style="border: 1px solid #ccc;"
  title="Interactive analysis report for the compute-configuration sweep">
</iframe>

Open **Benchmarking** under **Key Results** in the sidebar for three figures, one per
x axis (MPI ranks, OpenMP threads and total devices), each with the four panels. **Per
Simulation Results** holds the peak flood depth and conduit flow of each member. To embed a report of your own, follow
[Embedding a report in your own documentation](../reference/example-report.md#embedding-a-report-in-your-own-documentation).
For the event ensemble the same machine runs, see
[Observed events](norfolk-observed-events.md).
