# Observed events: an ensemble over six storms

An **observed-event ensemble** simulates storms that really happened, driven by the
measured rainfall and water level of each one. This tutorial shows what such an ensemble
looks like from the outside: the six events it contains, the config that selects them, the
one command that runs it, and the report it produces. The report is embedded at the end, so
you can read the result without running anything.

The ensemble simulates six events on the Norfolk, VA case study at a 3.5 m grid, each with
all three model families (the coupled TRITON-SWMM model, TRITON alone and SWMM alone). That
is 18 simulations. It is a demonstration of the output, not a flood result: nothing in it
should be cited as a hazard estimate for these storms.

## Prerequisites

--8<-- "prerequisites-base.md"

- The [Quickstart](quickstart.md) completed, so you have a working environment and the
  Norfolk example data on disk.
- A compiled solver on the machine that will run the simulations. See
  [Compile the solver](../how-to/compiling-the-solver.md).

The ensemble ran on a CPU partition of a university cluster, natively (no container), with
every simulation on one node. You can run the same definition on any machine that has the
compiled solver; only the dispatch fields of the analysis config change.

## The six events

The events are not typed into the config. They are a subset cut from the observed-event
index, the table of every storm in the weather record, and the subset file lists the six
rows the ensemble runs. Four of the six are attributed to a named storm, and each of those
attributions cites a public source. The other two are identified by their simulation window
alone, because no public source has yet been confirmed for them.

| Event | Simulation window | Event type | Public source |
|---|---|---|---|
| Event with no confirmed attribution | 2003-09-17 05:44 to 2003-09-19 07:34 | compound | none |
| Mid-Atlantic nor'easter of November 2009 (the remnant of Hurricane Ida) | 2009-11-11 07:10 to 2009-11-14 05:36 | compound | [weather.gov event review](https://www.weather.gov/mhx/nov132009eventreview) |
| Hurricane Irene | 2011-08-27 00:44 to 2011-08-28 08:58 | compound | [pilotonline.com, 28 August 2011](https://www.pilotonline.com/2011/08/28/storm-surge-from-hurricane-irene-falls-short-of-predictions-2/) |
| Event with no confirmed attribution | 2016-09-02 20:44 to 2016-09-04 16:32 | compound | none |
| Hurricane Matthew | 2016-10-08 08:04 to 2016-10-09 18:12 | compound | [nnva.gov storm surge page](https://www.nnva.gov/3052/Storm-Surge) |
| Hurricane Dorian | 2019-09-06 07:46 to 2019-09-07 04:54 | surge | [pilotonline.com, 6 September 2019](https://www.pilotonline.com/2019/09/06/hampton-roads-ducked-the-worst-of-hurricane-dorian/) |

The event type is the observed-event index's own classification: five events are compound
and one is a surge event. The rows are in date order, which is the order of the report's
event pages. The report labels each event from the event index, so treat the table above,
not a label inside the report, as the statement of which events carry a confirmed
attribution.

## The config

Two files describe the ensemble, a system config and an analysis config. The system config
turns on all three model families:

```yaml
toggle_triton_model: true
toggle_tritonswmm_model: true
toggle_swmm_model: true
target_dem_resolution: 3.5
```

The analysis config names the weather record, the event windows, and the subset of events to
run, and says how each simulation uses the machine:

```yaml
weather_events_to_simulate: observed_events_named_subset.csv
toggle_sensitivity_analysis: false
run_mode: hybrid
n_mpi_procs: 4
n_omp_threads: 2
n_nodes: 1
multi_sim_run_method: batch_job
execution_environment: native
```

`run_mode: hybrid` with four MPI ranks of two OpenMP threads gives each simulation eight
cores on one node, and `multi_sim_run_method: batch_job` submits each simulation as its own
job. That 4 rank by 2 thread split is what this ensemble used, and it is not a recommendation:
the [compute-configuration sweep](norfolk-compute-config-sweep.md) ranks it last of the four
8-core splits it tried (1.331 hours against 0.999 for eight OpenMP threads, one run each). The subset file is a table with one row per event and these columns:

```text
year,event_type,event_id,event_start,event_end,description,event_label
```

The first three columns are the event's key in the observed-event index, so the subset
selects rows by key and never restates a storm's rainfall or water level. Fill the remaining
fields of both configs from the Norfolk templates, as the
[Benchmarking study](benchmarking-study.md) describes, and see
[Config filling](../how-to/config-filling.md) for the fields this page leaves out.

## Run it

Validate first. A dry run checks the configs and prints the plan without executing anything.

```bash
hhemt run --system-config my_system.yaml --analysis-config observed_events.yaml --dry-run
hhemt run --system-config my_system.yaml --analysis-config observed_events.yaml
```

On a cluster, add your HPC profile as the Benchmarking study shows. The run simulates the
18 scenarios, processes and consolidates their outputs, validates the finished tree, and
renders the report.

## What the report shows

Open the report below, or [in its own tab](../reference/bundles/norfolk-observed-events-report.html).
Its sidebar follows the default reporting set, described in
[The interactive analysis report](../reference/example-report.md).

- **Errors and Warnings** holds the post-completion validation. For this ensemble all 13
  applicable checks passed and the status export carries 18 rows, one per model family and
  event.
- **Per Simulation Results** holds one page per event. Each page shows the rainfall and
  boundary-condition forcing, then the peak flood depth from TRITON and from TRITON-SWMM, and
  the conduit flow from TRITON-SWMM and from SWMM.
- **Workflow performance** and **Metadata** record what each job cost and which toolkit build
  produced the result.

Comparing the two peak-flood-depth panels on one event is the quickest way to see what the
coupled model adds to the surface-only model, and comparing the two conduit-flow panels shows
what the surface water adds to the pipe network alone.

## The embedded report

The report is one file of about 85 MiB that needs no sibling files, so the frame below takes a
moment to load. Its figure pages load the Plotly library from a CDN (7 of its 13 pages do), so
the figures stay blank without a network connection. The same file opens on its own through the
link above.

<iframe
  src="../../reference/bundles/norfolk-observed-events-report.html"
  width="100%"
  height="600"
  loading="lazy"
  style="border: 1px solid #ccc;"
  title="Interactive analysis report for the observed-event ensemble">
</iframe>

To embed a report of your own, follow
[Embedding a report in your own documentation](../reference/example-report.md#embedding-a-report-in-your-own-documentation).
Next, [choose a compute configuration](norfolk-compute-config-sweep.md) for an ensemble
like this one; the config above is a starting point, and that page shows what other splits cost.
