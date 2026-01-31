# Process Monitoring Utilities

This directory contains utilities for monitoring and verifying process concurrency during workflow execution.

## Quick Start

### Basic Process Count Monitoring

Use `ProcessMonitor` to detect process explosion bugs:

```python
from tests.utils.process_monitor import ProcessMonitor

# Set expected maximum (e.g., Snakemake + workers + overhead)
with ProcessMonitor(max_expected=7, sample_interval=0.2) as monitor:
    # Run your workflow
    analysis.submit_workflow(mode="local", ...)

# Check results
monitor.assert_no_explosion(margin=2.0)  # Allows 2x expected
report = monitor.get_report()
print(f"Max processes: {report['max_processes']}")
print(f"Average: {report['avg_processes']:.1f}")
```

### Detailed Runner Concurrency Tracking

Use `RunnerConcurrencyMonitor` for **deterministic verification** of concurrent runner counts:

```python
from tests.utils.process_monitor import RunnerConcurrencyMonitor

with RunnerConcurrencyMonitor(sample_interval=0.1) as monitor:
    analysis.submit_workflow(mode="local", ...)

# Get detailed breakdown by runner type
report = monitor.get_detailed_report()
print(f"Max concurrent prepare_scenario_runner: {report['max_concurrent']['prepare_scenario_runner']}")
print(f"Max concurrent run_simulation_runner: {report['max_concurrent']['run_simulation_runner']}")
print(f"Max concurrent process_timeseries_runner: {report['max_concurrent']['process_timeseries_runner']}")

# Print human-readable summary
monitor.print_summary()

# Export timeline for visualization
monitor.export_timeline("timeline.csv")
```

## Example Output

### RunnerConcurrencyMonitor.print_summary()

```
============================================================
Runner Concurrency Summary
============================================================
Duration: 95.3s (953 samples)

Max Total Concurrent Runners: 6
Avg Total Concurrent Runners: 2.4

Max Concurrent by Type:
  prepare_scenario_runner       : 4
  run_simulation_runner         : 4
  process_timeseries_runner     : 4
  consolidate_workflow          : 1
============================================================
```

### Timeline CSV Format

The exported CSV can be visualized in Excel, Python (matplotlib/pandas), or R:

```csv
timestamp_s,prepare_scenario_runner,run_simulation_runner,process_timeseries_runner,consolidate_workflow,total
0.00,0,0,0,0,0
0.10,2,0,0,0,2
0.20,4,0,0,0,4
0.30,4,0,0,0,4
1.50,2,2,0,0,4
2.10,0,4,0,0,4
...
```

## Use Cases

### 1. Regression Testing for Process Explosions

Catch bugs like recursive fork bombs:

```python
@pytest.mark.slow
def test_no_process_explosion(analysis):
    cores = analysis.cfg_analysis.local_cpu_cores_for_workflow
    expected_max = 1 + cores + 2  # Snakemake + workers + overhead

    with ProcessMonitor(max_expected=expected_max) as monitor:
        analysis.submit_workflow(...)

    monitor.assert_no_explosion(margin=2.0)
```

### 2. Verifying Concurrency Limits

Confirm that Snakemake respects configured core limits:

```python
@pytest.mark.slow
def test_runner_concurrency(analysis):
    cores = analysis.cfg_analysis.local_cpu_cores_for_workflow

    with RunnerConcurrencyMonitor() as monitor:
        analysis.submit_workflow(...)

    report = monitor.get_detailed_report()

    # Verify each runner type respects core limits
    assert report['max_concurrent']['prepare_scenario_runner'] <= cores + 2
    assert report['max_concurrent']['run_simulation_runner'] <= cores + 2
    assert report['max_concurrent']['process_timeseries_runner'] <= cores + 2
```

### 3. Performance Analysis

Identify bottlenecks by analyzing concurrent execution patterns:

```python
with RunnerConcurrencyMonitor() as monitor:
    analysis.submit_workflow(...)

monitor.export_timeline("performance_analysis.csv")

# Analyze in pandas/matplotlib
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("performance_analysis.csv")
df.plot(x="timestamp_s", y=["prepare_scenario_runner", "run_simulation_runner", "process_timeseries_runner"])
plt.xlabel("Time (seconds)")
plt.ylabel("Concurrent Runners")
plt.title("Runner Concurrency Timeline")
plt.show()
```

## Technical Details

### ProcessMonitor

- **What it counts**: All Python processes owned by the current user (optionally filtered by name)
- **Sampling**: Background thread samples at specified interval (default: 0.1s)
- **Overhead**: Minimal (<1% CPU) due to psutil efficiency
- **Thread safety**: Uses threading.Event for clean shutdown

### RunnerConcurrencyMonitor

- **What it counts**: Specific runner processes identified by cmdline patterns:
  - `prepare_scenario_runner`
  - `run_simulation_runner`
  - `process_timeseries_runner`
  - `consolidate_workflow`
  - `setup_workflow`
- **How it identifies**: Scans process cmdline for runner script names
- **Accuracy**: Deterministic process counting (not statistical)
- **Export format**: CSV with timestamp and per-runner counts

## Best Practices

1. **Choose the right tool**:
   - Use `ProcessMonitor` for **regression testing** (detect explosions)
   - Use `RunnerConcurrencyMonitor` for **verification** (confirm limits)

2. **Set appropriate margins**:
   - ProcessMonitor: Use 2.0x margin to tolerate brief phase-transition spikes
   - RunnerConcurrencyMonitor: Allow cores + 2 to account for interpreter overhead

3. **Sample rate trade-offs**:
   - Higher rate (0.05s): Better resolution, more overhead
   - Lower rate (0.5s): Less overhead, may miss brief spikes
   - Default (0.1s): Good balance for most use cases

4. **Export timelines for debugging**:
   - Always export when investigating unexpected concurrency
   - Visualize in Excel/Python to identify patterns
   - Compare before/after when optimizing

## Common Patterns

### Understanding Brief Process Spikes

During phase transitions (e.g., prepare → run → process), you may see brief spikes:

```
prepare_scenario_runner: 4 (finishing)
run_simulation_runner: 4 (starting)
Total: 8 (briefly exceeds cores=4)
```

This is **expected behavior** because:
1. Snakemake eagerly starts new jobs as dependencies resolve
2. Python interpreter creates short-lived parent processes
3. Processes take time to exit cleanly

The spikes are:
- **Brief** (1-2 seconds)
- **Bounded** (never recursive/exponential)
- **Harmless** (memory usage returns to normal)

### Distinguishing Normal Spikes from Bugs

| Characteristic | Normal Spike | Bug (Fork Bomb) |
|----------------|--------------|-----------------|
| Duration | Seconds | Sustained/growing |
| Pattern | Phase transitions | Exponential growth |
| Memory | Stable after spike | Continues growing |
| Test result | Passes | OOM or timeout |
| Max count | 2-3x cores | 10x+ cores |

Use `RunnerConcurrencyMonitor` to **deterministically confirm** that spikes are bounded and transient.
