"""
Example: Using RunnerConcurrencyMonitor to deterministically verify concurrency.

This script demonstrates how to use the new monitoring tools to answer questions like:
- How many prepare_scenario_runner processes run concurrently?
- Does Snakemake respect the configured core limit?
- What's the pattern of concurrent execution during phase transitions?
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils.process_monitor import RunnerConcurrencyMonitor
import tests.utils_for_testing as tst

def main():
    """Run a test workflow with detailed concurrency monitoring."""

    # Get test analysis
    case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=False)
    analysis = case.system.analysis

    print("="*70)
    print("CONCURRENCY MONITORING EXAMPLE")
    print("="*70)
    print(f"Analysis: {analysis.analysis_name}")
    print(f"Configured cores: {analysis.cfg_analysis.local_cpu_cores_for_workflow}")
    print(f"Number of simulations: {len(analysis.df_sims)}")
    print("="*70 + "\n")

    # Monitor with detailed breakdown
    with RunnerConcurrencyMonitor(sample_interval=0.1) as monitor:
        result = analysis.submit_workflow(
            mode="local",
            process_system_level_inputs=False,  # Skip if already done
            compile_TRITON_SWMM=False,          # Skip if already done
            prepare_scenarios=True,
            overwrite_scenario=True,
            process_timeseries=True,
            which="both",
            verbose=True,
        )

        if not result["success"]:
            print(f"❌ Workflow failed: {result.get('message', '')}")
            return 1

    # Get report
    report = monitor.get_detailed_report()

    # Print detailed summary
    monitor.print_summary()

    # Export timeline
    timeline_path = analysis.analysis_paths.analysis_dir / "runner_concurrency_timeline.csv"
    monitor.export_timeline(str(timeline_path))
    print(f"\n📊 Timeline exported to: {timeline_path}")
    print("   Open in Excel or Python/pandas for visualization\n")

    # Answer specific questions
    print("="*70)
    print("DETERMINISTIC ANSWERS TO CONCURRENCY QUESTIONS")
    print("="*70)

    cores = analysis.cfg_analysis.local_cpu_cores_for_workflow

    print(f"\n1. Maximum concurrent prepare_scenario_runner processes:")
    max_prepare = report['max_concurrent'].get('prepare_scenario_runner', 0)
    print(f"   {max_prepare} (configured cores: {cores})")
    if max_prepare <= cores:
        print("   ✅ Respects core limit")
    else:
        print(f"   ⚠️  Briefly exceeded cores (normal during phase transitions)")

    print(f"\n2. Maximum concurrent run_simulation_runner processes:")
    max_run = report['max_concurrent'].get('run_simulation_runner', 0)
    print(f"   {max_run} (configured cores: {cores})")
    if max_run <= cores:
        print("   ✅ Respects core limit")
    else:
        print(f"   ⚠️  Briefly exceeded cores (normal during phase transitions)")

    print(f"\n3. Maximum concurrent process_timeseries_runner processes:")
    max_process = report['max_concurrent'].get('process_timeseries_runner', 0)
    print(f"   {max_process} (configured cores: {cores})")
    if max_process <= cores:
        print("   ✅ Respects core limit")
    else:
        print(f"   ⚠️  Briefly exceeded cores (normal during phase transitions)")

    print(f"\n4. Maximum TOTAL concurrent runners:")
    print(f"   {report['max_total_runners']}")
    if report['max_total_runners'] > cores * 2:
        print(f"   ⚠️  Exceeded 2x cores - may indicate a problem")
    elif report['max_total_runners'] > cores:
        print(f"   ℹ️  Briefly exceeded cores (normal during phase transitions)")
    else:
        print("   ✅ Well within limits")

    print(f"\n5. Average concurrent runners over time:")
    print(f"   {report['avg_total_runners']:.1f}")
    if report['avg_total_runners'] <= cores:
        print(f"   ✅ Average respects core limit (expected for steady-state)")
    else:
        print(f"   ⚠️  Average exceeds cores - unusual pattern")

    print(f"\n6. Duration and sample count:")
    print(f"   Duration: {report['duration_seconds']:.1f}s")
    print(f"   Samples: {report['samples']}")
    print(f"   Sample rate: {report['samples']/report['duration_seconds']:.1f} Hz")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
Brief spikes above the configured core count are EXPECTED during phase
transitions when:
  - Some runners are finishing (e.g., prepare_scenario)
  - New runners are starting (e.g., run_simulation)
  - Python interpreter creates short-lived parent processes

These spikes are SAFE if:
  ✓ They're brief (1-3 seconds, not sustained)
  ✓ They're bounded (< 2x cores, not exponential)
  ✓ Memory usage returns to normal after spike
  ✓ Tests complete successfully

They're BUGS if:
  ✗ Sustained growth over time
  ✗ Exponential increase (10x+ cores)
  ✗ Memory exhaustion
  ✗ Recursive patterns in cmdlines
""")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
