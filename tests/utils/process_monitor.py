"""
Process monitoring utilities for regression testing.

Helps detect process explosion bugs by tracking process counts during test execution.
"""

import psutil
import os
import time
from threading import Thread, Event
from typing import Dict, List, Tuple
from collections import defaultdict


class ProcessMonitor:
    """
    Monitor process counts during test execution to detect process explosion.

    Usage:
        with ProcessMonitor(max_expected=10) as monitor:
            # Run test code that might spawn processes
            run_workflow()

        # After context exits, check results
        assert monitor.max_processes <= 10, f"Process explosion: {monitor.max_processes} processes"
    """

    def __init__(
        self,
        max_expected: int,
        sample_interval: float = 0.1,
        process_name_filter: str | None = None
    ):
        """
        Initialize process monitor.

        Parameters
        ----------
        max_expected : int
            Maximum expected number of processes (test fails if exceeded)
        sample_interval : float
            How often to sample process count (seconds)
        process_name_filter : str | None
            If provided, only count processes with names containing this string
        """
        self.max_expected = max_expected
        self.sample_interval = sample_interval
        self.process_name_filter = process_name_filter

        self.max_processes = 0
        self.process_counts: List[int] = []
        self.timestamps: List[float] = []

        self._stop_event = Event()
        self._monitor_thread: Thread | None = None

    def _count_processes(self) -> int:
        """Count processes matching filter criteria."""
        current_user = os.getuid()
        count = 0

        for proc in psutil.process_iter(['pid', 'name', 'uids']):
            try:
                # Only count processes owned by current user
                if proc.info['uids'].real != current_user:
                    continue

                # Apply name filter if specified
                if self.process_name_filter:
                    if self.process_name_filter not in proc.info['name']:
                        continue

                count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return count

    def _monitor_loop(self):
        """Background thread that samples process counts."""
        while not self._stop_event.is_set():
            count = self._count_processes()
            self.process_counts.append(count)
            self.timestamps.append(time.time())

            if count > self.max_processes:
                self.max_processes = count

            time.sleep(self.sample_interval)

    def __enter__(self):
        """Start monitoring."""
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Stop monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def assert_no_explosion(self, margin: float = 1.5):
        """
        Assert that process count didn't explode beyond expected.

        Parameters
        ----------
        margin : float
            Allow up to margin * max_expected processes
        """
        threshold = int(self.max_expected * margin)
        assert self.max_processes <= threshold, (
            f"Process explosion detected: {self.max_processes} processes "
            f"(expected ≤ {self.max_expected}, threshold with margin: {threshold})"
        )

    def get_report(self) -> Dict:
        """Get summary report of monitoring results."""
        return {
            "max_processes": self.max_processes,
            "max_expected": self.max_expected,
            "avg_processes": sum(self.process_counts) / len(self.process_counts) if self.process_counts else 0,
            "samples": len(self.process_counts),
            "explosion_detected": self.max_processes > self.max_expected * 1.5,
        }


class RunnerConcurrencyMonitor:
    """
    Monitor concurrent runner processes with detailed breakdown by runner type.

    Tracks prepare_scenario_runner, run_simulation_runner, and process_timeseries_runner
    separately to provide deterministic verification of concurrency limits.

    Usage:
        with RunnerConcurrencyMonitor(sample_interval=0.1) as monitor:
            analysis.submit_workflow(mode="local", ...)

        # Get detailed breakdown
        report = monitor.get_detailed_report()
        print(f"Max concurrent prepare: {report['max_concurrent']['prepare_scenario_runner']}")
        print(f"Max concurrent simulate: {report['max_concurrent']['run_simulation_runner']}")
        print(f"Max concurrent process: {report['max_concurrent']['process_timeseries_runner']}")

        # Export timeline for visualization
        monitor.export_timeline("runner_timeline.csv")
    """

    def __init__(self, sample_interval: float = 0.1):
        """
        Initialize runner concurrency monitor.

        Parameters
        ----------
        sample_interval : float
            How often to sample process counts (seconds)
        """
        self.sample_interval = sample_interval

        # Track runner-specific counts over time
        self.runner_counts_timeline: List[Dict[str, int]] = []
        self.timestamps: List[float] = []

        # Track maximum concurrent runners by type
        self.max_concurrent: Dict[str, int] = defaultdict(int)

        # Track total maximum
        self.max_total_runners = 0

        self._stop_event = Event()
        self._monitor_thread: Thread | None = None
        self._start_time = 0.0

    def _count_runners(self) -> Dict[str, int]:
        """
        Count active runner processes by type.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping runner type to count, e.g.:
            {
                'prepare_scenario_runner': 2,
                'run_simulation_runner': 4,
                'process_timeseries_runner': 1,
                'total': 7
            }
        """
        current_user = os.getuid()
        counts = defaultdict(int)

        runner_patterns = [
            'prepare_scenario_runner',
            'run_simulation_runner',
            'process_timeseries_runner',
            'consolidate_workflow',
            'setup_workflow'
        ]

        for proc in psutil.process_iter(['pid', 'cmdline', 'uids', 'name']):
            try:
                # Only count processes owned by current user
                if proc.info['uids'].real != current_user:
                    continue

                # Check if it's a Python process running a runner script
                cmdline = proc.info.get('cmdline') or []
                cmdline_str = ' '.join(cmdline)

                for pattern in runner_patterns:
                    if pattern in cmdline_str:
                        counts[pattern] += 1
                        counts['total'] += 1
                        break  # Don't double-count

            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                pass

        return dict(counts)

    def _monitor_loop(self):
        """Background thread that samples runner counts."""
        while not self._stop_event.is_set():
            counts = self._count_runners()
            timestamp = time.time() - self._start_time

            self.runner_counts_timeline.append(counts)
            self.timestamps.append(timestamp)

            # Update maximum concurrent counts
            for runner_type, count in counts.items():
                if count > self.max_concurrent[runner_type]:
                    self.max_concurrent[runner_type] = count

            # Track total maximum
            total = counts.get('total', 0)
            if total > self.max_total_runners:
                self.max_total_runners = total

            time.sleep(self.sample_interval)

    def __enter__(self):
        """Start monitoring."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Stop monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def get_detailed_report(self) -> Dict:
        """
        Get detailed concurrency report.

        Returns
        -------
        Dict
            Report containing:
            - max_concurrent: Dict of max concurrent runners by type
            - max_total_runners: Maximum total concurrent runners
            - avg_total_runners: Average number of concurrent runners
            - samples: Number of samples collected
            - duration_seconds: Total monitoring duration
        """
        total_counts = [counts.get('total', 0) for counts in self.runner_counts_timeline]

        return {
            'max_concurrent': dict(self.max_concurrent),
            'max_total_runners': self.max_total_runners,
            'avg_total_runners': sum(total_counts) / len(total_counts) if total_counts else 0,
            'samples': len(self.runner_counts_timeline),
            'duration_seconds': self.timestamps[-1] if self.timestamps else 0,
        }

    def export_timeline(self, filepath: str) -> None:
        """
        Export timeline to CSV for visualization.

        Parameters
        ----------
        filepath : str
            Path to output CSV file
        """
        import csv

        # Get all runner types that appeared
        all_runner_types = set()
        for counts in self.runner_counts_timeline:
            all_runner_types.update(counts.keys())
        all_runner_types.discard('total')  # Will compute this separately
        all_runner_types = sorted(all_runner_types)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['timestamp_s'] + all_runner_types + ['total'])

            # Data rows
            for timestamp, counts in zip(self.timestamps, self.runner_counts_timeline):
                row: List[str | int] = [f"{timestamp:.2f}"]
                for rt in all_runner_types:
                    row.append(counts.get(rt, 0))
                row.append(counts.get('total', 0))
                writer.writerow(row)

    def print_summary(self) -> None:
        """Print a human-readable summary of concurrency."""
        report = self.get_detailed_report()

        print("\n" + "="*60)
        print("Runner Concurrency Summary")
        print("="*60)
        print(f"Duration: {report['duration_seconds']:.1f}s ({report['samples']} samples)")
        print(f"\nMax Total Concurrent Runners: {report['max_total_runners']}")
        print(f"Avg Total Concurrent Runners: {report['avg_total_runners']:.1f}")
        print("\nMax Concurrent by Type:")
        for runner_type, max_count in sorted(report['max_concurrent'].items()):
            if runner_type != 'total':
                print(f"  {runner_type:30s}: {max_count}")
        print("="*60 + "\n")
