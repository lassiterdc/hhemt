"""
Execution strategies for TRITON-SWMM simulations.

This module provides different execution strategies for running simulations:
- SerialExecutor: Sequential execution on a single core
- LocalConcurrentExecutor: Parallel execution on local machine using ThreadPoolExecutor
- SlurmExecutor: Parallel execution on HPC using SLURM srun tasks

All executors follow the ExecutionStrategy protocol.
"""

import time
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class ExecutionStrategy(Protocol):
    """Protocol for simulation execution strategies."""

    def execute_simulations(
        self,
        launch_functions: List[Tuple],
        max_concurrent: Optional[int],
        verbose: bool,
    ) -> List[str]:
        """
        Execute simulations and return completion statuses.

        Parameters
        ----------
        launch_functions : List[Tuple]
            List of tuples (launcher, finalize_sim) from _create_subprocess_sim_run_launcher().
            launcher() starts the process (non-blocking), finalize_sim() waits and updates logs.
        max_concurrent : Optional[int]
            Maximum number of concurrent simulations
        verbose : bool
            If True, print progress messages

        Returns
        -------
        List[str]
            List of simulation statuses
        """
        ...


class SerialExecutor:
    """Sequential simulation execution."""

    def __init__(self, analysis: "TRITONSWMM_analysis"):
        """
        Initialize SerialExecutor.

        Parameters
        ----------
        analysis : TRITONSWMM_analysis
            Parent analysis object
        """
        self.analysis = analysis

    def execute_simulations(
        self,
        launch_functions: List[Tuple],
        max_concurrent: Optional[int] = None,
        verbose: bool = True,
    ) -> List[str]:
        """
        Execute simulations sequentially (one at a time).

        Parameters
        ----------
        launch_functions : List[Tuple]
            List of tuples (launcher, finalize_sim) from _create_subprocess_sim_run_launcher().
            launcher() starts the process (non-blocking), finalize_sim() waits and updates logs.
        max_concurrent : Optional[int]
            Ignored for serial execution (included for protocol compatibility)
        verbose : bool
            If True, print progress messages

        Returns
        -------
        List[str]
            List of "completed" statuses for each simulation
        """
        if verbose:
            print(
                f"[Serial] Running {len(launch_functions)} simulations sequentially",
                flush=True,
            )

        results = []
        for idx, (launcher, finalize_sim) in enumerate(launch_functions):
            if verbose:
                print(
                    f"[Serial] Starting simulation {idx + 1}/{len(launch_functions)}",
                    flush=True,
                )

            # Launch simulation
            proc, start_time, sim_logfile, lf = launcher()

            # Wait for completion and update logs
            finalize_sim(proc, start_time, sim_logfile, lf)

            results.append("completed")

            if verbose:
                print(
                    f"[Serial] Completed simulation {idx + 1}/{len(launch_functions)}",
                    flush=True,
                )

        # Update analysis log after all simulations
        self.analysis._update_log()

        return results


class LocalConcurrentExecutor:
    """Concurrent simulation execution on local machine using ThreadPoolExecutor."""

    def __init__(self, analysis: "TRITONSWMM_analysis"):
        """
        Initialize LocalConcurrentExecutor.

        Parameters
        ----------
        analysis : TRITONSWMM_analysis
            Parent analysis object
        """
        self.analysis = analysis

    def execute_simulations(
        self,
        launch_functions: List[Tuple],
        max_concurrent: Optional[int] = None,
        verbose: bool = True,
    ) -> List[str]:
        """
        Run simulations concurrently on a desktop/local machine.

        The launcher pattern is non-blocking - each launcher starts a process
        and returns immediately, while finalize_sim waits for completion.
        This method uses ThreadPoolExecutor to run multiple simulations concurrently.

        Parameters
        ----------
        launch_functions : List[Tuple]
            List of tuples (launcher, finalize_sim) from _create_subprocess_sim_run_launcher().
            launcher() starts the process (non-blocking), finalize_sim() waits and updates logs.
        max_concurrent : Optional[int]
            Maximum number of concurrent simulations. If None, calculated automatically.
        verbose : bool
            If True, print progress messages

        Returns
        -------
        List[str]
            List of simulation statuses ("completed" or "failed")
        """
        use_gpu = self.analysis.cfg_analysis.run_mode == "gpu"

        # ----------------------------
        # Determine parallelism
        # ----------------------------
        if max_concurrent is None:
            # ----------------------------
            # Determine GPU parallelism (TODO)
            # ----------------------------
            if use_gpu:
                raise ValueError(
                    "Currently desktop-based simulations are not designed to use GPUs. "
                    "Feature must be built out."
                )

            # ----------------------------
            # Calculate effective max parallel with all constraints
            # ----------------------------
            min_memory_per_sim_MiB = 1024  # Conservative default
            max_concurrent = self.analysis.calculate_effective_max_parallel(
                min_memory_per_function_MiB=min_memory_per_sim_MiB,
                max_concurrent=max_concurrent,
                verbose=verbose,
            )

        if verbose:
            print(
                f"[Local] Running up to {max_concurrent} simulations concurrently",
                flush=True,
            )

        # ----------------------------
        # Execute launchers with ThreadPoolExecutor
        # ----------------------------
        results = []

        def execute_sim(launcher, finalize_sim):
            """Execute a single simulation: launch and finalize."""
            proc, start_time, sim_logfile, lf = launcher()
            finalize_sim(proc, start_time, sim_logfile, lf)
            return "completed"

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(execute_sim, launcher, finalize_sim)
                for launcher, finalize_sim in launch_functions
            ]

            for idx, future in enumerate(as_completed(futures)):
                if verbose:
                    print(
                        f"[Local] Simulation {idx + 1}/{len(launch_functions)} completed",
                        flush=True,
                    )
                try:
                    results.append(future.result())
                except Exception as e:
                    if verbose:
                        print(f"[Local] Simulation failed with error: {e}", flush=True)
                    results.append("failed")

        # Update analysis log after all simulations
        self.analysis._update_log()

        return results


class SlurmExecutor:
    """Concurrent simulation execution on HPC using SLURM srun tasks."""

    def __init__(self, analysis: "TRITONSWMM_analysis"):
        """
        Initialize SlurmExecutor.

        Parameters
        ----------
        analysis : TRITONSWMM_analysis
            Parent analysis object
        """
        self.analysis = analysis

    def execute_simulations(
        self,
        launch_functions: List[Tuple],
        max_concurrent: Optional[int] = None,
        verbose: bool = True,
    ) -> List[str]:
        """
        Launch simulations concurrently on an HPC system using SLURM.

        Uses a pool-based approach with process polling to limit concurrent srun tasks
        and avoid resource contention.

        This method honors all relevant SLURM environment variables to ensure
        concurrent simulations respect the job's resource allocation:

        **CPU-based constraints:**
        - SLURM_JOB_NUM_NODES: Number of nodes allocated
        - SLURM_CPUS_ON_NODE: CPUs per node
        - SLURM_CPUS_PER_TASK: CPUs per task (from job allocation)
        - SLURM_NTASKS: Total tasks allocated
        - SLURM_TASKS_PER_NODE: Task distribution across nodes

        **GPU-based constraints (if run_mode == "gpu"):**
        - SLURM_GPUS: Total GPUs allocated
        - SLURM_GPUS_ON_NODE: GPUs per node
        - SLURM_GPUS_PER_TASK: GPUs per task (from config)

        **Memory constraints:**
        - SLURM_MEM_PER_NODE: Memory per node
        - SLURM_MEM_PER_CPU: Memory per CPU

        Parameters
        ----------
        launch_functions : List[Tuple]
            List of tuples (launcher, finalize_sim) from _create_subprocess_sim_run_launcher().
            launcher() starts the process (non-blocking), finalize_sim() waits and updates logs.
        max_concurrent : Optional[int]
            Maximum number of concurrent tasks. If None, automatically
            calculated from SLURM environment variables and job configuration.
        verbose : bool
            If True, prints detailed resource constraint information.

        Returns
        -------
        List[str]
            List of simulation statuses, in completion order.

        Raises
        ------
        RuntimeError
            If GPU mode is requested but no GPUs are detected, or if
            simulation resource requirements exceed job allocation.
        """
        # ----------------------------
        # Get SLURM resource constraints
        # ----------------------------
        constraints = self.analysis._resource_manager._get_slurm_resource_constraints(
            verbose=verbose
        )
        num_nodes = constraints["num_nodes"]
        total_cpus = constraints["total_cpus"]
        total_gpus = constraints["total_gpus"]

        if max_concurrent is None:
            max_concurrent = int(constraints["max_concurrent"])

        # ----------------------------
        # Validate simulation requirements
        # ----------------------------
        n_nodes_per_sim = self.analysis.cfg_analysis.n_nodes or 1
        mpi_ranks = self.analysis.cfg_analysis.n_mpi_procs or 1
        omp_threads = self.analysis.cfg_analysis.n_omp_threads or 1
        cpus_per_sim = mpi_ranks * omp_threads
        gpus_per_sim = self.analysis.cfg_analysis.n_gpus or 0

        if n_nodes_per_sim > num_nodes:  # type: ignore
            raise RuntimeError(
                f"Each simulation requires {n_nodes_per_sim} node(s), "
                f"but job only has {num_nodes}."  # type: ignore
            )

        if cpus_per_sim > total_cpus:
            raise RuntimeError(
                f"Each simulation requires {cpus_per_sim} CPUs, "
                f"but job only has {total_cpus}."
            )

        if self.analysis.cfg_analysis.run_mode == "gpu":
            if total_gpus == 0:
                raise RuntimeError(
                    "GPU run mode requested, but no GPUs detected via SLURM. "
                    "Check SLURM_GPUS or SLURM_GPUS_ON_NODE environment variables."
                )
            if gpus_per_sim > total_gpus:
                raise RuntimeError(
                    f"Each simulation requires {gpus_per_sim} GPU(s), "
                    f"but only {total_gpus} GPU(s) allocated to the job."
                )

        if verbose:
            print(
                f"[SLURM] Running {len(launch_functions)} simulations "
                f"(max {max_concurrent} concurrent tasks)",
                flush=True,
            )

        # ----------------------------
        # Process polling-based concurrent execution
        # ----------------------------
        results: List[str] = []
        running_processes: dict = (
            {}
        )  # {proc: (finalize_sim, start_time, sim_logfile, lf)}
        pending_launchers = list(
            launch_functions
        )  # Queue of (launcher, finalize_sim) tuples

        # Launch initial batch up to max_concurrent
        while len(running_processes) < max_concurrent and pending_launchers:
            launcher, finalize_sim = pending_launchers.pop(0)
            proc, start_time, sim_logfile, lf = launcher()
            running_processes[proc] = (finalize_sim, start_time, sim_logfile, lf)
            if verbose:
                print(
                    f"[SLURM] Launched simulation ({len(running_processes)} running, "
                    f"{len(pending_launchers)} pending)",
                    flush=True,
                )

        # Poll and manage running processes
        while running_processes or pending_launchers:
            # Check which processes have completed
            completed_procs = []
            for proc in list(running_processes.keys()):
                if proc.poll() is not None:  # Process finished
                    completed_procs.append(proc)

            # Finalize completed processes
            for proc in completed_procs:
                finalize_sim, start_time, sim_logfile, lf = running_processes.pop(proc)
                finalize_sim(proc, start_time, sim_logfile, lf)
                results.append("completed")

                if verbose:
                    print(
                        f"[SLURM] Simulation completed ({len(running_processes)} running, "
                        f"{len(pending_launchers)} pending)",
                        flush=True,
                    )

                # Launch next pending simulation
                if pending_launchers:
                    launcher, finalize_sim = pending_launchers.pop(0)
                    proc, start_time, sim_logfile, lf = launcher()
                    running_processes[proc] = (
                        finalize_sim,
                        start_time,
                        sim_logfile,
                        lf,
                    )
                    if verbose:
                        print(
                            f"[SLURM] Launched simulation ({len(running_processes)} running, "
                            f"{len(pending_launchers)} pending)",
                            flush=True,
                        )

            # Small sleep to avoid busy-waiting
            if running_processes:
                time.sleep(0.1)

        # Update analysis log after all simulations
        self.analysis._update_log()

        return results
