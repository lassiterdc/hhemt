"""
Resource Management for TRITON-SWMM Toolkit

This module handles compute resource allocation and SLURM constraint parsing.
Extracted from TRITONSWMM_analysis as part of Phase 1 refactoring.

Classes:
    ResourceManager: Manages resource allocation calculations and SLURM constraints
"""

import os
import psutil


class ResourceManager:
    """
    Manages compute resource allocation and SLURM constraint parsing.

    This class encapsulates logic for:
    - Calculating effective parallelism based on CPU, GPU, and memory constraints
    - Parsing SLURM environment variables and job allocations
    - Determining resource limits for concurrent task execution

    Attributes:
        in_slurm (bool): Whether code is running in a SLURM environment
        cfg_analysis: Analysis configuration object
    """

    def __init__(self, cfg_analysis, in_slurm: bool = False):
        """
        Initialize ResourceManager.

        Parameters
        ----------
        cfg_analysis : analysis_config
            Configuration object containing HPC and execution parameters
        in_slurm : bool
            Whether the code is running in a SLURM environment
        """
        self.cfg_analysis = cfg_analysis
        self.in_slurm = in_slurm

    def calculate_effective_max_parallel(
        self,
        min_memory_per_function_MiB: int | None = 1024,
        max_concurrent: int | None = None,
        verbose: bool = False,
    ) -> int:
        """
        Calculate the effective maximum parallelism based on CPU, GPU, memory, and SLURM constraints.

        This method respects SLURM job allocation constraints when running in a SLURM environment,
        ensuring that concurrent tasks don't exceed the job's resource allocation.

        Parameters
        ----------
        min_memory_per_function_MiB : int | None
            Minimum memory required per function (MiB).
            If provided, concurrency is reduced to avoid oversubscription.
        max_concurrent : int | None
            CPU-based upper bound on parallelism (e.g., based on cores/threads per task).
            If None, defaults to physical CPU count - 1 (or SLURM allocation if in SLURM).
        verbose : bool
            Print progress messages.

        Returns
        -------
        int
            The effective maximum number of parallel tasks.
        """
        # ----------------------------
        # Determine if we should use SLURM constraints
        # ----------------------------
        use_slurm_constraints = self.in_slurm is True

        # ----------------------------
        # CPU-based limit (with SLURM awareness)
        # ----------------------------
        if max_concurrent is None:
            if use_slurm_constraints:
                # Get SLURM-aware CPU limit
                constraints = self._get_slurm_resource_constraints(verbose=verbose)
                max_concurrent = int(constraints["max_concurrent"])
            else:
                # Pure hardware-based calculation
                physical_cores = psutil.cpu_count(logical=False)
                if isinstance(physical_cores, int) and physical_cores > 1:
                    physical_cores -= 1  # more conservative process count
                max_concurrent_cpu = physical_cores or 1
                # ----------------------------
                # Memory-based limit
                # ----------------------------
                mem_limit = max_concurrent_cpu
                if min_memory_per_function_MiB is not None:
                    available_mem_MiB = psutil.virtual_memory().available // (1024**2)
                    mem_limit = max(1, available_mem_MiB // min_memory_per_function_MiB)

                    if verbose:
                        print(
                            f"Memory-based limit: {mem_limit} "
                            f"(available {available_mem_MiB} MiB, "
                            f"{min_memory_per_function_MiB} MiB per task)",
                            flush=True,
                        )
                # ----------------------------
                # Final concurrency (apply all constraints)
                # ----------------------------
                limits = [int(max_concurrent_cpu), int(mem_limit)]

                max_concurrent = min(limits)

                if verbose and use_slurm_constraints and self.in_slurm:
                    print(
                        f"[SLURM] Using SLURM-aware concurrency limit: {max_concurrent}",
                        flush=True,
                    )

        return max_concurrent

    def _parse_slurm_tasks_per_node(self, tasks_per_node_str: str) -> list[int]:
        """
        Parse SLURM_TASKS_PER_NODE format.

        Examples:
            "4,4,4,4" -> [4, 4, 4, 4]
            "4(x4)" -> [4, 4, 4, 4]
            "8(x2),4" -> [8, 8, 4]

        Parameters
        ----------
        tasks_per_node_str : str
            SLURM_TASKS_PER_NODE environment variable value

        Returns
        -------
        list[int]
            Tasks per node for each allocated node
        """
        tasks = []
        for part in tasks_per_node_str.split(","):
            part = part.strip()
            if "(x" in part:
                # Format: "4(x4)" means 4 tasks repeated 4 times
                count_str, repeat_str = part.split("(x")
                count = int(count_str)
                repeat = int(repeat_str.rstrip(")"))
                tasks.extend([count] * repeat)
            else:
                # Simple format: "4" means 4 tasks
                tasks.append(int(part))
        return tasks

    def _get_slurm_resource_constraints(
        self, verbose: bool = False, min_mem_per_sim_MiB: int = 1024
    ) -> dict:
        """
        Extract and validate SLURM resource constraints from environment variables.

        This method reads all relevant SLURM environment variables and calculates
        the effective maximum concurrency based on:
        - CPU allocation (SLURM_CPUS_PER_TASK, SLURM_CPUS_ON_NODE)
        - GPU allocation (SLURM_GPUS, SLURM_GPUS_ON_NODE, SLURM_GPUS_PER_TASK)
        - Memory constraints (SLURM_MEM_PER_NODE, SLURM_MEM_PER_CPU)
        - Multi-node distribution (SLURM_TASKS_PER_NODE, SLURM_JOB_NUM_NODES)

        Returns
        -------
        dict
            Dictionary containing:
            - max_concurrent: int - Maximum concurrent simulations
            - total_cpus: int - Total CPUs allocated to job
            - total_gpus: int - Total GPUs allocated (if GPU mode)
            - cpus_per_task: int - CPUs per task from SLURM
            - gpus_per_task: int - GPUs per task from config
            - num_nodes: int - Number of nodes allocated
            - cpus_per_node: int - CPUs per node
            - memory_per_node_MiB: int - Memory per node in MiB
            - run_mode: str - CPU or GPU mode
        """
        # ----------------------------
        # Read basic SLURM allocation
        # ----------------------------
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        cpus_on_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 0))

        # If SLURM_CPUS_ON_NODE not set, use psutil
        if cpus_on_node == 0:
            cpus_on_node = psutil.cpu_count(logical=False) or 1

        # ----------------------------
        # Calculate total CPUs available
        # ----------------------------
        # Total CPUs = CPUs per node × number of nodes
        total_cpus = cpus_on_node * num_nodes

        # But respect SLURM_CPUS_PER_TASK if it's more restrictive
        # (i.e., if job was allocated fewer CPUs than node capacity)
        slurm_total_cpus = int(os.environ.get("SLURM_NTASKS", 1)) * cpus_per_task
        if slurm_total_cpus > 0 and slurm_total_cpus < total_cpus:
            total_cpus = slurm_total_cpus

        # ----------------------------
        # Memory constraints
        # ----------------------------
        mem_per_node_MiB = 0
        mem_per_cpu_MiB = 0

        # Try to get memory per node
        mem_per_node_str = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_per_node_str:
            # Format: "123456" (in MB) or "123456M" or "123G"
            mem_per_node_str = mem_per_node_str.rstrip("M")
            try:
                mem_per_node_MiB = int(mem_per_node_str)
            except ValueError:
                if mem_per_node_str.endswith("G"):
                    mem_per_node_MiB = int(mem_per_node_str[:-1]) * 1024

        # Try to get memory per CPU
        mem_per_cpu_str = os.environ.get("SLURM_MEM_PER_CPU")
        if mem_per_cpu_str:
            mem_per_cpu_str = mem_per_cpu_str.rstrip("M")
            try:
                mem_per_cpu_MiB = int(mem_per_cpu_str)
            except ValueError:
                if mem_per_cpu_str.endswith("G"):
                    mem_per_cpu_MiB = int(mem_per_cpu_str[:-1]) * 1024

        # Calculate effective memory per node
        if mem_per_node_MiB == 0 and mem_per_cpu_MiB > 0:
            mem_per_node_MiB = mem_per_cpu_MiB * cpus_on_node

        # ----------------------------
        # GPU constraints (if GPU mode)
        # ----------------------------
        total_gpus = 0
        gpus_per_task = self.cfg_analysis.n_gpus or 0

        if self.cfg_analysis.run_mode == "gpu":
            # Try SLURM_GPUS first (total GPUs)
            total_gpus = int(os.environ.get("SLURM_GPUS", 0))

            # If not set, try SLURM_GPUS_ON_NODE
            if total_gpus == 0:
                gpus_on_node = int(os.environ.get("SLURM_GPUS_ON_NODE", 0))
                total_gpus = gpus_on_node * num_nodes

            # Validate GPU allocation
            if total_gpus == 0:
                raise RuntimeError(
                    "GPU run mode requested, but no GPUs detected via SLURM. "
                    "Check SLURM_GPUS or SLURM_GPUS_ON_NODE environment variables."
                )

            if gpus_per_task > total_gpus:
                raise RuntimeError(
                    f"Each simulation requires {gpus_per_task} GPU(s), "
                    f"but only {total_gpus} GPU(s) allocated to the job."
                )

        # ----------------------------
        # Calculate max concurrency based on mode
        # ----------------------------
        if self.cfg_analysis.run_mode == "gpu":
            # GPU-based limit
            max_concurrent = max(1, total_gpus // gpus_per_task)
        else:
            # CPU-based limit
            mpi_ranks = self.cfg_analysis.n_mpi_procs or 1
            omp_threads = self.cfg_analysis.n_omp_threads or 1
            cpus_per_sim = mpi_ranks * omp_threads

            max_concurrent = max(1, total_cpus // cpus_per_sim) - 1
        # ----------------------------
        # Apply memory constraints
        # ----------------------------
        if mem_per_node_MiB > 0:
            # Estimate memory per simulation
            # This is conservative: assume each sim uses proportional memory
            available_mem_MiB = mem_per_node_MiB * num_nodes
            # For now, we don't have a per-sim memory requirement from config
            # But we can add a safety factor to prevent oversubscription
            # Assume each concurrent task uses ~1GB by default (can be overridden)
            mem_based_limit = max(1, available_mem_MiB // min_mem_per_sim_MiB)
            max_concurrent = min(max_concurrent, mem_based_limit)

        # ----------------------------
        # Respect multi-node task distribution
        # ----------------------------
        tasks_per_node_env = os.environ.get("SLURM_TASKS_PER_NODE")
        if tasks_per_node_env:
            tasks_per_node = self._parse_slurm_tasks_per_node(tasks_per_node_env)
            # Use minimum to be conservative
            if tasks_per_node:
                max_tasks_per_node = min(tasks_per_node)
                max_concurrent = min(max_concurrent, max_tasks_per_node * num_nodes)

        # ----------------------------
        # Verbose logging
        # ----------------------------
        if verbose:
            print(f"[SLURM] Resource Constraints:", flush=True)
            print(f"  Nodes: {num_nodes}", flush=True)
            print(f"  CPUs per node: {cpus_on_node}", flush=True)
            print(f"  Total CPUs Allocated (SLURM): {slurm_total_cpus}", flush=True)
            print(f"  CPUs per task (SLURM): {cpus_per_task}")
            if self.cfg_analysis.run_mode == "gpu":
                print(f"  Total GPUs: {total_gpus}", flush=True)
                print(f"  GPUs per task: {gpus_per_task}", flush=True)
            if mem_per_node_MiB > 0:
                print(f"  Memory per node: {mem_per_node_MiB} MiB", flush=True)
            print(f"  Max concurrent srun tasks: {max_concurrent}", flush=True)

        return {
            "max_concurrent": max_concurrent,
            "total_cpus": total_cpus,
            "total_gpus": total_gpus,
            "cpus_per_task": cpus_per_task,
            "gpus_per_task": gpus_per_task,
            "num_nodes": num_nodes,
            "cpus_per_node": cpus_on_node,
            "memory_per_node_MiB": mem_per_node_MiB,
        }
