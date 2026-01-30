# %%
from TRITON_SWMM_toolkit.config import load_system_config
import pandas as pd
import rioxarray as rxr
import numpy as np
import xarray as xr
from pathlib import Path
from rasterio.enums import Resampling
import sys
import TRITON_SWMM_toolkit.utils as ut
import tempfile
from TRITON_SWMM_toolkit.paths import SysPaths
from typing import Optional
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
from TRITON_SWMM_toolkit.plot_system import TRITONSWMM_system_plotting
import subprocess
import time


class TRITONSWMM_system:
    def __init__(self, system_config_yaml: Path) -> None:
        self.system_config_yaml = system_config_yaml
        self.cfg_system = load_system_config(system_config_yaml)

        system_dir = self.cfg_system.system_directory
        tritonswmm_dir = self.cfg_system.TRITONSWMM_software_directory

        # Initialize paths with backend split
        self.sys_paths = SysPaths(
            dem_processed=system_dir / "elevation.dem",
            mannings_processed=system_dir / "mannings.dem",
            # CPU paths (always present)
            TRITON_build_dir_cpu=tritonswmm_dir / "build_cpu",
            compilation_logfile_cpu=system_dir / "compilation_cpu.log",
            compilation_script_cpu=system_dir / "compile_cpu.sh",
            # GPU paths (optional, only if gpu_compilation_backend set)
            TRITON_build_dir_gpu=(
                tritonswmm_dir / "build_gpu"
                if self.cfg_system.gpu_compilation_backend
                else None
            ),
            compilation_logfile_gpu=(
                system_dir / "compilation_gpu.log"
                if self.cfg_system.gpu_compilation_backend
                else None
            ),
            compilation_script_gpu=(
                system_dir / "compile_gpu.sh"
                if self.cfg_system.gpu_compilation_backend
                else None
            ),
            # Backwards compatibility aliases (point to CPU versions)
            TRITON_build_dir=tritonswmm_dir / "build_cpu",
            compilation_logfile=system_dir / "compilation_cpu.log",
            compilation_script=system_dir / "compile_cpu.sh",
        )

        self._analysis: Optional["TRITONSWMM_analysis"] = None
        self.plot = TRITONSWMM_system_plotting(self)

    @property
    def analysis(self) -> "TRITONSWMM_analysis":
        if self._analysis is None:
            raise RuntimeError("No analysis defined. Call add_analysis() first.")
        return self._analysis

    def process_system_level_inputs(
        self, overwrite_if_exists: bool = False, verbose: bool = False
    ):
        self.create_dem_for_TRITON(overwrite_if_exists, verbose)
        if not self.cfg_system.toggle_use_constant_mannings:
            self.create_mannings_file_for_TRITON(overwrite_if_exists, verbose)

    def create_dem_for_TRITON(
        self, overwrite_if_exists: bool = False, verbose: bool = False
    ):
        dem_processed = self.sys_paths.dem_processed
        if dem_processed.exists() and not overwrite_if_exists:
            out = "DEM file already exists. Not rewriting."
        rds_dem_coarse = self._coarsen_dem()
        self._write_raster_formatted_for_TRITON(
            rds_dem_coarse, dem_processed, include_metadata=True
        )
        out = f"wrote {str(dem_processed)}"
        if verbose:
            print(out)
        return

    def create_mannings_file_for_TRITON(
        self, overwrite_if_exists: bool = False, verbose: bool = False
    ):
        mannings_processed = self.sys_paths.mannings_processed
        if mannings_processed.exists() and not overwrite_if_exists:
            out = "Mannings file already exists. Not rewriting."
        include_metadata = False
        rds_mannings_coarse = self._create_mannings_raster_matching_dem()
        self._write_raster_formatted_for_TRITON(
            rds_mannings_coarse,
            mannings_processed,
            include_metadata=include_metadata,
        )
        out = f"wrote {str(mannings_processed)}"
        if verbose:
            print(out)
        return

    def open_processed_mannings_as_rds(self):  # mannings_processed, dem_processed):
        mannings_processed = self.sys_paths.mannings_processed
        dem_processed = self.sys_paths.dem_processed

        dem_header = "".join(ut.read_header(dem_processed, 6))
        mannings_header = "".join(ut.read_header(mannings_processed, 6))
        if dem_header != mannings_header:
            mannings_data = ut.read_text_file_as_string(mannings_processed)
            mannings_with_header = dem_header + mannings_data
            with tempfile.NamedTemporaryFile(suffix=".asc") as tmp:
                tmp.write(mannings_with_header.encode("utf-8"))
                tmp.flush()
                rds_mannings_processed = rxr.open_rasterio(tmp.name).load()  # type: ignore
        else:
            rds_mannings_processed = rxr.open_rasterio(mannings_processed)
        return rds_mannings_processed

    @property
    def mannings_rds(self):
        return self.open_processed_mannings_as_rds()

    @property
    def processed_dem_rds(self):
        return rxr.open_rasterio(self.sys_paths.dem_processed)

    def _create_mannings_raster(self):
        landuse_lookup_file = self.cfg_system.landuse_lookup_file
        landuse_raster = self.cfg_system.landuse_raster
        landuse_colname = self.cfg_system.landuse_lookup_class_id_colname
        mannings_colname = self.cfg_system.landuse_lookup_mannings_colname

        df_lu_lookup = pd.read_csv(landuse_lookup_file).loc[  # type: ignore
            :, [landuse_colname, mannings_colname]
        ]
        rds_lu = rxr.open_rasterio(landuse_raster)
        assert isinstance(rds_lu, xr.DataArray)
        arr = rds_lu.data
        unique_values = np.unique(arr[~np.isnan(arr)])
        no_data_value = rds_lu.rio.nodata
        # create dataframe from landuse vals in the landuse raster
        df_lu_vals = pd.Series(index=unique_values, name="placeholder").to_frame()  # type: ignore
        df_lu_vals.index.name = landuse_colname
        # join the landuse values present in the raster with the lookup table
        df_lu_vals = df_lu_vals.join(
            df_lu_lookup.set_index(landuse_colname), how="left"
        )
        s_lu_mannings_mapping = df_lu_vals[mannings_colname].copy()
        dict_s_lu_mannings = s_lu_mannings_mapping.to_dict()

        rds_mannings_og = xr.apply_ufunc(
            lambda x: dict_s_lu_mannings.get(
                x, no_data_value
            ),  # Replace with mapped value, or keep original if not in dict
            rds_lu,
            keep_attrs=True,  # Keep original raster attributes
            vectorize=True,
        )
        return rds_mannings_og

    def _coarsen_dem(self):  # dem_unprocessed, target_resolution):
        dem_unprocessed = self.cfg_system.DEM_fullres
        target_resolution = self.cfg_system.target_dem_resolution

        rds_dem = rxr.open_rasterio(dem_unprocessed)
        # crs = rds_dem.rio.crs  # type: ignore
        # og_dem_res_xy, og_dem_avg_gridsize = compute_grid_resolution(rds_dem)
        if (rds_dem.data < -100).sum() > 0:  # type: ignore
            sys.exit(
                "Error - gaps found in DEM. Consider interpolating elevations using method = 'nearest' (see below in this function)"
            )
            rds_dem = rds_dem.rio.interpolate_na(method="nearest")
        # coarsen
        rds_dem_coarse = coarsen_georaster(rds_dem, target_resolution)
        return rds_dem_coarse

    def _create_mannings_raster_matching_dem(self, fillna_val=-9999):
        dem_unprocessed = self.cfg_system.DEM_fullres
        target_resolution = self.cfg_system.target_dem_resolution

        rds_mannings = self._create_mannings_raster()
        rds_dem = rxr.open_rasterio(dem_unprocessed)
        crs = rds_dem.rio.crs  # type: ignore
        assert rds_mannings.rio.crs == rds_dem.rio.crs  # type: ignore
        # resample mannings to og dem resolution to ensure exact alignment of final output
        rds_mannings = spatial_resampling(
            rds_mannings, rds_dem, missingfillval=fillna_val
        ).rio.write_crs(crs)
        assert (
            np.isclose(rds_mannings.rio.resolution(), rds_dem.rio.resolution())  # type: ignore
        ).sum() == 2
        rds_mannings_coarse = coarsen_georaster(rds_mannings, target_resolution)
        assert rds_mannings.min().values > 0
        return rds_mannings_coarse

    def _write_raster_formatted_for_TRITON(
        self, rds, output: Path, include_metadata: bool, fillna_val=-9999
    ):
        __, og_avg_gridsize = compute_grid_resolution(rds)
        ncols = rds.x.shape[0]
        nrows = rds.y.shape[0]
        xllcorner = (
            rds.x.values.min() - og_avg_gridsize / 2
        )  # adjusted from center to corner
        yllcorner = (
            rds.y.values.min() - og_avg_gridsize / 2
        )  # adjusted from center to corner
        # define DEM
        raster_metadata = {
            "ncols         ": ncols,
            "nrows         ": nrows,
            "xllcorner     ": xllcorner,
            "yllcorner     ": yllcorner,
            "cellsize      ": og_avg_gridsize,
            "NODATA_value  ": fillna_val,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        if include_metadata:
            self._write_raster(output, rds, raster_metadata)
        else:
            self._write_raster(output, rds)

    def _write_raster(self, fpath_raster, rds, raster_metadata=None):
        if raster_metadata is not None:
            data_write_mode = "a"
            f = open(fpath_raster, "w")
            for key in raster_metadata:
                f.write(key + str(raster_metadata[key]) + "\n")
            f.close()
        else:
            data_write_mode = "w"
        # create dataframe with the right shape
        df_long = (
            rds.to_dataframe("elevation").reset_index().loc[:, ["x", "y", "elevation"]]
        )
        df = df_long.pivot(index="y", columns="x", values="elevation")
        # ensure y is DESCENDING down and x is ASCENDING to the right
        df = df.sort_index(ascending=False)
        cols_sorted = df.columns.sort_values(ascending=True)
        df = df.loc[:, cols_sorted]
        # pad with zeros to achieve consistent spacing in the resulting file
        target_decimal_places = 5
        longest_num = (
            len(str(df.abs().max().max()).split(".")[0]) + target_decimal_places + 1
        )
        df_padded = df.apply(
            self._flt_to_str_certain_num_of_characters,
            args=(target_decimal_places, longest_num),
        )
        # df_padded = df_padded.astype(float)
        df_padded.to_csv(
            fpath_raster, mode=data_write_mode, index=False, header=False, sep=" "
        )

    def _flt_to_str_certain_num_of_characters(
        self, flt, target_decimal_places, longest_num
    ):
        flt = round(flt, target_decimal_places)  # type: ignore
        str_flt = flt.astype(str)
        str_flt = str_flt.apply(lambda x: str(x).ljust(longest_num, "0"))
        return str_flt

    # compilation stuff
    def compile_TRITON_SWMM(
        self,
        backends: Optional[list[str]] = None,
        recompile_if_already_done_successfully: bool = False,
        redownload_triton_swmm_if_exists: bool = False,
        verbose: bool = True,
    ):
        """
        Compile TRITON-SWMM for specified backend(s).

        Parameters
        ----------
        backends : Optional[list[str]]
            List of backends to compile ("cpu", "gpu", or both). If None:
            - Always compiles CPU
            - Compiles GPU only if gpu_compilation_backend is set in config
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        redownload_triton_swmm_if_exists : bool
            If True, re-download TRITON-SWMM source even if it exists
        verbose : bool
            If True, print progress messages
        """

        # Determine which backends to compile
        if backends is None:
            backends = ["cpu"]
            if self.cfg_system.gpu_compilation_backend:
                backends.append("gpu")

        # Download TRITON-SWMM source if needed (shared across backends)
        TRITONSWMM_software_directory = self.cfg_system.TRITONSWMM_software_directory
        if (
            redownload_triton_swmm_if_exists
            or not TRITONSWMM_software_directory.exists()
        ):
            self._download_tritonswmm_source(verbose=verbose)

        # Auto-patch TRITON's cmake for OpenMP builds (before compilation)
        for backend in backends:
            if backend == "cpu":
                self._patch_triton_machine_cmake_for_openmp(
                    TRITONSWMM_software_directory, backend, verbose=verbose
                )

        # Compile each backend sequentially
        for backend in backends:
            if verbose:
                print(f"\n{'=' * 60}", flush=True)
                print(f"Compiling {backend.upper()} Backend", flush=True)
                print("=" * 60, flush=True)

            if backend == "cpu":
                self._compile_backend(
                    backend="cpu",
                    build_dir=self.sys_paths.TRITON_build_dir_cpu,
                    compilation_script=self.sys_paths.compilation_script_cpu,
                    compilation_logfile=self.sys_paths.compilation_logfile_cpu,
                    cmake_backend_flag="-DKokkos_ENABLE_OPENMP=ON",
                    recompile=recompile_if_already_done_successfully,
                    verbose=verbose,
                )
            elif backend == "gpu":
                if self.cfg_system.gpu_compilation_backend is None:
                    raise ValueError(
                        "GPU backend requested but gpu_compilation_backend not set in config. "
                        "Set gpu_compilation_backend='HIP' or 'CUDA' in system config YAML."
                    )

                # Determine Kokkos flag based on config
                if self.cfg_system.gpu_compilation_backend == "HIP":
                    cmake_backend_flag = "-DKokkos_ENABLE_HIP=ON"
                elif self.cfg_system.gpu_compilation_backend == "CUDA":
                    cmake_backend_flag = "-DKokkos_ENABLE_CUDA=ON"
                else:
                    raise ValueError(
                        f"Invalid gpu_compilation_backend: {self.cfg_system.gpu_compilation_backend}. "
                        "Must be 'HIP' or 'CUDA'."
                    )

                self._compile_backend(
                    backend="gpu",
                    build_dir=self.sys_paths.TRITON_build_dir_gpu,  # type: ignore
                    compilation_script=self.sys_paths.compilation_script_gpu,  # type: ignore
                    compilation_logfile=self.sys_paths.compilation_logfile_gpu,  # type: ignore
                    cmake_backend_flag=cmake_backend_flag,
                    recompile=recompile_if_already_done_successfully,
                    verbose=verbose,
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")

        # Print summary
        if verbose:
            self.print_compilation_status()

    def _download_tritonswmm_source(self, verbose: bool = True):
        """Download TRITON-SWMM source code from git repository."""
        TRITONSWMM_software_directory = self.cfg_system.TRITONSWMM_software_directory

        clone_cmd = f"git clone {self.cfg_system.TRITONSWMM_git_URL}"
        branch_checkout_cmd = ""
        if self.cfg_system.TRITONSWMM_branch_key:
            branch_checkout_cmd = (
                f" && git checkout {self.cfg_system.TRITONSWMM_branch_key}"
            )

        if verbose:
            print(
                f"[Download] Cloning TRITON-SWMM to {TRITONSWMM_software_directory}",
                flush=True,
            )

        # Create parent directory
        TRITONSWMM_software_directory.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing directory
        if TRITONSWMM_software_directory.exists():
            import shutil

            shutil.rmtree(TRITONSWMM_software_directory)

        # Clone and checkout
        subprocess.run(
            f'cd "{TRITONSWMM_software_directory.parent}" && {clone_cmd} && '
            f"cd triton{branch_checkout_cmd} && git submodule update --init --recursive",
            shell=True,
            check=True,
        )

    def _patch_triton_machine_cmake_for_openmp(
        self, triton_dir: Path, backend: str, verbose: bool = True
    ) -> bool:
        """
        Patch TRITON's cmake/machine.cmake to filter GPU flags for OpenMP builds.

        On Frontier, machine detection loads cray_HIP.sh which sets HIP launcher flags.
        These cause compilation errors in CPU/OpenMP builds. This patch filters them out.

        Parameters
        ----------
        triton_dir : Path
            Path to TRITON source directory
        backend : str
            Backend being compiled ("cpu" or "gpu")
        verbose : bool
            If True, print detailed before/after diagnostics

        Returns
        -------
        bool
            True if patch was applied, False if already patched or not needed
        """
        import logging

        logger = logging.getLogger(__name__)

        machine_cmake = triton_dir / "cmake" / "machine.cmake"

        if not machine_cmake.exists():
            logger.warning(f"cmake/machine.cmake not found at {machine_cmake}")
            return False

        # Read current content
        content = machine_cmake.read_text()

        # Check if already patched
        if "# TRITON-SWMM toolkit auto-patch" in content:
            if verbose:
                print(
                    "[CMAKE PATCH] cmake/machine.cmake already patched (skipping)",
                    flush=True,
                )
            return False

        # Only patch for CPU backend
        if backend != "cpu":
            return False

        # Find the line to patch (should be around line 216)
        target_line = (
            'set(CMAKE_CXX_FLAGS "${COMPILER_FLAGS} ${COMPILER_FLAGS_APPEND}")'
        )

        if target_line not in content:
            warning_msg = (
                f"Target line not found in cmake/machine.cmake. "
                f"Expected: {target_line}\n"
                f"This may indicate a TRITON version change. Skipping patch."
            )
            logger.warning(warning_msg)
            if verbose:
                print(f"[CMAKE PATCH] ⚠ {warning_msg}", flush=True)
            return False

        # Create backup (only if not already backed up)
        backup_path = machine_cmake.with_suffix(".cmake.backup")
        if not backup_path.exists():
            machine_cmake.rename(backup_path)
            machine_cmake.write_text(content)  # Restore original for patching
            if verbose:
                print(f"[CMAKE PATCH] Created backup: {backup_path}", flush=True)

        # Create patch
        patch_code = """# TRITON-SWMM toolkit auto-patch: Filter GPU flags for OpenMP backend
if(BACKEND STREQUAL "OPENMP")
  # Remove GPU-specific flags from COMPILER_FLAGS for CPU-only builds
  string(REPLACE "-DTRITON_HIP_LAUNCHER" "" COMPILER_FLAGS "${COMPILER_FLAGS}")
  string(REPLACE "-DTRITON_CUDA_LAUNCHER" "" COMPILER_FLAGS "${COMPILER_FLAGS}")
  string(REPLACE "-DACTIVE_GPU=1" "" COMPILER_FLAGS "${COMPILER_FLAGS}")
  message(STATUS "OpenMP backend: Removed GPU launcher flags from COMPILER_FLAGS")
endif()

set(CMAKE_CXX_FLAGS "${COMPILER_FLAGS} ${COMPILER_FLAGS_APPEND}")"""

        # Apply patch
        patched_content = content.replace(target_line, patch_code)
        machine_cmake.write_text(patched_content)

        # Print detailed before/after if verbose
        if verbose:
            print("\n" + "=" * 70, flush=True)
            print(
                "⚠️  PATCHING TRITON cmake/machine.cmake FOR CPU COMPILATION", flush=True
            )
            print("=" * 70, flush=True)
            print(
                "\nWhy: On Frontier HPC, machine detection injects GPU flags even for",
                flush=True,
            )
            print(
                "     CPU builds, causing compilation errors. This patch filters them.",
                flush=True,
            )
            print("\nFile: " + str(machine_cmake), flush=True)
            print("Backup: " + str(backup_path), flush=True)
            print("\n--- BEFORE (original line) ---", flush=True)
            print(target_line, flush=True)
            print("\n--- AFTER (patched section) ---", flush=True)
            # Show the patch with proper indentation
            for line in patch_code.split("\n"):
                print(line, flush=True)
            print("\n" + "=" * 70, flush=True)
            print("✓ Patch applied successfully", flush=True)
            print("=" * 70 + "\n", flush=True)

        logger.info(
            "✓ Patched cmake/machine.cmake to filter GPU flags for OpenMP builds"
        )
        return True

    def _compile_backend(
        self,
        backend: str,
        build_dir: Path,
        compilation_script: Path,
        compilation_logfile: Path,
        cmake_backend_flag: str,
        recompile: bool,
        verbose: bool,
    ):
        """Internal method to compile a single backend."""

        # Check if already compiled
        if backend == "cpu":
            already_compiled = self.compilation_cpu_successful
        else:
            already_compiled = self.compilation_gpu_successful

        if already_compiled and not recompile:
            if verbose:
                print(
                    f"[{backend.upper()}] Already compiled successfully (skipping)",
                    flush=True,
                )
            return

        TRITONSWMM_software_directory = self.cfg_system.TRITONSWMM_software_directory

        # Generate compilation script
        bash_script_lines = [
            "#!/bin/bash",
            "set -e  # Exit on error",
            "",
            f"TRITON_DIR={TRITONSWMM_software_directory}",
            f"BUILD_DIR={build_dir}",
            "",
        ]

        # Optional: Load HPC modules
        if self.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc:
            modules = (
                self.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
            )
            bash_script_lines.extend(
                [
                    f"# Load HPC modules",
                    f"module load {modules}",
                    "",
                ]
            )

        # Build cmake flags and environment setup - explicitly enable one backend and disable others
        if backend == "cpu":
            # CPU: Enable OpenMP for Kokkos, explicitly disable GPU backends
            # CRITICAL: Set TRITON_BACKEND as environment variable BEFORE cmake runs
            # TRITON's machine detection scripts (e.g., frontier/default_default.sh) set
            # TRITON_BACKEND=HIP as an env var during cmake configuration, which enables
            # GPU-specific code compilation. Setting it beforehand prevents this override.
            # CRITICAL: Use CXXFLAGS environment variable to undefine TRITON_HIP_LAUNCHER
            # CMake's CMAKE_CXX_FLAGS command-line setting gets overridden by TRITON's CMakeLists.txt
            # but CXXFLAGS environment variable gets prepended and cannot be overridden.
            # Must undefine HIP/CUDA launcher macros to prevent CUDA syntax errors in CPU builds.
            # Also use -fopenmp flag to ensure OpenMP runtime is properly linked (prevents __kmpc_* errors)
            env_setup = [
                "export TRITON_BACKEND=OPENMP",
                "export TRITON_ARCH=''",  # Prevent GPU architecture detection
                "export CXXFLAGS='-fopenmp -UTRITON_HIP_LAUNCHER -UTRITON_CUDA_LAUNCHER'",
                "export CFLAGS='-fopenmp'",
                "export LDFLAGS='-fopenmp'",
            ]
            cmake_flags = (
                "-DKokkos_ENABLE_OPENMP=ON "
                "-DKokkos_ENABLE_HIP=OFF "
                "-DKokkos_ENABLE_CUDA=OFF"
            )
        else:
            # GPU: Enable GPU backend, disable OpenMP for Kokkos
            # SWMM's CMakeLists.txt unconditionally finds and links OpenMP, so we must ensure
            # the OpenMP runtime library is linked to prevent "undefined reference to __kmpc_*" errors
            # Need -fopenmp in BOTH shared library and executable linker flags since libswmm5.so
            # is a shared library that gets linked into runswmm executable
            # Kokkos won't use OpenMP since we explicitly disable it with -DKokkos_ENABLE_OPENMP=OFF
            env_setup = ""  # Let machine detection set HIP/CUDA backend
            cmake_flags = (
                f"{cmake_backend_flag} "
                "-DKokkos_ENABLE_OPENMP=OFF "
                "-DCMAKE_C_FLAGS='-fopenmp' "
                "-DCMAKE_SHARED_LINKER_FLAGS='-fopenmp' "
                "-DCMAKE_EXE_LINKER_FLAGS='-fopenmp'"
            )

        # Build commands
        bash_script_lines.extend(
            [
                'cd "${TRITON_DIR}"',
                'rm -rf "${BUILD_DIR}"',
                'mkdir -p "${BUILD_DIR}"',
                'cd "${BUILD_DIR}"',
                "",
                "echo '=== ENVIRONMENT BEFORE CMAKE ==='",
                'echo "TRITON_BACKEND=$TRITON_BACKEND"',
                'echo "TRITON_ARCH=$TRITON_ARCH"',
                'echo "CXXFLAGS=$CXXFLAGS"',
                "",
            ]
        )

        # Add environment variable exports if needed (for CPU builds)
        if env_setup:
            if isinstance(env_setup, list):
                bash_script_lines.extend(env_setup)
            else:
                bash_script_lines.append(env_setup)

            # Add diagnostic output after env setup
            bash_script_lines.extend(
                [
                    "",
                    "echo '=== ENVIRONMENT AFTER EXPORTS ==='",
                    'echo "TRITON_BACKEND=$TRITON_BACKEND"',
                    'echo "TRITON_ARCH=$TRITON_ARCH"',
                    'echo "CXXFLAGS=$CXXFLAGS"',
                    "",
                ]
            )

        bash_script_lines.extend(
            [
                f"cmake -DTRITON_ENABLE_SWMM=ON -DTRITON_SWMM_FLOODING_DEBUG=ON {cmake_flags} .. 2>&1 | tee cmake_output.txt",
                "",
                "echo '=== CMAKE FLAGS FROM CACHE ==='",
                "grep -E 'CMAKE_CXX_FLAGS|TRITON_BACKEND|TRITON_ARCH|Kokkos.*ENABLE' CMakeCache.txt | head -20 || echo 'CMakeCache.txt not found'",
                "",
                "make -j4",
                "",
                "echo 'script finished'",
            ]
        )

        # Write script
        compilation_script.parent.mkdir(parents=True, exist_ok=True)
        compilation_script.write_text("\n".join(bash_script_lines))
        compilation_script.chmod(0o755)

        if verbose:
            print(f"[{backend.upper()}] Starting compilation...", flush=True)
            print(f"[{backend.upper()}]   Script: {compilation_script}", flush=True)

        # Execute compilation
        with open(compilation_logfile, "w") as logfile:
            subprocess.run(
                ["/bin/bash", str(compilation_script)],
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )

        # Wait for completion marker
        start_time = time.time()
        while time.time() - start_time < 10:
            compilation_log = ut.read_text_file_as_string(compilation_logfile)
            if "script finished" in compilation_log:
                break
            time.sleep(0.1)

        # Check success
        if backend == "cpu":
            success = self.compilation_cpu_successful
        else:
            success = self.compilation_gpu_successful

        if verbose:
            if success:
                print(f"[{backend.upper()}] ✓ Compilation successful!", flush=True)
            else:
                print(f"[{backend.upper()}] ✗ Compilation failed", flush=True)
                print(f"[{backend.upper()}]   Log: {compilation_logfile}", flush=True)

    def retrieve_compilation_log(self, backend: str) -> str:
        """
        Retrieve compilation log for specified backend.

        Parameters
        ----------
        backend : str
            Which backend's log to retrieve ("cpu" or "gpu", default: "cpu")

        Returns
        -------
        str
            Log content, or error message if log doesn't exist
        """
        if backend == "cpu":
            logfile = self.sys_paths.compilation_logfile_cpu
        elif backend == "gpu":
            if self.sys_paths.compilation_logfile_gpu is None:
                return "GPU backend not configured (gpu_compilation_backend not set)"
            logfile = self.sys_paths.compilation_logfile_gpu
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if logfile.exists():
            return ut.read_text_file_as_string(logfile)
        return f"No compilation log found for {backend} backend at {logfile}"

    def print_compilation_log(self, backend: str):
        """
        Print compilation log(s).

        Parameters
        ----------
        backend : Optional[str]
            Which backend to print ("cpu" or "gpu"). If None, prints all available backends.
        """
        if backend == "cpu":
            # Print all available backends
            print("=== CPU Backend Compilation Log ===", flush=True)
            print(self.retrieve_compilation_log("cpu"), flush=True)
            if self.cfg_system.gpu_compilation_backend:
                print("\n=== GPU Backend Compilation Log ===", flush=True)
                print(self.retrieve_compilation_log("gpu"), flush=True)
        if backend == "gpu":
            print(f"=== {backend.upper()} Backend Compilation Log ===", flush=True)
            print(self.retrieve_compilation_log(backend), flush=True)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @property
    def compilation_cpu_successful(self) -> bool:
        """Check if CPU backend compiled successfully."""
        log = self.retrieve_compilation_log("cpu")
        swmm_check = "Built target swmm5" in log
        triton_check = "[100%] Built target triton.exe" in log
        return swmm_check and triton_check

    @property
    def compilation_gpu_successful(self) -> bool:
        """Check if GPU backend compiled successfully."""
        if self.sys_paths.compilation_logfile_gpu is None:
            return False
        if not self.sys_paths.compilation_logfile_gpu.exists():
            return False
        log = self.retrieve_compilation_log("gpu")
        swmm_check = "Built target swmm5" in log
        triton_check = "[100%] Built target triton.exe" in log
        return swmm_check and triton_check

    @property
    def compilation_successful(self) -> bool:
        """
        Returns True if CPU backend (and GPU if configured) compiled successfully.
        For individual backend checks, use compilation_cpu_successful and compilation_gpu_successful.
        """
        if self.cfg_system.gpu_compilation_backend:
            success = (
                self.compilation_cpu_successful and self.compilation_gpu_successful
            )
        else:
            success = self.compilation_cpu_successful
        return success

    @property
    def available_backends(self) -> list[str]:
        """Return list of successfully compiled backends."""
        backends = []
        if self.compilation_cpu_successful:
            backends.append("cpu")
        if self.compilation_gpu_successful:
            backends.append("gpu")
        return backends

    def print_compilation_status(self):
        """Print human-readable compilation status for both backends."""
        print("\n" + "=" * 60, flush=True)
        print("Compilation Status", flush=True)
        print("=" * 60, flush=True)

        # CPU backend (always required)
        if self.compilation_cpu_successful:
            print(f"✓ CPU backend: COMPILED SUCCESSFULLY", flush=True)
            print(f"  Build: {self.sys_paths.TRITON_build_dir_cpu}", flush=True)
        else:
            print(f"✗ CPU backend: FAILED", flush=True)
            print(f"  Log: {self.sys_paths.compilation_logfile_cpu}", flush=True)

        # GPU backend (optional)
        if self.cfg_system.gpu_compilation_backend is None:
            print("  GPU backend: NOT REQUESTED", flush=True)
        elif self.compilation_gpu_successful:
            print(f"✓ GPU backend: COMPILED SUCCESSFULLY", flush=True)
            print(f"  Build: {self.sys_paths.TRITON_build_dir_gpu}", flush=True)
        else:
            print(f"✗ GPU backend: FAILED", flush=True)
            print(f"  Log: {self.sys_paths.compilation_logfile_gpu}", flush=True)

        print(f"\nAvailable backends: {', '.join(self.available_backends)}", flush=True)
        print("=" * 60 + "\n", flush=True)


# %% helper functions
def spatial_resampling(xds_to_resample, xds_target, missingfillval=-9999):
    from rasterio.enums import Resampling
    import xarray as xr

    # resample
    ## https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_dataset.RasterDataset.reproject_match
    ## (https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling)
    xds_to_resampled = xds_to_resample.rio.reproject_match(  # type: ignore
        xds_target, resampling=Resampling.average
    )
    # fill missing values with prespecified val (this should just corresponds to areas where one dataset has pieces outside the other)
    xds_to_resampled = xr.where(
        xds_to_resampled >= 3.403e37, x=missingfillval, y=xds_to_resampled
    )
    return xds_to_resampled


def compute_grid_resolution(rds):
    res_xy = rds.rio.resolution()
    mean_grid_size = np.sqrt(abs(res_xy[0]) * abs(res_xy[1]))
    return res_xy, mean_grid_size


def coarsen_georaster(rds, target_resolution):
    crs = rds.rio.crs
    _, og_avg_gridsize = compute_grid_resolution(rds)
    res_multiplier = target_resolution / og_avg_gridsize
    target_res = og_avg_gridsize * res_multiplier
    rds_coarse = rds.rio.reproject(  # type: ignore
        crs, resolution=target_res, resampling=Resampling.average
    )  # aggregate cells
    _, coarse_avg_gridsize = compute_grid_resolution(rds_coarse)
    assert np.isclose(coarse_avg_gridsize, target_resolution)
    return rds_coarse
