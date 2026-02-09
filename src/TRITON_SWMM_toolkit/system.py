# %%
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling

import TRITON_SWMM_toolkit.utils as ut
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
from TRITON_SWMM_toolkit.config.loaders import load_system_config
from TRITON_SWMM_toolkit.exceptions import CompilationError, ConfigurationError
from TRITON_SWMM_toolkit.log import TRITONSWMM_system_log
from TRITON_SWMM_toolkit.paths import SysPaths
from TRITON_SWMM_toolkit.plot_system import TRITONSWMM_system_plotting


class TRITONSWMM_system:
    def __init__(self, system_config_yaml: Path) -> None:
        self.system_config_yaml = system_config_yaml
        self.cfg_system = load_system_config(system_config_yaml)

        system_dir = self.cfg_system.system_directory
        tritonswmm_dir = self.cfg_system.TRITONSWMM_software_directory
        swmm_dir = self.cfg_system.SWMM_software_directory

        # Initialize paths with backend split
        self.sys_paths = SysPaths(
            dem_processed=system_dir / "elevation.dem",
            mannings_processed=system_dir / "mannings.dem",
            # TRITON-SWMM build dirs (coupled model)
            TRITONSWMM_build_dir_cpu=tritonswmm_dir / "build_tritonswmm_cpu",
            TRITONSWMM_build_dir_gpu=(
                tritonswmm_dir / "build_tritonswmm_gpu"
                if self.cfg_system.gpu_compilation_backend
                else None
            ),
            # TRITON-only build dirs (no SWMM coupling)
            TRITON_build_dir_cpu=tritonswmm_dir / "build_triton_cpu",
            TRITON_build_dir_gpu=(
                tritonswmm_dir / "build_triton_gpu"
                if self.cfg_system.gpu_compilation_backend
                else None
            ),
            # SWMM standalone build dir
            SWMM_build_dir=(
                swmm_dir if (self.cfg_system.toggle_swmm_model and swmm_dir) else None
            ),
            # Compilation artifacts (shared across build types)
            compilation_script_cpu=system_dir / "compile_cpu.sh",
            compilation_script_gpu=(
                system_dir / "compile_gpu.sh"
                if self.cfg_system.gpu_compilation_backend
                else None
            ),
            compilation_logfile_cpu=(
                tritonswmm_dir / "build_tritonswmm_cpu" / "compilation.log"
            ),
            compilation_logfile_gpu=(
                tritonswmm_dir / "build_tritonswmm_gpu" / "compilation.log"
                if self.cfg_system.gpu_compilation_backend
                else None
            ),
            # Backwards compatibility aliases (point to TRITON-SWMM CPU versions)
            TRITON_build_dir=tritonswmm_dir / "build_tritonswmm_cpu",
            compilation_logfile=tritonswmm_dir
            / "build_tritonswmm_cpu"
            / "compilation.log",
            compilation_script=system_dir / "compile_cpu.sh",
        )

        # Initialize system log
        log_path = system_dir / "system_log.json"
        if log_path.exists():
            self.log = TRITONSWMM_system_log.from_json(log_path)
        else:
            self.log = TRITONSWMM_system_log(logfile=log_path)
            self.log.write()

        self._analysis: TRITONSWMM_analysis | None = None
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
        # Log DEM processing status
        self.log.dem_processed.set(True)
        self.log.dem_shape.set(tuple(rds_dem_coarse.shape))
        self.log.write()
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
        # Log Manning's processing status
        self.log.mannings_processed.set(True)
        self.log.mannings_shape.set(tuple(rds_mannings_coarse.shape))
        self.log.write()
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
        backends: list[str] | None = None,
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

        # Compile each backend sequentially
        for backend in backends:
            if verbose:
                print(f"\n{'=' * 60}", flush=True)
                print(f"Compiling {backend.upper()} Backend", flush=True)
                print("=" * 60, flush=True)

            if backend == "cpu":
                self._compile_backend(
                    backend="cpu",
                    build_dir=self.sys_paths.TRITONSWMM_build_dir_cpu,
                    compilation_script=self.sys_paths.compilation_script_cpu,
                    compilation_logfile=self.sys_paths.compilation_logfile_cpu,
                    cmake_backend_flag="-DKokkos_ENABLE_OPENMP=ON",
                    recompile=recompile_if_already_done_successfully,
                    verbose=verbose,
                )
            elif backend == "gpu":
                if self.cfg_system.gpu_compilation_backend is None:
                    raise ConfigurationError(
                        field="gpu_compilation_backend",
                        message="GPU backend requested but gpu_compilation_backend not set.\n"
                        "  Set gpu_compilation_backend='HIP' or 'CUDA' in system config YAML.",
                        config_path=self.system_config_yaml,
                    )

                # Determine Kokkos flag based on config
                if self.cfg_system.gpu_compilation_backend == "HIP":
                    cmake_backend_flag = "-DKokkos_ENABLE_HIP=ON"
                elif self.cfg_system.gpu_compilation_backend == "CUDA":
                    cmake_backend_flag = "-DKokkos_ENABLE_CUDA=ON"
                else:
                    raise ConfigurationError(
                        field="gpu_compilation_backend",
                        message=f"Invalid value '{self.cfg_system.gpu_compilation_backend}'.\n"
                        "  Must be 'HIP' or 'CUDA'.",
                        config_path=self.system_config_yaml,
                    )

                self._compile_backend(
                    backend="gpu",
                    build_dir=self.sys_paths.TRITONSWMM_build_dir_gpu,  # type: ignore
                    compilation_script=self.sys_paths.compilation_script_gpu,  # type: ignore
                    compilation_logfile=self.sys_paths.compilation_logfile_gpu,  # type: ignore
                    cmake_backend_flag=cmake_backend_flag,
                    recompile=recompile_if_already_done_successfully,
                    verbose=verbose,
                )
            else:
                raise ConfigurationError(
                    field="backends",
                    message=f"Unknown backend '{backend}'. Must be 'cpu' or 'gpu'.",
                )

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
                    "# Load HPC modules",
                    f"module load {modules}",
                    "",
                ]
            )

        # Build cmake flags - use TRITON_IGNORE_MACHINE_FILES to bypass machine-specific defaults
        # This gives us full control over compilation settings regardless of the HPC system
        if backend == "cpu":
            # CPU: Enable OpenMP, disable GPU backends
            # -fopenmp ensures OpenMP runtime is linked (SWMM's CMakeLists.txt uses it)
            cmake_flags = (
                "-DTRITON_IGNORE_MACHINE_FILES=ON "
                "-DKokkos_ENABLE_OPENMP=ON "
                "-DKokkos_ENABLE_HIP=OFF "
                "-DKokkos_ENABLE_CUDA=OFF "
                "-DCMAKE_CXX_FLAGS='-O3 -fopenmp' "
                "-DCMAKE_C_FLAGS='-fopenmp' "
                "-DCMAKE_SHARED_LINKER_FLAGS='-fopenmp' "
                "-DCMAKE_EXE_LINKER_FLAGS='-fopenmp'"
            )
        else:
            # GPU: Enable GPU backend, disable OpenMP for Kokkos
            # Still need -fopenmp for SWMM which unconditionally finds OpenMP
            cmake_flags = (
                "-DTRITON_IGNORE_MACHINE_FILES=ON "
                f"{cmake_backend_flag} "
                "-DKokkos_ENABLE_OPENMP=OFF "
                "-DCMAKE_CXX_FLAGS='-O3' "
                "-DCMAKE_C_FLAGS='-fopenmp' "
                "-DCMAKE_SHARED_LINKER_FLAGS='-fopenmp' "
                "-DCMAKE_EXE_LINKER_FLAGS='-fopenmp'"
            )

        # Build commands
        bash_script_lines.extend(
            [
                'cd "${TRITON_DIR}"',
                'mkdir -p "${BUILD_DIR}"',
                'rm -rf "${BUILD_DIR}/CMakeFiles" "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/Makefile" "${BUILD_DIR}/cmake_install.cmake"',
                'cd "${BUILD_DIR}"',
                "",
                f"cmake -DTRITON_ENABLE_SWMM=ON -DTRITON_SWMM_FLOODING_DEBUG=ON {cmake_flags} .. 2>&1 | tee cmake_output.txt",
                "",
                "echo '=== CMAKE CONFIGURATION ==='",
                "grep -E 'CMAKE_CXX_FLAGS|TRITON_IGNORE_MACHINE|Kokkos.*ENABLE' CMakeCache.txt | head -20 || echo 'CMakeCache.txt not found'",
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
        build_dir.mkdir(parents=True, exist_ok=True)
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
            # Log to system log
            self.log.compilation_tritonswmm_cpu_successful.set(success)
        else:
            success = self.compilation_gpu_successful
            # Log to system log
            self.log.compilation_tritonswmm_gpu_successful.set(success)
        self.log.write()

        if verbose:
            if success:
                print(f"[{backend.upper()}] ✓ Compilation successful!", flush=True)
            else:
                print(f"[{backend.upper()}] ✗ Compilation failed", flush=True)
                print(f"[{backend.upper()}]   Log: {compilation_logfile}", flush=True)

        # Raise exception if compilation failed
        if not success:
            raise CompilationError(
                model_type="tritonswmm",
                backend=backend,
                logfile=compilation_logfile,
                return_code=1,  # Subprocess didn't fail, but build markers missing
            )

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

    # ========== TRITON-only Compilation (no SWMM coupling) ==========

    def compile_TRITON_only(
        self,
        backends: list[str] | None = None,
        recompile_if_already_done_successfully: bool = False,
        redownload_triton_swmm_if_exists: bool = False,
        verbose: bool = True,
    ):
        """
        Compile TRITON-only (without SWMM coupling) for specified backend(s).

        This compiles TRITON with -DTRITON_ENABLE_SWMM=OFF, producing a standalone
        2D hydrodynamic model without any SWMM integration.

        Parameters
        ----------
        backends : Optional[list[str]]
            List of backends to compile ("cpu", "gpu", or both). If None:
            - Always compiles CPU
            - Compiles GPU only if gpu_compilation_backend is set in config
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        verbose : bool
            If True, print progress messages
        """
        # TODO - if TRITON-SWMM is enabled, re-downloading should only be allowed to occur once. Otherwise, previous compilations could be overwritten.
        if not self.cfg_system.toggle_triton_model:
            if verbose:
                print("[TRITON-only] Skipped (toggle_triton_model=False)", flush=True)
            return

        # Determine which backends to compile
        if backends is None:
            backends = ["cpu"]
            if self.cfg_system.gpu_compilation_backend:
                backends.append("gpu")

        # Download TRITON source if needed (shared across backends)
        TRITONSWMM_software_directory = self.cfg_system.TRITONSWMM_software_directory

        if (
            redownload_triton_swmm_if_exists
            or not TRITONSWMM_software_directory.exists()
        ):
            self._download_tritonswmm_source(verbose=verbose)

        # Compile each backend sequentially
        for backend in backends:
            if verbose:
                print(f"\n{'=' * 60}", flush=True)
                print(f"Compiling TRITON-only {backend.upper()} Backend", flush=True)
                print("=" * 60, flush=True)

            if backend == "cpu":
                self._compile_triton_only_backend(
                    backend="cpu",
                    build_dir=self.sys_paths.TRITON_build_dir_cpu,
                    recompile=recompile_if_already_done_successfully,
                    verbose=verbose,
                )
            elif backend == "gpu":
                if self.cfg_system.gpu_compilation_backend is None:
                    raise ValueError(
                        "GPU backend requested but gpu_compilation_backend not set in config."
                    )
                self._compile_triton_only_backend(
                    backend="gpu",
                    build_dir=self.sys_paths.TRITON_build_dir_gpu,  # type: ignore
                    recompile=recompile_if_already_done_successfully,
                    verbose=verbose,
                )
            else:
                raise ConfigurationError(
                    field="backends",
                    message=f"Unknown backend '{backend}'. Must be 'cpu' or 'gpu'.",
                )

    def _compile_triton_only_backend(
        self,
        backend: str,
        build_dir: Path,
        recompile: bool,
        verbose: bool,
    ):
        """Internal method to compile TRITON-only for a single backend."""
        # Check if already compiled
        if backend == "cpu":
            already_compiled = self.compilation_triton_only_cpu_successful
        else:
            already_compiled = self.compilation_triton_only_gpu_successful

        if already_compiled and not recompile:
            if verbose:
                print(
                    f"[TRITON-only {backend.upper()}] Already compiled successfully (skipping)",
                    flush=True,
                )
            return

        TRITONSWMM_software_directory = self.cfg_system.TRITONSWMM_software_directory
        logfile = build_dir / "compilation.log"
        script_file = build_dir.parent / f"compile_triton_only_{backend}.sh"

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
                    "# Load HPC modules",
                    f"module load {modules}",
                    "",
                ]
            )

        # Build cmake flags
        if backend == "cpu":
            cmake_flags = (
                "-DTRITON_IGNORE_MACHINE_FILES=ON "
                "-DKokkos_ENABLE_OPENMP=ON "
                "-DKokkos_ENABLE_HIP=OFF "
                "-DKokkos_ENABLE_CUDA=OFF "
                "-DCMAKE_CXX_FLAGS='-O3 -fopenmp' "
                "-DCMAKE_C_FLAGS='-fopenmp' "
                "-DCMAKE_SHARED_LINKER_FLAGS='-fopenmp' "
                "-DCMAKE_EXE_LINKER_FLAGS='-fopenmp'"
            )
        else:
            # GPU backend
            if self.cfg_system.gpu_compilation_backend == "HIP":
                cmake_backend_flag = "-DKokkos_ENABLE_HIP=ON"
            elif self.cfg_system.gpu_compilation_backend == "CUDA":
                cmake_backend_flag = "-DKokkos_ENABLE_CUDA=ON"
            else:
                raise ValueError(
                    f"Invalid gpu_compilation_backend: {self.cfg_system.gpu_compilation_backend}"
                )

            cmake_flags = (
                "-DTRITON_IGNORE_MACHINE_FILES=ON "
                f"{cmake_backend_flag} "
                "-DKokkos_ENABLE_OPENMP=OFF "
                "-DCMAKE_CXX_FLAGS='-O3'"
            )

        # Build commands - KEY DIFFERENCE: -DTRITON_ENABLE_SWMM=OFF
        bash_script_lines.extend(
            [
                'cd "${TRITON_DIR}"',
                'mkdir -p "${BUILD_DIR}"',
                'rm -rf "${BUILD_DIR}/CMakeFiles" "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/Makefile" "${BUILD_DIR}/cmake_install.cmake"',
                'cd "${BUILD_DIR}"',
                "",
                f"cmake -DTRITON_ENABLE_SWMM=OFF {cmake_flags} .. 2>&1 | tee cmake_output.txt",
                "",
                "make -j4",
                "",
                "echo 'script finished'",
            ]
        )

        # Write script
        script_file.parent.mkdir(parents=True, exist_ok=True)
        script_file.write_text("\n".join(bash_script_lines))
        script_file.chmod(0o755)

        if verbose:
            print(
                f"[TRITON-only {backend.upper()}] Starting compilation...", flush=True
            )
            print(
                f"[TRITON-only {backend.upper()}]   Script: {script_file}", flush=True
            )

        # Execute compilation
        build_dir.mkdir(parents=True, exist_ok=True)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "w") as lf:
            subprocess.run(
                ["/bin/bash", str(script_file)],
                stdout=lf,
                stderr=subprocess.STDOUT,
            )

        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < 10:
            if logfile.exists():
                log = ut.read_text_file_as_string(logfile)
                if "script finished" in log:
                    break
            time.sleep(0.1)

        # Check success
        if backend == "cpu":
            success = self.compilation_triton_only_cpu_successful
            # Log to system log
            self.log.compilation_triton_cpu_successful.set(success)
        else:
            success = self.compilation_triton_only_gpu_successful
            # Log to system log
            self.log.compilation_triton_gpu_successful.set(success)
        self.log.write()

        if verbose:
            if success:
                print(
                    f"[TRITON-only {backend.upper()}] ✓ Compilation successful!",
                    flush=True,
                )
            else:
                print(
                    f"[TRITON-only {backend.upper()}] ✗ Compilation failed", flush=True
                )
                print(f"[TRITON-only {backend.upper()}]   Log: {logfile}", flush=True)

        # Raise exception if compilation failed
        if not success:
            raise CompilationError(
                model_type="triton",
                backend=backend,
                logfile=logfile,
                return_code=1,  # Subprocess didn't fail, but build markers missing
            )

    @property
    def compilation_triton_only_cpu_successful(self) -> bool:
        """Check if TRITON-only CPU backend compiled successfully."""
        logfile = self.sys_paths.TRITON_build_dir_cpu / "compilation.log"
        if logfile.exists():
            log = ut.read_text_file_as_string(logfile)
            # TRITON-only does NOT have swmm5 target
            triton_check = "[100%] Built target triton.exe" in log
            if triton_check:
                return True

        # Fallback: if compilation log is missing but executable exists, accept as successful
        exe_path = self.sys_paths.TRITON_build_dir_cpu / "triton.exe"
        return exe_path.exists()

    @property
    def compilation_triton_only_gpu_successful(self) -> bool:
        """Check if TRITON-only GPU backend compiled successfully."""
        if self.sys_paths.TRITON_build_dir_gpu is None:
            return False
        logfile = self.sys_paths.TRITON_build_dir_gpu / "compilation.log"
        if logfile.exists():
            log = ut.read_text_file_as_string(logfile)
            triton_check = "[100%] Built target triton.exe" in log
            if triton_check:
                return True

        exe_path = self.sys_paths.TRITON_build_dir_gpu / "triton.exe"
        return exe_path.exists()

    @property
    def compilation_triton_only_successful(self) -> bool:
        """
        Returns True if TRITON-only (CPU and GPU if configured) compiled successfully.
        For individual backend checks, use compilation_triton_only_cpu_successful and compilation_triton_only_gpu_successful.
        """
        if self.cfg_system.gpu_compilation_backend:
            return (
                self.compilation_triton_only_cpu_successful
                and self.compilation_triton_only_gpu_successful
            )
        else:
            return self.compilation_triton_only_cpu_successful

    # ========== SWMM Standalone Compilation ==========

    def compile_SWMM(
        self,
        recompile_if_already_done_successfully: bool = False,
        redownload_swmm_if_exists: bool = False,
        verbose: bool = True,
    ):
        """
        Compile standalone EPA SWMM executable.

        Downloads SWMM source from the configured git URL and compiles it
        as a standalone executable for running SWMM-only simulations.

        Parameters
        ----------
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        redownload_swmm_if_exists : bool
            If True, re-download SWMM source even if it exists
        verbose : bool
            If True, print progress messages
        """
        if not self.cfg_system.toggle_swmm_model:
            if verbose:
                print("[SWMM] Skipped (toggle_swmm_model=False)", flush=True)
            return

        if (
            self.compilation_swmm_successful
            and not recompile_if_already_done_successfully
        ):
            if verbose:
                print("[SWMM] Already compiled successfully (skipping)", flush=True)
            return

        if verbose:
            print(f"\n{'=' * 60}", flush=True)
            print("Compiling Standalone EPA SWMM", flush=True)
            print("=" * 60, flush=True)

        build_dir = self.sys_paths.SWMM_build_dir
        if build_dir is None:
            raise ValueError(
                "SWMM build dir not configured (toggle_swmm_model may be False)"
            )

        swmm_source_dir = build_dir / "swmm_source"
        logfile = build_dir / "compilation.log"
        script_file = build_dir / "compile_swmm.sh"

        # Generate compilation script
        bash_script_lines = [
            "#!/bin/bash",
            "set -e  # Exit on error",
            "",
            f"SWMM_SOURCE_DIR={swmm_source_dir}",
            f"BUILD_DIR={build_dir}",
            "",
        ]

        # Download SWMM source if needed
        # template: git clone --branch v5.2.4 --depth 1 https://github.com/USEPA/Stormwater-Management-Model.git
        if redownload_swmm_if_exists or not swmm_source_dir.exists():
            tag_line = ""
            if self.cfg_system.SWMM_tag_key:
                tag_line = f"--branch {self.cfg_system.SWMM_tag_key} --depth 1 "
            bash_script_lines.extend(
                [
                    "# Clone SWMM source",
                    f'rm -rf "{swmm_source_dir}"',
                    f'git clone {tag_line}{self.cfg_system.SWMM_git_URL} "{swmm_source_dir}"',
                ]
            )
            bash_script_lines.append("")

        # Build SWMM
        bash_script_lines.extend(
            [
                f'cd "{build_dir}"',
                "rm -rf swmm_build",
                "mkdir -p swmm_build",
                "cd swmm_build",
                "",
                f'cmake "{swmm_source_dir}" -DCMAKE_BUILD_TYPE=Release 2>&1 | tee cmake_output.txt',
                "",
                "make -j4",
                "",
                "echo 'script finished'",
            ]
        )

        # Write script
        build_dir.mkdir(parents=True, exist_ok=True)
        script_file.write_text("\n".join(bash_script_lines))
        script_file.chmod(0o755)

        if verbose:
            print("[SWMM] Starting compilation...", flush=True)
            print(f"[SWMM]   Script: {script_file}", flush=True)

        # Execute compilation
        with open(logfile, "w") as lf:
            subprocess.run(
                ["/bin/bash", str(script_file)],
                stdout=lf,
                stderr=subprocess.STDOUT,
            )

        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < 10:
            if logfile.exists():
                log = ut.read_text_file_as_string(logfile)
                if "script finished" in log:
                    break
            time.sleep(0.1)

        # Check success
        success = self.compilation_swmm_successful
        # Log to system log
        self.log.compilation_swmm_successful.set(success)
        self.log.write()

        if success:
            if verbose:
                print("[SWMM] ✓ Compilation successful!", flush=True)
                print(f"[SWMM]   Executable: {self.swmm_executable}", flush=True)
        else:
            if verbose:
                print("[SWMM] ✗ Compilation failed", flush=True)
                print(f"[SWMM]   Log: {logfile}", flush=True)

    @property
    def compilation_swmm_successful(self) -> bool:
        """Check if standalone SWMM compiled successfully."""
        if self.sys_paths.SWMM_build_dir is None:
            return False
        logfile = self.sys_paths.SWMM_build_dir / "compilation.log"
        if not logfile.exists():
            return False
        log = ut.read_text_file_as_string(logfile)
        success_markers = ("Built target runswmm", "Built target swmm5")
        failure_markers = ("CMake Error", "error:", "FAILED")
        if any(marker in log for marker in success_markers):
            return True
        if any(marker in log for marker in failure_markers):
            return False
        return False

    @property
    def swmm_executable(self) -> Path | None:
        """Return path to compiled SWMM executable, or None if not compiled."""
        if self.sys_paths.SWMM_build_dir is None:
            return None
        # Check common locations (SWMM_build_dir already includes "swmm_build")
        for exe_name in ["runswmm", "swmm5", "swmm"]:
            # Check bin subdirectory first
            exe_path = self.sys_paths.SWMM_build_dir / "bin" / exe_name
            if exe_path.exists():
                return exe_path
            # Check build root directly
            exe_path = self.sys_paths.SWMM_build_dir / exe_name
            if exe_path.exists():
                return exe_path
            # Check swmm_build subdirectory if SWMM_build_dir points to parent
            exe_path = self.sys_paths.SWMM_build_dir / "swmm_build" / "bin" / exe_name
            if exe_path.exists():
                return exe_path
            exe_path = self.sys_paths.SWMM_build_dir / "swmm_build" / exe_name
            if exe_path.exists():
                return exe_path
            exe_path = (
                self.sys_paths.SWMM_build_dir
                / "swmm_build"
                / "bin"
                / "Release"
                / exe_name
            )
            if exe_path.exists():
                return exe_path
            exe_path = (
                self.sys_paths.SWMM_build_dir / "swmm_build" / "src" / "run" / exe_name
            )
            if exe_path.exists():
                return exe_path
        return None

    # ========== TRITON-SWMM Compilation Success Checks ==========

    @property
    def compilation_cpu_successful(self) -> bool:
        """Check if TRITON-SWMM CPU backend compiled successfully."""
        log = self.retrieve_compilation_log("cpu")
        swmm_check = "Built target swmm5" in log
        triton_check = "[100%] Built target triton.exe" in log
        return swmm_check and triton_check

    @property
    def compilation_gpu_successful(self) -> bool:
        """Check if TRITON-SWMM GPU backend compiled successfully."""
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
        """Print human-readable compilation status for all model types and backends."""
        print("\n" + "=" * 60, flush=True)
        print("Compilation Status", flush=True)
        print("=" * 60, flush=True)

        # TRITON-SWMM (coupled model)
        print("\n--- TRITON-SWMM (coupled) ---", flush=True)
        if self.compilation_cpu_successful:
            print("✓ CPU backend: COMPILED SUCCESSFULLY", flush=True)
            print(f"  Build: {self.sys_paths.TRITONSWMM_build_dir_cpu}", flush=True)
        else:
            print("✗ CPU backend: NOT COMPILED", flush=True)

        if self.cfg_system.gpu_compilation_backend is None:
            print("  GPU backend: NOT REQUESTED", flush=True)
        elif self.compilation_gpu_successful:
            print("✓ GPU backend: COMPILED SUCCESSFULLY", flush=True)
            print(f"  Build: {self.sys_paths.TRITONSWMM_build_dir_gpu}", flush=True)
        else:
            print("✗ GPU backend: NOT COMPILED", flush=True)

        # TRITON-only (no SWMM coupling)
        print("\n--- TRITON-only (no SWMM) ---", flush=True)
        if not self.cfg_system.toggle_triton_model:
            print("  (disabled via toggle_triton_model=False)", flush=True)
        else:
            if self.compilation_triton_only_cpu_successful:
                print("✓ CPU backend: COMPILED SUCCESSFULLY", flush=True)
                print(f"  Build: {self.sys_paths.TRITON_build_dir_cpu}", flush=True)
            else:
                print("✗ CPU backend: NOT COMPILED", flush=True)

            if self.cfg_system.gpu_compilation_backend is None:
                print("  GPU backend: NOT REQUESTED", flush=True)
            elif self.compilation_triton_only_gpu_successful:
                print("✓ GPU backend: COMPILED SUCCESSFULLY", flush=True)
                print(f"  Build: {self.sys_paths.TRITON_build_dir_gpu}", flush=True)
            else:
                print("✗ GPU backend: NOT COMPILED", flush=True)

        # Standalone SWMM
        print("\n--- Standalone SWMM ---", flush=True)
        if not self.cfg_system.toggle_swmm_model:
            print("  (disabled via toggle_swmm_model=False)", flush=True)
        else:
            if self.compilation_swmm_successful:
                print("✓ COMPILED SUCCESSFULLY", flush=True)
                print(f"  Build: {self.sys_paths.SWMM_build_dir}", flush=True)
                if self.swmm_executable:
                    print(f"  Executable: {self.swmm_executable}", flush=True)
            else:
                print("✗ NOT COMPILED", flush=True)

        print(
            f"\nTRITON-SWMM backends: {', '.join(self.available_backends) or 'none'}",
            flush=True,
        )
        print("=" * 60 + "\n", flush=True)


# %% helper functions
def spatial_resampling(xds_to_resample, xds_target, missingfillval=-9999):
    import xarray as xr
    from rasterio.enums import Resampling

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
