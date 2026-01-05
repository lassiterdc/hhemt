import subprocess
import shutil
from TRITON_SWMM_toolkit.utils import create_from_template, read_text_file_as_string
from pathlib import Path


def define_experiment_paths(experiment_id: str, system_directory: Path):
    compiled_software_directory = system_directory / experiment_id / "compiled_software"
    compiled_software_directory.mkdir(parents=True, exist_ok=True)
    TRITON_build_dir = compiled_software_directory / "build"
    compilation_script = compiled_software_directory / "compile.sh"
    simulation_directory = system_directory / experiment_id / "sims"

    exp_paths = dict(
        compiled_software_directory=compiled_software_directory,
        TRITON_build_dir=TRITON_build_dir,
        compilation_script=compilation_script,
        simulation_directory=simulation_directory,
    )
    return exp_paths


def compile_TRITON_SWMM(
    experiment_id,
    system_directory,
    TRITONSWMM_software_directory,
    TRITON_SWMM_make_command,
    TRITON_SWMM_software_compilation_script,
):
    exp_paths = define_experiment_paths(experiment_id, system_directory)
    sftwr_cmpld = exp_paths["compiled_software_directory"]
    if sftwr_cmpld.exists():
        shutil.rmtree(sftwr_cmpld)
    shutil.copytree(TRITONSWMM_software_directory, sftwr_cmpld)
    mapping = dict(
        COMPILED_MODEL_DIR=sftwr_cmpld, MAKE_COMMAND=TRITON_SWMM_make_command
    )
    comp_script_content = create_from_template(
        TRITON_SWMM_software_compilation_script,
        mapping,
        exp_paths["compilation_script"],
    )
    # subprocess.run(["/bin/bash", str(exp_paths["compilation_script"])], check=True)

    compilation_logfile = sftwr_cmpld / f"compilation.log"

    with open(compilation_logfile, "w") as logfile:
        proc = subprocess.run(  # type: ignore
            ["/bin/bash", str(exp_paths["compilation_script"])],
            stdout=logfile,
            stderr=subprocess.STDOUT,
            check=True,
        )

    import time

    start_time = time.time()
    compilation_log = read_text_file_as_string(compilation_logfile)
    while "Building finished: triton" not in compilation_log:
        time.sleep(0.1)
        compilation_log = read_text_file_as_string(compilation_logfile)
        elapsed = time.time() - start_time
        time.sleep(0.1)
        if elapsed > 5:
            break
    return compilation_log
