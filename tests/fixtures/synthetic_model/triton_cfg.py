"""Synthesize TRITON-SWMM definition template .cfg file."""

from __future__ import annotations

from pathlib import Path

# Template copied verbatim from test_data/TRITON_SWMM_test_model/TRITONSWMM.cfg.
# Python-level substitutions use {token} braces; TRITON-SWMM's own relative path
# strings (dem_filename, inp_filename, etc.) are literals at write time.
_TEMPLATE = """\
#---------------------------------------------------------------------------
# TRITON-SWMM config file - {project_name}
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Mandatory input files
#---------------------------------------------------------------------------
dem_filename="dem/elevation.dem"
inp_filename="swmm/hydraulics.inp"
#---------------------------------------------------------------------------
# Format=BIN or ASC
# Option=PAR or SEQ
#---------------------------------------------------------------------------
input_format=ASC
input_option=SEQ
outfile_pattern="%s/%s/%s_%02d_%02d"
output_format=BIN
output_option=SEQ

#---------------------------------------------------------------------------
# TRITON-SWMM specific inputs
#---------------------------------------------------------------------------
manhole_diameter={manhole_diameter}
manhole_loss={manhole_loss}

#---------------------------------------------------------------------------
# Hydrograph and flow locations information
#---------------------------------------------------------------------------
num_sources={num_sources}
hydrograph_filename="strmflow/tseries.hyg"
src_loc_file="strmflow/loc.txt"

#---------------------------------------------------------------------------
# Manning input files
#---------------------------------------------------------------------------
n_infile="mannings/mannings.dem"
#const_mann=None

#---------------------------------------------------------------------------
# Runoff related information
#---------------------------------------------------------------------------
num_runoffs=0
#runoff_filename=""
#runoff_file=""

#---------------------------------------------------------------------------
# External boundaries
#---------------------------------------------------------------------------
num_extbc=1
extbc_dir="extbc"
extbc_file="extbc/loc.extbc"

#---------------------------------------------------------------------------
# time_series_flag=1 to activate, 0 to deactivate
#---------------------------------------------------------------------------
observation_loc_file=""
time_series_flag=0

#---------------------------------------------------------------------------
# print_option=h to output just the h, huv to output all h,u and v
#---------------------------------------------------------------------------
print_option=huv
max_value_print_option=h

#---------------------------------------------------------------------------
# Start and duration of simulation time in seconds
#---------------------------------------------------------------------------
sim_start_time=0
sim_duration={sim_duration}

#---------------------------------------------------------------------------
# If checkpoint_id is 0 that means a clean start
#---------------------------------------------------------------------------
checkpoint_id=0

#---------------------------------------------------------------------------
# time_increment_fixed=0 is for variable dt, 1 for constant dt
#---------------------------------------------------------------------------
time_step={time_step}
time_increment_fixed=0

#---------------------------------------------------------------------------
# Print interval in seconds of simulation time
#---------------------------------------------------------------------------
print_interval={print_interval}

#---------------------------------------------------------------------------
# Initial input files
#---------------------------------------------------------------------------
#h_infile="input/inith/asc/case05.inith"
#qx_infile="input/initqx/asc/case05.initqx"
#qy_infile="input/initqy/asc/case05.initqy"

#---------------------------------------------------------------------------
# Other variables
#---------------------------------------------------------------------------
it_count=0
courant=0.5
hextra=0.001
gpu_direct_flag=0
domain_decomposition=static
open_boundaries={open_boundaries}
"""


def build_cfg(params, dest: Path) -> Path:
    body = _TEMPLATE.format(
        project_name="synth",
        manhole_diameter=params.manhole_diameter_m,
        manhole_loss=params.manhole_loss_coefficient,
        num_sources=params.n_rows * params.n_cols,
        sim_duration=params.sim_duration_min * 60,
        time_step=params.triton_timestep_s,
        print_interval=params.reporting_timestep_s,
        open_boundaries=1,
    )
    dest.write_text(body, encoding="utf-8")
    return dest
