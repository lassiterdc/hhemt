"""Synthesize TRITON-SWMM definition template .cfg file."""

from __future__ import annotations

from pathlib import Path

# Template mirrors Norfolk's TRITON_SWMM_definition_template.cfg (at
# examples/norfolk_coastal_flooding/data/contents/). Placeholders use the
# ${NAME} syntax consumed by TRITONSWMM_scenario._generate_TRITON_SWMM_cfg
# via utils.create_from_template -> string.Template.safe_substitute. Do not
# pre-substitute these at synth build time — the toolkit fills them in at
# scenario prep. The file is written verbatim.
_TEMPLATE = """\
#---------------------------------------------------------------------------
# TRITON-SWMM config file - ${CASE_DESC}
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Mandatory input files
#---------------------------------------------------------------------------
dem_filename="${DEM}"
inp_filename="${SWMM}"
#---------------------------------------------------------------------------
# Format=BIN or ASC
# Option=PAR or SEQ
#---------------------------------------------------------------------------
input_format=ASC
input_option=SEQ
outfile_pattern="%s/%s/%s_%02d_%02d"
output_format=${OUT_FORMAT}
output_option=SEQ

#---------------------------------------------------------------------------
# TRITON-SWMM specific inputs
#---------------------------------------------------------------------------
manhole_diameter=${MH_DIAM}
manhole_loss=${MH_LOSS}

#---------------------------------------------------------------------------
# Hydrograph and flow locations information
#---------------------------------------------------------------------------
num_sources=${NUM_SOURCES}
hydrograph_filename="${HYDROGRAPH}"
src_loc_file="${HYDO_SRC_LOC}"

#---------------------------------------------------------------------------
# Manning input files
#---------------------------------------------------------------------------
${MAN_FILE_TOGGLE}n_infile="${MANNINGS}"
${CONST_MAN_TOGGLE}const_mann=${CONST_MAN}

#---------------------------------------------------------------------------
# Runoff related information
#---------------------------------------------------------------------------
num_runoffs=0
#runoff_filename=""
#runoff_file=""

#---------------------------------------------------------------------------
# External boundaries
#---------------------------------------------------------------------------
num_extbc=${NUM_EXT_BC}
extbc_dir="${EXTBC_DIR}"
extbc_file="${EXTBC_FILE}"

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
sim_duration=${SIM_DUR_S}

#---------------------------------------------------------------------------
# If checkpoint_id is 0 that means a clean start
#---------------------------------------------------------------------------
checkpoint_id=0

#---------------------------------------------------------------------------
# time_increment_fixed=0 is for variable dt, 1 for constant dt
#---------------------------------------------------------------------------
time_step=${TSTEP_S}
time_increment_fixed=0

#---------------------------------------------------------------------------
# Print interval in seconds of simulation time
#---------------------------------------------------------------------------
print_interval=${REPORTING_TSTEP_S}

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
open_boundaries=${OPEN_BOUNDARIES}
"""


def build_cfg(params, dest: Path) -> Path:
    """Write the synth TRITON-SWMM definition template.

    `params` is accepted for signature parity with the other synthetic-model
    builders (geometry, landuse, weather) but is not referenced: the template
    is fully parametrized via ${NAME} placeholders that the toolkit fills in
    at scenario-prep time, so no synth-side substitution is needed.
    """
    del params  # explicit: no synth-side values go into the template
    dest.write_text(_TEMPLATE, encoding="utf-8")
    return dest
