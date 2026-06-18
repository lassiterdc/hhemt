#!/bin/bash
#SBATCH -o _slurm_outputs/%x/%A_outputs_%a_%N.out
#SBATCH -e _slurm_outputs/%x/%A_errors_%a_%N.out
#SBATCH -A ${allocation}
#SBATCH -t ${time}
#SBATCH -p ${partition}
#SBATCH -N ${nodes}
#SBATCH --exclusive
# SBATCH --cpus-per-task=${cpus_per_task}   # CPU=1, GPU=threads per GPU
# SBATCH --ntasks-per-node=${ntasks_per_node} # GPU=tasks per node, CPU optional
#${gpu_toggle}SBATCH --gres=${gres}
#SBATCH --mail-user=***REMOVED***@virginia.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-${number_of_events}

# Load modules
module purge
module ${modules}
conda activate hhemt

${run_command} ${SLURM_ARRAY_TASK_ID} --prepare-sims=True --process-sim-timeseries=True