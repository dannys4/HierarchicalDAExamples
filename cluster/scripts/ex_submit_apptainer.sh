#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 64 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_preemptable
#SBATCH -o log/linear_advection.log-%A-%a
#SBATCH -n 12
#SBATCH -a 1-32
#SBATCH --mem=10G

module load apptainer
export OUT_DATA_PATH=/pool001/$USER/HDA/linear_advection
export IN_DATA_PATH=/mnt/data
export APPTAINERENV_IN_DATA_PATH=/mnt/data
export APPTAINERENV_PROJ_PATH=/home/jovyan/.julia/environments/v1.11
export APPTAINERENV_RUN_CMD=./cluster/scripts/run_example

singularity exec -f -H /home/jovyan --no-home --writable-tmpfs --bind ..:/home/jovyan/cluster,$OUT_DATA_PATH:$IN_DATA_PATH docker://dannys4/hierarchical_data_assimilation ./cluster/scripts/ex_command.sh
