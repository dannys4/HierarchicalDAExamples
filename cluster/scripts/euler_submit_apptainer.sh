#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 64 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_preemptable
#SBATCH -o log/euler.log-%A-%a
#SBATCH -n 2
#SBATCH -a 1-2
#SBATCH --mem=10G
#SBATCH --time=10:00:00

module load apptainer
export CTNR_HOME=/home/jovyan
export OUT_DATA_PATH=$HOME/orcd/pool/HDA/burgers
export IN_DATA_PATH=/mnt/data
export APPTAINERENV_IN_DATA_PATH=/mnt/data
export APPTAINERENV_PROJ_PATH=$CTNR_HOME/.julia/environments/v1.11
export APPTAINERENV_RUN_CMD=$CTNR_HOME/cluster/scripts/run_example
export APPTAINERENV_JULIA_CPU_TARGET="generic;znver3,clone_all;znver4,base(1)"

singularity exec -f -H $CTNR_HOME --no-home --writable-tmpfs --bind .:$CTNR_HOME/cluster,$OUT_DATA_PATH:$IN_DATA_PATH docker://dannys4/hierarchical_data_assimilation $CTNR_HOME/cluster/scripts/euler_run.sh
