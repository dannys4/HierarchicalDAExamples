#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 64 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_preemptable
#SBATCH -o log/linear_advection.log-%A-%a
#SBATCH -n 16
#SBATCH -a 1-32
#SBATCH --mem=20G

# Pathing for commands
## local pathing
export IN_DATA_PATH=$HOME/orcd/pool/HDA/linear_advection
export PROJ_PATH=$PWD
export RUN_CMD=./cluster/scripts/run_example

./cluster/scripts/advection_run.sh
