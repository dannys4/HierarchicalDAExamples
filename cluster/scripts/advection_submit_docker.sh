#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 64 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_preemptable
#SBATCH -o log/linear_advection.log-%A-%a
#SBATCH -n 16
#SBATCH -a 1-32
#SBATCH --mem=20G

export IN_DATA_PATH=/mnt/data
export OUT_DATA_PATH=$PWD/scripts/data
export DOCKERENV_IN_DATA_PATH=/mnt/data
export DOCKERENV_PROJ_PATH=/home/jovyan/.julia/environments/v1.11
export DOCKERENV_RUN_CMD=./scripts/run_example
export EXAMPLE_SCRIPT=./scripts/advection_run.sh

docker compose up
