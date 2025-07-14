#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 12 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_normal
#SBATCH -o log/burgers.log-%A-%a
#SBATCH -n 16
#SBATCH -a 1-12
#SBATCH --mem=10G

export IN_DATA_PATH=/pool001/$USER/HDA/linear_advection
export PROJ_PATH=$PWD
export RUN_CMD=./run_example
export JULIA_CPU_TARGET="generic;znver3,clone_all;znver4,base(1)"
./ex_command.sh
