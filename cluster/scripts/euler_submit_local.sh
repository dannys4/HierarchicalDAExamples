#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 12 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_normal
#SBATCH -o log/euler.log-%A-%a
#SBATCH -n 8
#SBATCH -a 1-12
#SBATCH --mem=10G

# Pathing for commands
## local pathing
export IN_DATA_PATH=/pool001/$USER/HDA/euler
export PROJ_PATH=$PWD
export RUN_CMD=./run_example

./euler_run.sh
