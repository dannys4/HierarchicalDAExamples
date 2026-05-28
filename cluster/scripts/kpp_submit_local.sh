#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 12 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_normal
#SBATCH -o log/kpp.log-%A-%a
#SBATCH --exclusive
#SBATCH --mem=376GB
source /home/dannys4/miniconda3/bin/activate 

# Pathing for commands
## local pathing
export IN_DATA_PATH=/home/$USER/orcd/pool/HDA/kpp
export PROJ_PATH=$PWD/..
export RUN_CMD=./run_example

$RUN_CMD kpp -n 1 proj_path=$PROJ_PATH data_path=$IN_DATA_PATH
