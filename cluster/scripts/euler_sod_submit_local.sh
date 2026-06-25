#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 12 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_normal
#SBATCH -o log/burgers.log-%A-%a
#SBATCH -n 16
#SBATCH -a 1-12
#SBATCH --mem=10G

export IN_DATA_PATH=/home/dannys4/git-repos/HierarchicalDAExamples/cluster/scripts/data
export PROJ_PATH=$PWD/../..
export RUN_CMD=./run_example

./euler_sod.sh
