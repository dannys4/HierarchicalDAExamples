#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 64 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_preemptable
#SBATCH -o log/linear_advection.log-%A-%a
#SBATCH -n 8
#SBATCH -a 1-64
#SBATCH --mem=10G

TMPDIR=`mktemp -d`
mkdir "$TMPDIR/compiled"
export JULIA_DEPOT_PATH="/pool001/$USER/julia_pkgs:$JULIA_DEPOT_PATH"
./run_example linear_advection -n 2 -- tf=10.0 data_path=/pool001/$USER/HDAE/linear_advection/ sigma_x_filter=0.01,0.025,0.05,0.10 delta_y=20,40,80 Ne=25,50,75,100
