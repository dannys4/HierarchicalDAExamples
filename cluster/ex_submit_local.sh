#!/bin/bash

# Example slurm submission script for MIT engaging.
# partition: mit_normal, print logs to `./log` subdir, 12 jobs (do the same thing), 16 cores per job, 10GB RAM minimum.
#SBATCH -p mit_normal
#SBATCH -o log/burgers.log-%A-%a
#SBATCH -n 16
#SBATCH -a 1-12
#SBATCH --mem=10G

# Run the burgers example two time for three different noise levels and four ensemble sizes (2*3*4=24 trials)
# Save the data to the pool directory to allow for significant storage, and only simulate to time 0.5 for each trial.
./run_example burgers -n 2 -- data_path=/pool001/$USER/ tf=0.5 sigma_x_filter=1e-3,1e-2,1e-1, Ne=25,50,75,100
