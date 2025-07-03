#!/bin/bash

#SBATCH -p mit_normal
#SBATCH -o myjob.log-%A-%a
#SBATCH -a 0-3

# Example slurm submission script
echo $(nproc)
# ./run_example burgers -n 5 -- sigma_x_data=1e-3