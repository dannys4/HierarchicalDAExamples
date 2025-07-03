#!/bin/bash

#SBATCH -p mit_normal
#SBATCH -o log/burgers.log-%A-%a
#SBATCH -N 1
#SBATCH -a 1-12
#SBATCH --mem=18G

# Example slurm submission script
# echo $(nproc)
./run_example burgers -n 2
