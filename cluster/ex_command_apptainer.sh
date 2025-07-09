#!/bin/bash

# Pathing for commands
## Singularity pathing
export IN_DATA_PATH=/mnt/data
export PROJ_PATH=/home/jovyan/.julia/environments/v1.11
export RUN_CMD=./cluster/run_example

# Parameters for command
export SIGMA_X=0.01,0.025,0.05,0.10
export DELTA_Y=20,40,80
export NE=25,50,75,100
export RUNS=2
export TF=10.0

$RUN_CMD linear_advection -n $RUNS tf=$TF proj_path=$PROJ_PATH data_path=$IN_DATA_PATH sigma_x_filter=$SIGMA_X delta_y=$DELTA_Y Ne=$NE

