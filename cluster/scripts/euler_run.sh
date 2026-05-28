#!/bin/bash

# Parameters for run
export DELTA_Y=10,20,30
export NE=50,100,200
export RUNS=4

$RUN_CMD euler_sod -n $RUNS proj_path=$PROJ_PATH data_path=$IN_DATA_PATH sigma_x_filter=$SIGMA_X delta_y=$DELTA_Y Ne=$NE
