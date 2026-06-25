#!/bin/bash

# Parameters for command
export NE=25,38,56,84,127
# export NE=50,100,200
export RUNS=10

$RUN_CMD euler_sod -n $RUNS proj_path=$PROJ_PATH data_path=$IN_DATA_PATH Ne=$NE make_figs=false
