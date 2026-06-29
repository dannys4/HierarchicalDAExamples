#!/bin/bash

# Parameters for command
export SIGMA_X=0.025,0.05,0.1
export DELTA_Y=20,40,80
export NE=25,50,75
export RUNS=10
export TF=10.0

$RUN_CMD linear_advection -n $RUNS tf=$TF proj_path=$PROJ_PATH data_path=$IN_DATA_PATH sigma_x_filter=$SIGMA_X delta_y=$DELTA_Y Ne=$NE make_figs=false

