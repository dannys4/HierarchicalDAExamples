#!/bin/bash

# Parameters for command
export SIGMA_Y=0.025,0.05,0.1
export DELTA_Y=20,40,80
export NE=25,50,75,100
export RUNS=2
export TF=2.0

$RUN_CMD burgers -n $RUNS tf=$TF proj_path=$PROJ_PATH data_path=$IN_DATA_PATH sigma_y=$SIGMA_Y delta_y=$DELTA_Y Ne=$NE

