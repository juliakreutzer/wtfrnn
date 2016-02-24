#!/bin/bash

# verbose
set -x


#infile="models/RNN_wvecDim_25_step_1e-2_2.bin" # the pickled neural network
infile="models/RNTN_wvecDim_30_step_0.01_acti_relu.bin"
model="RNTN" # the neural network type
acti="relu"

echo $infile

# test the model on test data
python runNNet.py --inFile $infile --test --data "test" --model $model --acti $acti

# test the model on dev data
#python runNNet.py --inFile $infile --test --data "dev" --model $model --acti $acti

# test the model on training data
#python runNNet.py --inFile $infile --test --data "train" --model $model --acti $acti












