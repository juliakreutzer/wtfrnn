#!/bin/bash

# verbose
set -x

# training params
epochs=100
step=0.01
wvecDim=30
acti="relu"
rho=1e-6
optimizer="adagrad"
init=0.01
w2v=embeddings/w2v.w5.d30.c20.ella.model

model="RNTN" #either RNN or RNTN


name=${model}_wvecDim_${wvecDim}_step_${step}_acti_${acti}_rho_${rho}_opti_${optimizer}_init_${init}
outfile="models/${name}_full15.bin"
logfile="logs/${name}_full15.log"

#name=${model}_wvecDim_${wvecDim}_step_${step}_acti_${acti}_rho_${rho}_opti_${optimizer}_init_${init}_w2v_gensim
#outfile="models/${name}_w2v_gensim_partial.bin"
#logfile="logs/${name}__w2v_gensim_partial.log"


echo $outfile


python runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --outputDim 5 --wvecDim $wvecDim \
                --model $model --acti $acti --rho $rho --optimizer $optimizer \
                --init $init --partial &> $logfile
