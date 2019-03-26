#!/bin/csh
#$ -M tjiang2@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu=1
#$ -N graphSAGE

cd ..

/afs/crc.nd.edu/user/t/tjiang2/anaconda2/envs/pytorch/bin/python -m src.main --epochs 50 --learn_method unsup --cuda --num_neg 100
