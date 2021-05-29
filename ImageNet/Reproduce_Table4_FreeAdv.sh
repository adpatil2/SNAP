#!/bin/bash

# Evaluating pretrained model
python smthPGD_All_attacks_restarts_CustStart_FreeTrain_AllNoise_ImageNet.py --noise-dist laplace --noise-pw 4500.0 --attack-type linf --epsilon 0.0078 --num-steps 100 --num-restarts 1 --start-batch-ind 1 --end-batch-ind 1000
python smthPGD_All_attacks_restarts_CustStart_FreeTrain_AllNoise_ImageNet.py --noise-dist laplace --noise-pw 4500.0 --attack-type l1 --epsilon 72.0 --num-steps 100 --num-restarts 1 --start-batch-ind 1 --end-batch-ind 1000
python smthPGD_All_attacks_restarts_CustStart_FreeTrain_AllNoise_ImageNet.py --noise-dist laplace --noise-pw 4500.0 --attack-type l2 --epsilon 2.0 --num-steps 100 --num-restarts 1 --start-batch-ind 1 --end-batch-ind 1000

## Training a new model
python Train_ImgNet_FreeTrain_AllNoise_final.py --noise-dist laplace --noise-pw 4500.0