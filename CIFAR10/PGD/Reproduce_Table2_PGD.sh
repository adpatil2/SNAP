#!/bin/bash

# For Evaluating Pretrained PGD+SNAP network (see Table 2 in the main text)
python smthPGD_All_attacks_restarts_Madry_AllNoise.py --noise-dist laplace --noise-shp-basis std --noise-pw 160.0 --attack-type linf --epsilon 0.031 --num-steps 100
python smthPGD_All_attacks_restarts_Madry_AllNoise.py --noise-dist laplace --noise-shp-basis std --noise-pw 160.0 --attack-type l2 --epsilon 0.5 --num-steps 100
python smthPGD_All_attacks_restarts_Madry_AllNoise.py --noise-dist laplace --noise-shp-basis std --noise-pw 160.0 --attack-type l1 --epsilon 12.0 --num-steps 100
python Compute_UniRobAcc_restarts_Madry_AllNoise.py --noise-dist laplace --noise-shp-basis std --noise-pw 160.0

# For Training new model:
python Train_CIFAR_Madry_AllNoise_final.py --noise-dist laplace --noise-shp-basis std --noise-pw 160.0 