# SNAP: Robustifying *l<sub>&infin;</sub>* Adversarial Training to the Union of Perturbation Models

This repository contains the code and pretrained models necessary for reproducing results in our recent preprint: 

**Robustifying *l<sub>&infin;</sub>* Adversarial Training to the Union of Perturbation Models** <br>
*Ameya D. Patil, M. Tuttle, Alexander G. Schwing, Naresh R. Shanbhag, University of IlliNoise at Urbana-Champaign (UIUC)* <br> 
Paper: [https://arxiv.org/abs/1702.06119](https://arxiv.org/abs/1702.06119)

## Short Summary \& Results: 

* Classical adversarial training (AT) frameworks are designed to achieve high adversarial accuracy against a single attack type, typically *l<sub>&infin;</sub>* norm-bounded perturbations. Recent extensions in AT have focused on defending against the union of multiple perturbation models but this benefit is obtained at the expense of a significant (up to 10X) increase in training complexity over single-attack *l<sub>&infin;</sub>* AT.

* In this work, we expand the capabilities of widely popular single-attack *l<sub>&infin;</sub>* AT frameworks to provide robustness to the union of (*l<sub>&infin;</sub>*, *l<sub>2</sub>*, *l<sub>1</sub>*) perturbations models, while preserving the training efficiency of single-attack *l<sub>&infin;</sub>* AT.

* Our technique, referred to as Shaped Noise Augmented Processing (SNAP), exploits a well-established byproduct of AT frameworks -- the reduction in the curvature of the decision boundary of networks. SNAP prepends a given deep net with a shaped noise augmentation layer whose distribution is learned along with network parameters using any standard single-attack AT. 

* As a result, SNAP enhances adversarial accuracy of ResNet-18 on CIFAR-10 against the union of (*l<sub>&infin;</sub>*, *l<sub>2</sub>*, *l<sub>1</sub>*) perturbation models by 14%-to-20% for four state-of-the-art (SOTA) single-attack *l<sub>&infin;</sub>* AT frameworks (PGD, [TRADES](https://github.com/yaodongyu/TRADES), [FreeAdv](https://github.com/mahyarnajibi/FreeAdversarialTraining), [FastAdv](https://github.com/locuslab/fast_adversarial)) as shown in the plot below. SNAP augmentations achieve the highest adversarial accuracy when training time is <12 hours on a single Tesla P100 GPU. 

* Thanks to its simplicity, efficiency, and scalability, SNAP augmentation of [FreeAdvTraining](https://github.com/mahyarnajibi/FreeAdversarialTraining) establishes a first benchmark for ResNet-50 and ResNet-101 networks robust to union of (*l<sub>&infin;</sub>*, *l<sub>2</sub>*, *l<sub>1</sub>*) perturbation models on ImageNet. 

* SNAP is very easy to integrate into any given single-attack *l<sub>&infin;</sub>* AT framework to improve its effectiveness against the union of (*l<sub>&infin;</sub>*, *l<sub>2</sub>*, *l<sub>1</sub>*) perturbations models.
<p align="center">
<img src="IntroFig_More_Git.png" width="600" >
</p>
<p align="center">
<img src="ImageNet_Git.png" width="800" >
</p>

## This Repository:

### Overview

Here we share our code to reproduce SNAP results on both CIFAR-10 (Fig.~2(b,d,f) in the paper) and ImageNet (Fig.~3(b,d,f) in the paper). We also share corresponding pretrained models to facilitate quick reproduction of our results. 

Importantly, these codes illustrate how ElliNoise can be employed in a given training setup with minimal modifications and no additional hyperparameter tuning. 

This illustrative shared code is based on publicly available [SmoothAdv](https://github.com/Hadisalman/smoothing-adversarial) code. As mentioned in the paper, ElliNoise augmentation of SmoothAdv is achieved via drop-in replacement of white noise by ElliNoise. 

The repository is organized in the following folders: 
```
- ReadMe (this file)
- CIFAR10/ 
	- PGD/
		- TrainedModels/
	- TRADES/
		- TrainedModels/
- ImageNet/
	- FreeAdv/
		- TrainedModels/
```
Sub-folders TrainedModels/ contain our models trained with the corresponding AT+SNAP.

Sub-folders PGD/, TRADES/, and FreeAdv/ contain all the python files to evaluate the pretrained models or to train the new models. 

### Getting Started: 

To clearly illustrate example usage, we provide shell scripts (.sh) that execute all training and evaluation python codes sequentially with appropriate arguments to reproduce the results for CIFAR-10 (Table 2 in the main text) and ImageNet (Table 4 in the main text). The evaluation scripts are configured to load our PreTrained models. Those scripts are listed below. 

	CIFAR10/PGD: Reproduce_Table2_PGD.sh 

	CIFAR10/TRADES: Reproduce_Table2_TRADES.sh 

	ImageNet/FreeAdv: Reproduce_Table4_FreeAdv.sh 


### Points to note: 

	- ImageNet dataset directory: All ImageNet python codes have a default ImageNet dataset directory. That 'dataset-dir' arguement needs to be provided appropriately. 
	- In order to execute .sh scripts, one needs to make them executable first. For example, to make Reproduce_Table2_PGD.sh an executable file, type ``chmod a+x Reproduce_Table2_PGD.sh'' in command line. Then, it can be run via command  ``./Reproduce_Table2_PGD.sh''
	- All python/shell scripts are assumed to be executed in their current directory. 

## Contact and Citation Information 

If you face any issues in running these codes or if you have any questions about this work, please feel free to contact me at [adpatil2@illinois.edu](adpatil2@illinois.edu). I would be more than happy to help!

If you find the paper or the code useful for your research, please consider citing it as follows:
```bash
@article{patil2021snap,
  title={ElliNoise: Improving Accuracy vs Robustness Trade-off in Deep Nets via Ellipsoidal Noise Shaping},
  author={Patil, Ameya and Tuttle, Michael and Schwing, Alexander and Shanbhag, Naresh },
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2021}
}
```

## Acknowledgements

Parts of the code in this repository are based on following awesome public repositories: 

* [http://github.com/locuslab/smoothing](http://github.com/locuslab/smoothing)
* [https://github.com/locuslab/robust_union](https://github.com/locuslab/robust_union)
* [https://github.com/yaodongyu/TRADES](https://github.com/yaodongyu/TRADES)
* [https://github.com/ashafahi/free_adv_train](https://github.com/ashafahi/free_adv_train)
* [https://github.com/locuslab/fast_adversarial](https://github.com/locuslab/fast_adversarial)
