# SNAP: Robustifying *l<sub>&infin;</sub>* Adversarial Training to the Union of Perturbation Models

This repository contains the code and pretrained models necessary for reproducing results in our recent preprint: 

**Robustifying *l<sub>&infin;</sub>* Adversarial Training to the Union of Perturbation Models** <br>
*Ameya D. Patil, M. Tuttle, Alexander G. Schwing, Naresh R. Shanbhag, University of IlliNoise at Urbana-Champaign (UIUC)* <br> 
Paper: [https://arxiv.org/abs/1702.06119](https://arxiv.org/abs/1702.06119)

## Short Summary \& Results: 

* Adversarial training (AT) frameworks are designed to achieve high adversarial accuracy against a single attack type, typically *l<sub>&infin;</sub>* norm-bounded perturbations. Recent extensions in AT have focused on defending against the union of multiple perturbation models but this benefit is obtained at the expense of a significant (10X) increase in training complexity over single-attack AT.

* In this work, we strive to achieve the best of both worlds, *i.e.*, high adversarial accuracy against the union of multiple perturbation models with the  training time of single-attack AT frameworks.

* Our technique, referred to as Shaped Noise Augmented Processing (SNAP), exploits a well-established byproduct of AT frameworks -- the reduction in the curvature of the decision boundary of networks. SNAP prepends a given deep net with a shaped noise augmentation layer whose distribution is learned along with network parameters using any standard single-attack AT. 

* As a result, SNAP enhances adversarial accuracy of ResNet-18 on CIFAR-10 against the union of (*l<sub>&infin;</sub>*, *l<sub>2</sub>*, *l<sub>1</sub>*) perturbation models by 14%-to-20% for four state-of-the-art (SOTA) single-attack AT frameworks as shown in the plot below. SNAP augmentations achieve the highest adversarial accuracy when training time is <12 hours on a single Tesla P100 GPU. 

* Thanks to its computational efficiency, SNAP augmentation of [FreeAdvTraining](https://github.com/mahyarnajibi/FreeAdversarialTraining) establishes a first benchmark for ResNet-50 robust to union of (*l<sub>&infin;</sub>*, *l<sub>2</sub>*, *l<sub>1</sub>*) perturbation models on ImageNet. 
<p align="center">
<img src="Intro_Time_Fig_rest_github.pdf" width="600" >
</p>
