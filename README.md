# FER_C
FER-C: Benchmarking Out-of-Distribution Soft Calibration for Facial Expression Recognition

##  Installation Requirements
We have provided a environment.yml in our repository. 
Main libraries needed: torch, torchvision, numpy, sklearn and a CUDA supported machine.
## Usage
>1.) Call bash run.sh #This will start a for-loop across three random seeds for all three datasets with and without calibration as described in the main paper.\
>2.)The main scripts of interest are in the form "train_[ALGO].py". with [ALGO]. For example, train_DMUE.py\
>3.)MaxEnt Loss and the other loss functions are implemented in "losses.py"

## AffectNet-C
![AffectNet](https://github.com/dexterdley/FER_C/blob/master/figures/soft_affectnet.png)
AffectNet-C is built from the official validation set of AffectNet, which contains 4000 images of size 224x224 for eight discrete emotion class. The soft labels are determined based on a Gaussian Mixture Model built from the valence, arousal values obtained in the training set and further softened based on the corruption types.

## AffWild-C
![Affwild](https://github.com/dexterdley/FER_C/blob/master/figures/soft_affwild.png)
AffWild-C is extracted from video frames based on unique identities from AffWild with seven discrete emotion classes. There are 240 frames of size 224x224 which gives 4800 source images, which are then augmented into AffWild-C.

## RAF-DB-C
![RAFDB](https://github.com/dexterdley/FER_C/blob/master/figures/soft_rafdb.png)
The RAF-DB dataset contains roughly 3000 source test images with mixed labels of six discrete emotion classes. The provided test labels are further softened based on the corruptions applied.
