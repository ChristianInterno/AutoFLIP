# Automated Federated Learning via Informed Pruning
This is the repositoy for the paper AutoFLIP submitteed at AutoML  Conf'24.

Clon the AutoFLIP repo, checkout the branch and install it (assuming you already activated your favorite virtual env).

For the experimental code from this repository, see below.

## Installation
> :warning: Only tested on Linux.


First, create a fresh conda environment and activate it.


## Repository and Code Structure
With AutoFLIP we dynamically prunes and compresses DL models within both the local clients and the global server. It leverages a federated loss exploration phase to investigate model gradient behavior across diverse datasets and losses, providing insights into parameter significance for build a efficent mask for the pruining. Our experiments showcase notable enhancements in scenarios with strong non-IID data, underscoring AutoFLIP capacity to tackle computational constraints and achieve superior global convergence. 

Therefore we have to main components in our code:
- FedLEx: Where we complete the exploration sesison for AutoFLIP
- FL: Where we perform the Federated Leargnin traning with AutoFLIP

## Abstract
Federated learning (FL) represents a pivotal shift in machine learning (ML) as it enables collaborative training of local ML models coordinated by a central aggregator, all without the need to exchange local data. However, its application on edge devices is hindered by limited computational capabilities and data communication challenges, compounded by the inherent complexity of Deep Learning (DL) models. Model pruning is identified as a key technique for compressing DL models on devices with limited resources. Nonetheless, conventional pruning techniques typically rely on manually crafted heuristics and demand human expertise to achieve a balance between model size, speed, and accuracy, often resulting in sub-optimal solutions. In this study, we introduce an automated federated learning approach utilizing informed pruning, called AutoFLIP, which dynamically prunes and compresses DL models within both the local clients and the global server. It leverages a federated loss exploration phase to investigate model gradient behavior across diverse datasets and losses, providing insights into parameter significance. Our experiments showcase notable enhancements in scenarios with strong non-IID data, underscoring AutoFLIP capacity to tackle computational constraints and achieve superior global convergence.

