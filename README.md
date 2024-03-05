# Automated Federated Learning via Informed Pruning
This is the repositoy for the paper AutoFLIP submitteed at AutoML  Conf'24.

Clon the AutoFLIP repo, checkout the branch and install it (assuming you already activated your favorite virtual env).

For the experimental code from this repository, see below.

## Installation
> :warning: Only tested on Linux.


First, create a fresh conda environment and activate it:

tensorboard==2.12.1
numpy==1.24.2
pandas==1.5.3
scikit-learn==1.2.2
transformers==4.27.4
tqdm==4.65.0
einops==0.6.1
Plus, please install torch-related packages using one command provided by the official guide (See official installation guide); e.g., conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 torchtext==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge

## Repository and Code Structure
With AutoFLIP we dynamically prunes and compresses DL models within both the local clients and the global server. It leverages a federated loss exploration phase to investigate model gradient behavior across diverse datasets and losses, providing insights into parameter significance for build a efficent mask for the pruining. Our experiments showcase notable enhancements in scenarios with strong non-IID data, underscoring AutoFLIP capacity to tackle computational constraints and achieve superior global convergence. 

Therefore we have to main components in our code:
- Where we complete the exploration sesison for AutoFLIP (AutoFlip_Exploration)
- Where we perform the Federated Leargnin traning with AutoFLIP (AutoFLIP)
Additionaly we define different files for:
- Server
- Client
- Aggregation algorithm

```
└───src
    |   └───algorithm:
    |   |       └───basealgorithm.py:  For all FL algo
    |   |       └───`Autoflip.py` Aggregation for AutoFLIP
    |   |       └───`fedavg.ipynb`: Aggregation for FedAVG
    |   |       └───...
    |   └───server: server definition 
    |   |       └───Autoflipserver_exploration.py: For exploration phase of AutoFLIP
    |   |       └───`Autoflipserver.py` 
    |   |       └───`fedavgserver.py`: 
    |   |       └───...
    |   └───client: client definition 
    |   |       └───Autoflipclient_exploration.py: For exploration phase of AutoFLIP
    |   |       └───`Autoflipclient.py` 
    |   |       └───`fedavgclient.py`: 
    |   |       └───...
    |   └───datasets: ...(dataset used)
    |   └───metrics: ...(metrics zoo)
    |   └───models: ... (Deep Neural Network definition)
    └───GTL_utils.py: Functions to proces the guuidence pruning mask
    └───plot_utils.py
    └───utils.py
└───AutoFLIP
    └───main.py
```

## Abstract
Federated learning (FL) represents a pivotal shift in machine learning (ML) as it enables collaborative training of local ML models coordinated by a central aggregator, all without the need to exchange local data. However, its application on edge devices is hindered by limited computational capabilities and data communication challenges, compounded by the inherent complexity of Deep Learning (DL) models. Model pruning is identified as a key technique for compressing DL models on devices with limited resources. Nonetheless, conventional pruning techniques typically rely on manually crafted heuristics and demand human expertise to achieve a balance between model size, speed, and accuracy, often resulting in sub-optimal solutions. In this study, we introduce an automated federated learning approach utilizing informed pruning, called AutoFLIP, which dynamically prunes and compresses DL models within both the local clients and the global server. It leverages a federated loss exploration phase to investigate model gradient behavior across diverse datasets and losses, providing insights into parameter significance. Our experiments showcase notable enhancements in scenarios with strong non-IID data, underscoring AutoFLIP capacity to tackle computational constraints and achieve superior global convergence.

