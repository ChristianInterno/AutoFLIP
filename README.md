# Automated Federated Learning via Informed Pruning
This is the repositoy for the paper AutoFLIP submitteed at AutoML  Conf'24.

## Getting Started

Before diving into the specifics, ensure you have activated your preferred virtual environment.

### Clone and Set Up the AutoFLIP Repository

```bash
git clone [AutoFLIP-repo-url] 
cd AutoFLIP 
git checkout [branch-name]
```

For the experimental code from this repository, see below.

## Installation
:warning: Note: AutoFLIP has been tested exclusively on Linux environments.

First, create a fresh conda environment and activate it:

conda create --name autoflip python=3.8
conda activate autoflip

Install the required packages:

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 torchtext==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge

Please install torch-related packages using one command provided by the official guide (See official installation guide); e.g., conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 torchtext==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge

## Repository and Code Structure
With AutoFLIP we dynamically prunes and compresses DL models within both the local clients and the global server. It leverages a federated loss exploration phase to investigate model gradient behavior across diverse datasets and losses, providing insights into parameter significance for build a efficent mask for the pruining. Our experiments showcase notable enhancements in scenarios with strong non-IID data, underscoring AutoFLIP capacity to tackle computational constraints and achieve superior global convergence. 

Therefore we have to main components in our code:
-AutoFlip_Exploration: Where we complete the exploration sesison for AutoFLIP.
-AutoFLIP: Where we perform the Federated Leargnin traning with AutoFLIP.

Further, the repository structure is organized as follows:
src: Contains the source code for algorithms, server and client implementations, and utilities.
checkpoints: Stores pruned model states for both clients and the server.
logs: Saves all logs from the experiments.
plots: Contains the code for visualizations of the experimental results.
results_paper: Holds the experimental outcomes referenced in the paper.
AutoFLIP: Main directory for executing AutoFLIP experiments.

```
AutoFLIP/
├── src/
│   ├── algorithm/
│   │   ├── basealgorithm.py
│   │   ├── autoflip.py
│   │   ├── fedavg.ipynb
│   │   └── ...
│   ├── server/
│   │   ├── autoflipserver_exploration.py
│   │   ├── autoflipserver.py
│   │   └── ...
│   ├── client/
│   │   ├── autoflipclient_exploration.py
│   │   ├── autoflipclient.py
│   │   └── ...
│   ├── datasets/
│   ├── metrics/
│   └── models/
├── checkpoints/
│   ├── ClientPrunedGModel.py
│   └── InitPrunedGlobalModel.py
├── logs/
├── plots/
├── result_paper/
└── main.py

```

## Abstract
Federated learning (FL) represents a pivotal shift in machine learning (ML) as it enables collaborative training of local ML models coordinated by a central aggregator, all without the need to exchange local data. However, its application on edge devices is hindered by limited computational capabilities and data communication challenges, compounded by the inherent complexity of Deep Learning (DL) models. Model pruning is identified as a key technique for compressing DL models on devices with limited resources. Nonetheless, conventional pruning techniques typically rely on manually crafted heuristics and demand human expertise to achieve a balance between model size, speed, and accuracy, often resulting in sub-optimal solutions. In this study, we introduce an automated federated learning approach utilizing informed pruning, called AutoFLIP, which dynamically prunes and compresses DL models within both the local clients and the global server. It leverages a federated loss exploration phase to investigate model gradient behavior across diverse datasets and losses, providing insights into parameter significance. Our experiments showcase notable enhancements in scenarios with strong non-IID data, underscoring AutoFLIP capacity to tackle computational constraints and achieve superior global convergence.

## Example
Start your AutoFLIP journey with the following example, which reproduces the CIFAR10 experiment from the paper:

do
    python3 main.py\
        --exp_name "CIFAR10_${treeshold}_${n}" --device cuda --result_path ./result_paper/20K/RedDim/CIFAR10/AutoFLIP \
        --seed 1709566899 dataset CIFAR10 \
        --split_type patho --mincls 2 --test_fraction 0.2 \
        --rawsmpl 1.0 --resize 24  --randhf 0.5  --randjit 0.5\
        --model_name TwoCNN --hidden_size 32 --dropout 0 --num_layers 4 --init_type xavier \
        --algorithm autoflip --eval_type both --eval_every 10 --eval_metrics acc1 acc5 f1 precision recall\
        --K 20 --R 200 --E 5 --C 0.25 --B 350 --beta 0.9 \
        --optimizer Adam --lr  0.0003 --lr_decay 0.99 --lr_decay_step 1 --criterion CrossEntropyLoss\
        --Patience_mask 40 --epoochs_mask 150 --perc_clients_for_mask 1 --mask_pruining True --treeshold_pruining  0.30
done


## Experiments
All runcommands for the ablation and experiments of AUtoFLIP are in `commands/command_paper.sh`.

