# Automated Federated Learning via Informed Pruning
This is the repositoy for the paper AutoFLIP submitteed at AutoML  Conf'24.

We also have a lean and convenient AutoFLIP integration. For this clone the AutoFLIP repo, checkout the branch and install it (assuming you already activated your favorite virtual env):

git clone https://github.com/automl/AutoFLIP.git && cd AutoFLIP
git checkout feature/sawei
pip install -e .

You can find a usage example [here](INSERISCI LINK).
For the experimental code from this repository, see below.

## Installation
> :warning: Only tested on Linux.


First, create a fresh conda environment and activate it.
```bash
conda create -n SAWEI python=3.10.0 -y
conda activate SAWEI
```

Then, download the repository from [here](INSERISCI LINK):
```bash
cd SAWEI
```



## Repository and Code Structure
With AutoFLIP we dynamically prunes and compresses DL models within both the local clients and the global server. It leverages a federated loss exploration phase to investigate model gradient behavior across diverse datasets and losses, providing insights into parameter significance for build a efficent mask for the pruining. Our experiments showcase notable enhancements in scenarios with strong non-IID data, underscoring AutoFLIP’s capacity to tackle computational constraints and achieve superior global convergence. 

Therefore we have to main components in our code:
- FedLEx: Where we complete the exploration sesison
- FL: Where we perform the Federated Leargnin traning with AutoFLIP

```
└───AWEI
    |   └───runscripts
    |   └───awei: code for heuristics, baselines and our method SAWEI (`adaptive_weighted_ei.py`)
    |   |       └───configs: settings for all methods and experiments
    |   |       └───`optimize.py` runscript for evaluating methods
    |   |       └───`evaluate.ipynb`: Plotting script
    |   |       └───...
└───dacbo: defines the dynamic interface (gym) to SMAC in `new_env.py`
```



## Abstract
Federated learning (FL) represents a pivotal shift in machine learning (ML) as it enables 4
collaborative training of local ML models coordinated by a central aggregator, all without 5
the need to exchange local data. However, its application on edge devices is hindered by 6
limited computational capabilities and data communication challenges, compounded by 7
the inherent complexity of Deep Learning (DL) models. Model pruning is identified as a 8
key technique for compressing DL models on devices with limited resources. Nonetheless, 9
conventional pruning techniques typically rely on manually crafted heuristics and demand 10
human expertise to achieve a balance between model size, speed, and accuracy, often 11
resulting in sub-optimal solutions. 12
In this study, we introduce an automated federated learning approach utilizing informed 13
pruning, called AutoFLIP, which dynamically prunes and compresses DL models within 14
both the local clients and the global server. It leverages a federated loss exploration phase to 15
investigate model gradient behavior across diverse datasets and losses, providing insights into 16
parameter significance. Our experiments showcase notable enhancements in scenarios with 17
strong non-IID data, underscoring AutoFLIP’s capacity to tackle computational constraints 18
and achieve superior global convergence.

## Example
Our scripts use [hydra](INSERISCI LINK) making configuring experiments covenient and requires a special commandline syntax. 
Here we show how ''...''.
```bash
cd AWEI
python awei/optimize.py '+policy=awei' 'seed=89' +instance_set/BBOB=default 'instance_set.fid=15' 'instance_set.instance=1' 'n_eval_episodes=1' +dim=2d
```
If you want to configure the dimensions, checkout options in `AWEI/awei/configs/dim`.
You can find the logs in `AWEI/awei/awei_runs/DATE/TIME`.
The logs have following content:
- `initial_design.json`: The configurations for the initial design/design of experiments (DoE).
- `rollout_data.json`: The rollout data of the optimization ("step", "state", "action", "reward", "instance", "cost", "configuration", "episode", "policy_name", "seed")
- `wei_history.json`: The summands of the Weighted Expected Improvement (WEI) ("n_evaluated", "alpha", "pi_term", "ei_term", "pi_pure_term", "pi_mod_term"). "pi_term" corresponds to the exploitation term, "ei_term" to the exploration term of WEI.
- `ubr_history.json`: The upper bound regret (UBR) with its summands ("n_evaluated", "ubr", "min_ucb", "min_lcb")

## Experiments
You can find all runcommands for the BBOB benchmarks in `AWEI/run_BBOB.sh`. All runcommands for the ablation of SAWEI on BBOB are in `AWEI/run_BBOB_ablation.sh`.
The runcommands for HPOBench are in `AWEI/run_HPOBench.sh`.

:warning: You might need to specify your own slurm or local setup via `+slurm=yoursetupconfig` as a command line override. We set a local launcher per default so no slurm cluster is required. 

Please find the data in this [google drive](https://drive.google.com/drive/folders/12jmpJ1VRS3rzRcCd6rrrcjusP19RAmmV?usp=sharing).

### Plotting
All plots for the paper are generated in `AWEI/awei/evaluate.ipynb`.


## Cite Us
```bibtex
@inproceedings{benjamins-automl23a
    author    = {Carolin Benjamins and
                Elena Raponi and
                Anja Jankovic and
                Carola Doerr and
                Marius Lindauer},
    title     = {Self-Adjusting Weighted Expected Improvement for Bayesian Optimization},
    year      = {2023}
}
```

