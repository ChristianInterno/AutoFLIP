import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #set of devices that you want to use
import sys
import torch
import argparse
import traceback

import time


from importlib import import_module
from torch.utils.tensorboard import SummaryWriter



from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model
import gc
import copy

def main(args, writer):
    """Main program to run federated learning.

    Args:
        args: user input arguments parsed by argparser
        writer: `torch.utils.tensorboard.SummaryWriter` instance for TensorBoard tracking
    """
    # set seed for reproducibility
    set_seed(args.seed)

    server_dataset, client_datasets = load_dataset(args)

    # adjust device
    if 'cuda' in args.device:
        assert torch.cuda.is_available(), 'Please check if your GPU is available now!'
        args.device = 'cuda' if args.device_ids == [] else f'cuda:{args.device_ids[0]}'

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    #     raise ImportWarning('GPU NOT FOUND. CPU computation might be slow.')


    # get model
    model, args = load_model(args)

    # Multi GPU
    # if n_gpus > 1:
    #     model = torch.nn.DataParallel(model)

    # check all args before FL
    args = check_args(args)

    if args.algorithm == 'Autoflip':
        # # create copy of the central server for Autoflip pruinig
        pruned_model = copy.deepcopy(model)  # or create a new instance and load state
        server_class_pruining = import_module(f'src.server.{args.algorithm}server_Exploration').__dict__[f'{args.algorithm.title()}server']
        server_pruining  = server_class_pruining(args=args, writer=writer, server_dataset=server_dataset, client_datasets=client_datasets, model=pruned_model)
        server_pruining.update()

        # Explicit cleanup
        del server_pruining
        # Optionally force garbage collection, though usually not necessary
        gc.collect()

    # create central server
    server_class = import_module(f'src.server.{args.algorithm}server').__dict__[f'{args.algorithm.title()}server']
    server = server_class(args=args, writer=writer, server_dataset=server_dataset, client_datasets=client_datasets, model=model)

    # federated learning
    for curr_round in range(1, args.R + 1):

        ## update round indicator
        server.round = curr_round

        ## update after sampling clients randomly
        selected_ids  = server.update()

        ## evaluate on clients not sampled (for measuring generalization performance)
        if curr_round % args.eval_every == 0 or (curr_round == args.R):
            server.evaluate(excluded_ids=selected_ids)

        # torch.cuda.empty_cache() #check

    else:
        ## wrap-up
        server.finalize()



if __name__ == "__main__":

    # parse user inputs as arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    #####################
    # Default arguments #
    #####################
    parser.add_argument('--exp_name', help='experiment name', type=str, required=False, default=['Autoflip'])
    seed = int(time.time())
    parser.add_argument('--seed', help='global random seed', type=int, default=seed)
    parser.add_argument('--device', help='device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`', type=str, default='cuda')
    parser.add_argument('--device_ids',  nargs='+', type=int, help='GPU device ids for multi-GPU training (use all available GPUs if no number is passed)', default=[])
    parser.add_argument('--data_path', help='path to read data from', type=str, default='./data')
    parser.add_argument('--log_path', help='path to store logs', type=str, default='./log')
    parser.add_argument('--result_path', help='path to save results', type=str, default='./result_paper')
    parser.add_argument('--use_tb', help='use TensorBoard to track logs (atif passed)', action='store_true')
    parser.add_argument('--tb_port', help='TensorBoard port number (valid only if `use_tb`)', type=int, default=6006)
    parser.add_argument('--tb_host', help='TensorBoard host address (valid only if `use_tb`)', type=str, default='0.0.0.0')

    #####################
    # Dataset arguments #
    #####################
    ## dataset
    parser.add_argument('--dataset', help='''name of dataset to use for an experiment 
    * NOTE: case sensitive*
    - image classification datasets in `torchvision.datasets`,
    - text classification datasets in `torchtext.datasets`,
    - LEAF benchmarks [ FEMNIST | Sent140 | Shakespeare | CelebA | Reddit ],
    - among [ TinyImageNet | CINIC10 | BeerReviewsA | BeerReviewsL | Heart | Adult | Cover | GLEAM ]
        ''', type=str, required=False, default='CIFAR10')
    parser.add_argument('--test_fraction', help='fraction of local hold-out dataset for evaluation', type=float, choices=[Range(0., 9e-1)], default= 0.2)
    parser.add_argument('--rawsmpl', help='fraction of raw data to be used (only used when one of `LEAF` datasets is used)', type=float, choices=[Range(0., 1.)], default=1.0)

    ## data augmentation arguments
    parser.add_argument('--resize', help='resize input images (using `torchvision.transforms.Resize`)', type=int, default=24) #24
    parser.add_argument('--imnorm', help='normalize channels using ImageNet pre-trained mean & standard deviation (using `torchvision.transforms.Normalize`)', action='store_true')
    parser.add_argument('--randrot', help='randomly rotate input (using `torchvision.transforms.RandomRotation`)', type=int, default=None)
    parser.add_argument('--randhf', help='randomly flip input horizontaly (using `torchvision.transforms.RandomHorizontalFlip`)', type=float, choices=[Range(0., 1.)], default=0.5)#0.5
    parser.add_argument('--randvf', help='randomly flip input vertically (using `torchvision.transforms.RandomVerticalFlip`)', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randjit', help='randomly flip input vertically (using `torchvision.transforms.RandomVerticalFlip`)', type=float, default=0.5) #0.5

    ## statistical heterogeneity simulation arguments
    parser.add_argument('--split_type', help='''type of data split scenario
    - `iid`: statistically homogeneous setting,
    - `unbalanced`: unbalance in sample counts across clients,
    - `patho`: pathological non-IID split scenario proposed in (McMahan et al., 2016),
    - `diri`: Dirichlet distribution-based split scenario proposed in (Hsu et al., 2019),
    - `pre`: pre-defined data split scenario
    ''', type=str, choices=['iid', 'unbalanced', 'patho', 'diri', 'pre'], required=False, default='patho')
    parser.add_argument('--mincls', help='the minimum number of distinct classes per client (only used when `split_type` is `patho`)', type=int, default=2)
    parser.add_argument('--cncntrtn', help='a concentration parameter for Dirichlet distribution (only used when `split_type` is `diri`)', type=float, default=0.2)


    ###################
    # Model arguments#
    ###################
    ## model
    parser.add_argument('--model_name', help='a model to be used (note that it is case sensitive)', type=str,
        choices=[
            'TwoNN', 'TwoCNN', 'FEMNISTCNN',
            'LeNet', 'MobileNet', 'SqueezeNet',
            'VGG9', 'VGG9BN', 'VGG11', 'VGG11BN', 'VGG13', 'VGG13BN',
            'ResNet10', 'ResNet18', 'ResNet34',
            'MobileNeXt', 'SqueezeNeXt', 'MobileViT',
            'NextCharLSTM', 'NextWordLSTM',
            'DistilBert', 'SqueezeBert', 'MobileBert',
            'LogReg', 'GRUClassifier','twocnn_GTL', 'BIGCNN', 'WideResNet'
        ],
        required=False, default='TwoCNN'
    )
    parser.add_argument('--hidden_size', help='hidden channel size for vision models, or hidden dimension of language models', type=int, default=32) #32
    parser.add_argument('--dropout', help='dropout rate', type=float, choices=[Range(0., 1.)], default=0)
    parser.add_argument('--use_model_tokenizer', help='use a model-specific tokenizer (if passed)', action='store_true')
    parser.add_argument('--use_pt_model', help='use a pre-trained model weights for fine-tuning (if passed)', action='store_true')
    parser.add_argument('--seq_len', help='maximum sequence length used for `torchtext.datasets`)', type=int, default=512)
    parser.add_argument('--num_layers', help='number of layers in recurrent cells', type=int, default=4) #4
    parser.add_argument('--num_embeddings', help='size of embedding dictionary', type=int, default=1000)
    parser.add_argument('--embedding_size', help='embedding dimension of language models', type=int, default=512)
    parser.add_argument('--init_type', help='weight initialization method', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'])
    parser.add_argument('--init_gain', type=float, default=1.0, help='magnitude of variance used for weight initialization')


    ######################
    ######################
    # Learning arguments #
    ######################
    ## federated learning settings
    parser.add_argument('--algorithm', help='type of an federated aggragation learning algorithm to be used', type=str,
        choices=['fedavg', 'Autoflip'],
        required=False, default='Autoflip'
    )
    parser.add_argument('--eval_type', help='''the evaluation type of a model trained from FL algorithm
    - `local`: evaluation of personalization model on local hold-out dataset  (i.e., evaluate personalized models using each client\'s local evaluation set)
    - `global`: evaluation of a global model on global hold-out dataset (i.e., evaluate the global model using separate holdout dataset located at the server)
    - 'both': combination of `local` and `global` setting
    ''', type=str,
        choices=['local', 'global', 'both'],
        required=False, default='both'
    )
    parser.add_argument('--eval_fraction', help='fraction of randomly selected (unparticipated) clients for the evaluation (for `eval_type` is `local` or `both`)', type=float, choices=[Range(1e-8, 1.)], default=1.) #1
    parser.add_argument('--eval_every', help='frequency of the evaluation (i.e., evaluate peformance of a model every `eval_every` round)', type=int, default=10)
    parser.add_argument('--eval_metrics', help='metric(s) used for evaluation', type=str,
        choices=[
            'acc1', 'acc5', 'auroc', 'auprc', 'youdenj', 'f1', 'precision', 'recall',
            'seqacc', 'mse', 'mae', 'mape', 'rmse', 'r2', 'd2'
        ], nargs='+', required=False, default=['acc1','acc5','f1','precision','recall']
    )
    parser.add_argument('--K', help='number of total cilents participating in federated training', type=int, default=20)
    parser.add_argument('--R', help='number of total rounds', type=int, default=200)
    parser.add_argument('--C', help='sampling fraction of clietns per round (full participation when zero is passed)', type=float, choices=[Range(0., 1.)], default=0.05)
    parser.add_argument('--E', help='number of local epochs', type=int, default=5)
    parser.add_argument('--B', help='batch size for local update in each client (full-batch training when zero is passed)', type=int, default=350) #64
    parser.add_argument('--beta', help='global momentum factor for an update of a global model when aggregated at the server', type=float, choices=[Range(0., 1.)], default=0.9) #0.9

    # optimization arguments
    parser.add_argument('--no_shuffle', help='do not shuffle data (if passed)', action='store_true')
    parser.add_argument('--optimizer', help='type of optimization method (should be a sub-module of `torch.optim`)', type=str, default='Adam')
    parser.add_argument('--max_grad_norm', help='a constant required for gradient clipping', type=float, choices=[Range(0., float('inf'))], default=0.)
    parser.add_argument('--weight_decay', help='weight decay (L2 penalty)', type=float, choices=[Range(0., 1.)], default=0)
    parser.add_argument('--momentum', help='momentum factor', type=float, choices=[Range(0., 1.)], default=0.)
    parser.add_argument('--lr', help='learning rate for local updates in each client', type=float, choices=[Range(0., 100.)], default=0.0003)
    parser.add_argument('--lr_server', help='learning rate for the server opt', type=float, choices=[Range(0., 100.)], default=0.0003)
    parser.add_argument('--lr_decay', help='rate of learning rate decay applied per round', type=float, choices=[Range(0., 1.)], default=0.99)
    parser.add_argument('--lr_decay_step', help='rate of learning rate decay applied per round', type=int, default=1)
    parser.add_argument('--criterion', help='type of criterion for objective function (should be a submodule of `torch.nn`)', type=str, default='CrossEntropyLoss')
    parser.add_argument('--mu', help='constant for proximity regularization term (for algorithms `fedprox`)', type=float, choices=[Range(0., 100)], default=0.01)

    ##### Argoument for Eploraiton phase #####
    parser.add_argument('--Patience_mask', help='Pateince early stopping for scouting process', type=int, default=40)
    parser.add_argument('--epoochs_mask', help='Number of epochs for the exploration process', type=int, default=1)
    parser.add_argument('--perc_clients_for_mask', help=' specifies the percentage of clients for which to create a mask.', type=float,choices=[Range(0., 1)], default=0.1)
    parser.add_argument('--guidence_normalization', help=' specifies the noromalization type create the Global Guidence mask.',  type=str, default='MinMax')
    parser.add_argument('--mask_pruining', help='Pruining functionality', type=str, default='True')
    parser.add_argument('--treeshold_pruining', help='Treeshold for pruining in percentile', type=float, default=0.5)
    parser.add_argument('--initial_value_threshold', help='Treeshold for pruining in percentile', type=float, default=0.5)

    ##### Ablation studies #####treeshold_pruining
    # parser.add_argument('--sf', help='scaling factor', type=float, choices=[Range(0., 1.)], default=1)




    ##### Argoument for Fed Optimzation algs (Reddi et al., 2020) (https://arxiv.org/abs/2003.00295) #####
    parser.add_argument('--beta1', help='server momentum factor', type=float, choices=[Range(0., 1.)], default=0.9)
    parser.add_argument('--tau', help='server momentum factor', type=float, choices=[Range(0., 1.)], default=0.001)
    parser.add_argument('--plot', help='Do want to plot?', default=False)
    parser.add_argument('--CEXP2', help='N', type=int, default=1)

    parser.add_argument('--random_pruning', help='Do want random pruning', default=False)


    

    # parse arguments
    args = parser.parse_args()

    # make path for saving losses & metrics & models
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.exp_name = f'{args.exp_name}_{args.seed}_{args.dataset.lower()}_{args.model_name.lower()}_{curr_time}'
    args.result_path = os.path.join(args.result_path, f'{args.exp_name}')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # make path for saving logs
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # initialize logger
    set_logger(f'{args.log_path}/{args.exp_name}.log', args)

    # check TensorBoard execution
    if args.use_tb: # run TensorBaord for tracking losses and metrics
        tb = TensorBoardRunner(os.path.join(args.log_path, f'{args.exp_name}'), args.tb_host, args.tb_port)
    else:
        tb = None

    # define writer
    writer = SummaryWriter(
            log_dir=os.path.join(args.log_path, f'{args.exp_name}'),
            filename_suffix=f'_{curr_time}'
    )

    # run main program
    torch.autograd.set_detect_anomaly(True)
    try:
        main(args, writer)
        if args.use_tb:
            tb.finalize()
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        if args.use_tb:
            tb.interrupt()
        sys.exit(1)

    # # run main program
    # main(args, writer)
    #
    # # bye!
    # if tb is not None:
    #     tb.finalize()
    # os._exit(0)
