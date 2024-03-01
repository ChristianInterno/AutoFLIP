import torch
from torch.nn import Parameter, Module
import numpy as np

import logging
import warnings

from pytorch_lightning import Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
import os

from src.models.twocnn_GTL import twocnn_GTL
from src.models.twocnn import TwoCNN
from src.models.twonn import TwoNN
from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model
import argparse

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from pytorch_lightning.strategies import DDPStrategy
import random



# num_workers = 4
log_every_n_steps = 1
num_epochs = 10
batch_size = 64

checkpoint_dir_name = 'checkpoints'



####### Create PyTorch datasets and dataset loaders for a subset of CIFAR10 classe #####################################

# Transformations
# RC = transforms.RandomCrop(32, padding=4)
RHF = transforms.RandomHorizontalFlip(0.5)
# RVF = transforms.RandomVerticalFlip()
NRM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([TPIL, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([TT, NRM])


def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i

class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class
########################################################################################################################


def checkpoint_exists(name):
    return os.path.exists(f'{checkpoint_dir_name}/{name}.ckpt')

def mask_exists(name):
    return os.path.exists(f'{checkpoint_dir_name}/{name}.pt')

def save_checkpoint(model, name):
    model.save_checkpoint(f'{checkpoint_dir_name}/{name}.ckpt')


# def GT_train( name='twocnn_GTL', mask=None, checkpoint=None,
#                      convergence=None, class_data=None):
#
#     tensorboard = pl_loggers.TensorBoardLogger(save_dir='logs', name=name)
#
#     if checkpoint:
#         model = twocnn_GTL.load_from_checkpoint(f"{checkpoint_dir_name}/{checkpoint}.ckpt", mask=mask)
#     else:
#         model = twocnn_GTL(in_channels=2 ,hidden_size= 32, num_classes = 10, mask=None )
#
#     dataset = CIFAR10(root='/home/cinterno/storage/FL2/Federated-Learning-in-PyTorch/data', download=True, transform=ToTensor(), train=True)
#     # testdataset =  CIFAR10(root='/home/cinterno/storage/FL2/Federated-Learning-in-PyTorch/data', download=True, transform=ToTensor(), train=False)
#
#     if class_data:
#         classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
#                      'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
#
#         # classDict = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
#         #              5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
#
#         values = list(classDict.keys())
#         values.remove(class_data)
#         rand = random.choice(values)
#
#         x_train = dataset.data
#         # x_test = testdataset.data
#         y_train = dataset.targets
#         # y_test = testdataset.targets
#
#         class_train = DatasetMaker(
#             [get_class_i(x_train, y_train, classDict[class_data]),
#              get_class_i(x_train, y_train, classDict[rand])],
#             transform_with_aug
#         )
#         print(f"We get the final parameters value with our scout {class_data} and {rand}")
#
#         train_loader = DataLoader(class_train, batch_size=batch_size,shuffle=True)
#     else:
#         train_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
#
#
#     early_stop_callback = EarlyStopping(monitor="Acc/Epoch", min_delta=0.00, patience=0, verbose=True,
#                                         mode="max", stopping_threshold=0.000000000000000000000000000000000000001)
#     early_stop = EarlyStopping(monitor="Acc/Epoch", patience=1000, verbose=True)
#
#     callbacks = [early_stop_callback] if convergence else [early_stop]
#     trainer = Trainer(enable_model_summary=False, max_epochs=num_epochs, logger=tensorboard,
#                       callbacks=callbacks, log_every_n_steps=log_every_n_steps, accelerator='gpu')
#     # testset = CIFAR10(root='/home/cinterno/storage/FL2/Federated-Learning-in-PyTorch/data', train=False,
#     #                   download=True, transform=ToTensor())
#     # testloader = DataLoader(testset, batch_size=10,
#     #                         shuffle=False)
#
#     # classes = ('plane', 'car', 'bird', 'cat',
#     #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#     trainer.fit(model, train_loader)
#     return trainer
#
# def GT_train2( name='twocnn_GTL', mask=None, checkpoint=None,
#                      convergence=None, dataset=None):
#     tensorboard = pl_loggers.TensorBoardLogger(save_dir='logs', name=name)
#
#     model = twocnn_GTL(in_channels=2 ,hidden_size= 32, num_classes = 10, mask=None )
#
#     early_stop_callback = EarlyStopping(monitor="Acc/Epoch", min_delta=0.00, patience=1, verbose=True,
#                                         mode="max", stopping_threshold=0.000000000000000000000000000000000000001)
#     early_stop= EarlyStopping(monitor="Acc/Epoch", min_delta=0.00, patience=1000, verbose=True,
#                                         mode="max")
#     callbacks = [early_stop_callback] if convergence else [early_stop]
#     trainer = Trainer(enable_model_summary=False, max_epochs=num_epochs, logger=tensorboard,
#                       callbacks=callbacks, log_every_n_steps=log_every_n_steps, accelerator='gpu')
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     # testset = CIFAR10(root='/home/cinterno/storage/FL2/Federated-Learning-in-PyTorch/data', train=False,
#     #                   download=True, transform=ToTensor())
#     # testloader = DataLoader(testset, batch_size=10,
#     #                         shuffle=False)
#
#     # classes = ('plane', 'car', 'bird', 'cat',
#     #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#     trainer.fit(model, train_loader)
#     return trainer

###########MASK#########
def get_mask():
    return torch.load(f'{checkpoint_dir_name}/mask.pt')

def create_mask_pruining(model_checkpoint,scouts,name_file, args):
    def _compute_mask(model, scout_dict_list):
        def learning_spread(model, scout):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            scout = scout.to(device)
            ls = torch.mean((scout - model) ** 2, 0)

            if args.guidence_normalization == 'MinMax':
                min_ls = torch.min(ls)
                max_ls = torch.max(ls)
                spread = max_ls - min_ls
                return (ls - min_ls) / spread

            if args.guidence_normalization == 'RobustScaler':
                median_ls = torch.median(ls)
                q1 = torch.quantile(ls, 0.25)
                q3 = torch.quantile(ls, 0.75)
                iqr = q3 - q1

                return (ls - median_ls) / iqr

        out_mask = {}
        model_dict = model.state_dict()

        scouts = {}
        for scout_dict in scout_dict_list:
            for name, param in model.named_parameters():
                scouts.setdefault(name, [])
                scouts[name].append(scout_dict[name])  # TODO: this should be done differently...

        for name, param in model.named_parameters():
            out_mask[name] = learning_spread(model_dict[name], torch.stack(scouts[name]))

        return out_mask

    model, args = load_model(args)
    # model = TwoCNN(in_channels = 1, hidden_size = 32, num_classes = 10)#MINST
    # model = TwoCNN(in_channels = 3, hidden_size = 32, num_classes = 10)#CIFAR10
    model.load_state_dict(torch.load(f'{checkpoint_dir_name}/{model_checkpoint}.ckpt'))
    # model = TwoCNN.load_state_dict(model_ckp['model_state_dict'])
    mask = _compute_mask(model, scouts)
    torch.save(mask, f'{checkpoint_dir_name}/InitPruinedGlobalModel/mask{name_file}.pt')
    return mask

def create_mask(model_checkpoint,scouts,name_file, args):
    def _compute_mask(model, scout_dict_list):
        def learning_spread(model, scout):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            scout = scout.to(device)
            ls = torch.mean((scout - model) ** 2, 0)

            if args.guidence_normalization == 'MinMax':
                min_ls = torch.min(ls)
                max_ls = torch.max(ls)
                spread = max_ls - min_ls
                return (ls - min_ls) / spread

            if args.guidence_normalization == 'RobustScaler':
                median_ls = torch.median(ls)
                q1 = torch.quantile(ls, 0.25)
                q3 = torch.quantile(ls, 0.75)
                iqr = q3 - q1

                return (ls - median_ls) / iqr

        out_mask = {}
        model_dict = model.state_dict()

        scouts = {}
        for scout_dict in scout_dict_list:
            for name, param in model.named_parameters():
                scouts.setdefault(name, [])
                scouts[name].append(scout_dict[name])  # TODO: this should be done differently...

        for name, param in model.named_parameters():
            out_mask[name] = learning_spread(model_dict[name], torch.stack(scouts[name]))

        return out_mask

    model, args = load_model(args)
    # model = TwoCNN(in_channels = 1, hidden_size = 32, num_classes = 10)#MINST
    # model = TwoCNN(in_channels = 3, hidden_size = 32, num_classes = 10)#CIFAR10
    model.load_state_dict(torch.load(f'{checkpoint_dir_name}/{model_checkpoint}.ckpt'))
    # model = TwoCNN.load_state_dict(model_ckp['model_state_dict'])
    mask = _compute_mask(model, scouts)
    torch.save(mask, f'{checkpoint_dir_name}/mask{name_file}.pt')
    return mask

def predict(model, dataloader, device):
    prediction = []
    with torch.no_grad():
        model.eval()
        for data, _ in dataloader:
            data = data.to(device)
            output = model(data)
            prediction.append(output.gpu().numpy())
    prediction = np.concatenate(prediction, axis=0)
    return prediction




###New function from a fork on github GTL0
def _get_guidance_matrix(scout_param, model_param: Parameter) -> torch.Tensor:
    learning_spread = torch.mean((scout_param - model_param) ** 2, 0)
    min_ls = torch.min(learning_spread)
    max_ls = torch.max(learning_spread)
    return torch.tensor((learning_spread - min_ls) / (max_ls - min_ls))


def apply_guidance_matrices(model: Module, scouts) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.grad *= _get_guidance_matrix(scouts[name], param)


def debias_layers(model: Module) -> None: #TODO: get a look on what is this
    def debias(param):
        return torch.stack([torch.mean(param, dim=0)] * len(param), dim=0)

    last_layer = list(model.children())[-1]
    last_layer.bias = debias(last_layer.bias)
    last_layer.weight = debias(last_layer.weight)