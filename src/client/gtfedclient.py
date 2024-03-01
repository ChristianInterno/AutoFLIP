import copy
import torch
import torch.nn.utils.prune as prune
import inspect
import numpy as np

from .baseclient import BaseClient
from src import MetricManager
from src.GTL_utils import checkpoint_exists, mask_exists
import time
import matplotlib.pyplot as plt

import math



class GtfedClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(GtfedClient, self).__init__()
        self.args = args


        self.training_set = training_set
        self.test_set = test_set

        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)


    def print_model_info(self, model, title="Model information"):
        total_params = sum(p.numel() for p in model.parameters())
        total_zero_params = sum((p == 0).sum().item() for p in model.parameters())
        total_size = sum(p.element_size() * p.nelement() for p in model.parameters())
        sparsity = total_zero_params / total_params
        compression_rate = total_params / (total_params - total_zero_params) if total_zero_params else 1

        print(f"{title}:")
        print(f"Total parameters: {total_params}")
        print(f"Zero parameters: {total_zero_params}")
        print(f"Sparsity: {sparsity*100:.2f}%")
        print(f"Compression rate: {compression_rate:.2f}x")
        print(f"Total size (bytes): {total_size}")
        for name, param in model.named_parameters():
            print(f"{name}: {param.size()}, sparsity: {torch.sum(param == 0) / param.nelement()}")



    def create_scout(self, model):
        if not checkpoint_exists('tl_base'):
            tl_base = model.train()
            model.to(self.args.device)
            optimizer = self.optim(model.parameters(), **self._refine_optim_args(self.args))
            for e in range(1) :
                for inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                    outputs = model(inputs)
                    loss = self.criterion()(outputs, targets)

                    for param in model.parameters():
                        param.grad = None

                    loss.backward()
                    optimizer.step()  # !!! the application of those deltas is happening here

            torch.save(tl_base.state_dict(),
             f'checkpoints/tl_base{self.args.exp_name}.ckpt')

        if not mask_exists(f'mask{self.args.exp_name}'):
            mm = MetricManager(self.args.eval_metrics)
            mask = model.train()
            model.to(self.args.device)
            optimizer = self.optim(model.parameters(), **self._refine_optim_args(self.args))

            best_loss = float('inf')  # initialize best loss as infinity
            patience_counter = 0  # counter for early stopping
            patience = self.args.Patience_mask  # number of epochs to wait before stopping

            # for e in range(self.args.epoochs_mask):
            for e in range(1):
                epoch_correct = 0  # number of correct predictions in this epoch
                epoch_total = 0  # total number of predictions in this epoch
                for inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                    outputs = model(inputs)
                    loss = self.criterion()(outputs, targets)

                    for param in model.parameters():
                        param.grad = None

                    loss.backward()
                    optimizer.step()

                    # calculate accuracy
                    predicted = outputs.argmax(dim=1)
                    correct = (predicted == targets).sum().item()  # number of correct predictions
                    total = targets.shape[0]  # total number of predictions
                    epoch_correct += correct
                    epoch_total += total


                    mm.track(loss.item(), outputs, targets)
                else:
                    mm.aggregate(len(self.training_set), e + 1)

                epoch_accuracy = epoch_correct / epoch_total  # calculate accuracy for this epoch

                # early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0  # reset counter when loss improves
                else:
                    patience_counter += 1  # increment counter when loss does not improve

                if patience_counter >= patience:  # if counter reaches the threshold
                    break  # stop the training

            return mask


    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument):
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0 :
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)


    def update(self):

        def get_nested_module(model, module_path):
            print("Module path:", module_path)
            modules = module_path.split('.')
            for module in modules:
                print("Current module:", module)
                if hasattr(model, module):
                    model = getattr(model, module)
                    print("Submodule found:", model)
                else:
                    print("Submodule not found:", module)
                    return None
            return model

        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))

        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                for param in self.model.parameters():
                    param.grad = None

                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                optimizer.step()
                #Client Pruning
                if mask_exists(f'mask{self.args.exp_name}'):

                    if self.args.mask_pruining == 'True':

                        updated_mask = torch.load(f'checkpoints/InitPruinedGlobalModel/mask{self.args.exp_name}.pt')

                        for path, mask_value in updated_mask.items():
                            # Split the path to access specific submodule and parameter
                            submodule_path, param_name = path.rsplit('.', 1)
                            submodule = get_nested_module(self.model, submodule_path)

                            if submodule:
                                mask_tensor = torch.tensor(mask_value, device=submodule.weight.device)
                                # # Ensure the mask is correctly shaped for the parameter it's applied to
                                # target_param = getattr(submodule, param_name)
                                # assert mask_tensor.shape == target_param.shape, f"Shape mismatch for {path}: {mask_tensor.shape} vs {target_param.shape}"

                                try:
                                    # Apply pruning to the specific parameter of the submodule
                                    prune.custom_from_mask(submodule, name=param_name, mask=mask_tensor)
                                except AttributeError as e:
                                    print(f"Error applying pruning to {path}: {e}")
                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)

        def finalize_pruning(model):
            for name, module in model.named_modules():
                # Check if the module has the 'weight_orig' attribute, indicating pruning was applied to the weight
                if hasattr(module, 'weight_orig'):
                    # Remove pruning reparameterization for 'weight'
                    prune.remove(module, 'weight')

                # Similarly, check if pruning was applied to the bias
                if hasattr(module, 'bias_orig'):
                    # Remove pruning reparameterization for 'bias'
                    prune.remove(module, 'bias')

            return model

        self.model = finalize_pruning(self.model)
        self.print_model_info(self.model, f"Client{self.id} Model after pruning")
        torch.save(self.model.state_dict(), f'/home/cinterno/storage/FL2/Federated-Learning-in-PyTorch/checkpoints/ClientPruinedGModel/Client{self.id}for{self.args.exp_name}.pt')
        return mm.results

    @torch.inference_mode()
    def evaluate(self):

        if self.args._train_only:  # `args.test_fraction` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.test_set))
        return mm.results

    def download(self, model):

        self.model = copy.deepcopy(model)

    def upload(self):
        self.model.to('cuda')
        return self.model.named_parameters()

    def __len__(self):
        return len(self.training_set), len(self.test_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'

