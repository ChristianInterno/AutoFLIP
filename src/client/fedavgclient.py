import copy
import torch
import inspect
import logging

from .baseclient import BaseClient
from src import MetricManager
import torch.nn.utils.prune as prune
import random


logger = logging.getLogger(__name__)


class FedavgClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(FedavgClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set
        
        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

    # def random_prune(self, model, pruning_ratio=0.45):
    #     # Gather all parameters to prune
    #     parameters_to_prune = []
    #     for name, param in model.named_parameters():
    #         if 'weight' in name or 'bias' in name:  # Consider both weights and biases
    #             parameters_to_prune.append((param, 'weight' if 'weight' in name else 'bias'))
    #
    #     # Calculate total number of parameters and the number to prune
    #     total_params = sum(p.numel() for p, _ in parameters_to_prune)
    #     params_to_prune = int(total_params * pruning_ratio)
    #
    #     # Prune randomly until the desired ratio is achieved
    #     while params_to_prune > 0:
    #         # Randomly select a parameter and a pruning proportion
    #         param, name = random.choice(parameters_to_prune)
    #         total_param_num = param.numel()
    #         # Calculate how many parameters to prune from this layer
    #         # Ensure not to prune more than the remaining unpruned parameters
    #         prune_num = min(random.randint(1, total_param_num), params_to_prune)
    #
    #         # Create a random mask with the specified number of parameters set to zero
    #         mask = torch.ones(total_param_num, dtype=torch.bool,
    #                           device=self.args.device)  # Ensure mask is on the same device
    #         zero_indices = random.sample(range(total_param_num), prune_num)
    #         mask.view(-1)[zero_indices] = False  # Set selected positions to zero
    #         mask = mask.view(param.shape)  # Reshape the mask back to the original shape
    #
    #         # Apply the mask to the parameter
    #         masked_param = torch.where(mask, param.data, torch.zeros_like(param))
    #         param.data.copy_(masked_param)
    #
    #         # Update the number of parameters still to prune
    #         params_to_prune -= prune_num

    def random_prune(self, model, pruning_ratio=0.45):
        # Gather all parameters and their sizes
        param_list = [(name, p) for name, p in model.named_parameters() if 'weight' in name or 'bias' in name]
        total_params = sum(p.numel() for _, p in param_list)
        params_to_prune = int(total_params * pruning_ratio)

        # Calculate the number of parameters to prune for each parameter based on its proportion of the total count
        for name, param in param_list:
            # Calculate proportion of total params
            param_count = param.numel()
            proportion = param_count / total_params

            # Calculate how many parameters to prune from this layer, based on its proportion
            prune_count = int(params_to_prune * proportion)

            if prune_count > 0:  # Only proceed if there are parameters to prune
                # Randomly select indices to zero out
                indices = torch.randperm(param_count)[:prune_count].to(self.args.device)
                # Create a flat view of the parameter for easy indexing
                flat_param = param.data.view(-1)
                # Zero out the selected indices
                flat_param[indices] = 0
                # Update the number of parameters still to prune
                params_to_prune -= prune_count

    def print_model_info(self, model, title="Model information"):
        total_params = sum(p.numel() for p in model.parameters())
        total_zero_params = sum((p == 0).sum().item() for p in model.parameters())
        total_size = sum(p.element_size() * p.nelement() for p in model.parameters())
        sparsity = total_zero_params / total_params
        compression_rate = total_params / (total_params - total_zero_params) if total_zero_params else 1

        logger.info(f"{title}:")
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Zero parameters: {total_zero_params}")
        logger.info(f"Sparsity: {sparsity*100:.2f}%")
        logger.info(f"Compression rate: {compression_rate:.2f}x")
        logger.info(f"Total size (bytes): {total_size}")
        for name, param in model.named_parameters():
            logger.info(f"{name}: {param.size()}, sparsity: {torch.sum(param == 0) / param.nelement()}")

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

                if self.args.random_pruning:

                    self.random_prune(self.model)

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
        #torch.save(self.model.state_dict(), f'/home/cinterno/storage/FL2/Federated-Learning-in-PyTorch/ClientsUpdtPDist/ClientFedAvg{self.id}.pt')
        return mm.results

    @torch.inference_mode()
    def evaluate(self):
        if self.args._train_only: # `args.test_fraction` == 0
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
