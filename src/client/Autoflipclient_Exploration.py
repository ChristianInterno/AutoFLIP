import copy
import torch
import inspect
import logging

from .baseclient import BaseClient
from src import MetricManager
from src.GTL_utils import checkpoint_exists, mask_exists

logger = logging.getLogger(__name__)


class AutoflipClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(AutoflipClient, self).__init__()
        self.args = args


        self.training_set = training_set
        self.test_set = test_set

        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

    def create_scout(self, model):
        if not checkpoint_exists('tl_base_pruining'):
            logger.info('The explorers are getting ready to explore the different loss landscape!')
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
                    optimizer.step()

            torch.save(tl_base.state_dict(),
             f'checkpoints/tl_base_pruining{self.args.exp_name}.ckpt')

        if not mask_exists(f'mask{self.args.exp_name}'):
            logger.info(f'The client {self.id} is exploring...')
            mm = MetricManager(self.args.eval_metrics)
            mask = model.train()
            model.to(self.args.device)
            optimizer = self.optim(model.parameters(), **self._refine_optim_args(self.args))

            best_loss = float('inf')  # initialize best loss as infinity
            patience_counter = 0  # counter for early stopping
            patience = self.args.Patience_mask  # number of epochs to wait before stopping

            for e in range(self.args.epoochs_mask):
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
                logger.info(f'Exploration epoch: {e + 1}, Loss: {loss.item()}, Accuracy: {epoch_accuracy}')  # print the epoch number, loss, and accuracy

                # early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0  # reset counter when loss improves
                else:
                    patience_counter += 1  # increment counter when loss does not improve

                if patience_counter >= patience:  # if counter reaches the threshold
                    logger.info(f'Early stopping at epoch {e + 1}, best exploration loss was {best_loss}')
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

    def process_mask_pruining(self, mask):
        threshold = self.args.treeshold_pruining  # Retrieve the pruning threshold from the arguments
        for name, mask_value in mask.items():
            # Update the mask: Set to 0 if value is below the threshold, otherwise leave unchanged
            mask[name] = mask_value * (mask_value >= threshold).float()
        return mask

    def validate_model_shapes(client_model, global_model):
        for client_param, global_param in zip(client_model.parameters(), global_model.parameters()):
            assert client_param.shape == global_param.shape, f"Mismatch found: {client_param.shape} vs {global_param.shape}"

    def update(self):

        # Function to store initial model parameters
        def store_initial_model_params(model):
            initial_params = {}
            for name, param in model.named_parameters():
                if 'weight' in name:  # Assuming you're only interested in weights for pruning
                    initial_params[name] = param.data.clone()  # Store a copy of the initial parameter values
            return initial_params

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

                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
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

