import os
import gc
import json
import torch
import random
import logging
import numpy as np
import concurrent.futures
import torchvision
import torchvision.transforms as transforms

from importlib import import_module
from collections import ChainMap, defaultdict
import pandas as pd

from src import init_weights, TqdmToLogger, MetricManager
from .GTbaseserver import GTbaseserver
import matplotlib.pyplot as plt
from src.plot_utils import plot_metric_server, plot_metric_client, plot_Trajectory_l2, top_eigenvalues
from src.GTL_utils import create_mask, mask_exists, create_mask_pruining  # Import the function to apply the mask
from src.utils import compute_loss, compute_loss_landscape
import torch.nn as nn

logger = logging.getLogger(__name__)

checkpoint_dir_name = 'checkpoints'


class GtfedServer(GTbaseserver):
    def __init__(self, args, writer, server_dataset, client_datasets, model):

        self.args = args
        self.writer = writer

        # round indicator
        self.round = 0

        # global holdout set
        if self.args.eval_type != 'local':
            self.server_dataset = server_dataset

        # model
        checkpoint_path = f'checkpoints/InitPruinedGlobalModel/IPGB_{self.args.exp_name}.pt'
        
        self.model = self._init_model_with_pruning(model, checkpoint_path)
        #self.model = self._init_model(model)
        #torch.save(self.model.state_dict(),f'checkpointss/InitPruinedGlobalModel/NonPruining_{self.args.exp_name}.pt')


        # server aggregator
        self.server_optimizer = self._get_algorithm(self.model, lr=self.args.lr_server, momentum=self.args.beta)

        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.server_optimizer, gamma=self.args.lr_decay,
                                                            step_size=self.args.lr_decay_step)

        # clients
        self.clients = self._create_clients(client_datasets)

        # result container
        self.results = defaultdict(dict)

        self.trajectories = []


    def _init_model_with_pruning(self, model, checkpoint_path):
        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a model with pruning!')

        # Load the pruned model's state dict
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda')) #check if cpu or gpu
        if 'state_dict' in checkpoint:
            pruned_state_dict = checkpoint['state_dict']
        else:
            pruned_state_dict = checkpoint  # Adjust based on the actual structure of your checkpoint

        # Update model to use pruned weights
        model.load_state_dict(pruned_state_dict)

        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully initialized the model with pruned weights!')
        logger.info(f'Saving pruned version of model...')
        torch.save(model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}_Pruned_Init.pt'))

        return model

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; (Initialization type: {self.args.init_type.upper()}))!')
        logger.info(f'Saving first version of model...')
        torch.save(model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}_Init.pt'))
        return model

    def _get_algorithm(self, model, **kwargs):
        ALGORITHM_CLASS = import_module(f'..algorithm.{self.args.algorithm}', package=__package__).__dict__[
            f'{self.args.algorithm.title()}Optimizer']
        return ALGORITHM_CLASS(params=model.parameters(), **kwargs)

    def _create_clients(self, client_datasets):
        CLINET_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__[
            f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLINET_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[-1])
            client.id = identifier
            mask = client.create_scout(model=self.model)
            return client, mask

        # logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Create clients!')

        percentage = self.args.perc_clients_for_mask
        num_clients_to_create_mask = round(percentage * len(client_datasets))
        clients_to_create_mask = random.sample(range(len(client_datasets)), num_clients_to_create_mask)
        clients = []
        self.global_scout = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                    enumerate(client_datasets),
                    logger=logger,
                    desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}]',
                    total=len(client_datasets)
            ):
                ##########################################
                # Questo e per esperimento LEAF, per  non creare scout per ogni client ne selezionamo solo un percentuale
                if identifier in clients_to_create_mask:
                    client, mask = workhorse.submit(__create_client, identifier, datasets).result()
                    clients.append(client)
                    self.global_scout.append(mask.state_dict())
                else:
                    client = CLINET_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[-1])
                    client.id = identifier
                    clients.append(client)


        # logger.info(
        #     f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully created {self.args.K} clients!')

        return clients

    def _broadcast_models(self, ids):
        def __broadcast_model(client):
            client.download(self.model)

        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Broadcast the pruned global model at the server!')

        self.model.to('cuda')

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
            for identifier in TqdmToLogger(
                    ids,
                    logger=logger,
                    desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...broadcasting server model... ',
                    total=len(ids)
            ):
                workhorse.submit(__broadcast_model, self.clients[identifier]).result()
        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully broadcasted the model to selected {len(ids)} clients!')

    def _sample_clients(self, exclude=[]):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Sample clients!')

        if exclude == []:  # Update - randomly select max(floor(C * K), 1) clients
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample([i for i in range(self.args.K)], num_sampled_clients))

        else:  # Evaluation - randomly select unparticipated clients in amount of `eval_fraction` multiplied
            num_unparticipated_clients = self.args.K - len(exclude)
            if num_unparticipated_clients == 0:  # when C = 1, i.e., need to evaluate on all clients
                num_sampled_clients = self.args.K
                sampled_client_ids = sorted([i for i in range(self.args.K)])
            else:
                num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                sampled_client_ids = sorted(random.sample(
                    [identifier for identifier in [i for i in range(self.args.K)] if identifier not in exclude],
                    num_sampled_clients))
        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_ids

    def _log_results(self, resulting_sizes, results, eval, participated):
        losses, metrics, num_samples = list(), defaultdict(list), list()
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '

            # get loss and metrics
            if eval:
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)

                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)
            else:  # same, but retireve results of last epoch's
                # loss
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)

                # metrics
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)
                    # get sample size
            num_samples.append(resulting_sizes[identifier])
            # log per client
            logger.info(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)

        # aggregate intototal logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses = np.array(losses).astype(float)
        weighted = losses.dot(num_samples) / sum(num_samples)
        equal = losses.mean()
        std = losses.std()
        total_log_string += f'\n    - Loss: Weighted Avg. ({weighted:.4f}) | Equal Avg. ({equal:.4f}) | Std. ({std:.4f}) |'
        result_dict['loss'] = {'weighted': weighted, 'equal': equal, 'std': std}
        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss ' + eval * f'({"In" if participated else "Out"})',
            {f'Weighted Average': weighted, f'Equal Average': equal},
            self.round
        )

        # metrics
        for name, val in metrics.items():
            val = np.array(val).astype(float)
            weighted = val.dot(num_samples) / sum(num_samples)
            equal = val.mean()
            std = val.std()
            total_log_string += f'\n    - {name.title()}: Weighted Avg. ({weighted:.4f}) | Equal Avg. ({equal:.4f}) | Std. ({std:.4f}) |'
            result_dict[name] = {'weighted': weighted, 'equal': equal, 'std': std}
            for name in metrics.keys():
                self.writer.add_scalars(
                    f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f'({"In" if participated else "Out"})',
                    {f'Weighted Average': weighted, f'Equal Average': equal},
                    self.round
                )
            self.writer.flush()

        # log total message
        logger.info(total_log_string)
        return result_dict

    def _request(self, ids, eval=False, participated=False):

        def __update_clients(client):
            client.args.lr = self.lr_scheduler.get_last_lr()[-1]
            update_result = client.update()
            return {client.id: len(client.training_set)}, {client.id: update_result}

        def __evaluate_clients(client):
            eval_result = client.evaluate()
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Request {"updates" if not eval else "evaluation"} to {"all" if ids is None else len(ids)} clients!')
        if eval:
            if self.args._train_only: return
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
                for idx in TqdmToLogger(
                        ids,
                        logger=logger,
                        desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...evaluate clients... ',
                        total=len(ids)
                ):
                    results.append(workhorse.submit(__evaluate_clients, self.clients[idx]).result())
            eval_sizes, eval_results = list(map(list, zip(*results)))
            eval_sizes, eval_results = dict(ChainMap(*eval_sizes)), dict(ChainMap(*eval_results))
            self.results[self.round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
                eval_sizes,
                eval_results,
                eval=True,
                participated=participated
            )
            logger.info(
                f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')
        else:
            results = []
            # global_scout = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
                for idx in TqdmToLogger(
                        ids,
                        logger=logger,
                        desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...update clients... ',
                        total=len(ids)
                ):
                    results.append(workhorse.submit(__update_clients, self.clients[idx]).result())

                if self.args.perc_clients_for_mask == 1:
                    global_scout_filtered = [self.global_scout[i] for i in
                                             ids]  # we are selecting the scout for create the Mask for this round
                    create_mask_pruining(model_checkpoint=f'tl_base_pruining{self.args.exp_name}', scouts=global_scout_filtered,
                                name_file=self.args.exp_name, args=self.args)#use global_scout_filtered for update mask
                else:
                    create_mask(model_checkpoint=f'tl_base{self.args.exp_name}', scouts=self.global_scout,
                                name_file=self.args.exp_name,
                                args=self.args)  # usiamo sempre la mask creata all inizio dalla percentuale desiderate di C

            update_sizes, update_results = list(map(list, zip(*results)))
            update_sizes, update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*update_results))
            self.results[self.round]['clients_updated'] = self._log_results(
                update_sizes,
                update_results,
                eval=False,
                participated=True
            )
            logger.info(
                f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...completed updates of {"all" if ids is None else len(ids)} clients!')
            return update_sizes

    def _aggregate(self, ids, updated_sizes):
        print("Global model parameter shapes before aggregation:")
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.shape}")
        
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')

        # calculate mixing coefficients according to sample sizes
        coefficients = {identifier: float(coefficient / sum(updated_sizes.values())) for identifier, coefficient in
                        updated_sizes.items()}

        # accumulate weights
        for identifier in ids:
            locally_updated_weights_iterator = self.clients[identifier].upload()
            self.server_optimizer.accumulate(coefficients[identifier], locally_updated_weights_iterator)
        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')

    def _cleanup(self, indices):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Clean up!')

        for identifier in indices:
            if self.clients[identifier].model is not None:
                self.clients[identifier].model = None
            else:
                err = f'why clients ({identifier}) has no model? please check!'
                logger.exception(err)
                raise AssertionError(err)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully cleaned up!')
        gc.collect()

    @torch.inference_mode()
    def _central_evaluate(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in torch.utils.data.DataLoader(dataset=self.server_dataset, batch_size=self.args.B,
                                                           shuffle=False):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.server_dataset))

        # log result
        result = mm.results
        server_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        ## loss
        loss = result['loss']
        server_log_string += f'| loss: {loss:.4f} '

        ## metrics
        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        # log TensorBoard
        self.writer.add_scalar('Server Loss', loss, self.round)
        for name, value in result['metrics'].items():
            self.writer.add_scalar(f'Server {name.title()}', value, self.round)
        else:
            self.writer.flush()
        self.results[self.round]['server_evaluated'] = result

    def update(self):
        """Update the global model through federated learning.
        """
        # randomly select clients
        selected_ids = self._sample_clients()

        # broadcast the current model at the server to selected clients
        self._broadcast_models(selected_ids)

        # request update to selected clients
        updated_sizes = self._request(selected_ids, eval=False)

        # request evaluation to selected clients
        self._request(selected_ids, eval=True, participated=True)

        # receive updates and aggregate into a new weights
        self.server_optimizer.zero_grad()  # empty out buffer
        self._aggregate(selected_ids, updated_sizes)  # aggregate local updates
        self.server_optimizer.step()  # update global model with the aggregated update
        self.lr_scheduler.step()  # update learning rate

        self.trajectories.append({name: param.clone() for name, param in self.model.named_parameters()})

        # remove model copy in clients
        self._cleanup(selected_ids)
        return selected_ids

    def evaluate(self, excluded_ids):
        """Evaluate the global model located at the server.
        """
        # randomly select all remaining clients not participated in current round
        selected_ids = self._sample_clients(exclude=excluded_ids)
        self._broadcast_models(selected_ids)

        # request evaluation
        ## `local`: evaluate on selected clients' holdout set
        ## `global`: evaluate on the server's global holdout set
        ## `both`: conduct both `local` and `global` evaluations
        if self.args.eval_type == 'local':
            self._request(selected_ids, eval=True, participated=False)
        elif self.args.eval_type == 'global':
            self._central_evaluate()
        elif self.args.eval_type == 'both':
            self._request(selected_ids, eval=True, participated=False)
            self._central_evaluate()

        # remove model copy in clients
        self._cleanup(selected_ids)

        # calculate generalization gap
        if (not self.args._train_only) and (not self.args.eval_type == 'global'):
            gen_gap = dict()
            curr_res = self.results[self.round]
            for key in curr_res['clients_evaluated_out'].keys():
                for name in curr_res['clients_evaluated_out'][key].keys():
                    if name in ['equal', 'weighted']:
                        gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
                        gen_gap[f'gen_gap_{key}'] = {name: gap}
                        self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'],
                                                self.round)
                        self.writer.flush()
            else:
                self.results[self.round]['generalization_gap'] = dict(gen_gap)

    def finalize(self):
        """Save results.
        """
        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Save results and the global model checkpoint!')
        # save figure
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w',
                  encoding='utf8') as result_file:
            results = {key: value for key, value in self.results.items()}
            json.dump(results, result_file, indent=4)

        # save checkpoint
        logger.info(f'Saving last version of model...')
        torch.save(self.model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}_Fin.pt'))

        # Plot
        if self.args.plot:
            metrics = self.args.eval_metrics

            json_data = results
            # Call the plot_metric function
            for metric in metrics:
                save_path = os.path.join(self.args.result_path, f'server_evaluated_{metric}.pdf')
                plot_metric_server(json_data, metric, save_path)

            for metric in metrics:
                save_path = os.path.join(self.args.result_path, f'clients_evaluated_{metric}.pdf')
                plot_metric_client(json_data, metric, save_path)

        # save_path_t2 = os.path.join(self.args.result_path, 'trajectory_parameters_L2_during_training')
        # plot_Trajectory_l2(self.trajectories, save_path_t=save_path_t2)

        # logger.info(f'Saving loss landscape...')
        # self.compute_and_save_loss_landscape()

        self.writer.close()
        logger.info(
            f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...finished federated learning!')
