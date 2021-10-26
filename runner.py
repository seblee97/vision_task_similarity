import copy
from typing import Dict

import constants
import dataset
import network
import numpy as np
import torch
import torch.nn as nn
from run_modes import base_runner


class Runner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str = ""):
        self._first_task_epochs = config.switch_epoch
        self._second_task_epochs = config.total_epochs - config.switch_epoch

        self._input_dimension = config.input_dimension
        self._hidden_dimension = config.hidden_dimension

        self._network, self._optimiser = self._setup_network(config=config)
        self._train_dataloaders, self._test_dataloaders = self._setup_data(
            config=config
        )

        self._loss_function = nn.MSELoss()

        super().__init__(config=config, unique_id=unique_id)

    def _get_data_columns(self):
        columns = [
            constants.EPOCH_LOSS,
            f"{constants.TEST}_{constants.LOSS}_0",
            f"{constants.TEST}_{constants.LOSS}_1",
            f"{constants.TEST}_{constants.ACCURACY}_0",
            f"{constants.TEST}_{constants.ACCURACY}_1",
            constants.NODE_NORM_ENTROPY,
        ]

        columns.extend(
            [f"{constants.SELF_OVERLAP}_{i}" for i in range(self._hidden_dimension)]
        )
        columns.extend(
            [f"{constants.NODE_FISCHER}_{0}_{i}" for i in range(self._hidden_dimension)]
        )
        columns.extend(
            [f"{constants.NODE_FISCHER}_{1}_{i}" for i in range(self._hidden_dimension)]
        )

        return columns

    def _setup_data(self, config):

        mixed_train_1, mixed_test_1 = dataset.FashionMNISTSplitter.get_mixed_dataloader(
            config.indices[0],
            config.indices[1],
            mixing=config.mixing[0],
            batch_size=config.batch_size,
            shuffle=True,
        )
        mixed_train_2, mixed_test_2 = dataset.FashionMNISTSplitter.get_mixed_dataloader(
            config.indices[0],
            config.indices[1],
            mixing=config.mixing[1],
            batch_size=config.batch_size,
            shuffle=True,
        )

        train_dataloaders = [mixed_train_1, mixed_train_2]
        test_dataloaders = [mixed_test_1, mixed_test_2]

        return train_dataloaders, test_dataloaders

    def _setup_network(self, config):
        net = network.TwoLayerRegressionNetwork(
            input_dim=config.input_dimension,
            hidden_dim=config.hidden_dimension,
            output_dim=config.output_dimension,
            nonlinearity=config.nonlinearity,
            num_heads=len(config.indices),
        )
        optimiser = torch.optim.SGD(params=net.parameters(), lr=config.learning_rate)
        return net, optimiser

    def _obtain_target_mappings(self, config):
        target_mappings = {}
        label_mappings = {}
        for task_index, dataset_indices in enumerate(config.indices):
            target_mapping, label_mapping = dataset.FashionMNISTSplitter.get_mapping(
                index_1=dataset_indices[0], index_2=dataset_indices[1]
            )
            target_mappings = {**target_mappings, **target_mapping}
            label_mappings = {**label_mappings, **label_mapping}
        return target_mappings, label_mappings

    def train(self):

        self._pre_train_logging()

        for e in range(self._first_task_epochs):
            self._train_test_loop(epoch=e, task_index=0)

        for e in range(self._second_task_epochs):
            self._train_test_loop(epoch=e, task_index=1)

    def _pre_train_logging(self):
        node_norms = self._compute_node_norms()
        node_norm_entropy = self._compute_norms_entropy(node_norms=node_norms)

        node_fischers_0 = self._compute_node_fischers(task_index=0)
        node_fischers_1 = self._compute_node_fischers(task_index=1)

        base_logging_dict = {constants.NODE_NORM_ENTROPY: node_norm_entropy}

        overlap_logging_dict = {
            f"{constants.SELF_OVERLAP}_{i}": norm for i, norm in enumerate(node_norms)
        }

        fischer_0_logging_dict = {
            f"{constants.NODE_FISCHER}_{0}_{i}": fischer
            for i, fischer in enumerate(node_fischers_0)
        }

        fischer_1_logging_dict = {
            f"{constants.NODE_FISCHER}_{1}_{i}": fischer
            for i, fischer in enumerate(node_fischers_1)
        }

        logging_dict = {
            **base_logging_dict,
            **overlap_logging_dict,
            **fischer_0_logging_dict,
            **fischer_1_logging_dict,
        }

        self._epoch_log(logging_dict=logging_dict, epoch=0)

    def _train_test_loop(self, epoch: int, task_index: int):

        train_epoch_loss = self._train_loop(task_index=task_index)

        test_loss_0, test_accuracy_0 = self._test_loop(task_index=0)
        test_loss_1, test_accuracy_1 = self._test_loop(task_index=1)

        node_norms = self._compute_node_norms()
        node_norm_entropy = self._compute_norms_entropy(node_norms=node_norms)
        node_fischers_0 = self._compute_node_fischers(task_index=0)
        node_fischers_1 = self._compute_node_fischers(task_index=1)

        base_logging_dict = {
            constants.EPOCH_LOSS: train_epoch_loss,
            f"{constants.TEST}_{constants.LOSS}_0": test_loss_0,
            f"{constants.TEST}_{constants.LOSS}_1": test_loss_1,
            f"{constants.TEST}_{constants.ACCURACY}_0": test_accuracy_0,
            f"{constants.TEST}_{constants.ACCURACY}_1": test_accuracy_1,
            constants.NODE_NORM_ENTROPY: node_norm_entropy,
        }

        overlap_logging_dict = {
            f"{constants.SELF_OVERLAP}_{i}": norm for i, norm in enumerate(node_norms)
        }

        fischer_0_logging_dict = {
            f"{constants.NODE_FISCHER}_{0}_{i}": fischer
            for i, fischer in enumerate(node_fischers_0)
        }

        fischer_1_logging_dict = {
            f"{constants.NODE_FISCHER}_{1}_{i}": fischer
            for i, fischer in enumerate(node_fischers_1)
        }

        logging_dict = {
            **base_logging_dict,
            **overlap_logging_dict,
            **fischer_0_logging_dict,
            **fischer_1_logging_dict,
        }

        self._epoch_log(logging_dict=logging_dict, epoch=epoch)
        self._logger.info(f"Epoch {epoch + 1} loss: {train_epoch_loss}")

        self._data_logger.checkpoint()

    def _epoch_log(self, logging_dict: Dict[str, float], epoch: int):
        for tag, scalar in logging_dict.items():
            self._data_logger.write_scalar(tag=tag, step=epoch, scalar=scalar)

    def _train_loop(self, task_index: int):

        self._network.switch(new_task_index=task_index)

        loader = self._train_dataloaders[task_index]
        size = len(loader.dataset)

        epoch_loss = 0

        for batch, (x, y) in enumerate(loader):
            self._optimiser.zero_grad()
            prediction = self._network(x)
            loss = self._loss_function(prediction.flatten(), y.to(torch.float))
            loss.backward()
            self._optimiser.step()
            epoch_loss += loss.item()

        return epoch_loss / size

    def _test_loop(self, task_index: int):
        epoch_loss = 0
        correct_instances = 0

        loader = self._test_dataloaders[task_index]
        size = len(loader.dataset)

        with torch.no_grad():
            for batch, (x, y) in enumerate(loader):
                prediction = self._network.test_forward(x=x, head_index=task_index)
                loss = self._loss_function(prediction.flatten(), y)

                epoch_loss += loss.item()
                correct = (torch.sign(prediction) == y).item()
                correct_instances += correct

        return epoch_loss / size, correct_instances / size

    def _compute_norms_entropy(self, node_norms: np.ndarray) -> float:
        """Compute and log 'entropy' over node norms.

        This pseudo-entropy is computed by:
            - normalising the array of node norms
            - binning these normalised values
            - computing entropy over this binned distribution

        Args:
            epoch: epoch count (for logging).
            node_norms: magnitudes of hidden units.

        Returns:
            pseudo_entropy: pseudo measure of node norm entropy.
        """
        normalised_norms = node_norms / np.max(node_norms)
        binned_norms, _ = np.histogram(normalised_norms)
        dist = binned_norms / np.max(binned_norms)
        pseudo_entropy = -1 * np.sum(
            [(d + constants.EPS) * np.log(d + constants.EPS) for d in dist]
        )
        return pseudo_entropy

    def _compute_node_norms(self) -> None:
        network_copy = copy.deepcopy(self._network)
        layer = network_copy.layer_weights
        sel_sim = torch.mm(layer, layer.t()).numpy() / self._input_dimension
        norms = np.diagonal(sel_sim)
        return norms

    def _compute_node_fischers(self, task_index: int) -> None:
        loader = self._test_dataloaders[task_index]
        size = len(loader.dataset)

        self._network.switch(new_task_index=task_index)
        node_fischers = [0 for _ in range(self._hidden_dimension)]

        for batch, (x, y) in enumerate(loader):
            pre_activation = self._network.input_to_hidden(x=x)
            post_activation = self._network.activate(x=pre_activation)
            prediction = self._network.hidden_to_output(
                x=post_activation, head_index=task_index
            )

            loss = self._loss_function(prediction.flatten(), y.to(torch.float))

            derivative = torch.autograd.grad(loss, post_activation)[0]

            for node_index, node_derivative in enumerate(derivative[0]):
                node_fischers[node_index] += node_derivative.detach().item() ** 2 / size

        return node_fischers

    def post_process(self) -> None:
        """Solidify any data and make plots."""
        self._plotter.load_data()
        self._plotter.plot_learning_curves()
