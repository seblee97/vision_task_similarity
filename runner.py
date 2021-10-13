import copy

import constants
import dataset
import network
import numpy as np
import torch
import torch.nn as nn
from run_modes import base_runner


class Runner(base_runner.BaseRunner):
    def __init__(self, config):
        self._first_task_epochs = config.switch_epoch
        self._second_task_epochs = config.total_epochs - config.switch_epoch

        self._input_dimension = config.input_dimension
        self._hidden_dimension = config.hidden_dimension

        self._network, self._optimiser = self._setup_network(config=config)
        self._train_dataloaders, self._test_dataloaders = self._setup_data(
            config=config
        )

        self._loss_function = nn.MSELoss()

        super().__init__(config=config)

    def _get_data_columns(self):
        columns = [
            constants.EPOCH_LOSS,
            f"{constants.TEST}_{constants.LOSS}_0",
            f"{constants.TEST}_{constants.LOSS}_1",
            f"{constants.TEST}_{constants.ACCURACY}_0",
            f"{constants.TEST}_{constants.ACCURACY}_1",
        ]

        columns.extend(
            [f"{constants.SELF_OVERLAP}_{i}" for i in range(self._hidden_dimension)]
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

        for e in range(self._first_task_epochs):
            self._train_test_loop(epoch=e, task_index=0)

        for e in range(self._second_task_epochs):
            self._train_test_loop(epoch=e, task_index=1)

    def _train_test_loop(self, epoch: int, task_index: int):

        self._log_overlaps(epoch=epoch)

        train_epoch_loss = self._train_loop(task_index=task_index)

        test_loss_0, test_accuracy_0 = self._test_loop(task_index=0)
        test_loss_1, test_accuracy_1 = self._test_loop(task_index=1)

        self._data_logger.write_scalar(
            tag=constants.EPOCH_LOSS, step=epoch, scalar=train_epoch_loss
        )
        self._data_logger.write_scalar(
            tag=f"{constants.TEST}_{constants.LOSS}_0", step=epoch, scalar=test_loss_0
        )
        self._data_logger.write_scalar(
            tag=f"{constants.TEST}_{constants.LOSS}_1", step=epoch, scalar=test_loss_1
        )
        self._data_logger.write_scalar(
            tag=f"{constants.TEST}_{constants.ACCURACY}_0",
            step=epoch,
            scalar=test_accuracy_0,
        )
        self._data_logger.write_scalar(
            tag=f"{constants.TEST}_{constants.ACCURACY}_1",
            step=epoch,
            scalar=test_accuracy_1,
        )

        self._logger.info(f"Epoch {epoch + 1} loss: {train_epoch_loss}")

        self._data_logger.checkpoint()

    def _train_loop(self, task_index: int):

        self._network.switch(new_task_index=task_index)

        loader = self._train_dataloaders[task_index]
        size = len(loader.dataset)

        epoch_loss = 0

        for batch, (x, y) in enumerate(loader):
            prediction = self._network(x)
            loss = self._loss_function(prediction.flatten(), y.to(torch.float))
            self._optimiser.zero_grad()
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

    def _log_overlaps(self, epoch: int):
        network_copy = copy.deepcopy(self._network)
        layer = network_copy.layer_weights
        sel_sim = torch.mm(layer, layer.t()).numpy() / self._input_dimension
        for i, o in enumerate(np.diagonal(sel_sim)):
            self._data_logger.write_scalar(
                tag=f"{constants.SELF_OVERLAP}_{i}", step=epoch, scalar=o
            )
