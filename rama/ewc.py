"""Adapted from implementation by moskomule

https://github.com/moskomule/ewc.pytorch
"""
import copy

from rama import constants
from torch import nn


class EWC:

    def __init__(self, importance: float, device: str):
        self._importance = importance
        self._device = device

        self._previous_task_parameters = []

    def _store_previous_task_parameters(self):
        """Store the parameters from before task switches."""
        previous_task_paramters = {}
        for n, param in copy.deepcopy(self._params).items():
            previous_task_paramters[n] = param.data.to(self._device)

        self._previous_task_parameters.append(previous_task_paramters)

    def compute_first_task_importance(
        self,
        network: nn.Module,
        previous_task_index: int,
        loss_function,
        dataloader,
    ):
        self._network = network
        self._dataloader = dataloader 
        self._loss_function = loss_function

        self._previous_task_index = previous_task_index
        self._new_task_index = self._network.task

        self._params = {
            n: p for n, p in self._network.named_parameters() if "heads" not in n
        }

        self._precision_matrices = self._diag_fisher()
        
        self._store_previous_task_parameters()
        
    @property
    def precision_matrices(self):
        return self._precision_matrices

    def _diag_fisher(self):
        # to compute Fischer on previous task, switch heads
        self._network.switch(new_task_index=self._previous_task_index)

        precision_matrices = {}

        for n, param in copy.deepcopy(self._params).items():
            param.data.zero_()
            precision_matrices[n] = param.data.to(self._device)

        self._network.eval()
        for batch, (x, y) in enumerate(self._dataloader):
            self._network.zero_grad()
            output = self._network(x)
            loss = self._loss_function(output, y)
            loss.backward()

            for n, param in self._network.named_parameters():
                if "heads" not in n:
                    precision_matrices[n].data += param.grad.data ** 2 / len(
                        self._dataloader.dataset
                    )

        precision_matrices = {n: param for n, param in precision_matrices.items()}

        # return back head
        self._network.switch(new_task_index=self._new_task_index)

        return precision_matrices

    def penalty(self, network: nn.Module):
        loss = 0
        for n, param in network.named_parameters():
            if "heads" not in n:
                _loss = (
                    self._precision_matrices[n]
                    * (param - self._previous_task_parameters[0][n]) ** 2
                )
                loss += _loss.sum()
        return self._importance * loss
