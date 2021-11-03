from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
from rama import constants


class TwoLayerRegressionNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nonlinearity: str,
        num_heads: int,
    ):

        super().__init__()

        self._layer = nn.Linear(input_dim, hidden_dim)
        self._heads = nn.ModuleList(
            [nn.Linear(hidden_dim, output_dim) for _ in range(num_heads)]
        )

        self._activation = self._setup_activation(nonlinearity=nonlinearity)

        self._task = 0

    @property
    def layer_weights(self):
        return self._layer.state_dict()["weight"]

    def _setup_activation(self, nonlinearity: str) -> Callable:
        if nonlinearity == constants.RELU:
            activation = F.relu
        elif nonlinearity == constants.SIGMOID:
            activation = F.sigmoid
        else:
            raise ValueError(f"Unrecognised nonlinearity name: {nonlinearity}")
        return activation

    def switch(self, new_task_index: int):
        self._freeze_head(self._task)
        self._unfreeze_head(new_task_index)
        self._task = new_task_index

    def _freeze_head(self, head_index: int) -> None:
        """Freeze weights of head for task with index head_index."""
        for param in self._heads[head_index].parameters():
            param.requires_grad = False

    def _unfreeze_head(self, head_index: int) -> None:
        """Unfreeze weights of head for task with index head index."""
        for param in self._heads[head_index].parameters():
            param.requires_grad = True

    def test_forward(self, x, head_index: int):
        x = self.input_to_hidden(x)
        x = self.activate(x)
        y = self.hidden_to_output(x, head_index)

        return y

    def input_to_hidden(self, x):
        x = x.reshape(x.shape[0], -1)
        return self._layer(x)

    def activate(self, x):
        return self._activation(x)

    def hidden_to_output(self, x, head_index: int):
        return self._heads[head_index](x)

    def forward(self, x):
        x = self.input_to_hidden(x)
        x = self.activate(x)
        y = self.hidden_to_output(x, self._task)

        return y
