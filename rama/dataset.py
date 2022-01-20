import os
from typing import Dict, List, Union, Optional

import torch
import torchvision
from rama import constants

DATA_PATH = constants.FMNIST_PATH


def get_whitening_matrix(test_data_x):
    # import pdb; pdb.set_trace()
    mean_test_x = torch.mean(test_data_x, axis=0)
    centered_test_x = test_data_x - mean_test_x
    empirical_covariance = centered_test_x.T @ centered_test_x / len(test_data_x) # D x D
    evals, evecs = torch.symeig(empirical_covariance, eigenvectors=True)
    return evals, evecs

class MixedFashionMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_1, dataset_2, mixing: Union[float, int], mapping: Dict, whitening_matrix: Optional[torch.Tensor] = None):
        self._dataset_1 = dataset_1
        self._dataset_2 = dataset_2

        assert len(self._dataset_1) == len(
            self._dataset_2
        ), "Constituent datasets must be same size."

        self._mixing = mixing
        self._mapping = mapping

        self._whitening_matrix = whitening_matrix

    def __getitem__(
        self,
        index,
    ):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = (
            self._mixing * self._dataset_1[index][0]
            + (1 - self._mixing) * self._dataset_2[index][0]
        )
        target = (
            self._mixing * self._mapping[self._dataset_1[index][1]]
            + (1 - self._mixing) * self._mapping[self._dataset_2[index][1]]
        )

        x = img.reshape(1, -1) @ self._whitening_matrix[1] / torch.sqrt(self._whitening_matrix[0] + 1e-5)

        return x, target

    def __len__(self):
        return len(self._dataset_1)


class FashionMNIST:

    ALL_TRANSFORMS = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.5,), (0.5)),
        ]
    )

    FASHION_MNIST_TRAIN = torchvision.datasets.FashionMNIST(
        root=DATA_PATH, transform=ALL_TRANSFORMS, download=not os.path.isdir(DATA_PATH)
    )
    FASHION_MNIST_TEST = torchvision.datasets.FashionMNIST(
        root=DATA_PATH, transform=ALL_TRANSFORMS, train=False
    )


class FashionMNISTSplitter:

    FASHION_MNIST_TRAIN_SPLITS = [
        torch.utils.data.Subset(
            FashionMNIST.FASHION_MNIST_TRAIN,
            torch.where(FashionMNIST.FASHION_MNIST_TRAIN.targets == i)[0],
        )
        for i in range(10)
    ]
    FASHION_MNIST_TEST_SPLITS = [
        torch.utils.data.Subset(
            FashionMNIST.FASHION_MNIST_TEST,
            torch.where(FashionMNIST.FASHION_MNIST_TEST.targets == i)[0],
        )
        for i in range(10)
    ]

    @classmethod
    def get_binary_classification_dataset(cls, index_1: int, index_2: int):
        train_set = torch.utils.data.ConcatDataset(
            [
                cls.FASHION_MNIST_TRAIN_SPLITS[index_1],
                cls.FASHION_MNIST_TRAIN_SPLITS[index_2],
            ]
        )
        test_set = torch.utils.data.ConcatDataset(
            [
                cls.FASHION_MNIST_TEST_SPLITS[index_1],
                cls.FASHION_MNIST_TEST_SPLITS[index_2],
            ]
        )
        return train_set, test_set

    @classmethod
    def get_binary_classification_dataloader(
        cls, index_1: int, index_2: int, batch_size: int, shuffle: bool
    ):
        train_set, test_set = cls.get_binary_classification_dataset(index_1, index_2)
        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return train_dataloader, test_dataloader

    @classmethod
    def get_mapping(cls, index_1: int, index_2: int, label_1: int, label_2: int):
        target_mapping = {
            index_1: label_1,
            index_2: label_2,
        }
        label_mapping = {index_1: label_1, index_2: label_2}
        return target_mapping, label_mapping

    @classmethod
    def get_mixed_dataset(
        cls,
        indices_1: List[int],
        indices_2: List[int],
        mixing: Union[float, int],
        label_1: int,
        label_2: int,
        whiten: bool = True
    ):
        train_set_1, test_set_1 = cls.get_binary_classification_dataset(
            indices_1[0], indices_1[1]
        )
        train_set_2, test_set_2 = cls.get_binary_classification_dataset(
            indices_2[0], indices_2[1]
        )

        mapping_1, _ = cls.get_mapping(
            index_1=indices_1[0], index_2=indices_1[1], label_1=label_1, label_2=label_2
        )
        mapping_2, _ = cls.get_mapping(
            index_1=indices_2[0], index_2=indices_2[1], label_1=label_1, label_2=label_2
        )

        mapping = {**mapping_1, **mapping_2}

        if whiten:
            test_set_1_tensor = next(iter(torch.utils.data.DataLoader(test_set_1, batch_size=len(test_set_1))))[0]
            test_set_1_tensor_flat = test_set_1_tensor.reshape(len(test_set_1_tensor), -1)
            test_set_2_tensor = next(iter(torch.utils.data.DataLoader(test_set_2, batch_size=len(test_set_2))))[0]
            test_set_2_tensor_flat = test_set_2_tensor.reshape(len(test_set_2_tensor), -1)
            mixed_test_set_tensor = mixing * test_set_1_tensor_flat + (1 - mixing) * test_set_2_tensor_flat
            whitening_matrix = get_whitening_matrix(mixed_test_set_tensor)
        else:
            whitening_matrix = None

        mixed_train_set = MixedFashionMNISTDataset(
            dataset_1=train_set_1,
            dataset_2=train_set_2,
            mixing=mixing,
            mapping=mapping,
            whitening_matrix=whitening_matrix
        )
        mixed_test_set = MixedFashionMNISTDataset(
            dataset_1=test_set_1, dataset_2=test_set_2, mixing=mixing, mapping=mapping, whitening_matrix=whitening_matrix
        )
        return mixed_train_set, mixed_test_set

    @classmethod
    def get_mixed_dataloader(
        cls,
        indices_1: List[int],
        indices_2: List[int],
        mixing: Union[float, int],
        label_1: int,
        label_2: int,
        batch_size: int,
        shuffle: bool,
        whiten: bool
    ):
        mixed_train_set, mixed_test_set = cls.get_mixed_dataset(
            indices_1=indices_1,
            indices_2=indices_2,
            mixing=mixing,
            label_1=label_1,
            label_2=label_2,
            whiten=whiten
        )
        mixed_train_dataloader = torch.utils.data.DataLoader(
            mixed_train_set, batch_size=batch_size, shuffle=shuffle
        )
        mixed_test_dataloader = torch.utils.data.DataLoader(
            mixed_test_set, batch_size=batch_size, shuffle=shuffle
        )

        return mixed_train_dataloader, mixed_test_dataloader
