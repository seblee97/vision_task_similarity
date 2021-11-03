import os
from typing import Dict, List, Union

import torch
import torchvision
from rama import constants

DATA_PATH = constants.FMNIST_PATH


class MixedFashionMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_1, dataset_2, mixing: Union[float, int], mapping: Dict):
        self._dataset_1 = dataset_1
        self._dataset_2 = dataset_2

        assert len(self._dataset_1) == len(
            self._dataset_2
        ), "Constituent datasets must be same size."

        self._mixing = mixing
        self._mapping = mapping

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

        return img, target

    def __len__(self):
        return len(self._dataset_1)


class FashionMNIST:

    ALL_TRANSFORMS = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]
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
    def get_mapping(cls, index_1: int, index_2: int):
        target_mapping = {
            index_1: -1.0,
            index_2: 1.0,
        }
        label_mapping = {index_1: -1, index_2: 1}
        return target_mapping, label_mapping

    @classmethod
    def get_mixed_dataset(
        cls, indices_1: List[int], indices_2: List[int], mixing: Union[float, int]
    ):
        train_set_1, test_set_1 = cls.get_binary_classification_dataset(
            indices_1[0], indices_1[1]
        )
        train_set_2, test_set_2 = cls.get_binary_classification_dataset(
            indices_2[0], indices_2[1]
        )

        mapping_1, _ = cls.get_mapping(index_1=indices_1[0], index_2=indices_1[1])
        mapping_2, _ = cls.get_mapping(index_1=indices_2[0], index_2=indices_2[1])

        mapping = {**mapping_1, **mapping_2}

        mixed_train_set = MixedFashionMNISTDataset(
            dataset_1=train_set_1,
            dataset_2=train_set_2,
            mixing=mixing,
            mapping=mapping,
        )
        mixed_test_set = MixedFashionMNISTDataset(
            dataset_1=test_set_1, dataset_2=test_set_2, mixing=mixing, mapping=mapping
        )
        return mixed_train_set, mixed_test_set

    @classmethod
    def get_mixed_dataloader(
        cls,
        indices_1: List[int],
        indices_2: List[int],
        mixing: Union[float, int],
        batch_size: int,
        shuffle: bool,
    ):
        mixed_train_set, mixed_test_set = cls.get_mixed_dataset(
            indices_1=indices_1, indices_2=indices_2, mixing=mixing
        )
        mixed_train_dataloader = torch.utils.data.DataLoader(
            mixed_train_set, batch_size=batch_size, shuffle=shuffle
        )
        mixed_test_dataloader = torch.utils.data.DataLoader(
            mixed_test_set, batch_size=batch_size, shuffle=shuffle
        )

        return mixed_train_dataloader, mixed_test_dataloader
