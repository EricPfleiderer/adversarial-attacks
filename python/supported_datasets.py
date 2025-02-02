from torchvision import datasets
from enum import Enum
import torch.utils.data as data
import torchvision.transforms as transforms


class SupportedDatasets(Enum):
    MNIST = datasets.MNIST
    # CIFAR10 = datasets.CIFAR10
    # CIFAR100 = datasets.CIFAR100


def get_dataset(dataset: SupportedDatasets, root='data/', download=True) -> (data.DataLoader, data.DataLoader):
    if not isinstance(dataset, SupportedDatasets):
        raise ValueError(f'Please use the {SupportedDatasets} enumerator to specify the dataset.')

    else:

        # Normalize RGB images
        class NormalizeRGB:
            def __call__(self, sample):
                return sample/255

        tfs = transforms.ToTensor()
        if dataset.value == 'CIFAR10' or dataset.value == 'CIFAR100':
            tfs = transforms.Compose([tfs, NormalizeRGB])

        train_ds = dataset.value(root, train=True, download=download, transform=tfs)
        test_ds = dataset.value(root, train=False, download=download, transform=tfs)

        # TODO: Standardize supported datasets (3 channels, 0 to 1 range, channels dim first)

        return train_ds, test_ds


def get_loaders(dataset: SupportedDatasets, batch_size, root='data/', download=True, shuffle=True) -> (data.DataLoader, data.DataLoader):
    train_ds, test_ds = get_dataset(dataset, root, download)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader

