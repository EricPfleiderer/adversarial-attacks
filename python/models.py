import torch.nn as nn
import torch.nn.functional as F

from python.supported_datasets import SupportedDatasets


class ImageClassifier(nn.Module):

    def __init__(self, img_dims, n_classes):
        super(ImageClassifier, self).__init__()
        self.img_dims = img_dims
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(img_dims[0], 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        # self.conv2_drop = nn.Dropout2d()
        self.cnn_out_dim = (((img_dims[1]-4)//2)-4)//2  # TODO: Make kernel size, stride and number of layers dynamic
        self.nn_in_dim = 20*self.cnn_out_dim**2
        self.fc1 = nn.Linear(self.nn_in_dim, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x):

        """
        :param x: 3d tensor (color channel, spatial dim 1, spatial dim 2)
        :return:
        """

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self.nn_in_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def get_classifier(dataset: SupportedDatasets):
    if dataset.name == 'MNIST':
        return ImageClassifier((1, 28, 28), 10)
    elif dataset.name == 'CIFAR10':
        return ImageClassifier((3, 32, 32), 10)
    elif dataset.name == 'CIFAR100':
        return ImageClassifier((3, 32, 32), 100)
