import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import logging
from torch.utils.data import DataLoader
from python.models import get_classifier
from python.supported_datasets import get_loaders, get_dataset


class TorchTrainable:

    """
    Wrapper for torch models, their optimizer and their training procedure.
    """

    def __init__(self, params):

        logging.info('Initializing a trainable.')
        self.params = params
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        logging.info(f'Model device:{device}')

        self.train_ds, self.test_ds = get_dataset(params['dataset'])
        self.train_loader, self.test_loader = get_loaders(params['dataset'], params['classifier']['batch_size'])

        # Torch model
        self.model = get_classifier(params['dataset']).to(self.device)

        # Torch optimizer
        self.optimizer = params['classifier']['optimizer']['type'](self.model.parameters(), **params['classifier']['optimizer']['params'])

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
        }

    def reset_history(self):
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
        }

    def train(self):

        logging.info(f'Training the image classifier for ' + str(self.params['classifier']['epochs']) + ' epochs.')

        self.reset_history()

        for epoch in range(self.params['classifier']['epochs']):

            # Training
            train_loss = 0
            train_hits = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                predictions = output.data.max(1, keepdim=True)[1]
                train_hits += predictions.eq(target.data.view_as(predictions)).sum()
                train_loss = F.nll_loss(output, target)
                train_loss.backward()
                self.optimizer.step()
            train_accuracy = train_hits / self.train_loader.dataset.data.shape[0]

            # Validation
            test_loss = 0
            test_hits = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    predictions = output.data.max(1, keepdim=True)[1]
                    test_hits += predictions.eq(target.data.view_as(predictions)).sum()
            test_loss /= len(self.test_loader.dataset)
            test_accuracy = test_hits / self.test_loader.dataset.data.shape[0]

            # Update history
            self.history['train_loss'].append(train_loss.item())
            self.history['train_accuracy'].append(train_accuracy.data.cpu().detach().numpy())
            self.history['test_loss'].append(test_loss)
            self.history['test_accuracy'].append(test_accuracy.data.cpu().detach().numpy())

            # Update logs
            logging.info(f'Epoch: {epoch}/' + str(self.params['classifier']['epochs']) +
                         ', train loss: ' + '{:.4f}'.format(round(train_loss.item(), 4)) +
                         ', test_loss: ' + '{:.4f}'.format(round(test_loss, 4)))

    def __call__(self, x):
        return torch.exp(self.model(x.to(self.device)))

    def plot_history(self, path, save=True, show=False):
        epochs = self.params['classifier']['epochs']
        plt.figure()
        plt.plot(np.arange(1, epochs + 1), self.history['train_loss'], label='Training loss')
        plt.plot(np.arange(1, epochs + 1), self.history['test_loss'], label='Test loss')
        plt.legend(loc='best')
        if save:
            plt.savefig(path + 'classifier_loss.jpg')
        if show:
            plt.show()

        plt.figure()
        plt.plot(np.arange(1, epochs + 1), self.history['train_accuracy'], label='Training accuracy')
        plt.plot(np.arange(1, epochs + 1), self.history['test_accuracy'], label='Test accuracy')
        plt.legend(loc='best')
        if save:
            plt.savefig(path + 'classifier_accuracy.jpg')
        if show:
            plt.show()
