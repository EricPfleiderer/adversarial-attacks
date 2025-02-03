import sys
import logging
from enum import Enum

import matplotlib.pyplot as plt
import torch.cuda
import numpy as np
from datetime import datetime

from python.supported_datasets import SupportedDatasets
from python.adversarial import GeneticAdversary
from python.trainables import TorchTrainable
from python.utils import create_video


def set_logging(level) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add file handlers
    debug_handler = logging.FileHandler('logs/debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    info_handler = logging.FileHandler('logs/info.log')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    # Add stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


if __name__ == '__main__':

    set_logging(logging.INFO)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Driver device: {device}')
    device = torch.device(device)

    params = {
        'dataset': SupportedDatasets.MNIST,

        'classifier': {
            'batch_size': 256,
            'epochs': 5,
            'optimizer': {
                'type': torch.optim.Adam,
                'params': {
                    'lr': 0.001,
                }
            },
        },

        'adversary': {
            'N': 10,
            'epochs': 5000,
            'asexual_repro': 1.,
            'selective_pressure': 0.3,
            'mutation_size': 0.001,
            'epsilon': 0.01,
            'uncertainty_power': 2,
            'sameness_power': 2,  # Must be even.
            'history_iters': 50,
        },
    }

    trainable = TorchTrainable(params)

    trainable.train()
    # trainable.plot_history(path='outs/')

    # 10, 11
    index = 11

    image = trainable.train_ds[index][0].to(trainable.device)
    target = torch.unsqueeze(torch.tensor(trainable.train_ds[index][1]), dim=0)

    print(f'The target is a {target[0]}')
    preds = trainable(image)[0]

    print(f'The classifier believes the class to be {torch.argmax(preds)} with a confidence of {preds[torch.argmax(preds)]}')

    adversary = GeneticAdversary(trainable, **params['adversary'])
    perturbation = adversary.genetic_attack(image=image, target=target)

    adv_preds = trainable(image+perturbation)[0]

    print(f'The classifier believes the adversarial image is a {torch.argmax(adv_preds)} with a confidence of {adv_preds[torch.argmax(adv_preds)]}')

    # for pred, sol in zip(adversary.history['predictions'], adversary.history['solutions']):
    for i in range(len(adversary.history['predictions'])):

        solution = adversary.history['solutions'][i].reshape(28, 28, 1)
        preds = adversary.history['predictions'][i][0]
        distance = adversary.history['distance'][0:i]
        certainty = adversary.history['certainty'][0:i]

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(solution)
        axs[0, 0].set_title('Perturbation')
        axs[0, 1].imshow(image.reshape(28, 28, 1).cpu().detach().numpy()+solution)
        axs[0, 1].set_title('Adversarial sample')
        axs[1, 0].bar(np.arange(0, 10), preds)
        axs[1, 0].set_title('Predictions')
        axs[1, 0].set_xlabel('Class label')
        axs[1, 0].set_ylabel('Certainty')
        axs[1, 0].set_ylim(0, 1)
        axs[1, 1].plot(np.arange(0, len(distance)) * params['adversary']['history_iters'], distance, label='distance')
        axs[1, 1].plot(np.arange(0, len(certainty)) * params['adversary']['history_iters'], certainty, label='certainty')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].set_title('Losses')
        fig.tight_layout(pad=2.5)
        plt.savefig(f'outs/recap_{datetime.now()}.png')


    create_video('outs', 'recap.avi')




