import torch.utils.data
from nnvis import paths
import logging
import numpy as np
from torch import utils as utils
from torchvision import datasets, transforms


logger = logging.getLogger("vis_net")


def data_load(train_samples=60000, test_samples=10000):
    """
    Function prepares and loads data

    :param train_samples: size of training dataset subset
    :param test_samples: size of test dataset subset
    :return: train loader, test loader
    """
    logger.info("Loading data.")
    logger.debug(f"Training set size: {train_samples}")
    logger.debug(f"Test set size: {test_samples}")

    # preprocess data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081)])

    logger.debug(f"Data transformations: {transform}")

    # prepare subsets
    train_set = datasets.MNIST(paths.dataset, train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST(paths.dataset, train=False, download=True,
                              transform=transform)

    tr = list(range(0, len(train_set), 1))
    te = np.random.permutation(len(test_set))

    tr = tr[:train_samples]
    te = te[:test_samples]

    train_set = torch.utils.data.Subset(train_set, tr)
    test_set = torch.utils.data.Subset(test_set, te)

    # get data loaders
    train_loader = utils.data.DataLoader(train_set, 64, shuffle=True)
    test_loader = utils.data.DataLoader(test_set, 1000, shuffle=False)

    return train_loader, test_loader
