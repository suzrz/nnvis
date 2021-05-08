import logging
import copy
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch import optim
from nnvis import paths, net, data_loader

logger = logging.getLogger("vis_net")


def pre_train_subset(model, device, subset_list, epochs, test_loader):
    """
    Function to examine impact of different sizes of training subset.

    :param model: NN model
    :param device: device to be used
    :param subset_list: list of subsets sizes to be examinated
    :param epochs: number of training epoch
    :param test_loader: test dataset loader
    """
    logger.info("Subset preliminary experiment started")
    if paths.train_subs_loss.exists() and paths.train_subs_acc.exists():
        return

    loss_list = []
    acc_list = []
    theta_i = copy.deepcopy(torch.load(paths.init_state))
    theta_f = copy.deepcopy(torch.load(paths.final_state))

    for n_samples in subset_list:
        model.load_state_dict(theta_i)

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

        for epoch in range(1, epochs):
            train_loader, test_loader = data_loader.data_load(train_samples=n_samples)

            net.train(model, train_loader, optimizer, device, epoch)
            net.test(model, test_loader, device)

            scheduler.step()
            logger.debug(f"Finished epoch for tranining subset {epoch}, {n_samples}")

        loss, acc = net.test(model, test_loader, device)

        loss_list.append(loss)
        acc_list.append(acc)

    np.savetxt(paths.train_subs_loss, loss_list)
    np.savetxt(paths.train_subs_acc, acc_list)

    model.load_state_dict(theta_f)


def pre_test_subset(model, device, subset_list):
    """
    Function examines impact of test dataset size on stability of measurements

    :param model: NN model
    :param device: device to be used
    :param subset_list: list of subset sizes to be examined
    """
    if paths.test_subs_loss.exists() and paths.test_subs_acc.exists():
        return

    subset_losses = []
    subset_accs = []
    theta_f = copy.deepcopy(torch.load(paths.final_state))

    model.load_state_dict(theta_f)

    for n_samples in subset_list:
        losses = []
        accs = []
        for x in range(10):
            _, test_loader = data_loader.data_load(test_samples=n_samples)  # choose random data each time
            loss, acc = net.test(model, test_loader, device)
            losses.append(loss)
            accs.append(acc)
            logger.info(f"Subset size: {n_samples}\n"
                        f"Validation loss: {loss}\n"
                        f"Accuracy: {acc}\n")

        subset_losses.append(losses)
        subset_accs.append(accs)

    np.savetxt(paths.test_subs_loss, subset_losses)
    np.savetxt(paths.test_subs_acc, subset_accs)


def pre_epochs(model, device, epochs_list):
    """
    Function examines performance of the model after certain number of epochs

    :param model: NN model
    :param device: device to be used
    :param epochs_list: list of epochs numbers after which will be the model evaluated
    """
    logger.info("Epochs performance experiment started.")
    if paths.epochs_loss.exists() and paths.epochs_acc.exists():
        return

    loss_list = []
    acc_list = []

    theta_i = copy.deepcopy(torch.load(paths.init_state))

    model.load_state_dict(theta_i)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler
    train_loader, test_loader = data_loader.data_load()

    for epoch in range(max(epochs_list) + 1):
        net.train(model, train_loader, optimizer, device, epoch)
        net.test(model, test_loader, device)

        scheduler.step()

        logger.debug(f"Finished epoch {epoch}")
        if epoch in epochs_list:
            loss, acc = net.test(model, test_loader, device)

            loss_list.append(loss)
            acc_list.append(acc)
            logger.info(f"Performance of the model for epoch {epoch}"
                        f"Validation loss: {loss}"
                        f"Accuracy: {acc}")

    np.savetxt(paths.epochs_loss, loss_list)
    np.savetxt(paths.epochs_acc, loss_list)
