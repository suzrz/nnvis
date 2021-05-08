"""
This code is based on:

Title: Visualizing the Loss Landscape of Neural Nets
Authors: Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom
Date: 2021-02-1
Version: -
Availability: https://github.com/tomgoldstein/loss-landscape
"""
import os
import h5py
import torch
import numpy as np
import logging
from pathlib import Path
from nnvis import paths, net

logger = logging.getLogger("vis_net")


def get_random_direction(model, device):
    """
    Function generates random directions and normalizes them

    :param model: model to be respected when normalizing
    :param device: device to be used
    :return: normalized random direction
    """
    weights = [p.data for p in model.parameters()]
    direction = [torch.randn(w.size()).to(device) for w in weights]

    assert (len(direction) == len(weights))

    for d, w in zip(direction, weights):
        for dire, wei in zip(d, w):
            dire.mul_(wei.norm() / (dire.norm() + 1e-10))

    return direction


def get_directions(model, device):
    """
    Function prepares two random directions

    :param model: model
    :param device: device to be used
    :return: list of two random directions
    """
    x = get_random_direction(model, device)
    y = get_random_direction(model, device)

    return [x, y]


def set_surf_file(filename):
    """
    Function prepares h5py file for storing loss function values

    :param filename: Filename of a surface file
    """
    xmin, xmax, xnum = -1, 2, 20
    ymin, ymax, ynum = -1, 2, 20

    if filename.exists():
        return

    with h5py.File(filename, 'a') as fd:
        xcoord = np.linspace(xmin, xmax, xnum)
        fd["xcoordinates"] = xcoord

        ycoord = np.linspace(ymin, ymax, ynum)
        fd["ycoordinates"] = ycoord

        shape = (len(xcoord), len(ycoord))

        losses = -np.ones(shape=shape)
        fd["loss"] = losses


def get_indices(vals, xcoords, ycoords):
    """
    Function gets indices

    :param vals: values
    :param xcoords: x coordinates
    :param ycoords: y coordinates
    :return: indices
    """
    ids = np.array(range(vals.size))
    ids = ids[vals.ravel() <= 0]

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoords, ycoords)

    s1 = xcoord_mesh.ravel()[ids]
    s2 = ycoord_mesh.ravel()[ids]

    return ids, np.c_[s1, s2]


def overwrite_weights(model, init_weights, directions, step, device):
    """
    Function overwrite weights of the model according to actual step

    :param model: model which parameters are updated
    :param init_weights: initial parameters of the model
    :param directions: x, y directions
    :param step: step
    :param device: device
    """
    dx = directions[0]
    dy = directions[1]

    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w.to(device) + d.clone().detach().requires_grad_(True)


def calc_loss(model, test_loader, directions, device):
    """
    Function iterates over surface file and calculates loss over the surface

    :param model: model to be evaluated
    :param test_loader: test dataset loader
    :param directions: random projection directions
    :param device: device
    """
    logger.info("Calculating loss function surface")
    filename = Path(os.path.join(paths.random_dirs, "surf.h5"))
    logger.debug(f"Surface file: {filename}")

    set_surf_file(filename)

    init_weights = [p.data for p in model.parameters()]

    with h5py.File(filename, "r+") as fd:
        xcoords = fd["xcoordinates"][:]
        ycoords = fd["ycoordinates"][:]
        losses = fd["loss"][:]

        ids, coords = get_indices(losses, xcoords, ycoords)

        for count, idx in enumerate(ids):
            coord = coords[count]
            logger.debug(f"Index: {idx}")

            overwrite_weights(model, init_weights, directions, coord, device)

            loss, _ = net.test(model, test_loader, device)
            logger.debug(f"Loss: {loss}")

            losses.ravel()[idx] = loss

            fd["loss"][:] = losses

            fd.flush()
