"""
This code is based on:

Title: Animating the Optimization Trajectory of Neural Nets
Authors: Chao Yang
Date: 2021-02-1
Version: 0.1.9
Availability: https://github.com/logancyang/loss-landscape-anim
"""
import re
import os
import sys
import torch
import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from nnvis import paths, net

logger = logging.getLogger("vis_net")


def atof(text):
    """
    Function converts char representation of number if the parameter text is number to float representation.
    Else it returns the input unchaged.

    :param text: text to be converted
    :return: float number or unchanged input
    """
    try:
        retval = float(text)
    except ValueError:
        retval = text

    return retval


def natural_keys(text):
    """
    Function creates key for natural sort.

    :param text: input to create the key from
    :return: keys
    """
    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]


class Examinator2D:
    def __init__(self, model, device, checkpoints_dir=paths.checkpoints):
        self.model = model
        self.device = device
        self.directory = checkpoints_dir

    def __get_steps(self):
        """
        Method loads steps of optimization algorithm from specified directory

        """
        logger.debug(f"Loading checkpoints from directory {self.directory}")

        steps = {"flat_w": [], "loss": []}
        files = os.listdir(os.path.abspath(self.directory))
        files.sort(key=natural_keys)

        for filename in files:
            if re.search("step", filename):
                logger.debug(f"Loading from file {filename}")
                with open(os.path.join(os.path.abspath(self.directory), filename), "rb") as fd:
                    try:
                        checkpoint = pickle.load(fd)

                        steps["flat_w"].append(checkpoint["flat_w"])
                        steps["loss"].append(checkpoint["loss"])
                    except pickle.UnpicklingError:
                        continue

        return steps

    def __sample_path(self, steps, n_samples=300):
        """
        Method takes n_samples from steps dictionary

        :param steps: dictionary of sgd steps with members flat_w [] and loss []
        :param n_samples: number of samples to take
        :return: sampled dict
        """
        samples = {"flat_w": [], "loss": []}

        if n_samples > len(steps["flat_w"]):
            logger.warning(f"Less steps ({len(steps)} than final samples ({n_samples}). Using whole set of steps.")
            n_samples = len(steps)

        interval = len(steps["flat_w"]) // n_samples
        logger.debug(f"Samples interval: {interval}")
        count = 0
        for i in range(len(steps["flat_w"]) - 1, -1, -1):  # TODO investigate
            if i % interval == 0 and count < n_samples:
                samples["flat_w"].append(steps["flat_w"][i])
                samples["loss"].append(steps["loss"][i])
                count += 1

        # samples["flat_w"] = reversed(samples["flat_w"])
        # samples["loss"] = reversed(samples["loss"])
        return samples

    def __calc_step(self, resolution, grid):
        """
        Method calculates the length of steps to take during the visualization.

        :param resolution: resolution of the visualization
        :param grid: prepared grid for the results
        :return: step size
        """
        dist_2d = grid[-1] - grid[0]
        dist = (dist_2d[0] ** 2 + dist_2d[1] ** 2) ** 0.5
        return dist * (1 + 0.3) / resolution

    def __pca_dim_reduction(self, params):
        """
        Performs PCA dimension reduction.

        :param params: parameters of the model
        :return: path of the optimization algorithm, pca directions, reduced directions and pcvariances
        """
        logger.debug("PCA dimension reduction")
        optim_path_np = []
        for tensor in params:
            optim_path_np.append(np.array(tensor[0].cpu()))

        pca = PCA(n_components=2)
        path_2d = pca.fit_transform(optim_path_np)
        reduced_dirs = pca.components_
        logger.debug(f"Reduced directions: {reduced_dirs}")
        logger.debug(f"PCA variances: {pca.explained_variance_ratio_}")

        return {
            "optim_path": optim_path_np,
            "path_2d": path_2d,
            "reduced_dirs": reduced_dirs,
            "pcvariances": pca.explained_variance_ratio_
        }

    def __compute_loss_2d(self, test_loader, params_grid):
        """
        Calculates the loss of the model on 2D grid

        :param test_loader: test data set loader
        :param params_grid: parameter grid
        :return: 2D array of validation loss, position of minimum, value of minumum
        """
        loss_2d = []
        n = len(params_grid)
        m = len(params_grid[0])
        loss_min = sys.float_info.max
        arg_min = ()

        logger.info("Calculating loss values for PCA directions")
        for i in range(n):
            loss_row = []
            for j in range(m):
                logger.debug(f"Calculating loss for coordinates: {i}, {j}")
                w_ij = torch.Tensor(params_grid[i][j].float()).to(self.device)

                self.model.load_from_flat_params(w_ij)
                loss, acc = net.test(self.model, test_loader, self.device)
                logger.debug(f"Loss for {i}, {j} = {loss}")
                if loss < loss_min:
                    loss_min = loss
                    logger.debug(f"New min loss {loss_min}")
                    arg_min = (i, j)
                loss_row.append(loss)
            loss_2d.append(loss_row)

        loss_2darray = np.array(loss_2d).T
        return loss_2darray, arg_min, loss_min

    def __convert_coord(self, idx, ref_p, step_size):
        """
        Converts the coordinates to matrix coordinates

        :param idx: ids
        :param ref_p: reference point
        :param step_size: step size
        :return converted coordinate
        """
        return idx * step_size + ref_p

    def __get_coords(self, step_size, resolution, optim_point2d):
        """
        Calculates coordinates

        :param step_size: step size
        :param resolution: resolution of the visualization
        :param optim_point2d: reference point
        :return: x coordinate, y coordinate
        """
        converted_x = []
        converted_y = []
        for i in range(-resolution, resolution):
            converted_x.append(self.__convert_coord(i, optim_point2d[0], step_size))
            converted_y.append(self.__convert_coord(i, optim_point2d[1], step_size))

        return converted_x, converted_y

    def get_loss_grid(self, test_loader, resolution=50):
        """
        Calculated the validation loss over a PCA projection grid.

        :param test_loader: test data set loader
        :param resolution: resolution of the visualization
        :return: path of the optimizer in 2D, validation loss grid, minimum position, minimum value, coordinates,
                pcvariances
        """
        grid_file = Path(os.path.join(paths.pca_dirs, "loss_grid"))
        logger.info(f"Surface grid file {grid_file}")

        steps = self.__get_steps()
        logger.debug(f"Steps len: {len(steps)}, type: {type(steps)}\n"
                     f"Steps flat_w len: {len(steps['flat_w'])}\n"
                     f"Steps loss len: {len(steps['loss'])}")
        sampled_optim_path = self.__sample_path(steps)
        logger.debug(f"Sampled len: {len(sampled_optim_path)}, type: {type(sampled_optim_path)}\n"
                     f"Sample flat_w len: {len(sampled_optim_path['flat_w'])}\b"
                     f"Sample loss len: {len(sampled_optim_path['loss'])}")

        optim_path = sampled_optim_path["flat_w"]
        logger.debug(f"Optim path len: {len(optim_path)}")

        l_path = sampled_optim_path["loss"]
        logger.debug(f"Loss path len : {len(l_path)}")

        reduced_dict = self.__pca_dim_reduction(optim_path)
        path_2d = reduced_dict["path_2d"]
        directions = reduced_dict["reduced_dirs"]
        pcvariances = reduced_dict["pcvariances"]

        d1 = directions[0]
        d2 = directions[1]

        optim_point = optim_path[-1]
        optim_point_2d = path_2d[-1]

        alpha = self.__calc_step(resolution, path_2d)
        logger.debug(f"Step size: {alpha}")

        grid = []
        # prepare grid
        for i in range(-resolution, resolution):
            r = []
            for j in range(-resolution, resolution):
                updated = optim_point[0].cpu() + (i * d1 * alpha) + (j * d2 * alpha)
                r.append(updated)
            grid.append(r)

        if not grid_file.exists():
            loss, argmin, loss_min = self.__compute_loss_2d(test_loader, grid)

            with open(grid_file, "wb") as fd:
                pickle.dump((loss, argmin, loss_min), fd)

        else:
            with open(grid_file, "rb") as fd:
                loss, argmin, loss_min = pickle.load(fd)

        coords = self.__get_coords(alpha, resolution, optim_point_2d)

        return {
            "path_2d": path_2d,
            "loss_grid": loss,
            "argmin": argmin,
            "loss_min": loss_min,
            "coords": coords,
            "pcvariances": pcvariances
        }
