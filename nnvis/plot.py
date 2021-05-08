import os
import re
import copy
import h5py
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.font_manager import FontProperties
from nnvis import paths

color_loss = "red"
color_trained = "dimgrey"
color_acc = "blue"

font = FontProperties()
font.set_size(20)

logger = logging.getLogger("vis_net")


def _plot_line(x, y, xlabel, ylabel, annotate=False, color="blue"):
    """
    Helper function to plot annotated line.

    :param x: x-axis data
    :param y: y-axis data
    :param xlabel: label for x-axis
    :param ylabel: label for y-axis
    :param annotate: annotate last three values
    :param color: color of the plot
    """
    logging.debug("[plot]: plotting line")
    fig, ax = plt.subplots(figsize=(6.4, 2))

    if xlabel:
        logging.debug("[plot]: xlabel = {}".format(xlabel))
        ax.set_xlabel(xlabel, fontproperties=font)
    if ylabel:
        logging.debug("[plot]: ylabel = {}".format(ylabel))
        ax.set_ylabel(ylabel, fontproperties=font)

    ax.plot(x, y, ".-", color=color, linewidth=1)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if annotate:
        if y[-1] < 1:
            # loss
            k = 2
            if x[-1] < 1000:
                k = 0.25
        else:
            # accuracy
            k = 0.02
            if x[-1] < 1000:
                k = 0.002

        ax.annotate("{:.3f}".format(y[-1]), xy=(x[-1], y[-1]), xytext=(x[-1], y[-1] + y[-1]*k))
        ax.annotate("{:.3f}".format(y[-2]), xy=(x[-2], y[-2]), xytext=(x[-2], y[-2] + y[-2]*k))
        ax.annotate("{:.3f}".format(y[-3]), xy=(x[-3], y[-3]), xytext=(x[-3], y[-3] + y[-3]*k))

    fig.tight_layout()
    plt.savefig(f"{os.path.join(os.path.join(paths.prelim_img), ylabel)}.pdf", format="pdf")
    plt.close("all")


def plot_impact(x, loss, acc, loss_only=False, acc_only=False, annotate=True, xlabel=None):
    """
    Function plots results of the training subset size and number of epochs preliminary experiments.

    :param x: x ticks
    :param loss: validation loss data
    :param acc: accuracy data
    :param loss_only: plot only loss
    :param acc_only: plot only acc
    :param annotate: annotate last three values
    :param xlabel: xlabel
    """
    logging.debug("[plot]: Plotting preliminary experiments results")
    if not acc_only:
        if not loss.exists():
            logging.error("[plot]: No loss data found")
            return
        _plot_line(x, np.loadtxt(loss), xlabel, "Validation loss", annotate, color_loss)

    if not loss_only:
        if not acc.exists():
            logging.error("[plot]: No accuracy data found")
            return
        _plot_line(x, np.loadtxt(acc), xlabel, "Accuracy", annotate, color_acc)


def plot_box(x, loss_only=False, acc_only=False, show=False, xlabel=None):
    """
    Function plots box plot representation of test data set subset size preliminary experiment.

    :param x: data
    :param loss_only: plot only validation loss examination
    :param acc_only: plot only accuracy examination
    :param show: show the plot
    :param xlabel: xlabel to be shown
    """
    logging.debug("Plotting preliminary experiments results (test subset size)")

    if not acc_only:
        fig, ax = plt.subplots(figsize=(9, 6))

        if not paths.epochs_loss.exists():
            logging.warning("No loss data found")
            return

        loss = np.loadtxt(paths.test_subs_loss)

        ax.set_ylabel("Validation loss", fontproperties=font)
        ax.set_xlabel(xlabel, fontproperties=font)
        ax.set_xticklabels(x)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.boxplot(loss)

        if show:
            plt.show()
        plt.savefig(f"{os.path.join(paths.prelim_img, 'test_loss.pdf')}", format="pdf")

    if not loss_only:
        fig, ax = plt.subplots()

        if not paths.epochs_acc.exists():
            logging.warning("No accuracy data found")
            return

        acc = np.loadtxt(paths.test_subs_acc)

        ax.set_ylabel("Accuracy", fontproperties=font)
        ax.set_xlabel(xlabel, fontproperties=font)
        ax.set_xticklabels(x)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.boxplot(acc)

        if show:
            plt.show()

        plt.savefig(f"{os.path.join(paths.prelim_img, 'test_acc.pdf')}", format="pdf")


def plot_metric(alpha, ydata, img_path, metric):
    """
    Function plots result of experiment focused on specified metric.

    :param alpha: interpolation coefficient
    :param ydata: results of the experiment to be plotted
    :param img_path: path to where save the graph
    :param metric: observed metric
    """
    ylab = "Validation loss" if metric == "loss" else "Accuracy"
    clr = color_loss if metric == "loss" else color_acc

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(alpha, ydata, ".-", color=clr, linewidth=1)
    ax.set_xlabel(r"$\alpha$", fontproperties=font)
    ax.set_ylabel(ylab, fontproperties=font)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.savefig(f"{img_path}.pdf", format="pdf")
    plt.close("all")


def map_distance(directory):
    """
    Maps calculated distances to values from interval <0, 1>

    :param directory: directory with distance files
    :return: dictionary of mapped distances assigned to according names
    """
    logger.debug(f"Mapping distance to <0;1>. Data directory: {directory}")
    a_files = os.listdir(directory)
    distances = {}
    for file in a_files:
        # get all distances and associate them to right values in dictionary
        if re.search("distance", file):
            f = open(os.path.join(directory, file), 'r')
            distances[file] = float(f.readline())

    result = copy.deepcopy(distances)

    # get max and min values
    mx = distances[max(distances, key=lambda k: distances[k])]
    mn = distances[min(distances, key=lambda k: distances[k])]

    for key, value in distances.items():
        # map the distances
        result[key] = (value - mn)/(mx - mn) * (1 - 0) + 0
        if result[key] < 0.1:
            # low cap
            result[key] = 0.1

    return result


def plot_params_by_layer(x, layer, opacity_dict, show=False):
    """
    Function plots all examined parameters of a layer in one plot

    :param x: data for x-axis (usually interpolation coefficient)
    :param layer: examined layer
    :param opacity_dict: dictionary with travelled distances of each parameter
    :param show: show the plots
    """
    logger.debug(f"Plotting all parameters of layer {layer} together.")
    files = os.listdir(paths.individual)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for file in files:
        if re.search(layer, file) and re.search("loss", file) and not \
                re.search("distance", file) and not re.search("q", file):
            k = file + "_distance"  # key for opacity dictionary
            lab = file.split("_")  # get label (parameter position)
            try:
                ax.plot(x, np.loadtxt(os.path.join(paths.individual, file)), label=lab[-1],
                        alpha=opacity_dict[k], color="blueviolet")
            except KeyError:
                logger.warning(f"No distance file for {file}")
                continue

    ax.set_ylabel("Validation loss", fontproperties=font)
    ax.set_xlabel(r"$\alpha$", fontproperties=font)

    logger.debug(f"Saving parameters of layer {layer} to {Path(paths.individual_img, layer).resolve()}")
    plt.savefig("{}.pdf".format(Path(paths.individual_img, layer)), format="pdf")

    if show:
        plt.show()

    plt.close("all")


def plot_vec_all_la(x):
    """
    Function plots all performance of the model with modified layers in one figure

    :param x: data for x-axis (interpolation coefficient)
    """
    files = os.listdir(paths.layers)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Validation loss")
    ax.set_xlabel(r"$\alpha$")

    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel(r"$\alpha$")

    for file in files:
        if not re.search("distance", file) and not re.search("q", file):
            lab = file.split('_')
            k = file + "_distance"
            try:
                if re.search("loss", file):
                    ax.plot(x, np.loadtxt(os.path.join(paths.layers, file)), label=lab[-1], lw=1)
                if re.search("acc", file):
                    ax2.plot(x, np.loadtxt(os.path.join(paths.layers, file)), lw=1)
            except KeyError:
                logger.warning(f"Missing key {k} in opacity dict, will not plot line for {file}")
                continue

    ax.plot(x, np.loadtxt(paths.loss_path), label="all", color=color_trained, linewidth=1)
    ax2.plot(x, np.loadtxt(paths.acc_path), color=color_trained, linewidth=1)

    fig.legend()
    fig.subplots_adjust(bottom=0.17)

    plt.savefig("{}_{}.pdf".format(paths.layers_img, "all_la"), format="pdf")

    plt.close("all")


def plot_lin_quad_real(alpha):
    epochs = np.arange(0, 14)

    if paths.loss_path.exists():
        lin = np.loadtxt(paths.loss_path)
    else:
        raise FileNotFoundError("Linear interpolation on the level of model not found. "
                                "Please run Linear.interpolate_all_linear.")

    if paths.q_loss_path.exists():
        quadr = np.loadtxt(paths.q_loss_path)
    else:
        raise FileNotFoundError("Linear interpolation on the level of model not found. "
                                "Please run Quadratic.interpolate_all_quadratic first.")

    if paths.actual_loss_path.exists():
        real = np.loadtxt(paths.actual_loss_path)
    else:
        raise FileNotFoundError("Linear interpolation on the level of model not found. "
                                "Please create and train model first.")

    fig, ax1 = plt.subplots()

    ax2 = ax1.twiny()

    c1, = ax1.plot(alpha, lin, label="Linear interpolation", color="orange")
    c2, = ax1.plot(alpha, quadr, label="Quadratic interpolation", color="blue")
    c3, = ax2.plot(epochs, real, label="Real values", color="black")

    curves = [c1, c2, c3]
    ax2.legend(curves, [curve.get_label() for curve in curves])

    ax1.set_xlabel(r"$\alpha$", fontproperties=font)
    ax2.set_xlabel("Epochs", fontproperties=font)
    ax1.set_ylabel("Validation Loss", fontproperties=font)

    plt.savefig(os.path.join(paths.layers_img, "lin_quadr_real.pdf"), format="pdf")

    plt.close("all")


def plot_individual_lin_quad(x):
    """
    Function plots comparison between linear and quadratic interpolation on the level of individual parameter.

    :param x: x-axis data
    """
    data = {}
    for fil in os.listdir(paths.individual):  # get all linear interpolation results
        if re.search("svloss", fil) and not re.search("q", fil) and not re.search("distance", fil):
            data[fil] = ""

    for fil in os.listdir(paths.individual):  # get all quadratic interpolation results
        if re.search("svloss", fil) and re.search("q", fil):
            k = fil[:-2]
            try:
                data[k] = fil
            except KeyError:
                continue

    for key, value in data.items():
        linear = Path(os.path.join(paths.individual, key))
        quadratic = Path(os.path.join(paths.individual, value))

        try:
            linear = np.loadtxt(linear)
            quadratic = np.loadtxt(quadratic)
        except OSError:
            continue  # only one type of the 1D experiment available

        fig = plt.figure(figsize=(8.5, 6))
        ax = fig.add_subplot()

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.plot(x, linear, color="orange", label="Linear")
        ax.plot(x, quadratic, color="blue", label="Quadratic")

        ax.set_xticks(ticks=[], minor=[])
        ax.set_yticks(ticks=[], minor=[])

        plt.savefig(os.path.join(paths.individual_img, f"{key}_comparison.pdf"), format="pdf")
        plt.close("all")


def contour_path(steps, loss_grid, coords, pcvariances):
    """
    Function plots a SGD path on loss grid

    :param steps: SGD steps
    :param loss_grid: validation loss surface
    :param coords: coordinates for the grid
    :param pcvariances: chosen pca variances
    """
    _, ax = plt.subplots()
    coords_x, coords_y = coords

    im = ax.contourf(coords_x, coords_y, loss_grid, levels=40, alpha=0.9)
    w1s = [step[0] for step in steps]
    w2s = [step[1] for step in steps]
    ax.plot(w1s, w2s, color='r', lw=1)
    ax.plot(steps[0][0], steps[0][1], "ro")
    plt.colorbar(im)
    ax.set_xlabel(f"PCA 0 {pcvariances[0]:.2%}")
    ax.set_ylabel(f"PCA 1 {pcvariances[1]:.2%}")

    plt.savefig(os.path.join(paths.pca_dirs_img, "loss_contour_path.pdf"), format="pdf")

    plt.close("all")


def surface_contour(loss_grid, coords):
    """
    Function plots the loss surface around a trained model

    :param loss_grid: validation loss grid
    :param coords: coordinates
    """
    _, ax = plt.subplots()

    im = ax.contourf(coords[0], coords[1], loss_grid, levels=45, alpha=0.9)
    plt.colorbar(im)

    plt.savefig(os.path.join(paths.pca_dirs_img, "loss_surface.pdf"), format="pdf")

    plt.close("all")


def surface3d_rand_dirs():
    # vmin = 0
    # vmax = 100

    # vlevel = 0.5
    surface = Path(os.path.join(paths.random_dirs, "surf.h5"))
    surf_name = "loss"

    with h5py.File(surface, 'r') as fd:
        x = np.array(fd["xcoordinates"][:])
        y = np.array(fd["ycoordinates"][:])

        X, Y = np.meshgrid(x, y)
        Z = np.array(fd[surf_name][:])

        """3D"""
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        surface = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, cmap="plasma")
        fig.colorbar(surface, shrink=0.5, aspect=5)
        plt.savefig(os.path.join(paths.random_dirs_img, "surface_3d.pdf"), format="pdf")


def surface_heatmap_rand_dirs():
    surface = Path(os.path.join(paths.random_dirs, "surf.h5"))
    surf_name = "loss"

    with h5py.File(surface, 'r') as fd:

        Z = np.array(fd[surf_name][:])

        """HEAT MAP"""
        fig, ax = plt.subplots()
        im = ax.imshow(Z, cmap="plasma")
        plt.colorbar(im)
        plt.savefig(os.path.join(paths.random_dirs_img, "surface_heatmap.pdf"), format="pdf")
