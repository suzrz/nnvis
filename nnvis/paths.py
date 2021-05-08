import os
import logging
from pathlib import Path

logger = logging.getLogger("vis_net")


# general directories
dirname = os.path.dirname
results = Path(dirname(dirname(__file__)), "results")
imgs = Path(dirname(dirname(__file__)), "imgs")

checkpoints = Path(dirname(dirname(__file__)), "model_states")

dataset = Path(dirname(dirname(__file__)), "data")

init_state = Path(results, "init_state.pt")
final_state = Path(results, "final_state.pt")

# directories for individual parameters experiments
individual = Path(results, "individualParam")
individual_img = Path(imgs, "individualParam")

# directories for vector paramaters experiments
layers = Path(results, "layers")
layers_img = Path(imgs, "layers")

# directory for preliminary experiments results
prelim = Path(results, "preliminary")
prelim_img = Path(imgs, "preliminary")

# directory for random directions experiment
random_dirs = Path(results, "randDirs")
random_dirs_img = Path(imgs, "randDirs")

# directory for PCA directions
pca_dirs = Path(results, "PCADirs")
pca_dirs_img = Path(imgs, "PCADirs")

# actual loss and accuracy progress of the model
actual_loss_path = Path(results, "actual_loss")
actual_acc_path = Path(results, "actual_acc")

# final loss and accuracy of the model
sf_loss_path = Path(results, "final_loss")
sf_acc_path = Path(results, "final_acc")

# interpolation
loss_path = Path(results, "loss_all")
acc_path = Path(results, "acc_all")

# quadratic interpolation
q_loss_path = Path(results, "q_loss_all")
q_acc_path = Path(results, "q_acc_all")

# individual loss experiments paths
svloss_path = Path(individual, "svloss")  # results\svloss
svloss_img_path = Path(individual_img, "svloss")

# individual accuracy experiments paths
sacc_path = Path(individual, "sacc")
sacc_img_path = Path(individual_img, "sacc")

# vector loss experiments paths
vvloss_path = Path(layers, "vvloss")
vvloss_img_path = Path(layers_img, "vvloss")

# vector accuracy experiments paths
vacc_path = Path(layers, "vacc")
vacc_img_path = Path(layers_img, "vacc")

# preliminary experiments paths
train_subs_loss = Path(prelim, "train_subs_loss")
train_subs_acc = Path(prelim, "train_subs_acc")

test_subs_loss = Path(prelim, "test_subs_loss")
test_subs_acc = Path(prelim, "test_subs_acc")

epochs_loss = Path(os.path.join(prelim, "epochs_loss"))
epochs_acc = Path(os.path.join(prelim, "epochs_acc"))

# random directions experiment surface file
surf = Path(os.path.join(random_dirs, "surf_file.h5"))


def init_dirs():
    """
    Function initializes directories
    """
    logger.info("Initializing directories")
    dirs = [results, imgs, checkpoints, individual, individual_img, layers, layers_img, prelim,
            prelim_img, random_dirs, random_dirs_img, pca_dirs, pca_dirs_img]

    for d in dirs:
        logger.debug(f"Searching for {d}")
        if not d.exists():
            logger.debug(f"{d} not found. Creating...")
            os.makedirs(d)
