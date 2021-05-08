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

# directories for single parameters experiments
single = Path(results, "singleParam")
single_img = Path(imgs, "singleParam")

# directories for vector paramaters experiments
vec = Path(results, "vec")
vec_img = Path(imgs, "vec")

# directory for preliminary experiments results
prelim = Path(results, "preliminary")
prelim_img = Path(imgs, "preliminary")

# directory for random directions experiment
random_dirs = Path(results, "rand_dirs")
random_dirs_img = Path(imgs, "rand_dirs")

# directory for PCA directions
pca_dirs = Path(results, "PCA_dirs")
pca_dirs_img = Path(imgs, "PCA_dirs")

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

# single loss experiments paths
svloss_path = Path(single, "svloss")  # results\svloss
svloss_img_path = Path(single_img, "svloss")

# single accuracy experiments paths
sacc_path = Path(single, "sacc")
sacc_img_path = Path(single_img, "sacc")

# vector loss experiments paths
vvloss_path = Path(vec, "vvloss")
vvloss_img_path = Path(vec_img, "vvloss")

# vector accuracy experiments paths
vacc_path = Path(vec, "vacc")
vacc_img_path = Path(vec_img, "vacc")

# preliminary experiments paths
train_subs_loss = Path(prelim, "train_subs_loss")
train_subs_acc = Path(prelim, "train_subs_acc")

test_subs_loss = Path(prelim, "test_subs_loss")
test_subs_acc = Path(prelim, "test_subs_acc")

epochs_loss = Path(os.path.join(prelim, "epochs_loss"))
epochs_acc = Path(os.path.join(prelim, "epochs_acc"))

# random directions experiment surface file
surf = Path(os.path.join(results, "surf_file.h5"))


def init_dirs():
    """
    Function initializes directories
    """
    logger.info("Initializing directories")
    dirs = [results, imgs, checkpoints, single, single_img, vec, vec_img, prelim,
            prelim_img, random_dirs, random_dirs_img, pca_dirs, pca_dirs_img]

    for d in dirs:
        logger.debug(f"Searching for {d}")
        if not d.exists():
            logger.debug(f"{d} not found. Creating...")
            os.makedirs(d)
