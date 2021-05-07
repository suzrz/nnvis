import torch
import pickle
from lib.paths import *
from torch import nn as nn
import torch.nn.functional as f

logger = logging.getLogger("vis_net")


class BaseNN(nn.Module):
    def __init__(self):
        super(BaseNN, self).__init__()

    def get_flat_params(self, device):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.data

        flat_params = torch.Tensor().to(device)

        for _, param in params.items():
            flat_params = torch.cat((flat_params, torch.flatten(param)))

        return flat_params

    def load_from_flat_params(self, f_params):
        shapes = []
        for name, param in self.named_parameters():
            shapes.append((name, param.shape, param.numel()))

        state = {}
        c = 0
        for shape in shapes:
            name, tsize, tnum = shape
            param = f_params[c: c + tnum].reshape(tsize)
            state[name] = torch.nn.Parameter(param)
            c += tnum

        self.load_state_dict(state, strict=True)


class SimpleCNN(BaseNN):
    """
    Neural network class

    This net consists of 4 layers. Two are convolutional and
    the other two are dropout.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # define layers of network
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        logger.info(f"Network was initialized.")
        logger.debug(f"Network architecture:\n{self}")

    def forward(self, x):
        """
        Forward pass data

        :param x: Input data
        :return: Output data. Probability of a data sample belonging to one of the classes
        """

        x = self.conv1(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.fc3(x)

        output = f.log_softmax(x, dim=1)
        return output


def train(model, train_loader, optimizer, device, epoch, checkpoint_file=True):
    """ Trains the network.

    :param model : Neural network model to be trained
    :param train_loader : Data loader
    :param optimizer : Optimizer
    :param device : Device on which will be the net trained
    :param epoch : Number of actual epoch
    :param checkpoint_file: creates checkpoint file
    :return: training loss for according epoch
    """
    model.train()  # put net into train mode
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optim_path = {"flat_w": [], "loss": []}
        data, target = data.to(device), target.to(device)  # load data
        optimizer.zero_grad()  # zero all gradients
        output = model.forward(data)  # feed data through net

        loss = f.nll_loss(output, target)  # compute train loss
        train_loss += f.nll_loss(output, target, reduction="sum").item()
        loss.backward()
        optimizer.step()

        if checkpoint_file:
            filename = Path(os.path.join(checkpoints), f"checkpoint_epoch_{epoch}_step_{batch_idx}.pkl")

            logger.debug(f"Creating checkpoint file {filename}")

            optim_path["flat_w"].append(model.get_flat_params(device))
            optim_path["loss"].append(loss)

            with open(filename, "wb") as fd:
                pickle.dump(optim_path, fd)

    train_loss /= len(train_loader.dataset)
    logger.info(f"Training in epoch {epoch} has finished (loss = {train_loss})")
    return train_loss


def test(model, test_loader, device):
    """ Validates the neural network.

    :param model : Neural network model to be validated
    :param test_loader : Data loader
    :device : Device on which will be validation performed
    """
    model.eval()  # put model in evaluation mode
    test_loss = 0  # validation loss
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            test_loss += f.nll_loss(output, target, reduction="sum").item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # compute validation loss of neural network
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.debug(f"Validation has finished:\n"
                 f"\n      Validation loss: {test_loss}\n"
                 f"\n      Accuracy: {accuracy} %\n")
    return test_loss, accuracy
