# -----------------------------------------------------------
# neural net (MLP) for binary classification
#
# Torjus Nilsen, Kongsberg, Norway
# email tornil1996@hotmail.com
# -----------------------------------------------------------

"""
TODO
[ ] Plot train/test loss
"""

# standard lib
import argparse

# previous sklearn dataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

# pytorch
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split

# data
import numpy as np

# Binary classificiation:
# Continuation of classifying the breastcancer dataset using a Pytorch neural net.
#
# We will need:
# - Pytorch Dataset and Dataloader: (dataloader is needed to iterate over the dataset and to divide it up in batches)
# - Model: An architecture describing the layers of our network with a forward pass method.
# - Training logic: forward pass, backward pass (updating weights from loss criterion with an optimizer)


# run params
epochs = 50
batch_size = 32
learning_rate = 0.001  # how much to change weights during training

# make experiments reproducible (a seed makes random numbers from that package generator non-varying between runs)
torch.manual_seed(42)
np.random.seed(42)


###
# DATASET AND DATALOADER
###


def get_dataset():

    # load dataset from sklearn as before
    X, y = load_breast_cancer(return_X_y=True)
    X = normalize(X)

    # creating a custom dataset:
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

    class MyDataset(Dataset):
        """
        The Custom Pytorch Dataset Class should inherit from Dataset and overwrite 3 methods:
            __init__: Store paths to data, or data if it fits in memory
            __len__: Length of dataset, often calculated from length of paths
            __getitem__: retrieve one sample and apply transforms if any (often ToTensor as model accepted input)

        This is a mock-example and not necessary if you can load all data in memory, but often with larger datasets the 
        class will fetch a path rather than the actual data.
        """

        def __init__(self, feature_vars, target_vars, transforms=None):
            self.feature_vars = feature_vars
            self.target_vars = target_vars
            self.transforms = transforms

        def __len__(self):
            # faster to get dataset length from target than data
            return len(self.target_vars)

        def __getitem__(self, idx):

            sample = self.feature_vars[idx, :]
            target = self.target_vars[idx]

            # Normally we include ToTensor in transforms.Compose()
            # ToTensor only accepts >2D data and our data is 1D dimensional feature vectors
            sample = torch.from_numpy(sample).float()
            target = torch.from_numpy(np.asarray(target)).float()

            if self.transforms:
                sample = self.transforms(sample)

            return sample, target

    # use compose to gather several transforms for our data
    # my_transforms = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.7317], std=[0.0836]),
    #     ]
    # )

    dataset = MyDataset(feature_vars=X, target_vars=y, transforms=None)

    # split into train and test datasets the pytorch way
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train, test = random_split(dataset, [train_size, test_size])

    return train, test  #  TODO remove, test_loader


###
# MODEL ARCHITECTURE
###


# creating a model:
# https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html


class BinaryClassifier(torch.nn.Module):
    def __init__(self, in_channels):
        super(BinaryClassifier, self).__init__()
        """
        This part only defines and initialise layers, they aren't used until the forward pass. 
        Backward pass is done automatically when calling <loss.backward>.

        A strategy for optimizing your network is to start with few layers and hidden channels (width), increase them 
        (increasing capacity of model) until able to overfit the training data, then try regularizers (dropout, early 
        stopping, batch normalisation) and different run options until getting a model that adequately learns from the 
        training data and generalises well to unseen data.

        ARGS:
            in_channels: the first layers input channels is the number of feature variables in our data = 30
            out_channels: number of classes we wish to predict binary=2
        """

        # these are layers of trainable weights of form:
        # y = x*A^T + b
        # where y is output, x is input, A is trainable weights and b is a constant bias.
        self.linear1 = torch.nn.Linear(in_channels, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 1)

        # ReLU is the most common activation. We use a non-linear activation after most layers
        # with trainable weights since it makes an otherwise linear weight able to make non-linear representations
        self.activation = torch.nn.ReLU()

    def forward(self, x):

        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))

        # With no final activation we would output "raw" predictions from (-inf, inf)
        # Normally we wish to squeeze these to same range as our target range, e.g. [0,1], so our model doesn't need to
        # learn this mapping on its own.
        # Our loss BCEWithLogits however includes a final activation sigmoid->[0,1] which is preferred since it uses a
        # trick that is more numerically stable than doing BCE+ sigmoid manually

        x = self.linear3(x)

        return x


class ImprovedBinaryClassifier(torch.nn.Module):
    def __init__(self, in_channels):
        super(ImprovedBinaryClassifier, self).__init__()
        """
        To improve on the barebones example you could experiment with:
            - run options & hyperparameters (epochs, learning rate, random seed)
            - number of layers
            - number of hidden channels
            - different optimizer
            - activation function (e.g. LeakyReLU)
            - add regularisers (dropout, batchnorm)

        ARGS:
            in_channels: the first layers input channels is the number of feature variables in our data = 30
            out_channels: number of classes we wish to predict binary=2
        """

        hidden_channels = [128, 256]

        # these are layers of trainable weights of form:
        # y = x*A^T + b
        # where y is output, x is input, A is trainable weights and b is a constant bias.
        self.linear1 = torch.nn.Linear(in_channels, hidden_channels[0])
        self.linear2 = torch.nn.Linear(hidden_channels[0], hidden_channels[1])
        self.linear3 = torch.nn.Linear(hidden_channels[1], 1)

        # ReLU is the most common activation. We use a non-linear activation after most layers
        # with trainable weights since it makes an otherwise linear weight able to make non-linear representations
        self.activation = torch.nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)

        self.batchnorm1 = nn.BatchNorm1d(hidden_channels[0])
        self.batchnorm2 = nn.BatchNorm1d(hidden_channels[1])

    def forward(self, x):

        x = self.batchnorm1(self.activation(self.linear1(x)))
        x = self.dropout(x)
        x = self.batchnorm2(self.activation(self.linear2(x)))
        x = self.dropout(x)

        # With no final activation we would output "raw" predictions from (-inf, inf)
        # Normally we wish to squeeze these to same range as our target range, e.g. [0,1], so our model doesn't need to
        # learn this mapping on its own.
        # Our loss BCEWithLogits however includes a final activation sigmoid->[0,1] which is preferred since it uses a
        # trick that is more numerically stable than doing BCE+ sigmoid manually

        x = self.linear3(x)

        return x


###
# FUNCTIONS
###


def train(train_loader, model, criterion, optimizer):
    """
    Train one epoch and return average epoch loss.

    We sometimes omit tracking training accuracy since a large enough model can always "remember" predictions for the 
    training data without actually learning any generalized representation.

    ARGS:
        train_loader: normal data
        model: 
        criterion:
        optimizer:
    OUT:
        train_loss: avg training epoch loss
    """

    # use trainmode to activate layers we only want during train procedure (dropout, batch_norm, etc.)
    model.train()

    batch_loss = 0.0
    batch_count = len(train_loader)  # number of total batches, not samples in a batch!

    for samples, targets in train_loader:

        # zero parameter gradients
        # we need to do this because we generally only want to update with the current gradients, but by default torch
        # accumulates/stores previous gradients too.
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(samples)  # logits ligger mellom (-inf, inf)

        # predictions will be [batch_size, 1] so we need to unsqueeze target from [batch_size,] -> [batch_size, 1]
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

        # multiply running loss by batch size since it by default uses mean reduction
        batch_loss += loss.item() * samples.size(0)

    # avg loss of 1 epoch => running_loss / batch_count
    train_loss = batch_loss / batch_count

    return train_loss


def validate(val_loader, model, criterion):
    """
    During validation we don't wish to change our network or update weights, only to send data through the forward pass,
    calculate validation accuracy, and use them to evalute our current model (weights). 

    ARGS:
        val_loader:
        model:
        criterion:
    OUT:
        val_loss: avg validation epoch loss
        val_acc: avg validation epoch accuracy
    """

    # turn off dropout, batch_norm, etc.
    model.eval()

    batch_loss = 0.0
    batch_count = len(val_loader)

    # acc metrics
    correct_count = 0
    total_count = 0

    # detach gradients from computational graph for faster computation during inference
    with torch.no_grad():

        for samples, targets in val_loader:

            # predictions will be [batch_size, 1] so we need to unsqueeze target from [batch_size,] -> [batch_size, 1]
            targets = targets.unsqueeze(1)

            outputs = model(samples)
            loss = criterion(outputs, targets)

            # multiply by num in batch since mse/bce-loss default uses mean reduction w.r.t. batch size
            batch_loss += loss.detach().item() * samples.size(0)

            # in python 1 evals True and 0 evals False
            predictions = outputs > 0.5

            correct_count += (predictions == targets).sum().item()
            total_count += samples.size(0)

    # avg loss of 1 epoch => running_loss / batch_count
    val_loss = batch_loss / batch_count

    # avg epoch accuracy
    val_acc = correct_count / total_count

    return val_loss, val_acc


###
# MAIN SCRIPT
###


def main():
    """
    Our main function should:
        - prepare datasets->dataloaders
        - Alternatively run train(forward + backward + optimize) and validate for 
    """

    train_dataset, test_dataset = get_dataset()
    in_channels = 30  # feature dimension of dataset

    # creating a dataloader:
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    # creating a dataloader when we already have a "dataset" might seem unnecessary but
    # we want to be able to easily split the data into batches and iterate over it

    # dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )

    model = BinaryClassifier(in_channels=in_channels)
    # model = ImprovedBinaryClassifier(in_channels=in_channels)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        train_loss = train(train_loader, model, criterion, optimizer)
        test_loss, test_acc = validate(test_loader, model, criterion)

        print(
            f"Epoch: [{epoch}/{epochs}], Train Loss: [{train_loss:.3f}], Test Loss: [{test_loss:.3f}], Test Acc: [{test_acc:.3f}]"
        )

    return


if __name__ == "__main__":
    main()
