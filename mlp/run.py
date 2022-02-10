# Original tutorial from:
# https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
from typing import Union
import numpy as np
import pandas as pd
import torch 

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch import nn

from torch.optim import SGD
from torch.nn import BCELoss

from addict import Dict
from common import CSVDataset


from torch.utils.tensorboard import SummaryWriter
# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html


class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs, last_layer_activation, enable_tracing=False):
        super(MLP, self).__init__()
        self.enable_tracing = enable_tracing
        # input to first hidden layer
        self.hidden1 = torch.nn.Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = torch.nn.ReLU()
        # second hidden layer
        self.hidden2 = torch.nn.Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = torch.nn.ReLU()
        # third hidden layer and output
        self.hidden3 = torch.nn.Linear(8, n_outputs)
        xavier_uniform_(self.hidden3.weight)

        #print(last_layer_activation)
        self.act3 = last_layer_activation
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X

def prepare_data(dataset: CSVDataset) -> Union[DataLoader, DataLoader]:
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

def train_model(train_dl, model, criterion=BCELoss(), num_epochs=200):
    # define the optimization
    #criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(num_epochs):
        # enumerate mini batches

        epoch_loss = 0
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad() # accumulation is done inside the optimizers
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            epoch_loss += loss
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

        # Only logging below:
        if epoch % 5 == 0:
            writer.add_scalar('training loss',
                            epoch_loss,
                            epoch)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss}")
    writer.close()

def evaluate_model(test_dl: Dataset, model: MLP):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
# %%
# prepare the data
id = 'ionosphere'
writer = SummaryWriter(f'runs/zerotest_{id}') # default `log_dir` is "runs" - we'll be more specific here

print(f"Problem: {id}")
PROBLEM = dict()
PROBLEM["iris"] = Dict({"path": 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv',
                    "activation": nn.Softmax(dim=1),
                    "criterion": nn.CrossEntropyLoss(),
                    "n_outputs": 3})
PROBLEM["ionosphere"] =  Dict({"path": 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv',
                          "activation": nn.Sigmoid(),
                          "criterion": nn.BCELoss(),
                          "n_outputs": 1})


# load the dataset
dataset = CSVDataset(PROBLEM[id].path)

print(PROBLEM[id].n_outputs)
print(PROBLEM[id].criterion)
print(dataset.df.shape)

n_features = dataset.df.shape[1] - 1 # subtracting one as the last is input col
print(n_features)

dataset.df.sample(5)

# %%
model = MLP(n_features,
            PROBLEM[id].n_outputs,
            PROBLEM[id].activation)

#writer.add_graph(model) # https://pytorch.org/docs/stable/tensorboard.html
#writer.close()

# %%
train_dl, test_dl = prepare_data(dataset)
train_model(train_dl,
            model,
            PROBLEM[id].criterion)

print("Done")
#evaluate_model(train_dl, model)

# %%
