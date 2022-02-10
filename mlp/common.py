from typing import Union
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = pd.read_csv(path, header=None)

        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        num_labels = len(np.unique(self.y))

        # TODO: replace this with OneHotEncoder:
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        self.y = LabelEncoder().fit_transform(self.y)

        if num_labels == 2: # BCELoss
            self.y = self.y.astype('float32')
            print("Reshaping to 2D array")
            self.y = self.y.reshape((len(self.y), 1))

        elif num_labels>2: # CrossEntropyLoss (with class indices, see pytorch doc https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 
            self.y = self.y.astype(np.int64) # dtypes: https://github.com/wkentaro/pytorch-for-numpy-users <3

        self.df = df  # so that I can view it

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return torch.utils.data.random_split(self, [train_size, test_size])
