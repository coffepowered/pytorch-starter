import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import random_split

from sklearn.preprocessing import OneHotEncoder


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path: str):
        # load the csv file as a dataframe
        df = pd.read_csv(path, header=None)

        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype("float32")
        self.y = (
            OneHotEncoder(sparse=False, drop="if_binary")
            .fit_transform(self.y.reshape(-1, 1))
            .astype("float32")
        )

        # Now works also with CE loss using class probabilities instead of indices
        # CrossEntropyLoss (see diff wrt class indices class indices, see pytorch doc https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

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
        return random_split(self, [train_size, test_size])
