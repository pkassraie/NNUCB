from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

class data_loader:
    def __init__(self, name, is_shuffled=True, seed=None, underrep= False, train = True):
        # Fetch data
        if name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        else:
            raise RuntimeError('Dataset does not exist')

        # train/test split
        len = X.shape[0]
        if train:
            X = X[0:int(len*0.8)]
            y = y[0:int(len*0.8)]
        else:
            X = X[int(len*0.8):]
            y = y[int(len*0.8):]

        # Shuffle data
        if is_shuffled:
            self.X, self.y = shuffle(X, y, random_state=seed)
        else:
            self.X, self.y = X, y

        if underrep:
            self.y = self.y.astype(np.int64)
            class1 = self.X[np.where((self.y==0))]
            y1 = self.y[np.where((self.y == 0))]
            class2 = self.X[np.where((self.y==1))]
            y2 = self.y[np.where((self.y == 1))]
            class1 = class1[0:int(y1.shape[0]/20)]
            y1 = y1[0:int(y1.shape[0]/20)]
            self.X = np.concatenate((class1, class2))
            self.y = np.concatenate((y1, y2))
            self.X, self.y = shuffle(self.X, self.y, random_state = seed)

        # generate one_hot coding:
        self.y_arm = OrdinalEncoder(
            dtype=np.int).fit_transform(self.y.reshape((-1, 1)))

        # set cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]
        self.contextdim = self.X.ndim - 1

    def step(self): # reads one data point and presents it to the learner
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                self.act_dim] = self.X[self.cursor]
        arm = self.y_arm[self.cursor][0]
        rwd = np.zeros((self.n_arm,))
        rwd[arm] = 1
        self.cursor += 1
        return X, rwd

    def finish(self):
        return self.cursor == self.size

    def reset(self):
        self.cursor = 0

class data_loader_ambig(data_loader):
    def __init__(self, name, is_shuffled=True, seed=None, train=True):
        if name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        else:
            raise RuntimeError('Dataset does not exist')

        # add ambiguity information
        inds = np.arange(y.shape[0])
        ambig_ind = np.array(pd.read_csv('data/ambig_indices.csv')['0'])

        for i in inds:
            if i in ambig_ind:
                inds[i] = 1
            else:
                inds[i] = 0

        # train/test split
        len = X.shape[0]
        if train:
            X = X[0:int(len*0.8)]
            y = y[0:int(len*0.8)]
            inds = inds[0:int(len*0.8)]
        else:
            X = X[int(len*0.8):]
            y = y[int(len*0.8):]
            inds = inds[int(len*0.8):]

        # Shuffle data
        if is_shuffled:
            self.X, self.y, self.ambig = shuffle(X, y, inds, random_state=seed)
        else:
            self.X, self.y, self.ambig = X, y, inds

        # generate one_hot coding:
        self.y_arm = OrdinalEncoder(
            dtype=np.int).fit_transform(self.y.reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] * self.n_arm
        if name == 'CIFAR_10_small':
            self.channels = 3
        else:
            self.channels = 1
        self.act_dim = self.X.shape[1]
        self.contextdim = self.X.ndim-1

    def step(self):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                self.act_dim] = self.X[self.cursor]
        ambig = self.ambig[self.cursor]
        arm = self.y_arm[self.cursor][0]
        rwd = np.zeros((self.n_arm,))
        rwd[arm] = 1
        self.cursor += 1
        return X, rwd, ambig

