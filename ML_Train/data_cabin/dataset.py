import os
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
FILEPATH = "label_map.csv"
PATH = "./alldata/"

def get_train_val_test_loaders(task, batch_size, **kwargs):

    tr, va, te, _ = getAlldataset(task = task, **kwargs)
    tr_loader = DataLoader(tr, batch_size = batch_size, shuffle= True)
    va_loader = DataLoader(va, batch_size = batch_size, shuffle= False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader, tr.get_numeric_label

def getAlldataset(task = "default", **kwargs):
    tr = CabinsDataset("train", task, **kwargs)
    va = CabinsDataset("val", task, **kwargs)
    te = CabinsDataset("test", task, **kwargs)

    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)

    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)

    tr.X = tr.X.transpose(0, 3, 1, 2)
    va.X = va.X.transpose(0, 3, 1, 2)
    te.X = te.X.transpose(0, 3, 1, 2)

    return tr, va, te, standardizer

def resize(X):
    """
        resize according to a certain ratio:
    """

    raise NotImplementedError

class ImageStandardizer(object):
    """Standardize a batch of images to mean 0 and variance 1.

    The standardization should be applied separately to each channel.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """

    def __init__(self):
        """Initialize mean and standard deviations to None."""
        super().__init__()
        self.image_mean = None
        self.image_std = None

    def fit(self, X):
        """Calculate per-channel mean and standard deviation from dataset X."""
        # TODO: Complete this function
        self.image_mean = np.mean(X, axis = (0, 1, 2))
        self.image_std = np.std(X, axis = (0, 1, 2))

    def transform(self, X):
        """Return standardized dataset given dataset X."""
        # TODO: Complete this function

        newX = (X - self.image_mean)/self.image_std
        return newX



class CabinsDataset(Dataset):
    def __init__(self, partition, task="target", augment=False):
        '''
            Read in the data from the disk
        '''
    
        super().__init__()

        if partition not in ["train", "val", "test"]:
            raise ValueError("Partition {} does not exist".format(partition))

            
        seed = 0
        np.random.seed(seed) # set the seed for random
        torch.manual_seed(seed)
        random.seed = seed 
        self.task = task
        self.partition = partition
        self.metadata = pd.read_csv(FILEPATH)
        self.augment = augment

        
        if self.augment == False:
            self.metadata = pd.read_csv(FILEPATH)
            print('loading data from csv file')
        
        self.X, self.y = self._load_data()

    def _load_data(self):
        '''
            load_data from memory
        '''
        print("loading %s..." % self.partition)
        df = self.metadata[self.metadata.partition == self.partition]
        X, y = [], []
        for i, row in df.iterrows():
            label = row["numeric_label"]
            image = imread(os.path.join(PATH, row["filename"]))
            X.append(label)
            y.append(image)
        
        return np.array(X), np.array(y)

    def get_numeric_label(self):
        return self.y



if __name__ == "__main__":
    tr, va, te, standardizer = getAlldataset()