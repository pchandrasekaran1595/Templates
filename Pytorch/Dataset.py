""" Preferably do not include any computationally intensive operations within __getitem__. """

import torch
from torch.utils.data import Dataset

# ********************************************************************************************* #

"""
1. Used for Tabluar Data.
2. Use LongTensor for MultiClass Classification Problems.
"""

class DS_1(Dataset):
    def __init__(this, X=None, y=None, mode="train"):
        this.mode = mode
        this.X = X
        if mode == "train":
            this.y = y

    def __len__(this):
        return this.X.shape[0]

    def __getitem__(this, idx):
        if this.mode == "train":
            return torch.FloatTensor(this.X[idx]), torch.FloatTensor(this.y[idx])
            # return torch.FloatTensor(this.X[idx]), torch.LongTensor(this.y[idx])
        else:
            return torch.FloatTensor(this.X[idx])
        
# ********************************************************************************************* #

"""
1. Used for Images.
2. Assumes that images are stored as (N, H/W, W/H, C).
3. Preferably store images as 'uint8' and use the torch.ToTensor() transform.
4. Use LongTensor for MultiClass Classification Problems.
"""

class DS_2(Dataset):
    def __init__(this, X=None, y=None, transform=None, mode="train"):
        this.transform = transform
        this.mode = mode
        this.X = X
        if mode == "train":
            this.y = y

    def __len__(this):
        return this.X.shape[0]

    def __getitem__(this, idx):
        img = this.transform(this.X[idx])
        if this.mode == "train":
            return img, torch.FloatTensor(this.y[idx])
            # return img, torch.LongTensor(this.y[idx])
        else:
            return img
        
# ********************************************************************************************* #
