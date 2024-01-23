import torch.utils.data as tordata
import os.path as osp
import numpy as np
from torchvision.datasets.folder import pil_loader
import pandas as pd
import random


class BaseImageDataset(tordata.Dataset):
    def __init__(self, dataset_name, transforms=None):
        self.transforms = transforms
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'IEFP')
        df = pd.read_csv(osp.join(self.root, '{}.txt'.format(
            dataset_name)), header=None, index_col=False, sep=' ')
        self.data = df.values
        self.image_list = np.array([osp.join(self.root, x)
                                   for x in self.data[:, 1]])

    def __len__(self):
        return len(self.image_list)


class EvaluationImageDataset(BaseImageDataset):
    def __init__(self, dataset_name, transforms=None):
        super(EvaluationImageDataset, self).__init__(
            dataset_name, transforms=transforms)
        self.ids = self.data[:, 0].astype(int)
        self.classes = np.unique(self.ids)

    def __getitem__(self, index):
        img = pil_loader(self.image_list[index])
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.ids[index]
        return img, label


class TrainImageDataset(BaseImageDataset):
    def __init__(self, dataset_name, transforms=None):
        super(TrainImageDataset, self).__init__(
            dataset_name, transforms=transforms)
        self.ids = self.data[:, 0].astype(int)
        self.classes = np.unique(self.ids)
        self.ages = self.data[:, 2].astype(int)
        self.genders = self.data[:, 3].astype(int)
        self.races = self.data[:, 4].astype(int)

    def __getitem__(self, index):
        img = pil_loader(self.image_list[index])
        if self.transforms is not None:
            img = self.transforms(img)
        age = self.ages[index]
        """Custom dataset that includes face image age group, categorized by [0-12, 13-18, 19-25,
        26-35, 36-45, 46-55, 56-65, >= 66]. Extends torchvision.datasets.ImageFolder
        """
        if age <= 12:
            age_group = 0
        elif age <= 18:
            age_group = 1
        elif age <= 25:
            age_group = 2
        elif age <= 35:
            age_group = 3
        elif age <= 45:
            age_group = 4
        elif age <= 55:
            age_group = 5
        elif age <= 65:
            age_group = 6
        else:
            age_group = 7
        gender = self.genders[index]
        race = self.races[index]
        label = self.ids[index]
        return img, label, age_group, gender, race
