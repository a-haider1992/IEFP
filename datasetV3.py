## Date: 06 Feb 2023

import torch.utils.data as tordata
import os.path as osp
import os
import pandas
from torchvision.datasets.folder import pil_loader
import numpy as np
import pdb
from PIL import Image

def find_image_paths(images, image_num1, image_num2):
    for image in images:
        parts = image.split('_')
        num = parts[len(parts)-1]
        image_num1 = image_num2 = ''
        if image_num1 in num:
            for idx in range(0, len(parts)-1):
                image_num1 += parts[idx]+'_'
            image_num1 += num
        if image_num2 in num:
            for idx in range(0, len(parts)-1):
                image_num2 += parts[idx]+'_'
            image_num2 += num
    return image_num1, image_num2

class LFWDataset(tordata.Dataset):
    def __init__(self, file, mode=0, transform=None):
        # 0: same
        # 1: not same
        # pdb.set_trace()
        self.transform = transform
        self.root = osp.join(osp.dirname(__file__), 'face_dataset', 'LFW', 'lfw')
        df = pandas.read_csv(osp.join(osp.dirname(__file__), 'face_dataset', 'LFW', file), delimiter=',')
        if mode==0:
            self.names = df['name'].to_list()
            image1 = df['imagenum1'].to_list()
            image2 = df['imagenum2'].to_list()
            self.images = {}
            self.classes = self.names
            for idx in range(0, len(image1)):
                images = os.listdir(os.path.join(self.root, self.names[idx]))
                im1, im2 = find_image_paths(images, str(image1[idx]), str(image2[idx]))
                self.images[idx] = [os.path.join(self.root, self.names[idx], im1), os.path.join(self.root, self.names[idx], im2)]
        elif mode==1:
            names1 = df['name1'].to_list()
            names2 = df['name2'].to_list()
            image1 = df['imagenum1'].to_list()
            image2 = df['imagenum2'].to_list()
            self.classes = names1
            self.images = {}
            for idx in range(0, len(names1)):
                name_1_img = os.listdir(os.path.join(self.root, names1[idx]))[0]
                name_2_img = os.listdir(os.path.join(self.root, names2[idx]))[0]
                self.images[idx] = [os.path.join(self.root, names1[idx], name_1_img), os.path.join(self.root, names2[idx], name_2_img)]
        # for idx in range(0, len(images_)):
        #     self.images.append(os.path.join(self.root, self.names[idx], images_[idx]))
        # self.labels = df['encoded_name'].astype(int).to_list()
        # self.classes = np.unique(self.labels)
        
    def __getitem__(self, index):
        image1 = Image.open(self.images[index][0]).convert('RGB')
        image2 = Image.open(self.images[index][1]).convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2
    def __len__(self):
        return len(self.images)
    

class Casia(tordata.Dataset):
    def __init__(self, file, transform=None):
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', 'casia-webface')
        df = pandas.read_csv(osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', file), delimiter=',')
        image1 = df['img_num1'].to_list()
        image2 = df['img_num2'].to_list()
        self.images = {}
        self.classes = image1
        for idx in range(0, len(image1)):
            img1 = image1[idx].split("\\")
            img2 = image2[idx].split("\\")
            self.images[idx] = [os.path.join(self.root, img1[0], img1[1]), os.path.join(self.root, img2[0], img2[1])]
        # for idx in range(0, len(images_)):
        #     self.images.append(os.path.join(self.root, self.names[idx], images_[idx]))
        # self.labels = df['encoded_name'].astype(int).to_list()
        # self.classes = np.unique(self.labels)
        
    def __getitem__(self, index):
        image1 = pil_loader(self.images[index][0])
        image2 = pil_loader(self.images[index][1])
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2
    def __len__(self):
        return len(self.images)

class TrainingData(LFWDataset):
    def __init__(self, file, mode, transform):
        super().__init__(file, mode, transform)

    def __getitem__(self, index):
        return super().__getitem__(index)

class EvaluationData(LFWDataset):
    def __init__(self, file, transform):
        super().__init__(file, transform)
    def __getitem__(self, index):
        return super().__getitem__(index)
    
class AgeDB(tordata.Dataset):
    def __init__(self, file, transform=None):
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', 'AgeDB')
        df = pandas.read_csv(osp.join(osp.dirname(osp.dirname(__file__)), 'dataset', file), delimiter=',')
        image1 = df['img_num1'].to_list()
        image2 = df['img_num2'].to_list()
        age_gap = df['age_gap'].astype(np.float32).to_list()
        self.images = {}
        self.classes = image1
        for idx in range(0, len(image1)):
            self.images[idx] = [os.path.join(self.root,image1[idx]), 
                                os.path.join(self.root,image2[idx]), age_gap[idx]]
        # self.images = []
        # for path in self.paths:
        #     path = os.path.join(self.root, path)
        #     self.images.append(path)
        
    def __getitem__(self, index):
        image1 = pil_loader(self.images[index][0])
        image2 = pil_loader(self.images[index][1])
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        age_gap = self.images[index][2]
        return image1, image2
    def __len__(self):
        return len(self.images)
    
class UTK(tordata.Dataset):
    def __init__(self, file, transform=None):
        self.transform = transform
        self.root = osp.join(osp.dirname(osp.dirname(__file__)), 'face_dataset', 'UTK')
        df = pandas.read_csv(osp.join(osp.dirname(osp.dirname(__file__)), 'face_dataset', file), delimiter=',')
        self.paths = df['path'].to_list()
        self.ages = df['age'].to_list()
        self.genders = df['gender'].to_list()
        self.races = df['race'].to_list()
        self.images = []
        self.classes = self.paths
        for path in self.paths:
            path = os.path.join(self.root, path)
            self.images.append(path)
        
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        age = self.ages[index]
        gender = self.genders[index]
        race = self.races[index]
        return image, age, gender, race
    def __len__(self):
        return len(self.images)

class TrainingDataAge(AgeDB):
    def __init__(self, file, transform):
        super().__init__(file, transform)

    def __getitem__(self, index):
        return super().__getitem__(index)
    
class EvaluationDataAge(AgeDB):
    def __init__(self, file, transform):
        super().__init__(file, transform)

    def __getitem__(self, index):
        return super().__getitem__(index)