import os
import random
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class INGAN_Dataset(Dataset):
    def __init__(self,config, torchvision_transform, is_test=False):
        self.root_dir = config.root_path
        self.config = config
        self.transform = torchvision_transform

        if is_test:
            tmp_list = open(os.path.join(self.root_dir, config.data_path, 'test_list.txt')).readlines()
        else:
            tmp_list = open(os.path.join(self.root_dir, config.data_path, 'train_list.txt')).readlines()
        self.data_list = [[i.split()[0] + '.png', float(i.split()[1])] for i in tmp_list]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_name = os.path.join(self.root_dir, self.config.data_path, 'discriminator_data', self.data_list[idx][0])
        target = Image.open(target_name)
        target = transforms.ToTensor()(target)

        data_name = os.path.join(self.root_dir, self.config.data_path, 'dataset', self.data_list[idx])
        data = Image.open(data_name)
        data = self.transform(data)
        if random.random() < 0.5:
            data = torch.roll(data, random.randint(10, 700), dims=2)

        height = np.array([self.data_list[idx][1]])
        
        return {'X': data, 'target': target, 'height': torch.from_numpy(height)}


class INGAN_DatasetV2(Dataset):
    def __init__(self, config, torchvision_transform, is_test=False):
        self.root_dir = config.root_path
        self.config = config
        self.transform = torchvision_transform

        if is_test:
            tmp_list = open(os.path.join(self.root_dir, config.data_path, 'test_list.txt')).readlines()
        else:
            tmp_list = open(os.path.join(self.root_dir, config.data_path, 'train_list.txt')).readlines()
        self.data_list = [[i.split()[0], float(i.split()[1])] for i in tmp_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_name = os.path.join(self.root_dir, self.config.data_path, 'discriminator_data',
                                   self.data_list[idx][0] + '.png')
        target = Image.open(target_name)
        target = transforms.ToTensor()(target)

        data_name = os.path.join(self.root_dir, self.config.data_path, 'dataset', self.data_list[idx][0] + '.png')
        data = Image.open(data_name)
        data = self.transform(data)
        
        corner = np.load(os.path.join(self.root_dir, self.config.data_path, 'corner', self.data_list[idx][0] + '.npy'))
        corner = Image.fromarray(corner)
        corner = transforms.Resize((1, 1024))(corner)
        corner = transforms.ToTensor()(corner)

        if random.random() < 0.5:
            r_size = random.randint(10, 700)
            data = torch.roll(data, r_size, dims=2)
            corner = torch.roll(corner, r_size, dims=2)

        height = np.array([self.data_list[idx][1]])

        return {'X': data, 'target': target, 'height': torch.from_numpy(height), 'corner': corner}


class INGAN_DatasetV3(Dataset):
    def __init__(self, config, torchvision_transform, is_test=False, is_pretrain=False):
        self.root_dir = config.root_path
        self.config = config
        self.transform = torchvision_transform
        self.is_pretrain = is_pretrain

        if is_test:
            tmp_list = open(os.path.join(self.root_dir, config.data_path, 'test_list.txt')).readlines()
        else:
            tmp_list = open(os.path.join(self.root_dir, config.data_path, 'train_list.txt')).readlines()
        self.data_list = [[i.split()[0], float(i.split()[1])] for i in tmp_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        flip, roll = random.random() < 0.5, random.random() < 0.5

        data_name = os.path.join(self.root_dir, self.config.data_path, 'dataset', self.data_list[idx][0] + '.png')
        data = Image.open(data_name)
        data = transforms.Resize((512, 1024))(data)

        corner = np.load(os.path.join(self.root_dir, self.config.data_path, 'corner', self.data_list[idx][0] + '.npy'))
        corner = Image.fromarray(corner)
        corner = transforms.Resize((1, 1024))(corner)

        if flip:
            data = transforms.RandomHorizontalFlip(p=1.0)(data)
            corner = transforms.RandomHorizontalFlip(p=1.0)(corner)

        data = transforms.ToTensor()(data)
        corner = transforms.ToTensor()(corner)

        if roll:
            r_size = random.randint(10, 700)
            data = torch.roll(data, r_size, dims=2)
            corner = torch.roll(corner, r_size, dims=2)
            
        data = transforms.RandomErasing(p=0.5, scale=(0.02, 0.04), ratio=(0.5, 1.5))(data)

        if not self.is_pretrain:
            target_name = os.path.join(self.root_dir, self.config.data_path, 'discriminator_data',
                                       self.data_list[idx][0] + '.png')
            target = Image.open(target_name)
            if flip:
                target = transforms.RandomHorizontalFlip(p=1.0)(target)
            target = transforms.ToTensor()(target)

            height = np.array([self.data_list[idx][1]])

            return {'X': data, 'floor': target, 'height': torch.from_numpy(height), 'corner': corner}

        return {'X': data, 'corner': corner}