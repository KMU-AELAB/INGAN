import os
from PIL import Image
import random
import numpy as np

import torch
from torch.utils.data import Dataset


class DiscriminatorDataset(Dataset):
    def __init__(self,config, torchvision_transform, is_test=False):
        self.root_dir = config.root_path
        self.config = config
        self.transform = torchvision_transform

        if is_test:
            self.data_list = os.path.join(self.root_dir, config.data_path, 'test_list.txt')
        else:
            self.data_list = os.path.join(self.root_dir, config.data_path, 'train_list.txt')


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_name = os.path.join(self.root_dir, self.config.data_path, '', self.data_list[idx])
        img = Image.open(data_name)

        data1 = self.transform(img)
        data2 = self.transform(img)

        return {'X': torch.tensor(img), 'X1': data1, 'X2': data2}
