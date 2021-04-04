import os
from PIL import Image
import random
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class DiscriminatorDataset(Dataset):
    def __init__(self,config, torchvision_transform, is_test=False):
        self.root_dir = config.root_path
        self.config = config
        self.transform = torchvision_transform

        if is_test:
            tmp_list = open(os.path.join(self.root_dir, config.data_path, 'test_list.txt')).readlines()
        else:
            tmp_list = open(os.path.join(self.root_dir, config.data_path, 'train_list.txt')).readlines()
        self.data_list = [i.split()[0] + '.png' for i in tmp_list]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_name = os.path.join(self.root_dir, self.config.data_path, 'discriminator_data',
                                 self.data_list[idx])
        fdata_name = os.path.join(self.root_dir, self.config.data_path, 'discriminator_data',
                                  self.data_list[(idx+random.randint(5, 1000)) % len(self.data_list)])
        
        img = Image.open(data_name)
        fimg = Image.open(fdata_name)

        data1 = self.transform(img)
        data2 = self.transform(img)
        
        img = transforms.ToTensor()(img)
        fimg = transforms.ToTensor()(fimg)
        
        if torch.sum(data1) <= torch.sum(img)*.98 or torch.sum(data2) < torch.sum(img)*.98:
            print(torch.sum(data1))
            print(torch.sum(data2))
            print(torch.sum(img))

        return {'X': img, 'X1': data1, 'X2': data2, 'Xf': fimg}
