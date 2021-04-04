import os
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
        self.data_list = [i.split()[0] + '.png' for i in tmp_list]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_name = os.path.join(self.root_dir, self.config.data_path, 'discriminator_data', self.data_list[idx])
        target = Image.open(target_name)
        target = transforms.ToTensor()(target)

        data_name = os.path.join(self.root_dir, self.config.data_path, 'dataset', self.data_list[idx])
        data = Image.open(data_name)
        data = self.transform(data)
        
        return {'X': data, 'target': target}
