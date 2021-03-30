import os
from PIL import Image
import random
import numpy as np

import torch
from torch.utils.data import Dataset


class DiscriminatorDataset(Dataset):
    def __init__(self,config, torchvision_transform):
        self.root_dir = config.root_path
        self.data_list = os.listdir(os.path.join(self.root_dir, config.data_path))
        self.config = config
        self.transform = torchvision_transform

        self.ratio_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                           2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.6, 4.7,
                           4.9, 5.0, 5.2, 5.3, 5.4, 5.5, 5.7, 5.9, 6.0, 6.3, 6.5, 6.9, 7.2, 7.3, 7.4, 7.6, 7.7, 7.9,
                           8.0, 8.1, 8.2, 8.4, 8.6, 8.7, 8.8, 8.9, 9.1, 9.5, 9.6, 9.8, 10.0, 10.1, 10.2, 10.3, 10.4,
                           10.6, 10.7, 11.0, 12.1, 12.6, 13.0, 13.1, 13.7, 17.1, 20.2, 22.6]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_name = os.path.join(self.root_dir, self.config.data_path, self.data_list[idx])
        img = Image.open(data_name)

        data = self.transform(img)

        rand_v = random.random()
        if rand_v > 0.3:
            dx = np.random.randint(int(1024 * rand_v))
            data = torch.roll(data, dx, 2)

        return {'X': data}
