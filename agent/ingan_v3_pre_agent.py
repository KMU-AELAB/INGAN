import os
import shutil
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter

from graph.model.horizon_base import HorizonBase, Corner
from graph.loss.horizon_base_loss import HorizonBaseLoss as Loss
from data.dataset import INGAN_DatasetV3 as INGAN_Dataset

from utils.metrics import AverageMeter
from utils.train_utils import set_logger, count_model_prameters

cudnn.benchmark = True


class INGANAgent(object):
    def __init__(self, config):
        self.config = config
        self.flag_gan = False
        self.train_count = 0

        self.torchvision_transform = transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.04), ratio=(0.5, 1.5)),
        ])

        self.pretraining_step_size = self.config.pretraining_step_size
        self.batch_size = self.config.batch_size

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        self.dataset = INGAN_Dataset(self.config, self.torchvision_transform, 'corner_train_list.txt', True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        self.dataset_test = INGAN_Dataset(self.config, self.torchvision_transform, 'test_list.txt', True)
        self.testloader = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        # define models
        self.feature = HorizonBase().cuda()
        self.corner = Corner().cuda()

        # define loss
        self.loss = Loss().cuda()

        # define lr
        self.lr = self.config.learning_rate

        # define optimizer
        self.opt = torch.optim.Adam([{'params': self.feature.parameters()},
                                     {'params': self.corner.parameters()},],
                                    lr=self.lr)

        # define optimize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.8, cooldown=20)

        # initialize train counter
        self.epoch = 265
        self.accumulate_iter = 0
        self.total_iter = (len(self.dataset) + self.config.batch_size - 1) // self.config.batch_size

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.feature = nn.DataParallel(self.feature, device_ids=gpu_list)
        self.corner = nn.DataParallel(self.corner, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='Discriminator')
        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of feature parameters: {}'.format(count_model_prameters(self.feature)))
        print('Number of corner parameters: {}'.format(count_model_prameters(self.corner)))

    def collate_function(self, samples):
        X = torch.cat([sample['X'].view(-1, 3, 512, 1024) for sample in samples], axis=0)
        corner = torch.cat([sample['corner'].view((1, 1, 1024)) for sample in samples], axis=0)

        return tuple([X, corner])

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.feature.load_state_dict(checkpoint['feature_state_dict'])
            self.corner.load_state_dict(checkpoint['corner_state_dict'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, epoch):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                'checkpoint_{}.pth.tar'.format(epoch))

        state = {
            'feature_state_dict': self.feature.state_dict(),
            'corner_state_dict': self.corner.state_dict(),
        }

        torch.save(state, tmp_name)
        shutil.copyfile(tmp_name, os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                               self.config.checkpoint_file))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def record_image(self, X, out, target, step='train'):
        self.summary_writer.add_image(step + '/img 1', X[0], self.epoch)
        self.summary_writer.add_image(step + '/img 2', X[1], self.epoch)
        self.summary_writer.add_image(step + '/img 3', X[2], self.epoch)

        self.summary_writer.add_image(step + '/result 1', out[0].view(1, 1024, 1).permute(0, 2, 1).repeat(1, 200, 1),
                                      self.epoch)
        self.summary_writer.add_image(step + '/result 2', out[1].view(1, 1024, 1).permute(0, 2, 1).repeat(1, 200, 1),
                                      self.epoch)
        self.summary_writer.add_image(step + '/result 3', out[2].view(1, 1024, 1).permute(0, 2, 1).repeat(1, 200, 1),
                                      self.epoch)

        self.summary_writer.add_image(step + '/target 1', target[0].view(1, 1024, 1).permute(0, 2, 1).repeat(1, 200, 1),
                                      self.epoch)
        self.summary_writer.add_image(step + '/target 2', target[1].view(1, 1024, 1).permute(0, 2, 1).repeat(1, 200, 1),
                                      self.epoch)
        self.summary_writer.add_image(step + '/target 3', target[2].view(1, 1024, 1).permute(0, 2, 1).repeat(1, 200, 1),
                                      self.epoch)

    def train(self):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_by_epoch()

            if self.epoch > self.pretraining_step_size:
                self.save_checkpoint(self.config.checkpoint_file)

    def train_by_epoch(self):
        tqdm_batch = tqdm(self.dataloader, total=self.total_iter, desc="epoch-{}".format(self.epoch))

        avg_loss = AverageMeter()
        for curr_it, (X, corner) in enumerate(tqdm_batch):
            self.feature.train()
            self.corner.train()
            self.opt.zero_grad()

            X = X.cuda(async=self.config.async_loading)
            corner = corner.cuda(async=self.config.async_loading)
            
            feat = self.feature(X)
            pred_cor = self.corner(feat)
            
            loss = self.loss(corner, pred_cor)

            loss.backward()
            self.opt.step()
            avg_loss.update(loss)

            if curr_it == 4:
                self.record_image(X, pred_cor, corner)

        tqdm_batch.close()

        self.summary_writer.add_scalar('train/loss', avg_loss.val, self.epoch)
        self.scheduler.step(avg_loss.val)

        with torch.no_grad():
            tqdm_batch = tqdm(self.testloader,
                              total=(len(self.dataset_test) + self.config.batch_size - 1) // self.config.batch_size,
                              desc="epoch-{}".format(self.epoch))

            avg_loss = AverageMeter()
            for curr_it, (X, corner) in enumerate(tqdm_batch):
                self.feature.eval()
                self.corner.eval()

                X = X.cuda(async=self.config.async_loading)
                corner = corner.cuda(async=self.config.async_loading)

                feat = self.feature(X)
                pred_cor = self.corner(feat)

                loss = self.loss(corner, pred_cor)

                avg_loss.update(loss)

                if curr_it == 2:
                    self.record_image(X, pred_cor, corner, 'test')

            tqdm_batch.close()

            self.summary_writer.add_scalar('eval/loss', avg_loss.val, self.epoch)
