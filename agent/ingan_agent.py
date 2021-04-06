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

from graph.model.hourglass_generator import Generator
from graph.model.discrimanator import Discriminator
from graph.model.regressor import Regressor
from graph.loss.hourglass_loss import HourglassLoss as Loss
from data.dataset import INGAN_Dataset

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
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.04), ratio=(0.5, 1.5)),
        ])

        self.pretraining_step_size = self.config.pretraining_step_size
        self.batch_size = self.config.batch_size

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        self.dataset = INGAN_Dataset(self.config, self.torchvision_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        self.dataset_test = INGAN_Dataset(self.config, self.torchvision_transform, True)
        self.testloader = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        # define models
        self.generator = Generator().cuda()
        self.regressor = Regressor().cuda()
        self.discriminator = Discriminator().cuda()

        # define loss
        self.loss = Loss().cuda()

        # define lr
        self.lr = self.config.learning_rate

        # define optimizer
        self.opt = torch.optim.Adam([{'params': self.generator.parameters()},
                                     {'params': self.regressor.parameters()},],
                                    lr=self.lr)

        # define optimize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.8, cooldown=20)

        # initialize train counter
        self.epoch = 0
        self.accumulate_iter = 0
        self.total_iter = (len(self.dataset) + self.config.batch_size - 1) // self.config.batch_size

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.generator = nn.DataParallel(self.generator, device_ids=gpu_list)
        self.regressor = nn.DataParallel(self.regressor, device_ids=gpu_list)
        self.discriminator = nn.DataParallel(self.discriminator, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='Discriminator')
        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of generator parameters: {}'.format(count_model_prameters(self.generator)))
        print('Number of regressor parameters: {}'.format(count_model_prameters(self.regressor)))
        print('Number of discriminator parameters: {}'.format(count_model_prameters(self.discriminator)))

    def collate_function(self, samples):
        X = torch.cat([sample['X'].view(-1, 3, 512, 1024) for sample in samples], axis=0)
        target = torch.cat([sample['target'].view(-1, 1, 512, 512) for sample in samples], axis=0)
        height = torch.cat([sample['height'] for sample in samples], axis=0)

        return tuple([X, target, height])

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.regressor.load_state_dict(checkpoint['regressor_state_dict'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, 'discriminator.pth.tar')
            print("Loading checkpoint '{}'".format(filename))

            checkpoint = torch.load(filename)
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

            print("**First time to train**")

    def save_checkpoint(self, epoch):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                'checkpoint_{}.pth.tar'.format(epoch))

        state = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'regressor_state_dict': self.regressor.state_dict(),
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

        self.summary_writer.add_image(step + '/result 1', out[0], self.epoch)
        self.summary_writer.add_image(step + '/result 2', out[1], self.epoch)
        self.summary_writer.add_image(step + '/result 3', out[2], self.epoch)

        self.summary_writer.add_image(step + '/target 1', target[0], self.epoch)
        self.summary_writer.add_image(step + '/target 2', target[1], self.epoch)
        self.summary_writer.add_image(step + '/target 3', target[2], self.epoch)

    def train(self):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_by_epoch()

            if self.epoch > self.pretraining_step_size:
                self.save_checkpoint(self.config.checkpoint_file)

    def train_by_epoch(self):
        tqdm_batch = tqdm(self.dataloader, total=self.total_iter, desc="epoch-{}".format(self.epoch))

        avg_loss = AverageMeter()
        for curr_it, (X, target, height) in enumerate(tqdm_batch):
            self.generator.train()
            self.regressor.train()
            self.discriminator.eval()
            self.opt.zero_grad()

            X = X.cuda(async=self.config.async_loading)
            target = target.cuda(async=self.config.async_loading)
            height = height.cuda(async=self.config.async_loading)
            
            out, inter_out2, inter_out1 = self.generator(X)
            pred_h = self.regressor(out, inter_out1, inter_out2)

            feature_origin, feature_out, disc_out = self.discriminator(target, out)
            _, feature_inter2, disc_inter2 = self.discriminator(target, inter_out2)
            _, feature_inter1, disc_inter1 = self.discriminator(target, inter_out1)

            loss = self.loss([target, out, inter_out2, inter_out1],
                             [feature_origin, feature_out, feature_inter2, feature_inter1],
                             [disc_out, disc_inter2, disc_inter1],
                             [height, pred_h])

            loss.backward()
            self.opt.step()
            avg_loss.update(loss)

            if curr_it == 4:
                self.record_image(X, out, target)

        tqdm_batch.close()

        self.summary_writer.add_scalar('train/loss', avg_loss.val, self.epoch)
        self.scheduler.step(avg_loss.val)

        with torch.no_grad():
            tqdm_batch = tqdm(self.testloader,
                              total=(len(self.dataset_test) + self.config.batch_size - 1) // self.config.batch_size,
                              desc="epoch-{}".format(self.epoch))

            avg_loss = AverageMeter()
            for curr_it, (X, target, height) in enumerate(tqdm_batch):
                self.generator.eval()
                self.regressor.eval()
                self.discriminator.eval()

                X = X.cuda(async=self.config.async_loading)
                target = target.cuda(async=self.config.async_loading)
                height = height.cuda(async=self.config.async_loading)

                out, inter_out2, inter_out1 = self.generator(X)

                pred_h = self.regressor(out, inter_out1, inter_out2)

                feature_origin, feature_out, disc_out = self.discriminator(target, out)
                _, feature_inter2, disc_inter2 = self.discriminator(target, inter_out2)
                _, feature_inter1, disc_inter1 = self.discriminator(target, inter_out1)

                loss = self.loss([target, out, inter_out2, inter_out1],
                                 [feature_origin, feature_out, feature_inter2, feature_inter1],
                                 [disc_out, disc_inter2, disc_inter1],
                                 [height, pred_h])

                avg_loss.update(loss)

                if curr_it == 2:
                    self.record_image(X, out, target, 'test')

            tqdm_batch.close()

            self.summary_writer.add_scalar('eval/loss', avg_loss.val, self.epoch)
