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

from graph.model.discrimanator import Discriminator as Model
from graph.loss.discriminator_loss import DiscriminatorLoss as Loss
from data.discriminator_dataset import DiscriminatorDataset

from utils.metrics import AverageMeter
from utils.train_utils import set_logger, count_model_prameters


cudnn.benchmark = True


class DiscriminatorAgent(object):
    def __init__(self, config):
        self.config = config
        self.flag_gan = False
        self.train_count = 0

        self.torchvision_transform = transforms.Compose([
            transforms.RandomRotation((-4, 4), fill='black'),
            transforms.ToTensor(),
        ])

        self.pretraining_step_size = self.config.pretraining_step_size
        self.batch_size = self.config.batch_size

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        self.dataset = DiscriminatorDataset(self.config, self.torchvision_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        self.dataset_test = DiscriminatorDataset(self.config, self.torchvision_transform, True)
        self.testloader = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=1,
                                     pin_memory=self.config.pin_memory, collate_fn=self.collate_function)

        # define models
        self.model = Model().cuda()

        # define loss
        self.loss = Loss().cuda()

        # define lr
        self.lr = self.config.learning_rate

        # define optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
        self.model = nn.DataParallel(self.model, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='Discriminator')
        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of model parameters: {}'.format(count_model_prameters(self.model)))

    def collate_function(self, samples):
        X = torch.cat([sample['X'].view(-1,1,512,512) for sample in samples], axis=0)
        X1 = torch.cat([sample['X1'].view(-1,1,512,512) for sample in samples], axis=0)
        X2 = torch.cat([sample['X2'].view(-1,1,512,512) for sample in samples], axis=0)
        Xf = torch.cat([sample['Xf'].view(-1,1,512,512) for sample in samples], axis=0)

        return tuple([X, X1, X2, Xf])

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.model.load_state_dict(checkpoint['discriminator_state_dict'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, epoch):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                'checkpoint_{}.pth.tar'.format(epoch))

        state = {
            'discriminator_state_dict': self.model.state_dict(),
        }

        torch.save(state, tmp_name)
        shutil.copyfile(tmp_name, os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                               self.config.checkpoint_file))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def record_image(self, X, X1, X2, Xf):
        self.summary_writer.add_image('origin/img 1', X[0], self.epoch)
        self.summary_writer.add_image('origin/img 2', X[1], self.epoch)

        self.summary_writer.add_image('variation_1/img 1', X1[0], self.epoch)
        self.summary_writer.add_image('variation_1/img 2', X1[1], self.epoch)

        self.summary_writer.add_image('variation_2/img 1', X2[0], self.epoch)
        self.summary_writer.add_image('variation_2/img 2', X2[1], self.epoch)

        self.summary_writer.add_image('negative/img 1', Xf[0], self.epoch)
        self.summary_writer.add_image('negative/img 2', Xf[1], self.epoch)




    def train(self):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_by_epoch()

            if self.epoch > self.pretraining_step_size:
                self.save_checkpoint(self.config.checkpoint_file)

    def train_by_epoch(self):
        tqdm_batch = tqdm(self.dataloader, total=self.total_iter, desc="epoch-{}".format(self.epoch))

        avg_loss = AverageMeter()
        for curr_it, (X, X1, X2, Xf) in enumerate(tqdm_batch):
            self.model.train()
            self.opt.zero_grad()
            
            X = X.cuda(async=self.config.async_loading)
            X1 = X1.cuda(async=self.config.async_loading)
            X2 = X2.cuda(async=self.config.async_loading)
            Xf = Xf.cuda(async=self.config.async_loading)

            out_origin = self.model(X)
            out_var1 = self.model(X1)
            out_var2 = self.model(X2)
            out_f = self.model(Xf)

            loss = self.loss(out_origin, out_var1, out_var2, out_f)

            loss.backward()
            self.opt.step()
            avg_loss.update(loss)

            if curr_it == 4:
                self.record_image(X, X1, X2, Xf)

        tqdm_batch.close()

        self.summary_writer.add_scalar('train/loss', avg_loss.val, self.epoch)
        self.scheduler.step(avg_loss.val)

        with torch.no_grad():
            tqdm_batch = tqdm(self.testloader,
                              total=(len(self.dataset_test) + self.config.batch_size - 1) // self.config.batch_size,
                              desc="epoch-{}".format(self.epoch))

            avg_loss = AverageMeter()
            for curr_it, (X, X1, X2, Xf) in enumerate(tqdm_batch):
                self.model.eval()

                X = X.cuda(async=self.config.async_loading)
                X1 = X1.cuda(async=self.config.async_loading)
                X2 = X2.cuda(async=self.config.async_loading)
                Xf = Xf.cuda(async=self.config.async_loading)

                out_origin = self.model(X)
                out_var1 = self.model(X1)
                out_var2 = self.model(X2)
                out_f = self.model(Xf)

                loss = self.loss(out_origin, out_var1, out_var2, out_f)
                avg_loss.update(loss)

            tqdm_batch.close()

            self.summary_writer.add_scalar('eval/loss', avg_loss.val, self.epoch)
