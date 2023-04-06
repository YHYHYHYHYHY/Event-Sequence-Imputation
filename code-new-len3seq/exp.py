import torch
import torch.nn as nn
import numpy as np
from Models import PILES
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pickle
import time
from data_provider import data_provider


class Exp():
    def __init__(self, args):
        self.args = args
        self.model = PILES(args).to(self.args.device)
        self.device = args.device


    def _getdata(self, data_dir):
        train_obs_dataset, train_full_dataset, test_obs_dataset, test_full_dataset, marker_num = data_provider(self.args, data_dir)
        # self.args.marker_num = marker_num
        train_obs_loader = DataLoader(train_obs_dataset, batch_size=self.args.batch_size, shuffle=True)
        train_full_loader = DataLoader(train_full_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_obs_loader = DataLoader(test_obs_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_full_loader = DataLoader(test_full_dataset, batch_size=self.args.batch_size, shuffle=True)
        return train_obs_loader, train_full_loader, test_obs_loader, test_full_loader

    def _rev(self, y):
        x = y.clone()
        batch_size = x.shape[0]
        for i in range(batch_size):
            seq = x[i]
            fin = 0
            for j in range(self.args.max_seq_len):
                if seq[j][1] < 0:
                    fin = j - 1
                    break
            for j in range(int(fin / 2) + 1):
                st = seq[j].clone()
                seq[j] = seq[fin - j]
                seq[fin - j] = st
        return x


    def train(self, data_dir):
        train_loader, _, _, _ = self._getdata(data_dir)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_rate)
        time_now = time.time()
        train_steps = len(train_loader)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_reverse = self._rev(batch_x)
                batch_x_reverse = batch_x_reverse.float().to(self.device)


                outputs = self.model(batch_x)
                outputs_rev = self.model(batch_x_reverse)
                loss = (outputs + outputs_rev).sum()
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))



