from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import pickle

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def data_provider(args, data_dir):


    # Dataset Loading
    # data_dir = "../nhps_data/pilotelevator-0.3/"

    train_data = pickle.load(open(data_dir + "train.pkl", 'rb'))
    test_data = pickle.load(open(data_dir + "test.pkl", 'rb'))

    def data_extract(data):
        seqs = data['seqs']
        seqs_obs = data['seqs_obs']
        marker_num = data['total_num']
        seqs_list = []
        seqs_obs_list = []
        for seq in seqs:
            event_list = []
            for event in seq:
                event_list.append([event['time_since_start'], event['type_event']])
            seqs_list.append(event_list)
        for seq in seqs_obs:
            event_list = []
            for event in seq:
                event_list.append([event['time_since_start'], event['type_event']])
            seqs_obs_list.append(event_list)
        return seqs_list, seqs_obs_list, marker_num


    train_full_seqs, train_obs_seqs, marker_num = data_extract(train_data)
    test_full_seqs, test_obs_seqs, _ = data_extract(test_data)


    def pad(x, max_len): # x为多条时间序列组成的list
        for ts in x:
            for i in range(max_len - len(ts)):
                ts.append([-1.0, -1])
        return torch.tensor(x)

    train_obs_seqs = pad(train_obs_seqs, args.max_seq_len)
    test_obs_seqs = pad(test_obs_seqs, args.max_seq_len)

    train_obs_dataset = MyDataset(train_obs_seqs)
    train_full_dataset = MyDataset(train_full_seqs)
    test_obs_dataset = MyDataset(test_obs_seqs)
    test_full_dataset = MyDataset(test_full_seqs)

    return train_obs_dataset, train_full_dataset, test_obs_dataset, test_full_dataset, marker_num