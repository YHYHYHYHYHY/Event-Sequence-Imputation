import torch
import torch.nn as nn
import numpy as np
from Models import PILES
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
import time
import argparse
from exp import Exp

parser = argparse.ArgumentParser(description='PILES')

parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')


# model define
parser.add_argument('--marker_num', type=int, default=10, help='# of distinct events')
parser.add_argument('--beta', type=float, default=0.5, help='adjust the ratio at embedding')
parser.add_argument('--d_emb', type=int, default=20, help='dimension after embedding (t, m) pair')
parser.add_argument('--f', type=float, default=2.0, help='The multiple of delta_t expansion ')

# -----------------Transformer params---------------------------------------------
parser.add_argument('--H_dim', type=int, default=32)
parser.add_argument('--max_seq_len', type=int, default=500, help='maximum input size')
parser.add_argument('--mid_seq_len', type=int, default=3, help='transformer input length')
parser.add_argument('--enc_in', type=int, default=20, help='encoder input size')
# parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')

#parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
#parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--device', type=str, default='cuda')

# optimization
# parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
# parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
# parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--lr_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
# parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
# parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

args = parser.parse_args()
args.max_seq_len = 700
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.f = 2.0

Exp = Exp(args)
Exp.train("../nhps_data/pilotelevator-0.3/")


def getlen(x):
    for j in range(x.shape[0]):
        for i in range(x.shape[1]):
            if(x[j][i][1].item()<0):
                print(i)
                break







