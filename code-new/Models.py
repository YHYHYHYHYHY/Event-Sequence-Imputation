import torch
import torch.nn as nn
import numpy as np
from models.Transformer import Transformer
import random

class Embedding(nn.Module):
    def __init__(self, d_model, m, beta, device, dropout=0.1): # m = # of event types
        super(Embedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.W_m = nn.Parameter(torch.randn((d_model, m)))
        self.W_t = nn.Parameter(torch.randn(d_model))
        self.b_t = nn.Parameter(torch.randn(d_model))
        self.beta = torch.tensor(beta)
        self.m = m
        self.d_model = d_model
        self.device = device

    def forward(self, x, is_reverse = False):
        seq_len = x.shape[0]
        for i in range(seq_len):
            t = x[i][0].item()
            marker = int(x[i][1].item())
            if t < 0:
                out = torch.zeros(self.d_model).to(self.device)
            else:
                one_hot = torch.zeros(self.m).to(self.device)
                one_hot[marker] = 1
                out = self.beta * torch.mv(self.W_m, one_hot) + (1 - self.beta) * (self.W_t * t + self.b_t)
            out = out.view(1, -1)
            if i == 0:
                ret = out
            else:
                ret = torch.cat((ret, out), dim=0)
        return ret


class PILES(nn.Module):
    def __init__(self, args):
        super(PILES, self).__init__()
        self.H_dim = args.H_dim # 过Transformer之后的维度
        self.d_emb = args.d_emb # 将(t, m) embedding 之后的维度
        self.marker_num = args.marker_num
        self.beta = args.beta
        self.dropout = args.dropout
        self.embedding = Embedding(self.d_emb, self.marker_num, self.beta, args.device, self.dropout).to(args.device)
        self.max_seq_len = args.max_seq_len
        self.linear1 = nn.Linear(2 * args.H_dim, args.marker_num) # 求lambda用的线性层
        self.mode = "argmax" # 默认在impute过程中采样marker选取概率最大的marker， 若为其他则根据softmax概率分布随机采样
        self.transformer = Transformer(args).to(args.device)
        self.cur_index = 0 # 目前正在进行impute的位置
        self.cur_timestamp = 0 # 目前最新的timestamp
        self.device = args.device

    def impute(self, H, F):

        emb_H = self.embedding(H)
        emb_F = self.embedding(F)
        H_mask = torch.zeros(self.max_seq_len)
        F_mask = torch.zeros(self.max_seq_len)
        for i in range(emb_H.shape[0]):
            H_mask[i] = 1
        for i in range(emb_F.shape[0]):
            if F[i][0].item() < 0:
                break
            F_mask[i] = 1

        emb_H = torch.cat((emb_H, torch.zeros(self.max_seq_len - emb_H.shape[0], self.d_emb).to(self.device)), 0).view(1, -1, self.d_emb)
        emb_F = torch.cat((emb_F, torch.zeros(self.max_seq_len - emb_F.shape[0], self.d_emb).to(self.device)), 0).view(1, -1, self.d_emb)

        h_hat = self.transformer(emb_H, H_mask)
        z_hat = self.transformer(emb_F, F_mask)
        x_hat = torch.concat((h_hat, z_hat), dim=1)
        lam = self.linear1(x_hat)
        lam = torch.softmax(lam, dim=1)
        if self.mode == "argmax":
            mu = torch.argmax(lam)
        else:
            mu = np.random.choice(np.arange(self.marker_num), p=lam.numpy())
        lam_val = lam[0][int(mu)]
        return torch.tensor(mu), lam_val


    def forward(self, x_batch): # x为装入batch的整个event-sequence x = [[[t0, m0], [t1, m1], ..., [tn, mn]], [[t0, m0], [t1, m1], ..., [tn, mn]], ... ,[[t0, m0], [t1, m1], ..., [tn, mn]]]
        for i in range(x_batch.shape[0]):
            x = x_batch[i]
            self.cur_index = 0
            self.cur_timestamp = x[0][0]
            l = x.shape[0]
            H = x[0].view(-1,2)
            F = x[1:]
            while self.cur_index < l - 1:
                marker, lam = self.impute(H, F)
                # lam.backward()  这里已确定lam可以进行backward
                # 选用指数分布进行delta_t的采样，采用重参数技巧使其能够进行反向传播
                epsilon = random.expovariate(1)
                delta_t = epsilon / lam
                if self.cur_timestamp + delta_t < F[0][0]: # 允许impute
                    self.cur_timestamp = self.cur_timestamp + delta_t

                    imp = torch.cat((self.cur_timestamp.view(-1), marker.view(-1)), dim=0).to(self.device)
                    H = torch.cat((H, imp.view(1, -1)), dim=0)
                else:
                    self.cur_index += 1
                    self.cur_timestamp = x[self.cur_index][0].item()
                    H = torch.cat((H, F[0].view(-1, 2)), 0)
                    F = F.cpu().numpy()
                    F = np.delete(F, 0, axis=0)
                    F = torch.tensor(F).to(self.device)
                if self.cur_timestamp < 0:
                    break
            pad_len = self.max_seq_len - H.shape[0]
            for j in range(pad_len):
                H = torch.cat((H, torch.tensor([[-1.0, -1]]).to(self.device)))
            H = H.view(1, -1, 2)
            if i == 0:
                ret = H
            else:
                ret = torch.cat((ret, H), dim=0)
        return ret



'''
def pad(x, max_len): # x为多条时间序列组成的list
    for ts in x:
        for i in range(max_len - len(ts)):
            ts.append([-1.0, -1])
    return torch.tensor(x)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')


    # model define
    parser.add_argument('--marker_num', type=int, default=8, help='# of distinct events')
    parser.add_argument('--beta', type=float, default=0.5, help='adjust the ratio at embedding')
    parser.add_argument('--d_emb', type=int, default=20, help='dimension after embedding (t, m) pair')

    # -----------------Transformer params---------------------------------------------
    parser.add_argument('--H_dim', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=500, help='maximum input size')
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
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.max_seq_len = 10

    x = [[[1.1, 0], [20.4, 3]], [[1.1, 0], [2.4, 4], [3.5, 7]]]
    x = pad(x, 5).to(args.device)


    model = PILES(args).to(args.device)
    params = model.parameters()

    y = model(x)



    print(y)

'''

