import os
import torch
import torch.optim as optim
import time
import numpy as np
from ret_benchmark.losses.registry import LOSS
from tqdm import tqdm
@LOSS.register("dch_loss")
class DCHLoss(torch.nn.Module):
    def __init__(self, config):
        super(DCHLoss, self).__init__()
        self.gamma = 20.0
        self.lambda1 = 0.1
        self.K = config.MODEL.HEAD.DIM
        self.one = torch.ones((config.DATA.TRAIN_BATCHSIZE, self.K)).cuda()

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, u, y, u1, y1):
        s = (y @ y1.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(u, u1)
        # formula 8
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        # formula 9
        quantization_loss = torch.log(1 + self.d(u.abs(), self.one) / self.gamma)
        # formula 7
        loss = cauchy_loss.mean() + self.lambda1 * quantization_loss.mean()

        return loss