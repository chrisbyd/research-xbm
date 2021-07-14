import torch
from torch import nn

from ret_benchmark.modeling.registry import HEADS
from ret_benchmark.utils.init_methods import weights_init_kaiming


@HEADS.register("linear_norm")
class LinearNorm(nn.Module):
    def __init__(self, cfg):
        super(LinearNorm, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(cfg.MODEL.HEAD.IN_CHANNELS, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, cfg.MODEL.HEAD.DIM),
            nn.Tanh()
        )
        
        self.module.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.module(x)
        return x
