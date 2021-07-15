import torch.nn as nn
from torchvision import models
from ret_benchmark.modeling.registry import HEADS
from ret_benchmark.utils.init_methods import weights_init_kaiming


@HEADS.register("alexhead")
class AlexHead(nn.Module):
    def __init__(self, cfg):
        super(AlexHead,self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias
        self.module =  nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, cfg.MODEL.HEAD.DIM),
        )
        self.module.apply(weights_init_kaiming)
    
    def forward(self,x):
        x = self.module(x)
        return x





