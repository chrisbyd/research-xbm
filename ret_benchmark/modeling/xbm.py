# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch


class XBM:
    def __init__(self, cfg):
        self.K = cfg.XBM.SIZE
        self.feats = torch.zeros(self.K, cfg.MODEL.HEAD.DIM).cuda()
        self.targets = torch.zeros(self.K, cfg.DATA.LABEL_DIM).cuda()
        self.ptr = 0
        self.full = False

    @property
    def is_full(self):
        return self.full

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
            self.full = True
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size
