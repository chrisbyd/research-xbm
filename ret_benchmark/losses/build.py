# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from .multi_similarity_loss import MultiSimilarityLoss
from .contrastive_loss import ContrastiveLoss
from .triplet_loss import TripletLoss
from .registry import LOSS
from .dch_loss import DCHLoss

def build_loss(cfg):
    loss_name = cfg.LOSSES.NAME
    assert loss_name in LOSS, f"loss name {loss_name} is not registered in registry"
    return LOSS[loss_name](cfg)
