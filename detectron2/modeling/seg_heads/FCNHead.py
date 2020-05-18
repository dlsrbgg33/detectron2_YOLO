# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn
import torch.nn.functional as F

from detectron2.utils import DeformConvWithOffset, ConvModule

import torch
import numpy as np
import pycocotools.mask as mask_util
import pdb
from detectron2.utils.registry import Registry
from typing import Dict
from detectron2.layers import Conv2d, ShapeSpec


SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
SEM_SEG_HEADS_REGISTRY.__doc__ = """
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""


def build_sem_seg_head(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)


@SEM_SEG_HEADS_REGISTRY.register()
class FCNHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`PanopticFPN`.
    It takes FPN features as input and merges information from all
    levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_channels = cfg.MODEL.SEM_SEG_HEAD.IN_CHANNELS
        self.out_channels = cfg.MODEL.SEM_SEG_HEAD.OUT_CHANNELS
        self.num_levels =  cfg.MODEL.SEM_SEG_HEAD.NUM_LEVELS
        self.num_classes =  cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES # 8
        self.ignore_label =  cfg.MODEL.SEM_SEG_HEAD.IGNORE_LABEL
        self.loss_weight =  cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.conv_cfg = None
        self.norm_cfg = None        # fmt: on

        self.deform_convs = nn.ModuleList()
        self.deform_convs.append(nn.Sequential( 
            DeformConvWithOffset(self.in_channels, self.in_channels, 
                                 kernel_size=3, padding=1),
            nn.GroupNorm(32, self.in_channels),
            nn.ReLU(inplace=True),
            DeformConvWithOffset(self.in_channels, self.out_channels,
                                 kernel_size=3, padding=1),
            nn.GroupNorm(32, self.out_channels),
            nn.ReLU(inplace=True),
            DeformConvWithOffset(self.out_channels, self.out_channels,
                                 kernel_size=3, padding=1),
            nn.GroupNorm(32, self.out_channels),
            nn.ReLU(inplace=True),            
            ))
        self.conv_pred = ConvModule(self.out_channels * 4, 
                                    self.num_classes, 1, 
                                    padding=0, 
                                    conv_cfg=self.conv_cfg, 
                                    norm_cfg=self.norm_cfg,
                                    activation=None)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


    def forward(self, inputs, targets):

        assert len(inputs) == self.num_levels
        fpn_px = []
        inputs = list(inputs.values())
        for i in range(self.num_levels):
            fpn_px.append(self.deform_convs[0](inputs[i]))
        # import pdb
        # pdb.set_trace()

        fpn_p2 = fpn_px[0]
        fpn_p3 = F.interpolate(fpn_px[1], None, 2, mode='bilinear', align_corners=False)
        fpn_p4 = F.interpolate(fpn_px[2], None, 4, mode='bilinear', align_corners=False)
        if fpn_p4.size(2) != fpn_p2.size(2):
            fpn_p4 = fpn_p4[:,:,:fpn_p2.size(2),:]
        fpn_p5 = F.interpolate(fpn_px[3], None, 8, mode='bilinear', align_corners=False)
        if fpn_p5.size(2) != fpn_p2.size(2):
            fpn_p5 = fpn_p5[:,:,:fpn_p2.size(2),:]
        feat = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5], dim=1)

        fcn_score = self.conv_pred(feat)
        fcn_output = self.upsample(fcn_score)
        if self.training:
            loss = self.loss(fcn_output, targets)
            return fcn_output, loss
        else:
            return fcn_output, {}
        

    def loss(self, segm_pred, segm_label):
        loss = dict()
        loss_segm = F.cross_entropy(segm_pred, segm_label, reduction='mean', ignore_index=255)
        loss['loss_segm'] = self.loss_weight * loss_segm
        return loss
