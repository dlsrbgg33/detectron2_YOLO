# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import xavier_init

from ..plugins import NonLocal2D, GeneralizedAttention, AsyncNonLocal2D, GlobalNonLocal2D
from detectron2.utils import ConvModule
from detectron2.utils.registry import Registry
from typing import Dict
from detectron2.layers import Conv2d, ShapeSpec
import pdb


EXTRA_NECK_REGISTRY = Registry("EXTRA_NECK")
EXTRA_NECK_REGISTRY.__doc__ = """
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""

def build_extra_neck(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.EXTRA_NECK.NAME
    return EXTRA_NECK_REGISTRY.get(name)(cfg, input_shape)


@EXTRA_NECK_REGISTRY.register()
class BFP(nn.Module):
    """
    A semantic segmentation head described in :paper:`PanopticFPN`.
    It takes FPN features as input and merges information from all
    levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        in_strides = {k: v.stride for k, v in input_shape.items()}
        self.in_channels = cfg.MODEL.EXTRA_NECK.IN_CHANNELS
        self.num_levels = cfg.MODEL.EXTRA_NECK.NUM_LEVELS
        self.conv_cfg = None
        self.norm_cfg = None

        self.refine_level = cfg.MODEL.EXTRA_NECK.REFINE_LEV
        self.refine_type = cfg.MODEL.EXTRA_NECK.REFINE_TYPE
        self.stack_type = 'add'

        self.f_name = list(in_strides.keys())

        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type =='conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'async_non_local':
            self.refine = AsyncNonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'general_attention':
            self.refine = GeneralizedAttention(
                self.in_channels,
                num_heads=8,
                attention_type='1100')
        elif self.refine_type == 'res':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                # activation=None,
                )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m)

    def gather(self, inputs):
        # gather multi-level features by resize and average
        feats = []
        inputs = list(inputs.values())
        torch_size = inputs[self.refine_level].size()[2:] 
        gather_size = tuple(torch_size)

        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)
        bsf = sum(feats) / len(feats)
        
        return bsf

    def forward(self, inputs, inputs_ref=None, res_only=False):
        assert len(inputs) == self.num_levels

        # step1: gather multi-level features by resize and average
        # feats = []
        # gather_size = inputs[self.refine_level].size()[2:]
        # for i in range(self.num_levels):
        #     if i < self.refine_level:
        #         gathered = F.adaptive_max_pool2d(
        #             inputs[i], output_size=gather_size)
        #     else:
        #         gathered = F.interpolate(
        #             inputs[i], size=gather_size, mode='nearest')
        #     feats.append(gathered)
        # bsf = sum(feats) / len(feats)
        bsf = self.gather(inputs)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by residual path
        # import pdb
        # pdb.set_trace()
        outs = []
        for i in range(self.num_levels):
            out_size = list(inputs.values())[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + list(inputs.values())[i])
        assert len(self.f_name) == len(outs)
        return dict(zip(self.f_name, outs))