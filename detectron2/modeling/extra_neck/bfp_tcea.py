# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import xavier_init

from ..plugins import NonLocal2D, GeneralizedAttention, AsyncNonLocal2D, GlobalNonLocal2D
from detectron2.utils import ConvModule, TCEA_Fusion, ST_Aggregator
from detectron2.utils.registry import Registry
from typing import Dict
from detectron2.layers import Conv2d, ShapeSpec
import pdb
from .bfp import EXTRA_NECK_REGISTRY

### Flow module
from ..flow_modules import (WarpingLayer, LiteFlowNetCorr, MaskEstimator,
                            SELayer, ImgWarpingLayer)
from ..flow_modules.resample2d_package.resample2d import Resample2d


@EXTRA_NECK_REGISTRY.register()
class BFP_TCEA(nn.Module):
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

        self.nframes = cfg.MODEL.EXTRA_NECK.NFRAMES
        self.center = cfg.MODEL.EXTRA_NECK.CENTER

        assert 0 <= self.refine_level < self.num_levels

        class Object():
            pass
        flow_args = Object()
        flow_args.search_range=4
        self.liteflownet = LiteFlowNetCorr(flow_args, self.in_channels+2)

        # self.tcea_fusion = TCEA_Fusion(nf=self.in_channels,
        #                                nframes=self.nframes,
        #                                center=self.center)
        self.st_agg = ST_Aggregator(in_ch=self.in_channels, T=2)

        self.flow_warping = WarpingLayer()

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


    def forward(self, inputs, ref_inputs, flow_init, 
                              next_inputs=None, next_flow_init=None):
        assert len(inputs) == self.num_levels
        # inputs: B,C,H,W
        # step1: gather multi-level features by resize and average
        bsf = self.gather(inputs)
        ref_bsf = self.gather(ref_inputs)
        B,C,H,W = bsf.size()

        warp_bsf = self.flow_warping(ref_bsf, flow_init)
        flow_fine = self.liteflownet(bsf, warp_bsf, flow_init)
        warp_bsf = self.flow_warping(warp_bsf, flow_fine)

        if next_inputs is not None:
            next_bsf = self.gather(next_inputs)
            next_warp_bsf = self.flow_warping(next_bsf, next_flow_init)
            next_flow_fine = self.liteflownet(bsf, next_warp_bsf, next_flow_init)
            next_warp_bsf = self.flow_warping(next_warp_bsf, next_flow_fine)
            bsf_stack = torch.stack([warp_bsf, bsf, next_warp_bsf], dim=1)
            # B,3,C,H,W
        else:
            bsf_stack = torch.stack([bsf, warp_bsf], dim=1)
        # B,2,C,H,W
        # bsf = self.tcea_fusion(bsf_stack)
        bsf_stack = torch.stack([bsf, warp_bsf], dim=2)
        # # # B,C,2,H,W
        bsf = self.st_agg(bsf_stack).squeeze(2) # B,C,H,W

        # step 3: refinement
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 4: scatter refined features to multi-levels by residual path
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