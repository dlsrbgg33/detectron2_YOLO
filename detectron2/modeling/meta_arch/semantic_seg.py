# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from ..backbone import build_backbone
from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY
from ..seg_heads import build_sem_seg_head
from ..extra_neck import build_extra_neck

#### Flow module
from ..flow_modules import (FlowNet2, LiteFlowNetCorr)
from ..flow_modules.resample2d_package.resample2d import Resample2d
from detectron2.utils import flow_utils
import os
import os.path as osp
import warnings
warnings.filterwarnings("ignore")

__all__ = ["SemanticSegmentor", "VideoSemanticSegmentor", "VideoSemanticSegmentor_DA"]


@META_ARCH_REGISTRY.register()
class SemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.extra_on = cfg.MODEL.EXTRA_NECK.EXTRA_ON
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.extra_neck = build_extra_neck(cfg, self.backbone.output_shape())
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model, used in inference.
                     See :meth:`postprocess` for details.

        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor of the output resolution that represents the
              per-pixel segmentation prediction.
        """

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        images = ImageList.from_tensors(images, 0)
    
        ### Feature Extractor with FPN
        features = self.backbone(images.tensor)

        ### Extra Neck for refining FPN
        if self.extra_on:
            features_ = self.extra_neck(features)
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, 0, self.sem_seg_head.ignore_label
            ).tensor
        else:
            targets = None
        # import pdb
        # pdb.set_trace()
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results



@META_ARCH_REGISTRY.register()
class VideoSemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        # import pdb
        # pdb.set_trace()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.extra_neck = build_extra_neck(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.mean=[123.675, 116.28, 103.53]
        self.std=[58.395, 57.12, 57.375]
        class Object(object):
          pass
        flow2_args = Object()
        flow2_args.rgb_max = 255.0
        flow2_args.fp16 = False
        self.flownet2 = FlowNet2(flow2_args, requires_grad=False)
        model_filename = osp.join(os.getcwd(), 'flow_pretrained',
        "FlowNet2_checkpoint.pth.tar")
        print("===> Load: %s" %model_filename)
        checkpoint = torch.load(model_filename)
        self.flownet2.load_state_dict(checkpoint['state_dict'])
        self.flownet2 = self.flownet2.cuda()
        self.flownet2.eval()
        self.flow_warping = Resample2d().cuda()

    @property
    def device(self):
        return self.pixel_mean.device


    #### denormalize the images
    def denormalize(self, img):
        # img: B,C,H,W
        img[:,0]= img[:,0]*self.std[0] + self.mean[0]
        img[:,1]= img[:,1]*self.std[1] + self.mean[1]
        img[:,2]= img[:,2]*self.std[2] + self.mean[2]
        return img

    def rgb_denormalize(self, img):
        # img: B,C,H,W
        img[:,:,0]= img[:,:,0]*self.std[0] + self.mean[0]
        img[:,:,1]= img[:,:,1]*self.std[1] + self.mean[1]
        img[:,:,2]= img[:,:,2]*self.std[2] + self.mean[2]
        return img


    def compute_flow(self, img, ref_img, scale_factor=1):
        H,W = img.size(2), img.size(3)
        rgb = self.denormalize(img)
        ref_rgb = self.denormalize(ref_img)        
        rgbs = torch.stack([rgb, ref_rgb], dim=2)
        # import pdb
        # pdb.set_trace()
        #### ZERO PAD
        if H == 800 and W == 1600:
            rgbs = F.pad(rgbs,(0,64,0,32))
        else:
            rgbs = F.pad(rgbs,(0,0,0,8))
        self.flownet2.cuda(rgbs.device)
        self.flownet2.eval()
        flow = self.flownet2(rgbs)
        #### ZERO TRIM
        indH = torch.arange(0,H).cuda(rgbs.device)
        indW = torch.arange(0,W).cuda(rgbs.device)
        flow = torch.index_select(flow,-2,indH)
        flow = torch.index_select(flow,-1,indW)
        # flow_ = flow[:,:,:H,:W]
        warp_img = self.flow_warping(ref_img, flow)
        occ_mask = torch.exp( -50. * torch.sum(img - warp_img, dim=1).pow(2) ).unsqueeze(1)
        if scale_factor != 1:
            flow = F.interpolate(flow, scale_factor=scale_factor, mode='bilinear', align_corners=False) * scale_factor
            occ_mask = F.interpolate(occ_mask, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return flow, occ_mask


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model, used in inference.
                     See :meth:`postprocess` for details.

        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor of the output resolution that represents the
              per-pixel segmentation prediction.
        """
        # import pdb
        # pdb.set_trace()
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 0)

        ref_images = [x["ref_image"].to(self.device) for x in batched_inputs]
        ref_images = [(x - self.pixel_mean) / self.pixel_std for x in ref_images]
        ref_images = ImageList.from_tensors(ref_images, 0)
        
        flowR2T, occ_mask = self.compute_flow(images.tensor.clone(), ref_images.tensor.clone(), scale_factor=0.25)

        features_main = self.backbone(images.tensor)
        features_ref  = self.backbone(ref_images.tensor)

        features_main = self.extra_neck(features_main, features_ref, flowR2T)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, 0, self.sem_seg_head.ignore_label
            ).tensor
        else:
            targets = None

        results, losses = self.sem_seg_head(features_main, targets)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results





@META_ARCH_REGISTRY.register()
class VideoSemanticSegmentor_DA(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        # import pdb
        # pdb.set_trace()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.extra_neck = build_extra_neck(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.mean=[123.675, 116.28, 103.53]
        self.std=[58.395, 57.12, 57.375]
        class Object(object):
          pass
        flow2_args = Object()
        flow2_args.rgb_max = 255.0
        flow2_args.fp16 = False
        self.flownet2 = FlowNet2(flow2_args, requires_grad=False)
        model_filename = osp.join(os.getcwd(), 'flow_pretrained',
        "FlowNet2_checkpoint.pth.tar")
        print("===> Load: %s" %model_filename)
        checkpoint = torch.load(model_filename)
        self.flownet2.load_state_dict(checkpoint['state_dict'])
        self.flownet2 = self.flownet2.cuda()
        self.flownet2.eval()
        self.flow_warping = Resample2d().cuda()

    @property
    def device(self):
        return self.pixel_mean.device


    #### denormalize the images
    def denormalize(self, img):
        # img: B,C,H,W
        img[:,0]= img[:,0]*self.std[0] + self.mean[0]
        img[:,1]= img[:,1]*self.std[1] + self.mean[1]
        img[:,2]= img[:,2]*self.std[2] + self.mean[2]
        return img

    def rgb_denormalize(self, img):
        # img: B,C,H,W
        img[:,:,0]= img[:,:,0]*self.std[0] + self.mean[0]
        img[:,:,1]= img[:,:,1]*self.std[1] + self.mean[1]
        img[:,:,2]= img[:,:,2]*self.std[2] + self.mean[2]
        return img


    def compute_flow(self, img, ref_img, scale_factor=1):
        H,W = img.size(2), img.size(3)
        rgb = self.denormalize(img)
        ref_rgb = self.denormalize(ref_img)        
        rgbs = torch.stack([rgb, ref_rgb], dim=2)
        # import pdb
        # pdb.set_trace()
        #### ZERO PAD
        if H == 800 and W == 1600:
            rgbs = F.pad(rgbs,(0,64,0,32))
        else:
            rgbs = F.pad(rgbs,(0,0,0,8))
        self.flownet2.cuda(rgbs.device)
        self.flownet2.eval()
        flow = self.flownet2(rgbs)
        #### ZERO TRIM
        indH = torch.arange(0,H).cuda(rgbs.device)
        indW = torch.arange(0,W).cuda(rgbs.device)
        flow = torch.index_select(flow,-2,indH)
        flow = torch.index_select(flow,-1,indW)
        # flow_ = flow[:,:,:H,:W]
        warp_img = self.flow_warping(ref_img, flow)
        occ_mask = torch.exp( -50. * torch.sum(img - warp_img, dim=1).pow(2) ).unsqueeze(1)
        if scale_factor != 1:
            flow = F.interpolate(flow, scale_factor=scale_factor, mode='bilinear', align_corners=False) * scale_factor
            occ_mask = F.interpolate(occ_mask, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return flow, occ_mask


    # def forward(self, batched_inputs_src, batched_inputs_tgt):

    def forward(self, batched_inputs_src, batched_inputs_tgt):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model, used in inference.
                     See :meth:`postprocess` for details.

        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor of the output resolution that represents the
              per-pixel segmentation prediction.
        """
        import pdb
        pdb.set_trace()
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 0)

        ref_images = [x["image"].to(self.device) for x in batched_inputs]
        ref_images = [(x - self.pixel_mean) / self.pixel_std for x in ref_images]
        ref_images = ImageList.from_tensors(ref_images, 0)
        
        flowR2T, occ_mask = self.compute_flow(images.tensor.clone(), ref_images.tensor.clone(), scale_factor=0.25)

        features_main = self.backbone(images.tensor)
        features_ref  = self.backbone(ref_images.tensor)

        # features_main = self.extra_neck(features_main, features_ref, flowR2T)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, 0, self.sem_seg_head.ignore_label
            ).tensor
        else:
            targets = None

        results, losses = self.sem_seg_head(features_main, targets)

        if self.training:
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r})
        return processed_results