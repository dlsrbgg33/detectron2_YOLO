from .build import META_ARCH_REGISTRY
import torch.nn as nn
from ..backbone import build_backbone
from detectron2.layers import ShapeSpec
from ..proposal_generator import build_proposal_generator
from typing import Dict
import torch
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from ..postprocessing import detector_postprocess

__all__ = ["OneStageDetector"]

@META_ARCH_REGISTRY.register()
class OneStageDetector(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Uses "instances" as the return key instead of using "proposal".
    """
    def __init__(self, cfg):
        super().__init__()
        ### ResNet with "build_retinanet_resnet_fpn_backbone" 
        self.backbone = build_backbone(cfg)
        # self.proposal_gen = build_proposal_generator(cfg, Dict[str, ShapeSpec])
        self.proposal_gen = build_proposal_generator(cfg, self.backbone.output_shape())
        ### Normalize ready
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        # self.training = False
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # if self.training:
        #     return super().forward(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        #### Image
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        
        #### Label
        # gt_instances = ImageList.from_tensors(gt_instances, self.backbone.size_divisibility)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, losses = self.proposal_gen(images, features, gt_instances)

        if self.training:
            return losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
