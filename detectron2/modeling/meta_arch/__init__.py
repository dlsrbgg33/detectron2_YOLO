# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

from .panoptic_fpn import PanopticFPN

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
# from .rcnn_fcos import GeneralizedRCNN, ProposalNetwork
from .retinanet import RetinaNet
from .semantic_seg import SemanticSegmentor, VideoSemanticSegmentor, VideoSemanticSegmentor_DA
from .one_stage_detector import OneStageDetector
# from .Semantic_seg import SemanticSeg
