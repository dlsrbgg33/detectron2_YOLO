# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .build import SEM_SEG_HEAD_REGISTRY, build_model  # isort:skip

from .panopFCNHead import PanopFCNHead

from .FCNHead import FCNHead, build_sem_seg_head, SEM_SEG_HEADS_REGISTRY