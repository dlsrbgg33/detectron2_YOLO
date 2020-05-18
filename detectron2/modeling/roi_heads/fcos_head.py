import math
import torch.nn as nn

from detectron2.layers import Conv2d, DeformConv, ShapeSpec, Scale, normal_init, get_norm
from detectron2.utils import ConvModule, Scale, bias_init_with_prob
from typing import List
from .box_head import ROI_BOX_HEAD_REGISTRY


@ROI_BOX_HEAD_REGISTRY.register()
class FCOSHead(nn.Module):
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    In our Implementation, schemetic structure is as following:

                                    /-> logits
                    /-> cls convs ->
                   /                \-> centerness
    shared convs ->
                    \-> reg convs -> regressions
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        # import pdb
        # pdb.set_trace()
        self.in_channels       = input_shape[0].channels
        self.num_classes       = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides       = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_shared_convs  = cfg.MODEL.FCOS.NUM_SHARED_CONVS
        self.num_stacked_convs = cfg.MODEL.FCOS.NUM_STACKED_CONVS
        self.prior_prob        = cfg.MODEL.FCOS.PRIOR_PROB
        self.use_deformable    = cfg.MODEL.FCOS.USE_DEFORMABLE
        self.ctr_on_reg        = cfg.MODEL.FCOS.CTR_ON_REG
        # fmt: on
        self.norm_layer        = cfg.MODEL.FCOS.NORM

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """
        Initializes six convolutional layers for FCOS head and a scaling layer for bbox predictions.
        """
        """ your code starts here """    
        activation = nn.ReLU()
        normalization = get_norm(self.norm_layer, self.in_channels)

        self.shared_convs = nn.ModuleList()
        self.cls_convs    = nn.ModuleList()
        self.reg_convs    = nn.ModuleList()

        # Initialize shared convolutional layer for FCOS - stacking num: self.num_shared_convs
        for i in range(self.num_shared_convs):
            chn = self.in_channels if i == 0 else self.in_channels
            self.shared_convs.append(
                Conv2d(
                    chn,
                    self.in_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=True,
                    norm=normalization,
                    activation=activation))
        num_indenp_convs = self.num_stacked_convs-self.num_shared_convs

        # Initialize cls and reg convolutional layer for FCOS 
        # - stacking num: num_indenp_convs
        for i in range(num_indenp_convs):
            chn = self.in_channels if i == 0 else self.in_channels
            self.cls_convs.append(
                Conv2d(
                    chn,
                    self.in_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=True,
                    norm=normalization,
                    activation=activation))
            self.reg_convs.append(
                Conv2d(
                    chn,
                    self.in_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=True,
                    norm=normalization,
                    activation=activation))

        self.cls_logits = nn.Conv2d(
            self.in_channels, self.num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(self.in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(self.in_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.fpn_strides])
        """ your code ends here """

    def _init_weights(self):
        if len(self.shared_convs) != 0:
            for m in self.shared_convs:
                normal_init(m, std=self.prior_prob)
        for m in self.cls_convs:
            normal_init(m, std=self.prior_prob)
        for m in self.reg_convs:
            normal_init(m, std=self.prior_prob)
        normal_init(self.bbox_pred, std=self.prior_prob)
        normal_init(self.centerness, std=self.prior_prob)
        normal_init(self.cls_logits, std=self.prior_prob)

        bias_cls = bias_init_with_prob(self.prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_cls)


    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            cls_scores (list[Tensor]): list of #feature levels, each has shape (N, C, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of C object classes.
            bbox_preds (list[Tensor]): list of #feature levels, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (l, t, r, b) box regression values for
                every position of featur map. These values are the distances from
                a specific point to each (left, top, right, bottom) edge
                of the corresponding ground truth box that the point belongs to.
            centernesses (list[Tensor]): list of #feature levels, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness logits, where these values used to
                downweight the bounding box scores far from the center of an object.
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []


        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16

        for feat_level, feature in enumerate(features):
            """ your code starts here """
            cls_feat = feature
            reg_feat = feature

            #### Shared convs
            if len(self.shared_convs) != 0:
                for shared_layer in self.shared_convs:
                    cls_feat = shared_layer(cls_feat)
                    reg_feat = shared_layer(reg_feat)
            
            #### Class convs after sharing
            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.cls_logits(cls_feat)
            cls_scores.append(cls_score)

            #### Reg convs after sharing
            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)
            bbox_pred = self.scales[feat_level](self.bbox_pred(reg_feat)).float().exp()
            bbox_preds.append(bbox_pred)

            ##### Determine where I add centerness branch between cls and reg #####
            if not self.ctr_on_reg:
                centerness = self.centerness(cls_feat)
                centernesses.append(centerness)
            else:
                centerness = self.centerness(reg_feat)
                centernesses.append(centerness)
                
            """ your code ends here """
        return cls_scores, bbox_preds, centernesses
