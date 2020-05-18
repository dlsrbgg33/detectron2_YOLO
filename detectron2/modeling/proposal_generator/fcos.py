import torch
from torch import nn

from detectron2.layers import ShapeSpec, IOULoss, ml_nms
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.structures import Boxes, Instances
from typing import Dict
from ..roi_heads import build_box_head

from ..roi_heads import FCOSHead
# from ..roi_heads import FCOSHead_ori
from .fcos_losses import FCOSLosses
from .fcos_targets import FCOSTargets, get_points
from detectron2.layers import ShapeSpec, batched_nms, cat
from ..box_regression import Box2BoxTransform

__all__ = ["FCOS"]

INF = 100000000

"""
Shape shorthand in this module:
    N: number of images in the minibatch.
    Hi, Wi: height and width of the i-th level feature map.
    4: size of the box parameterization.
Naming convention:
    labels: refers to the ground-truth class of an position.
    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the
        ground-truth box.
    logits_pred: predicted classification scores in [-inf, +inf];
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets
    ctrness_pred: predicted centerness scores
"""


@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOS(nn.Module):

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.mask_on               = cfg.MODEL.MASK_ON
        self.num_classes           = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features           = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides           = cfg.MODEL.FCOS.FPN_STRIDES
        self.normalize_reg_targets = cfg.MODEL.FCOS.NORMALIZE_REG_TARGETS
        # inference parameters
        self.score_threshold       = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.nms_pre_topk          = cfg.MODEL.FCOS.NMS_PRE_TOPK
        self.nms_threshold         = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.nms_post_topk         = cfg.MODEL.FCOS.NMS_POST_TOPK
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        # fmt: on
        self.cfg = cfg
        self.fcos_head = build_box_head(cfg, [input_shape[f] for f in self.in_features])
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)

        reg_loss_type = cfg.MODEL.FCOS.LOC_LOSS_TYPE
        self.reg_loss = IOULoss(reg_loss_type)
        self.min_size = 0

    def forward(self, images, features, gt_instances):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "pred_boxes", "scores", "pred_classes",
            "locations".
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]

        # Step 1. FCOS head implementation
        fcos_preds = self.fcos_head(features)
        all_level_points = get_points(features, self.fpn_strides)
        # self.training = False
        if self.training:
            # Step 2. training target generation
            training_targets = FCOSTargets(all_level_points, gt_instances, self.cfg)

            # Step 3. loss computation
            loss_inputs = fcos_preds + training_targets
            losses = FCOSLosses(*loss_inputs, self.reg_loss, self.cfg)
            if self.mask_on:  # Proposal generation for Instance Segmentation (ExtraExtra)
                # compute proposals for ROI sampling
                proposals = self.predict_proposals(
                    *fcos_preds,
                    all_level_points,
                    images.image_sizes,
                )
                return proposals, losses
            else:
                return None, losses

        # Step 4. Inference phase
        proposals = self.predict_proposals(*fcos_preds, all_level_points, images.image_sizes)
        return proposals, None

    def predict_proposals(
        self,
        cls_scores,
        bbox_preds,
        centernesses,
        all_level_points,
        image_sizes
    ):
        """
        Arguments:
            cls_scores, bbox_preds, centernesses: Same as the output of :meth:`FCOSHead.forward`
            all_level_points (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (Hi*Wi, 2), a set of point coordinates (xi, yi) of all feature map
                locations on 'feature level i' in image coordinate.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        num_imgs = len(image_sizes)
        num_levels = len(cls_scores)

        # recall that during training, we normalize regression targets with FPN's stride.
        # we denormalize them here.
        if self.normalize_reg_targets:
            bbox_preds = [bbox_preds[i] * self.fpn_strides[i] for i in range(num_levels)]

        result_list = []
        for img_id in range(num_imgs):
            # each entry of list corresponds to per-level feature tensor of single image.
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]

            # per-image proposal comutation
            det_bboxes = self.predict_proposals_single_image(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                all_level_points,
                image_sizes[img_id]
            )
            result_list.append(det_bboxes)
        return result_list

    def remove_small_boxes(self, boxlist, min_size):
        """
        Only keep boxes with both sides >= min_size

        Arguments:
            boxlist (Boxlist)
            min_size (int)
        """
        # TODO maybe add an API for querying the ws / hs
        xywh_boxes = boxlist.convert("xywh").bbox
        _, _, ws, hs = xywh_boxes.unbind(dim=1)
        keep = (
            (ws >= min_size) & (hs >= min_size)
        ).nonzero().squeeze(1)
        return boxlist[keep]


    def _cat(self, tensors, dim=0):
        """
        Efficient version of torch.cat that avoids a copy if there is only a single element in a list
        """
        assert isinstance(tensors, (list, tuple))
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim)

    def cat_boxlist(self, bboxes):
        """
        Concatenates a list of BoxList (having the same image size) into a
        single BoxList

        Arguments:
            bboxes (list[BoxList])
        """
        assert isinstance(bboxes, (list, tuple))
        assert all(isinstance(bbox, BoxList) for bbox in bboxes)

        size = bboxes[0].size
        assert all(bbox.size == size for bbox in bboxes)

        mode = bboxes[0].mode
        assert all(bbox.mode == mode for bbox in bboxes)

        fields = set(bboxes[0].fields())
        assert all(set(bbox.fields()) == fields for bbox in bboxes)

        cat_boxes = BoxList(self._cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

        for field in fields:
            data = self._cat([bbox.get_field(field) for bbox in bboxes], dim=0)
            cat_boxes.add_field(field, data)

        return cat_boxes


    def predict_proposals_single_image(
        self,
        cls_scores,
        bbox_preds,
        centernesses,
        all_level_points,
        image_size
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            cls_scores (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (C, Hi, Wi), where i denotes a specific feature level.
            bbox_preds (list[Tensor]): Same shape as 'cls_scores' except that C becomes 4.
            centernesses (list[Tensor]): Same shape as 'cls_scores' except that C becomes 1.
            all_level_points (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (Hi*Wi, 2), a set of point coordinates (xi, yi) of all feature map
                locations on 'feature level i' in image coordinate.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `predict_proposals`, but for only one image.
        """
        assert len(cls_scores) == len(bbox_preds) == len(all_level_points)
        pred_bboxes_list = []
        scores_list = []
        cls_idx_list = []
        results = []
        boxlists = []
        # Iterate over every feature level
        for (cls_score, bbox_pred, centerness, points) in zip(
            cls_scores, bbox_preds, centernesses, all_level_points
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # (C, Hi, Wi) -> (Hi*Wi, C)
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()

            # (4, Hi, Wi) -> (Hi*Wi, 4)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # (1, Hi, Wi) -> (Hi*Wi, )
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            """ Your code starts here """
            candidate_ids = scores > self.score_threshold
            scores =  scores * centerness.view(-1,1)
            pre_nms_top_k = candidate_ids.sum()
            pre_nms_top_k = pre_nms_top_k.clamp(max=self.nms_pre_topk)
            # pre_nms_top_k = candidate_ids

            scores = scores[candidate_ids]

            box_loc = candidate_ids.nonzero()[:,0]
            box_class = candidate_ids.nonzero()[:,1]
            box_regress = bbox_pred[box_loc]
            points = points[box_loc]

            if candidate_ids.sum().item() > pre_nms_top_k.item():
                scores, top_k_indices = scores.topk(pre_nms_top_k, sorted=False)
                # scores, top_indices = scores.sort(descending=True)
                # top_k_indices = top_indices[:pre_nms_top_k]
                box_class = box_class[top_k_indices]
                box_regress = box_regress[top_k_indices]
                points = points[top_k_indices]
            
            detections = torch.stack([
                points[:,0] - box_regress[:,0],
                points[:,1] - box_regress[:,1],
                points[:,0] + box_regress[:,2],
                points[:,1] + box_regress[:,3],
                ], dim=1)

            boxlist = Instances(image_size)
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(scores)
            boxlist.pred_classes = box_class

            boxlists.append(boxlist)

        boxlists = Instances.cat(boxlists)
        # import pdb
        # pdb.set_trace()
        # # non-maximum suppression per-image.
        results = ml_nms(
            boxlists,
            self.nms_threshold,
            # Limit to max_per_image detections **over all classes**
            max_proposals=self.nms_post_topk
        )
        return results
        
