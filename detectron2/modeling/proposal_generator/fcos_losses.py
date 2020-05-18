import torch
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.layers import cat

from .fcos_targets import compute_centerness_targets


def FCOSLosses(
    cls_scores,
    bbox_preds,
    centernesses,
    labels,
    bbox_targets,
    reg_loss,
    cfg
):
    """
    Arguments:
        cls_scores, bbox_preds, centernesses: Same as the output of :meth:`FCOSHead.forward`
        labels, bbox_targets: Same as the output of :func:`FCOSTargets`

    Returns:
        losses (dict[str: Tensor]): A dict mapping from loss name to loss value.
    """
    # fmt: off

    num_classes          = cfg.MODEL.FCOS.NUM_CLASSES
    focal_loss_alpha     = cfg.MODEL.FCOS.LOSS_ALPHA
    focal_loss_gamma     = cfg.MODEL.FCOS.LOSS_GAMMA
    # fmt: on

    # Collect all logits and regression predictions over feature maps
    # and images to arrive at the same shape as the labels and targets
    # The final ordering is L, N, H, W from slowest to fastest axis.
    flatten_cls_scores = cat(
        [
            # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            cls_score.permute(0, 2, 3, 1).reshape(-1, num_classes)
            for cls_score in cls_scores
        ], dim=0)

    flatten_bbox_preds = cat(
        [
            # Reshape: (N, 4, Hi, Wi) -> (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ], dim=0)
    flatten_centernesses = cat(
        [
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            centerness.reshape(-1) for centerness in centernesses
        ], dim=0)

    # flatten classification and regression targets.
    flatten_labels = cat(labels)
    flatten_bbox_targets = cat(bbox_targets)

    # retain indices of positive predictions.
    pos_inds = torch.nonzero(flatten_labels != num_classes).squeeze(1)
    num_pos = max(len(pos_inds), 1.0)

    # prepare one_hot label.
    class_target = torch.zeros_like(flatten_cls_scores)
    class_target[pos_inds, flatten_labels[pos_inds]] = 1

    # classification loss: Focal loss
    loss_cls = sigmoid_focal_loss_jit(
        flatten_cls_scores,
        class_target,
        alpha=focal_loss_alpha,
        gamma=focal_loss_gamma,
        reduction="sum",
    ) / num_pos

    # compute regression loss and centerness loss only for positive samples.
    pos_bbox_preds = flatten_bbox_preds[pos_inds]
    pos_centernesses = flatten_centernesses[pos_inds]
    pos_bbox_targets = flatten_bbox_targets[pos_inds]

    # compute centerness targets.
    pos_centerness_targets = compute_centerness_targets(pos_bbox_targets)
    centerness_norm = max(pos_centerness_targets.sum(), 1e-6)

    # regression loss: IoU loss
    loss_bbox = reg_loss(
        pos_bbox_preds,
        pos_bbox_targets,
        weight=pos_centerness_targets
    ) / centerness_norm

    # centerness loss: Binary CrossEntropy loss
    loss_centerness = F.binary_cross_entropy_with_logits(
        pos_centernesses,
        pos_centerness_targets,
        reduction="sum"
    ) / num_pos

    # final loss dict.
    losses = dict(
        loss_fcos_cls=loss_cls,
        loss_fcos_loc=loss_bbox,
        loss_fcos_ctr=loss_centerness
    )
    return losses
