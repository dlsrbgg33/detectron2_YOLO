from detectron2.layers import batched_nms


def ml_nms(
    boxlist,
    nms_thresh,
    max_proposals=-1,
):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    Arguments:
        boxlist(BoxList): should contain 'pred_boxes', 'scores', 'pred_classes' fields.
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist
