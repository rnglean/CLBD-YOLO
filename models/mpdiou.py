def bbox_mpdiou(box1, box2, xywh=True, mpdiou_hw=1, eps=1e-7):
    """
    Calculate MPDIoU between predicted and target bounding boxes.

    Args:
        box1: Predicted bounding boxes.
        box2: Target bounding boxes.
        xywh: Whether boxes are represented as (x, y, w, h).
        mpdiou_hw: Normalization factor used by MPDIoU.
        eps: Small value to avoid division by zero.

    Returns:
        MPDIoU values.
    """

    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = (
            box1.chunk(4, -1),
            box2.chunk(4, -1),
        )

        w1_, h1_ = w1 / 2, h1 / 2
        w2_, h2_ = w2 / 2, h2 / 2

        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_

        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_

    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

        w1 = b1_x2 - b1_x1
        h1 = b1_y2 - b1_y1 + eps

        w2 = b2_x2 - b2_x1
        h2 = b2_y2 - b2_y1 + eps

    inter = (
        (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
        *
        (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    )

    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    d1 = (
        (b2_x1 - b1_x1) ** 2
        + (b2_y1 - b1_y1) ** 2
    )

    d2 = (
        (b2_x2 - b1_x2) ** 2
        + (b2_y2 - b1_y2) ** 2
    )

    return (
        iou
        - d1 / mpdiou_hw.unsqueeze(1)
        - d2 / mpdiou_hw.unsqueeze(1)
    )
