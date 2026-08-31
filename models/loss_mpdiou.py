import torch
import torch.nn as nn

from ultralytics.utils.tal import bbox2dist
from .mpdiou import bbox_mpdiou


def build_mpdiou_hw(imgsz, stride_tensor, batch_size):
    """
    Build the normalization term used by MPDIoU.

    Args:
        imgsz: Input image size tensor [height, width].
        stride_tensor: Stride corresponding to each anchor point.
        batch_size: Training batch size.

    Returns:
        Tensor used as mpdiou_hw in MPDIoU loss.
    """
    return (
        (imgsz[0] ** 2 + imgsz[1] ** 2)
        / torch.square(stride_tensor)
    ).repeat(1, batch_size).transpose(1, 0)




class DFLoss(nn.Module):
    """Distribution Focal Loss."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl

        return (
            torch.nn.functional.cross_entropy(
                pred_dist, tl.view(-1), reduction="none"
            ).view(tl.shape) * wl
            +
            torch.nn.functional.cross_entropy(
                pred_dist, tr.view(-1), reduction="none"
            ).view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLossMPDIoU(nn.Module):
    """Bounding-box regression loss using MPDIoU."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        mpdiou_hw,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou = bbox_mpdiou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            xywh=False,
            mpdiou_hw=mpdiou_hw[fg_mask],
        )

        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
                self.dfl_loss.reg_max - 1,
            )

            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )

            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl
