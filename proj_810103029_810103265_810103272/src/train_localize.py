# filename: train_localize.py
import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou
from utils_localize import move_target_to_device 
from torchvision.ops import generalized_box_iou_loss
from utils_localize import apply_deltas_to_boxes


def assign_targets_to_proposals(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_fg: float = 0.7,
    iou_bg: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = proposals.device
    N = proposals.shape[0]
    labels = torch.full((N,), -1, dtype=torch.int64, device=device)
    matched = torch.zeros((N, 4), dtype=torch.float32, device=device)
    if proposals.numel() == 0 or gt_boxes.numel() == 0:
        return labels, matched
    ious = box_iou(proposals, gt_boxes)
    max_iou, idxs = ious.max(dim=1)
    matched = gt_boxes[idxs]
    labels[max_iou >= iou_fg] = 1
    labels[max_iou <  iou_bg] = 0
    matched[labels == 0] = 0.0
    return labels, matched

def train_detector(
    backbone,
    rpn,
    roi_head,
    dataloader,
    device: torch.device,
    num_epochs: int = 100,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    valid_dataloader=None,
    print_every: int = 1,
) -> Dict[str, list]:

    for p in backbone.parameters(): p.requires_grad = True
    for p in rpn.parameters():      p.requires_grad = True
    for p in roi_head.parameters(): p.requires_grad = True

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(rpn.parameters()) + list(roi_head.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    history = {"train_loss": []}

    for epoch in range(num_epochs):
        backbone.train(); rpn.train(); roi_head.train()
        total_loss = 0.0
        steps = 0

        last_rpn_obj = 0.0
        last_rpn_reg = 0.0
        last_cls     = 0.0
        last_reg     = 0.0

        if epoch < 4:
            iou_fg, iou_bg = 0.55, 0.45
        elif epoch < 10:
            iou_fg, iou_bg = 0.60, 0.50
        else:
            iou_fg, iou_bg = 0.70, 0.50

        for batch_idx, (images, targets) in enumerate(dataloader):
            steps += 1
            images = images.to(device)
            targets = [move_target_to_device(t, device) for t in targets]


            feats = backbone(images)
            features = {"0": feats}

            image_shapes = [t["size"] for t in targets]
            image_list = ImageList(images, image_shapes)

            proposals, rpn_losses = rpn(image_list, features, targets)
            rpn_loss = rpn_losses['loss_objectness'] + rpn_losses['loss_rpn_box_reg']

            last_rpn_obj = float(rpn_losses['loss_objectness'].detach().item())
            last_rpn_reg = float(rpn_losses['loss_rpn_box_reg'].detach().item())

            cls_logits, bbox_deltas = roi_head(features["0"], proposals, image_shapes)
            if cls_logits is None:
                cls_loss = torch.tensor(0.0, device=device)
                reg_loss = torch.tensor(0.0, device=device)
                loss = rpn_loss + cls_loss + reg_loss
            else:
                all_labels, all_bbox_targets = [], []
                for i, props in enumerate(proposals):
                    gt_boxes = targets[i]["boxes"]
                    labels, matched_gt_boxes = assign_targets_to_proposals(
                        props, gt_boxes, iou_fg=iou_fg, iou_bg=iou_bg
                    )
                    all_labels.append(labels)
                    all_bbox_targets.append(matched_gt_boxes)

                labels = torch.cat(all_labels, dim=0)
                bbox_targets = torch.cat(all_bbox_targets, dim=0)

                w = torch.tensor([1.0, 3.0], device=cls_logits.device, dtype=cls_logits.dtype)
                cls_loss = F.cross_entropy(cls_logits, labels, ignore_index=-1, weight=w)

                fg_inds = torch.where(labels == 1)[0]
                if fg_inds.numel() > 0:
                    bbox_deltas_fg = bbox_deltas[fg_inds, 4:8]
                    all_props = torch.cat(list(proposals), dim=0)
                    props_fg = all_props[fg_inds]
                    target_boxes = bbox_targets[fg_inds]

                    ex_w  = (props_fg[:, 2] - props_fg[:, 0]).clamp(min=1e-6)
                    ex_h  = (props_fg[:, 3] - props_fg[:, 1]).clamp(min=1e-6)
                    ex_cx = props_fg[:, 0] + 0.5 * ex_w
                    ex_cy = props_fg[:, 1] + 0.5 * ex_h

                    gt_w  = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1e-6)
                    gt_h  = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1e-6)
                    gt_cx = target_boxes[:, 0] + 0.5 * gt_w
                    gt_cy = target_boxes[:, 1] + 0.5 * gt_h

                    dx = (gt_cx - ex_cx) / ex_w
                    dy = (gt_cy - ex_cy) / ex_h
                    dw = torch.log(gt_w / ex_w)
                    dh = torch.log(gt_h / ex_h)
                    targets_reg = torch.stack((dx, dy, dw, dh), dim=1)

                    reg_l1 = F.smooth_l1_loss(bbox_deltas_fg, targets_reg, reduction="mean")

                    pred_boxes = apply_deltas_to_boxes(props_fg, bbox_deltas_fg)  # [N,4] x1y1x2y2
                    reg_giou  = generalized_box_iou_loss(pred_boxes, target_boxes, reduction="mean")

                    reg_loss = 0.5 * reg_l1 + 0.5 * reg_giou
                else:
                    reg_loss = torch.tensor(0.0, device=device)

                loss = rpn_loss + cls_loss + 5.0 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            last_cls = float(cls_loss.detach().item())
            last_reg = float(reg_loss.detach().item())

        avg_loss = total_loss / max(1, steps)
        history["train_loss"].append(float(avg_loss))

        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
        #print(f"RPN obj {last_rpn_obj:.3f} | RPN reg {last_rpn_reg:.3f} | "
        #      f"ROI cls {last_cls:.3f} | ROI reg {last_reg:.3f}")

    return history
