# filename: evaluate_localize.py
import os
import torch
import matplotlib.pyplot as plt

from torchvision.ops import nms, box_iou
from torchvision.models.detection.image_list import ImageList

from utils_localize import apply_deltas_to_boxes


def assign_targets_to_proposals(proposals, gt_boxes, iou_fg=0.7, iou_bg=0.5):
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


def evaluate_detector(
    backbone, rpn, roi_head, dataloader, device,
    iou_thresh=0.7, score_thresh=0.05, nms_thresh=0.2, max_dets=30
):
    backbone.eval(); rpn.eval(); roi_head.eval()
    TP, FP, FN = 0, 0, 0

    with torch.inference_mode():
        for images, targets in dataloader:
            images = images.to(device)
            image_shapes = [t["size"] for t in targets] 

            feats = {"0": backbone(images)}
            ilist = ImageList(images, image_shapes)
            proposals, _ = rpn(ilist, feats, None)

            cls_logits, bbox_deltas = roi_head(feats["0"], proposals, image_shapes)
            if cls_logits is None:
                continue

            idx_offset = 0
            for b in range(images.shape[0]):
                props = proposals[b]
                n_props = props.shape[0]
                logits = cls_logits[idx_offset:idx_offset + n_props]
                deltas = bbox_deltas[idx_offset:idx_offset + n_props, 4:8]
                idx_offset += n_props

                scores = torch.softmax(logits, dim=1)[:, 1]
                boxes  = apply_deltas_to_boxes(props, deltas)

                H, W = image_shapes[b]
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W - 1)  # x1/x2
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H - 1)  # y1/y2

                keep = scores > score_thresh
                boxes_kept  = boxes[keep]
                scores_kept = scores[keep]
                if boxes_kept.numel() > 0:
                    keep_idx   = nms(boxes_kept, scores_kept, nms_thresh)[:max_dets]
                    pred_boxes = boxes_kept[keep_idx].cpu()
                else:
                    pred_boxes = torch.empty((0, 4), device=boxes.device).cpu()

                gt_boxes = targets[b].get('boxes', torch.empty((0, 4))).cpu()

                if pred_boxes.shape[0] > 0 and gt_boxes.shape[0] > 0:
                    ious = box_iou(pred_boxes, gt_boxes)  
                else:
                    ious = torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]))

                gt_matched = torch.zeros(len(gt_boxes), dtype=torch.uint8)
                for i in range(len(pred_boxes)):
                    if len(gt_boxes) == 0:
                        FP += 1
                        continue
                    max_iou, max_idx = ious[i].max(0)
                    j = int(max_idx.item())
                    if max_iou.item() >= iou_thresh and gt_matched[j] == 0:
                        TP += 1
                        gt_matched[j] = 1
                    else:
                        FP += 1

                FN += int((gt_matched == 0).sum().item())

    TP, FP, FN = float(TP), float(FP), float(FN)
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  [TP={TP} FP={FP} FN={FN}]")
    return precision, recall, f1


def rpn_recall(backbone, rpn, loader, device, iou_thresh=0.5):
    backbone.eval(); rpn.eval()
    total, covered = 0, 0

    with torch.inference_mode():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            image_shapes = [t["size"] for t in targets]
            feats = {"0": backbone(imgs)}
            ilist = ImageList(imgs, image_shapes)
            proposals, _ = rpn(ilist, feats, None)

            for props, t in zip(proposals, targets):
                gts = t.get("boxes", torch.empty((0, 4))).to(props.device)
                if gts.numel() == 0:
                    continue
                ious = box_iou(props, gts)
                covered += (ious.max(dim=0).values >= iou_thresh).sum().item()
                total   += gts.shape[0]

    print(f"RPN recall @ IoU {iou_thresh}: {covered/max(total,1):.3f} ({covered}/{total})")


def visualize_n_detections(
    backbone, rpn, roi_head, dataloader, device,
    num_images=20, score_thresh=0.05, nms_thresh=0.2, max_dets=10,
    save_dir=None, draw_gt=False
):
    backbone.eval(); rpn.eval(); roi_head.eval()
    shown = 0

    with torch.inference_mode():
        for images, targets in dataloader:
            images = images.to(device)
            image_shapes = [t["size"] for t in targets]

            feats = {"0": backbone(images)}
            ilist = ImageList(images, image_shapes)
            proposals, _ = rpn(ilist, feats, None)
            cls_logits, bbox_deltas = roi_head(feats["0"], proposals, image_shapes)
            if cls_logits is None:
                continue

            idx_offset = 0
            for b in range(images.shape[0]):
                props   = proposals[b]
                n_props = props.shape[0]
                logits  = cls_logits[idx_offset:idx_offset + n_props]
                deltas  = bbox_deltas[idx_offset:idx_offset + n_props, 4:8]
                idx_offset += n_props

                scores = torch.softmax(logits, dim=1)[:, 1]
                boxes  = apply_deltas_to_boxes(props, deltas)

                H, W = image_shapes[b]
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W - 1)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H - 1)

                keep_mask   = scores > score_thresh
                boxes_kept  = boxes[keep_mask]
                scores_kept = scores[keep_mask]
                if boxes_kept.numel() > 0:
                    keep_idx   = nms(boxes_kept, scores_kept, nms_thresh)[:max_dets]
                    boxes_kept = boxes_kept[keep_idx]
                else:
                    boxes_kept = torch.empty((0, 4), device=boxes.device)

                img_t = images[b, 0, :H, :W].detach().cpu().numpy()
                mn, mx = img_t.min(), img_t.max()
                img_disp = (img_t - mn) / (mx - mn + 1e-8)

                plt.figure(figsize=(8, 8))
                plt.imshow(img_disp, cmap='gray')
                ax = plt.gca()

                for box in boxes_kept.detach().cpu():
                    x1, y1, x2, y2 = box.tolist()
                    ax.add_patch(plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        fill=False, edgecolor='lime', linewidth=2
                    ))

                if draw_gt and "boxes" in targets[b]:
                    for g in targets[b]["boxes"].cpu():
                        gx1, gy1, gx2, gy2 = g.tolist()
                        ax.add_patch(plt.Rectangle(
                            (gx1, gy1), gx2 - gx1, gy2 - gy1,
                            fill=False, edgecolor='yellow', linewidth=1.5, linestyle='--'
                        ))

                plt.axis('off')

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    out_path = os.path.join(save_dir, f"pred_{shown+1:02d}.png")
                    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                else:
                    plt.show()

                shown += 1
                if shown >= num_images:
                    print(f"Done. Generated {shown} images.")
                    return

    print(f"Done. Generated {shown} images (dataset may be small).")
