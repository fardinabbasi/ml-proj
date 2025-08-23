# filename: inference_localize.py
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import torch
from tqdm import tqdm
from torchvision.ops import nms
from torchvision.models.detection.image_list import ImageList

from utils_localize import apply_deltas_to_boxes, get_original_image_size


def run_inference(
    backbone,
    rpn,
    roi_head,
    test_dataloader,
    device: torch.device,
    test_img_dir: Union[str, Path],
    score_thresh: float = 0.05,
    nms_thresh: float = 0.2,
    max_dets: int = 25,
    out_csv: Union[str, Path] = "output.csv",
    include_scores: bool = False,
) -> pd.DataFrame:

    test_img_dir = Path(test_img_dir)
    out_csv = Path(out_csv)

    backbone.eval()
    rpn.eval()
    roi_head.eval()

    results = []

    with torch.inference_mode():
        for batch_images, batch_targets in tqdm(test_dataloader, desc="Inference"):
            img_names = [t["img_name"] for t in batch_targets]
            image_shapes = [t["size"] if "size" in t else (t["resized_h"], t["resized_w"])
                            for t in batch_targets]

            batch_images = batch_images.to(device)

            feats = {"0": backbone(batch_images)}
            ilist = ImageList(batch_images, image_shapes)
            proposals, _ = rpn(ilist, feats, None)


            cls_logits, bbox_deltas = roi_head(feats["0"], proposals, image_shapes)
            if cls_logits is None:
                continue

            scores_all = torch.softmax(cls_logits, dim=1)[:, 1]   
            boxes_all = torch.cat(proposals, dim=0)
            deltas_fg = bbox_deltas[:, 4:8]                     
            pred_boxes_all = apply_deltas_to_boxes(boxes_all, deltas_fg)


            num_props = [p.shape[0] for p in proposals]
            boxes_split = pred_boxes_all.split(num_props, dim=0)
            scores_split = scores_all.split(num_props, dim=0)

            for i, (boxes_i, scores_i) in enumerate(zip(boxes_split, scores_split)):
                H, W = image_shapes[i]

                boxes_i[:, 0::2] = boxes_i[:, 0::2].clamp(0, W - 1)  
                boxes_i[:, 1::2] = boxes_i[:, 1::2].clamp(0, H - 1)  

                keep = scores_i >= score_thresh
                boxes_i = boxes_i[keep]
                scores_i = scores_i[keep]
                if boxes_i.numel() == 0:
                    continue

                keep_idx = nms(boxes_i, scores_i, nms_thresh)[:max_dets]
                boxes_out = boxes_i[keep_idx]
                scores_out = scores_i[keep_idx]

                img_name = img_names[i]
                orig_w, orig_h = get_original_image_size(test_img_dir, img_name)
                x_scale = orig_w / float(W)
                y_scale = orig_h / float(H)

                boxes_out_cpu = boxes_out.cpu()
                scores_out_cpu = scores_out.cpu()

                for j, box in enumerate(boxes_out_cpu):
                    x_min, y_min, x_max, y_max = box.tolist()

                    x_min = max(0.0, min(x_min, W - 1))
                    y_min = max(0.0, min(y_min, H - 1))
                    x_max = max(0.0, min(x_max, W - 1))
                    y_max = max(0.0, min(y_max, H - 1))

                    x = x_min * x_scale
                    y = y_min * y_scale
                    width = (x_max - x_min) * x_scale
                    height = (y_max - y_min) * y_scale

                    row = {
                        "image_id": Path(img_name).stem,
                        "x": float(x),
                        "y": float(y),
                        "width": float(width),
                        "height": float(height),
                    }
                    if include_scores:
                        row["score"] = float(scores_out_cpu[j].item())
                    results.append(row)

    # save CSV
    df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(df)} predictions.")
    return df
