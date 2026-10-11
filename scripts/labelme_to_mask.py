"""
labelme_to_mask.py

Converts LabelMe polygon annotations (fully manual) into binary masks in the
same format as our pipeline, so a small hand-labeled subset can be compared
directly against SAM2's auto-propagated labels -- this is the "manual"
validation piece: not meant to replace SAM2 for all 26 videos (too slow),
just to spot-check that SAM2's propagation can be trusted.

Suggested use: pick 5-10 frames across 2-3 videos (ideally including at
least one frame where SAM2's label looked uncertain), hand-annotate them
with LabelMe, then compare pixel overlap (IoU) against the SAM2 label for
the same frame.

Install once:
    pip install labelme --break-system-packages

Usage:
    1. Open the tool and draw a polygon around the object on chosen frames:
         labelme D:\\Minor_Project\\training_data\\frames --output D:\\Minor_Project\\labelme_annotations
       (In the LabelMe window: Edit > Create Polygon, click around the
       object, close the polygon, name it "object", File > Save.)

    2. Convert the resulting JSON files to masks:
         python labelme_to_mask.py --json_dir D:\\Minor_Project\\labelme_annotations ^
                                    --out D:\\Minor_Project\\labelme_masks --img_size 256
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def shape_to_mask(shape, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    points = np.array(shape["points"], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="folder of LabelMe .json annotation files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img_size", type=int, default=256, help="must match dataset_prep.py's --img_size")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json_paths = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    print(f"Found {len(json_paths)} LabelMe annotation file(s)")

    for jp in json_paths:
        with open(jp) as f:
            data = json.load(f)

        img_h, img_w = data["imageHeight"], data["imageWidth"]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for shape in data.get("shapes", []):
            mask = cv2.bitwise_or(mask, shape_to_mask(shape, img_h, img_w))

        mask_rs = cv2.resize(mask, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        base = os.path.splitext(os.path.basename(jp))[0] + ".png"
        cv2.imwrite(os.path.join(args.out, base), mask_rs)
        print(f"  {base}: {len(data.get('shapes', []))} shape(s) -> mask saved")

    print(f"\nSaved to {args.out}")
    print("Compare against SAM2 labels for the same frame with, e.g.:")
    print("  python -c \"import cv2,numpy as np; a=cv2.imread('labelme_masks/X.png',0)>0; "
          "b=cv2.imread('training_data_sam2/labels/X.png',0)>0; "
          "print('IoU:', (a&b).sum()/(a|b).sum())\"")


if __name__ == "__main__":
    main()
