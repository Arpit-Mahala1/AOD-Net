"""
find_runs.py

Lists every eval_report.csv under D:\\Minor_Project (two levels deep) with its
mean IoU, F1 and second-level detection accuracy, so you can see which run
produced the numbers in the paper (Table 2: IoU 0.652, F1 0.744, det 0.902;
Table 4 ABODA: IoU 0.617, F1 0.677, det 0.925).

Usage:
    python find_runs.py
    python find_runs.py --root D:\\Minor_Project
"""

import argparse
import csv
import glob
import os


def mean(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, "nan")
        try:
            f = float(v)
        except ValueError:
            f = float("nan")
        vals.append(0.0 if f != f else f)  # NaN counted as 0, as in the paper
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\Minor_Project")
    args = ap.parse_args()

    paths = sorted(set(glob.glob(os.path.join(args.root, "*", "eval_report.csv"))
                       + glob.glob(os.path.join(args.root, "*", "*", "eval_report.csv"))))
    if not paths:
        print(f"No eval_report.csv found under {args.root}")
        return

    print(f"{'run folder':55s} {'n':>3s} {'IoU':>6s} {'F1':>6s} {'det':>6s}")
    for p in paths:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        folder = os.path.relpath(os.path.dirname(p), args.root)
        print(f"{folder:55s} {len(rows):3d} {mean(rows, 'pixel_iou'):6.3f} {mean(rows, 'pixel_f1'):6.3f} "
              f"{mean(rows, 'second_level_detection_accuracy'):6.3f}")


if __name__ == "__main__":
    main()
