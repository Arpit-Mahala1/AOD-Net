#!/usr/bin/env python3
r"""
extract_dataset_stats.py

Walks the abandoned-object-detection dataset (one folder per sample, each
containing a video_*.txt description and mask*.png ground-truth masks),
parses the text files, measures the masks, and dumps everything needed to
argue novelty (object size range, threshold diversity, multi-object scenes,
etc.) into one JSON + one CSV + one printed summary.

USAGE
    python extract_dataset_stats.py --root D:\Minor_Project\dataset --out stats

EXPECTED LAYOUT (one of two)
    dataset/
      sample_01/
        video_1.txt
        mask1.png
        mask2.png
      sample_02/
        video_2.txt
        mask1.png
        ...
  or, if you point --root directly at a single sample folder, it will just
  process that one folder.

OUTPUT
    stats/per_sample.json   -> full structured dump, one entry per sample
    stats/masks.csv         -> one row per mask (for quick spreadsheet/plots)
    stats/summary.txt       -> printed dataset-wide numbers for the paper

No dependency beyond opencv-python, numpy, and the stdlib.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Text-file parsing
# ---------------------------------------------------------------------------

DURATION_RE = re.compile(r"Video Duration:\s*(\d+)\s*seconds", re.IGNORECASE)
P_RE = re.compile(r"Persistence Threshold\s*\(P\):\s*(\d+)\s*seconds", re.IGNORECASE)
C_RE = re.compile(r"Cooldown Period\s*\(C\):\s*(\d+)\s*seconds", re.IGNORECASE)

# "mask1.png: 0s – 61s"  (dash can be -, en-dash, or em-dash)
MASK_LINE_RE = re.compile(
    r"(mask\d+\.png)\s*:\s*(\d+)\s*s\s*[-\u2013\u2014]\s*(\d+)\s*s", re.IGNORECASE
)

# "placed at 32 seconds" / "appears at 32 seconds" / "placed on the ground at 32 seconds"
PLACED_RE = re.compile(
    r"(?:placed(?:\s+on the ground)?|appears)\s+at\s+(\d+)\s*seconds", re.IGNORECASE
)

# "A first bike is placed at 32 seconds" / "A bike appears at 18 seconds" /
# "A blue bucket is placed on the ground at 26 seconds"
# -> ordinal (optional) + category + time
OBJECT_PLACEMENT_RE = re.compile(
    r"\b(?:A|An|The)\s+(?:(first|second|third|fourth|fifth|sixth|seventh)\s+)?"
    r"([a-zA-Z][a-zA-Z\s\-]*?)\s+"
    r"(?:appears|is placed(?:\s+on the ground)?|was placed(?:\s+on the ground)?)\s+"
    r"at\s+(\d+)\s*seconds",
    re.IGNORECASE,
)

ORDINAL_WORDS = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh"}

COLOR_WORDS = {
    "red", "blue", "black", "white", "green", "yellow", "orange", "brown",
    "gray", "grey", "pink", "purple", "silver", "gold", "beige",
}

# phrases indicating the object is partially visible/occluded, not a new category
PARTIAL_PHRASE_RE = re.compile(
    r"^(?:rear half of a|front half of a|back half of a|half of a|partial)\s+",
    re.IGNORECASE,
)


def _clean_category(raw: str) -> str:
    """Strip leading articles/ordinals that leak into the captured noun phrase."""
    words = raw.strip().split()
    while words and words[0].lower() in ({"a", "an", "the"} | ORDINAL_WORDS):
        words.pop(0)
    return " ".join(words).strip().lower()


def normalize_category(raw: str) -> tuple:
    """Split a raw category phrase like 'red gas cylinder' or 'rear half of a
    bike' into a base object type ('gas cylinder', 'bike') plus a list of
    descriptor tags (['red'], ['partial']) so the dataset-wide rollup groups
    by object type instead of fragmenting on every color/occlusion variant."""
    detail_tags = []
    s = raw
    m = PARTIAL_PHRASE_RE.match(s)
    if m:
        detail_tags.append("partial")
        s = s[m.end():]
    words = s.split()
    while words and words[0].lower() in COLOR_WORDS:
        detail_tags.append(words.pop(0).lower())
    base = " ".join(words).strip().lower() or raw.strip().lower()
    return base, detail_tags


def parse_txt(txt_path: Path) -> dict:
    text = txt_path.read_text(encoding="utf-8", errors="ignore")

    duration = _first_int(DURATION_RE, text)
    p_thresh = _first_int(P_RE, text)
    cooldown = _first_int(C_RE, text)

    # section split: Scene Description / Ground Truth Masks / Explanation
    scene_desc = _section(text, "Scene Description", "Ground Truth Masks")
    masks_section = _section(text, "Ground Truth Masks", "Explanation")
    explanation_section = _section(text, "Explanation", None)

    mask_ranges = {}
    for m in MASK_LINE_RE.finditer(masks_section or text):
        fname, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        mask_ranges[fname] = {"start_s": start, "end_s": end}

    placements = [int(x) for x in PLACED_RE.findall(scene_desc or text)]

    # object categories: [{"ordinal": "first"/None, "category_phrase": "black bike",
    #                       "category": "bike", "category_detail": ["black"], "time_s": 32}, ...]
    object_categories = []
    for m in OBJECT_PLACEMENT_RE.finditer(scene_desc or text):
        ordinal, raw_category, time_s = m.group(1), m.group(2), int(m.group(3))
        phrase = _clean_category(raw_category)
        base, detail = normalize_category(phrase)
        object_categories.append(
            {
                "ordinal": ordinal.lower() if ordinal else None,
                "category_phrase": phrase,
                "category": base,
                "category_detail": detail,
                "time_s": time_s,
            }
        )

    # per-mask explanation text (best-effort: split on "maskN.png:" markers)
    explanations = {}
    if explanation_section:
        parts = re.split(r"(mask\d+\.png:)", explanation_section)
        # parts looks like ['', 'mask1.png:', ' text...', 'mask2.png:', ' text...']
        for i in range(1, len(parts) - 1, 2):
            key = parts[i].rstrip(":")
            explanations[key] = parts[i + 1].strip()

    return {
        "duration_s": duration,
        "persistence_threshold_s": p_thresh,
        "cooldown_period_s": cooldown,
        "num_objects_placed": len(placements),
        "object_placement_times_s": placements,
        "object_categories": object_categories,
        "mask_time_ranges": mask_ranges,
        "mask_explanations": explanations,
        "scene_description": (scene_desc or "").strip(),
    }


def _first_int(pattern: re.Pattern, text: str):
    m = pattern.search(text)
    return int(m.group(1)) if m else None


def _section(text: str, start_header: str, end_header: str | None) -> str | None:
    start = text.find(start_header)
    if start == -1:
        return None
    start += len(start_header)
    if end_header:
        end = text.find(end_header, start)
        if end == -1:
            end = len(text)
    else:
        end = len(text)
    return text[start:end]


# ---------------------------------------------------------------------------
# Mask image analysis
# ---------------------------------------------------------------------------

def analyze_mask(mask_path: Path) -> dict:
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": f"could not read {mask_path}"}

    h, w = img.shape
    total_px = h * w
    binary = (img > 127).astype(np.uint8)
    fg_px = int(binary.sum())
    coverage_pct = round(100.0 * fg_px / total_px, 4)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    components = []
    for lbl in range(1, num_labels):  # skip background label 0
        x, y, cw, ch, area = stats[lbl]
        cx, cy = centroids[lbl]
        components.append(
            {
                "bbox_xywh": [int(x), int(y), int(cw), int(ch)],
                "area_px": int(area),
                "area_pct_of_frame": round(100.0 * area / total_px, 4),
                "centroid": [round(float(cx), 1), round(float(cy), 1)],
                "aspect_ratio_w_over_h": round(cw / ch, 3) if ch else None,
                "position_in_frame": _classify_position(cx, cy, w, h),
                "edge_distance_pct": _edge_distance_pct(cx, cy, w, h),
            }
        )
    # largest first
    components.sort(key=lambda c: -c["area_px"])

    return {
        "width": w,
        "height": h,
        "is_blank": fg_px == 0,
        "foreground_coverage_pct": coverage_pct,
        "num_objects_in_mask": len(components),
        "components": components,
    }


def _classify_position(cx: float, cy: float, w: int, h: int) -> str:
    """3x3 grid label (e.g. 'bottom-right', 'center') based on centroid location."""
    col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    row = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
    if row == "middle" and col == "center":
        return "center"
    return f"{row}-{col}" if col != "center" else row


def _edge_distance_pct(cx: float, cy: float, w: int, h: int) -> float:
    """How close the centroid is to the nearest frame edge, as %% of the shorter
    frame dimension (0%% = touching an edge, larger = closer to center)."""
    dist = min(cx, w - cx, cy, h - cy)
    return round(100.0 * dist / min(w, h), 2)


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

def tag_mask_categories(explanation: str, object_categories: list) -> list:
    """Best-effort: which of the sample's named objects (e.g. 'first black bike')
    does this mask's explanation text refer to? Matches on the full descriptive
    phrase (so 'black' isn't lost) but returns the normalized base category
    (so 'black bike' and 'bike' land in the same rollup bucket). Falls back to
    the single object's category when the sample only has one object total."""
    if not explanation:
        return []
    text = explanation.lower()
    if len(object_categories) == 1:
        return [object_categories[0]["category"]]
    hits = []
    for obj in object_categories:
        phrase = f"{obj['ordinal']} {obj['category_phrase']}" if obj["ordinal"] else obj["category_phrase"]
        if phrase in text:
            hits.append(obj["category"])
    return sorted(set(hits))


def process_sample(sample_dir: Path) -> dict | None:
    txt_files = list(sample_dir.glob("video_*.txt")) or list(sample_dir.glob("*.txt"))
    if not txt_files:
        return None
    txt_path = txt_files[0]

    parsed = parse_txt(txt_path)
    mask_paths = sorted(
        sample_dir.glob("mask*.png"),
        key=lambda p: int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 0,
    )

    masks_out = {}
    resolutions = set()
    for mp in mask_paths:
        info = analyze_mask(mp)
        time_range = parsed["mask_time_ranges"].get(mp.name)
        info["time_range_s"] = time_range
        info["explanation"] = parsed["mask_explanations"].get(mp.name)
        info["categories_mentioned"] = tag_mask_categories(
            info["explanation"], parsed["object_categories"]
        )
        if "width" in info and "height" in info:
            resolutions.add((info["width"], info["height"]))
        masks_out[mp.name] = info

    active_masks = [m for m in masks_out.values() if not m.get("is_blank", True)]
    coverages = [m["foreground_coverage_pct"] for m in active_masks]

    if len(resolutions) == 1:
        resolution = list(resolutions)[0]
    elif len(resolutions) > 1:
        resolution = f"INCONSISTENT: {sorted(resolutions)}"
    else:
        resolution = None

    return {
        "sample_id": sample_dir.name,
        "txt_file": txt_path.name,
        "resolution_wh": resolution,
        "duration_s": parsed["duration_s"],
        "persistence_threshold_s": parsed["persistence_threshold_s"],
        "cooldown_period_s": parsed["cooldown_period_s"],
        "num_objects_placed": parsed["num_objects_placed"],
        "object_placement_times_s": parsed["object_placement_times_s"],
        "object_categories": parsed["object_categories"],
        "num_masks": len(masks_out),
        "num_active_masks": len(active_masks),
        "min_object_coverage_pct": min(coverages) if coverages else None,
        "max_object_coverage_pct": max(coverages) if coverages else None,
        "masks": masks_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_sample_dirs(root: Path) -> list[Path]:
    # a "sample dir" is any directory (including root itself) that directly
    # contains a txt description file.
    candidates = []
    if list(root.glob("video_*.txt")) or list(root.glob("*.txt")):
        candidates.append(root)
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if list(sub.glob("video_*.txt")) or list(sub.glob("*.txt")):
            candidates.append(sub)
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root or single sample folder")
    ap.add_argument("--out", default="stats", help="output directory")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = find_sample_dirs(root)
    if not sample_dirs:
        print(f"No sample folders with a .txt file found under {root}", file=sys.stderr)
        sys.exit(1)

    results = []
    for sd in sample_dirs:
        r = process_sample(sd)
        if r:
            results.append(r)
        else:
            print(f"[skip] {sd} — no txt file", file=sys.stderr)

    # --- JSON dump (full detail) ---
    with open(out_dir / "per_sample.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- CSV, one row per mask (easy to pivot / plot) ---
    csv_path = out_dir / "masks.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id", "mask_file", "resolution_wh", "start_s", "end_s", "is_blank",
                "coverage_pct", "num_objects_in_mask", "largest_object_pct",
                "largest_aspect_ratio_w_over_h", "largest_position_in_frame",
                "largest_edge_distance_pct", "categories_mentioned",
            ]
        )
        for r in results:
            for mask_name, m in r["masks"].items():
                tr = m.get("time_range_s") or {}
                largest = m["components"][0] if m.get("components") else {}
                writer.writerow(
                    [
                        r["sample_id"], mask_name, r.get("resolution_wh"),
                        tr.get("start_s"), tr.get("end_s"),
                        m.get("is_blank"), m.get("foreground_coverage_pct"),
                        m.get("num_objects_in_mask"),
                        largest.get("area_pct_of_frame", 0),
                        largest.get("aspect_ratio_w_over_h"),
                        largest.get("position_in_frame"),
                        largest.get("edge_distance_pct"),
                        ";".join(m.get("categories_mentioned", [])),
                    ]
                )

    # --- dataset-wide summary ---
    all_coverages = [
        m["foreground_coverage_pct"]
        for r in results
        for m in r["masks"].values()
        if not m.get("is_blank")
    ]
    all_p = [r["persistence_threshold_s"] for r in results if r["persistence_threshold_s"] is not None]
    all_c = [r["cooldown_period_s"] for r in results if r["cooldown_period_s"] is not None]
    all_durations = [r["duration_s"] for r in results if r["duration_s"] is not None]
    obj_counts = [r["num_objects_placed"] for r in results]

    # per-category coverage: only masks unambiguously tagged with ONE category
    category_coverage = {}
    resolutions_seen = set()
    position_counts = {}
    for r in results:
        rw = r.get("resolution_wh")
        if isinstance(rw, (list, tuple)):
            resolutions_seen.add(tuple(rw))
        for m in r["masks"].values():
            if m.get("is_blank"):
                continue
            cats = m.get("categories_mentioned") or []
            if len(cats) == 1:
                category_coverage.setdefault(cats[0], []).append(m["foreground_coverage_pct"])
            for c in m.get("components", []):
                pos = c.get("position_in_frame")
                if pos:
                    position_counts[pos] = position_counts.get(pos, 0) + 1

    def stat_block(name, vals):
        if not vals:
            return f"{name}: (no data)"
        return (
            f"{name}: n={len(vals)}  min={min(vals):.2f}  max={max(vals):.2f}  "
            f"mean={sum(vals)/len(vals):.2f}"
        )

    summary_lines = [
        f"Samples processed: {len(results)}",
        stat_block("Video duration (s)", all_durations),
        stat_block("Persistence threshold P (s)", all_p),
        stat_block("Cooldown period C (s)", all_c),
        stat_block("Objects placed per video", obj_counts),
        f"Multi-object samples (>1 object): {sum(1 for c in obj_counts if c > 1)}",
        stat_block("Object mask coverage %% of frame (active masks only)", all_coverages),
        f"Smallest object footprint seen: {min(all_coverages):.3f}% of frame" if all_coverages else "",
        f"Largest object footprint seen: {max(all_coverages):.3f}% of frame" if all_coverages else "",
        f"Distinct mask resolutions across dataset: {sorted(resolutions_seen) if resolutions_seen else '(none detected)'}",
        f"Object position-in-frame counts: {position_counts if position_counts else '(none)'}",
        "",
        "Coverage %% of frame by object category (unambiguous masks only):",
    ] + [
        f"  {cat}: {stat_block('', vals).replace(': ', '', 1)}"
        for cat, vals in sorted(category_coverage.items())
    ]
    summary_text = "\n".join(l for l in summary_lines if l is not None)
    (out_dir / "summary.txt").write_text(summary_text)

    print(summary_text)
    print(f"\nWrote: {out_dir/'per_sample.json'}, {out_dir/'masks.csv'}, {out_dir/'summary.txt'}")


if __name__ == "__main__":
    main()
