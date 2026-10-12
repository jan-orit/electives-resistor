"""Build a YOLO detection dataset (one class: resistor) from labelled captures.

    python3 main/make_yolo_dataset.py

Reads <label>_NNN.jpg from captures/white_bg_captures_b2/, white_bg_captures_b3/ and captures/:
  * label in resistor_cv.VALUES -> positive image; box = bounding box of the body found by the
    contour detector (find_body). Boxes are auto-generated, then checked on the preview sheets.
  * label "empty"              -> negative image (empty chamber / distractors), empty label file.
Everything else (snap_*, probe.jpg, ...) is ignored.

Writes yolo_dataset/ (train/val/test 70/20/10, stratified per label), data.yaml, preview sheets
in yolo_dataset/preview/, and yolo_dataset.zip to copy to the training PC.
"""
import glob
import os
import random
import re
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import resistor_cv as rc  # noqa: E402

OUT = os.path.join(ROOT, "yolo_dataset")
SOURCES = [("b2", os.path.join(ROOT, "captures", "white_bg_captures_b2")),
           ("b3", os.path.join(ROOT, "captures", "white_bg_captures_b3")),
           ("new", os.path.join(ROOT, "captures"))]
random.seed(0)


def collect():
    items = []
    for tag, folder in SOURCES:
        for p in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            m = re.match(r"^(\w+?)_(\d+)\.jpg$", os.path.basename(p))
            if not m:
                continue
            label = m.group(1)
            if label in rc.VALUES or label == "empty":
                items.append((f"{tag}_{os.path.basename(p)}", label, p))
    return items


def split(items):
    """70/20/10 per label so every value appears in every split."""
    by = {}
    for it in items:
        by.setdefault(it[1], []).append(it)
    out = {"train": [], "val": [], "test": []}
    for label, lst in by.items():
        random.shuffle(lst)
        n = len(lst)
        n_test = max(1, round(0.1 * n)) if n >= 3 else 0
        n_val = max(1, round(0.2 * n)) if n >= 3 else 0
        out["test"] += lst[:n_test]
        out["val"] += lst[n_test:n_test + n_val]
        out["train"] += lst[n_test + n_val:]
    return out


def main():
    items = collect()
    if not items:
        raise SystemExit("no labelled images found")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, "images", sp))
        os.makedirs(os.path.join(OUT, "labels", sp))
    os.makedirs(os.path.join(OUT, "preview"))

    sets = split(items)
    failed, preview = [], []
    counts = {sp: [0, 0] for sp in sets}  # [positives, negatives]
    for sp, lst in sets.items():
        for name, label, path in lst:
            frame = cv2.imread(path)
            h, w = frame.shape[:2]
            box_line = ""
            if label != "empty":
                cnt, _ = rc.find_body(frame)
                if cnt is None:
                    failed.append(name)
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                box_line = f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n"
                counts[sp][0] += 1
                vis = frame.copy()
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                preview.append((name, vis))
            else:
                counts[sp][1] += 1
            shutil.copy(path, os.path.join(OUT, "images", sp, name))
            with open(os.path.join(OUT, "labels", sp, os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.write(box_line)

    with open(os.path.join(OUT, "data.yaml"), "w") as f:  # no 'path:' key: ultralytics then uses this file's folder
        f.write("train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: resistor\n")

    # preview sheets: 12 tiles each, filename on every tile
    for i in range(0, len(preview), 12):
        tiles = []
        for name, vis in preview[i:i + 12]:
            t = cv2.resize(vis, (320, 240))
            cv2.putText(t, name[:-4], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            tiles.append(t)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)]
        cv2.imwrite(os.path.join(OUT, "preview", f"sheet_{i // 12 + 1:02d}.jpg"), np.vstack(rows))

    shutil.make_archive(OUT, "zip", ROOT, "yolo_dataset")
    print("split          positives  negatives")
    for sp, (p, n) in counts.items():
        print(f"  {sp:5s}        {p:6d}     {n:6d}")
    tot_p = sum(c[0] for c in counts.values())
    tot_n = sum(c[1] for c in counts.values())
    print(f"total {tot_p + tot_n} images ({tot_p} with a resistor, {tot_n} empty/negative)")
    if failed:
        print("boxes could not be generated (skipped):", failed)
    print("previews:", os.path.join(OUT, "preview"))
    print("zip for the PC:", OUT + ".zip")


if __name__ == "__main__":
    main()
