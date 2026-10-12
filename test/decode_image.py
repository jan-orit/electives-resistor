"""Run the classical reader on saved images and write debug sheets to captures/debug/.

    python3 test/decode_image.py captures/white_bg_captures_b2/47k_001.jpg
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "main"))  # resistor_cv lives in main/
DEBUG = os.path.normpath(os.path.join(HERE, "..", "captures", "debug"))

import resistor_cv as rc


def debug_sheet(frame, res):
    ann = frame.copy()
    if res["box"] is not None:
        cv2.polylines(ann, [res["box"]], True, (0, 255, 0), 2)
    label = rc.fmt(res["ohms"]) + (" ohm" if res["ohms"] is not None else "")
    cv2.putText(ann, label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
    if res["crop"] is None:
        return ann
    crop = res["crop"].copy()
    ch, cw = crop.shape[:2]
    cv2.rectangle(crop, (0, int(ch * rc.ROWS[0])), (cw, int(ch * rc.ROWS[1])), (0, 255, 255), 1)
    strip = np.zeros((40, cw, 3), np.uint8)
    prof = np.nan_to_num(res["profile"]).astype(np.uint8)
    strip[:] = prof[None]
    marks = np.full((60, cw, 3), 30, np.uint8)
    for i, b in enumerate(res["bands"]):
        cv2.rectangle(marks, (b["x0"], 0), (b["x1"], 20), (0, 200, 0), -1)
        cv2.putText(marks, f"{i+1}:{b['name']}", (b["x0"], 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(marks, "%d,%d,%d" % b["hsv"], (b["x0"], 56), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)
    right = np.vstack([crop, strip, marks])
    scale = 2 if cw < 400 else 1
    right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    W = max(ann.shape[1], right.shape[1])
    pad = lambda im: cv2.copyMakeBorder(im, 0, 0, 0, W - im.shape[1], cv2.BORDER_CONSTANT)
    return np.vstack([pad(ann), pad(right)])


MODEL = rc.load_model()

if __name__ == "__main__":
    os.makedirs(DEBUG, exist_ok=True)
    for path in sys.argv[1:]:
        frame = cv2.imread(path)
        res = rc.read_resistor(frame, MODEL)
        names = " - ".join(f"{b['name']}{b['hsv']}" for b in res["bands"])
        print(f"{os.path.basename(path)}: {rc.fmt(res['ohms'])} {res['reason']}\n    {names}")
        out = os.path.join(DEBUG, f"{os.path.splitext(os.path.basename(path))[0]}_dbg.jpg")
        cv2.imwrite(out, debug_sheet(frame, res))
