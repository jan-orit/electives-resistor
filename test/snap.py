"""Live camera preview for lining up resistors and capturing dataset frames.

    python3 test/snap.py -l 4k7   # SPACE saves captures/4k7_001.jpg, 002, ...
    python3 test/snap.py          # SPACE saves captures/snap_001.jpg, ...

Keys:  SPACE save   d toggle live decode overlay   g toggle alignment guides   q/ESC quit
The window opens on the Pi's own screen even if you launch this over SSH.
The sharpness number is variance of Laplacian: turn the focus ring until it peaks.
Saved files are the raw frame (no overlays).
"""
import argparse
import sys
import os
import time

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ["DISPLAY"] = ":0"  # SSH session: draw on the Pi's desktop

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "main"))  # resistor_cv lives in main/
CAPTURES = os.path.normpath(os.path.join(HERE, "..", "captures"))

import resistor_cv as rc

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--label", default="snap", help="resistor label used in filenames, e.g. 4k7")
ap.add_argument("--dev", type=int, default=0)
args = ap.parse_args()

os.makedirs(CAPTURES, exist_ok=True)
model = rc.load_model()  # None until calibrate.py has been run
cam = cv2.VideoCapture(args.dev)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def put(img, text, org, scale, color, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, org, font, scale, color, thick)


WIN = "Smart Component Bin - capture"
cv2.namedWindow(WIN)
show_decode, show_guides = True, True
flash_until, last_saved = 0.0, ""
res, frame_i = None, 0

while True:
    ok, frame = cam.read()
    if not ok:
        print("camera read failed")
        break
    raw = frame.copy()
    view = frame.copy()
    h, w = view.shape[:2]

    sharp = cv2.Laplacian(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    if show_decode:
        if frame_i % 3 == 0:  # decode is slower than the camera; refresh a few times a second
            res = rc.read_resistor(raw, model)
        if res is not None and res["box"] is not None:
            cv2.polylines(view, [res["box"]], True, (0, 255, 0), 2)
            txt = rc.fmt(res["ohms"]) + (" ohm" if res["ohms"] is not None else "")
            if res["ohms"] is None and res.get("label"):
                txt += f"  (unsure: {res['label']})"
            put(view, txt, (10, 80), 0.9, (0, 255, 255), 2, cv2.FONT_HERSHEY_DUPLEX)
            put(view, "-".join(b["name"] for b in res["bands"]), (10, 102), 0.5, (255, 255, 255))
        else:
            put(view, "no resistor found", (10, 80), 0.8, (0, 0, 255), 2, cv2.FONT_HERSHEY_DUPLEX)
    if show_guides:
        cv2.line(view, (0, h // 2), (w, h // 2), (255, 255, 0), 1)
        cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        m = 40  # keep the whole body inside this margin
        cv2.rectangle(view, (m, m), (w - m, h - m), (255, 255, 0), 1)

    put(view, f"sharp {sharp:.0f}   [{args.label}]", (10, 20), 0.55, (0, 255, 255))
    put(view, "SPACE save  d decode  g guides  q quit", (10, h - 40), 0.5, (255, 255, 255))
    if time.time() < flash_until:
        put(view, f"saved {last_saved}", (10, h - 15), 0.7, (0, 255, 0), 2, cv2.FONT_HERSHEY_DUPLEX)
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 255, 0), 6)

    cv2.imshow(WIN, view)
    frame_i += 1
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("d"):
        show_decode = not show_decode
    if key == ord("g"):
        show_guides = not show_guides
    if key == ord(" "):
        n = len([f for f in os.listdir(CAPTURES) if f.startswith(args.label + "_")]) + 1
        last_saved = os.path.join(CAPTURES, f"{args.label}_{n:03d}.jpg")
        cv2.imwrite(last_saved, raw)
        print(f"{last_saved}  sharpness={sharp:.1f}  read={rc.fmt(res['ohms']) if res else '-'}")
        flash_until = time.time() + 0.6

cam.release()
cv2.destroyAllWindows()
