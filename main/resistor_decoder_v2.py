"""Live resistor reader v2: YOLOv8n localizes the resistor, the HSV decoder reads it.

    python3 main/resistor_decoder_v2.py                  # YOLO + decoder
    python3 main/resistor_decoder_v2.py --no-yolo        # same screen, contour detector only (= v1)
    python3 main/resistor_decoder_v2.py --weights main/best_320_ncnn_model --imgsz 320

    # one-time: convert the trained weights (best.pt from the PC) to NCNN for speed on the Pi
    python3 main/resistor_decoder_v2.py --export 320     # or 640 -> main/best_<size>_ncnn_model

Pipeline (paper section 3.4 / 3.5): YOLO finds the resistor box; the colour decoder then reads
only inside that box. If YOLO finds nothing the frame counts as "no resistor". Press q or ESC to quit.
v1 (resistor_decoder_v1.py) is untouched and works without any YOLO weights.
"""
import argparse
import os
import sys
import threading
import time

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ["DISPLAY"] = ":0"

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import resistor_cv as rc  # noqa: E402
import resistor_decoder_v1 as v1  # noqa: E402  (reuses its display + smoothing helpers)

WIN = "Resistor Decoder v2 (YOLO)"
PAD = 12  # pixels added around the YOLO box before the decoder searches inside it


class YoloLocalizer:
    """Wraps an ultralytics model (.pt or exported NCNN folder); returns the best resistor box."""

    def __init__(self, weights, imgsz=320, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights, task="detect")
        self.imgsz, self.conf = imgsz, conf

    def __call__(self, frame):
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device="cpu", verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = (float(v) for v in r.boxes.xyxy[i])
        return (x1, y1, x2, y2, float(r.boxes.conf[i]))


def read(frame, model, localizer=None):
    """YOLO box (if a localizer is given) -> decode inside it. Adds res["yolo"] and res["ms"]."""
    t0 = time.perf_counter()
    det = localizer(frame) if localizer else None
    t1 = time.perf_counter()
    if localizer and det is None:  # paper 3.5: no detection -> nothing to read
        res = {"ohms": None, "label": None, "bands": [], "reason": "yolo: no resistor",
               "crop": None, "box": None}
    else:
        roi = None if det is None else (det[0] - PAD, det[1] - PAD, det[2] + PAD, det[3] + PAD)
        res = rc.read_resistor(frame, model, roi=roi)
    t2 = time.perf_counter()
    res["yolo"] = det
    res["ms"] = ((t1 - t0) * 1000, (t2 - t1) * 1000)
    return res


class Decoder(v1.Decoder):
    """v1's background thread, with the YOLO-aware read()."""

    def __init__(self, model, localizer):
        super().__init__(model)
        self.localizer = localizer

    def run(self):
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            res = read(frame, self.model, self.localizer)
            with self.lock:
                if res["ohms"] is not None:
                    self.history.append(res["label"])
                    self.reject_since = None
                elif res["box"] is not None:
                    self.history.append(v1.REJECT)
                    if self.reject_since is None:
                        self.reject_since = time.time()
                else:
                    self.history.append(None)
                    self.reject_since = None
                self.result = res
                self.seq += 1


def overlay(view, res, mode):
    """YOLO box + timing on top of v1's annotation."""
    h, w = view.shape[:2]
    if res is not None and res.get("yolo"):
        x1, y1, x2, y2, c = res["yolo"]
        cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (255, 128, 0), 1)
        cv2.putText(view, f"resistor {c:.2f}", (int(x1), max(12, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1)
    if res is not None and res.get("ms"):
        y_ms, d_ms = res["ms"]
        cv2.putText(view, f"{mode}: yolo {y_ms:.0f} ms + decode {d_ms:.0f} ms", (10, h - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view


def export(size):
    src = os.path.join(HERE, "best.pt")
    if not os.path.exists(src):
        raise SystemExit("main/best.pt not found - copy it from the training PC first")
    from ultralytics import YOLO
    out = YOLO(src).export(format="ncnn", imgsz=size, half=False)
    dst = os.path.join(HERE, f"best_{size}_ncnn_model")
    if os.path.exists(dst):
        import shutil
        shutil.rmtree(dst)
    os.rename(out, dst)
    print("exported ->", dst)


def default_weights(imgsz):
    for cand in (f"best_{imgsz}_ncnn_model", "best.pt"):
        p = os.path.join(HERE, cand)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--export", type=int, metavar="SIZE")
    args = ap.parse_args()
    if args.export:
        return export(args.export)

    model = rc.load_model()
    if model is None:
        raise SystemExit("color_model.json not found - run main/calibrate.py first")
    localizer, mode = None, "contour only"
    if not args.no_yolo:
        weights = args.weights or default_weights(args.imgsz)
        if weights is None:
            print("WARNING: no YOLO weights found (main/best.pt or main/best_<size>_ncnn_model) - "
                  "falling back to the contour detector")
        else:
            localizer, mode = YoloLocalizer(weights, args.imgsz, args.conf), f"YOLO@{args.imgsz}"
            print("YOLO weights:", weights)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow(WIN)
    dec = Decoder(model, localizer)
    dec.start()
    last_printed = None
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("camera read failed")
                break
            dec.submit(frame)
            with dec.lock:
                res, shown, unsupported = dec.result, v1.stable_value(dec.history), dec.is_unsupported()
            state = rc.fmt(rc.OHMS[shown]) + " ohm" if shown else ("not supported" if unsupported else "-")
            if state != last_printed:
                print(state)
                last_printed = state
            cv2.imshow(WIN, overlay(v1.annotate(frame, res, shown, unsupported), res, mode))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        dec.running = False
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
