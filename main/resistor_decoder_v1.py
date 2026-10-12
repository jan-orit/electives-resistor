"""Live resistor reader: camera feed on screen, resistor value overlaid.

    python3 main/resistor_decoder_v1.py

Place one resistor in view (whole body inside the yellow margin, on the same background and
under the same lighting as the calibration shots). Press q or ESC to quit.

Uses color_model.json (written by calibrate.py). Window opens on the Pi's own screen even
when launched over SSH.
"""
import collections
import os
import threading
import time

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ["DISPLAY"] = ":0"

import cv2

import resistor_cv as rc

WIN = "Resistor Decoder v1"
HISTORY = 5   # last N decode results
AGREE = 3     # a value is shown once it wins this many of the last N
REJECT = "reject"  # history marker: a resistor was found but could not be read as a supported value
UNSUPPORTED_AFTER = 3.0  # seconds of continuous rejection before "not supported" is shown


def stable_value(history):
    """Most common label in the recent history, if it has enough votes; else None."""
    votes = collections.Counter(h for h in history if h and h != REJECT)
    if not votes:
        return None
    label, n = votes.most_common(1)[0]
    return label if n >= AGREE else None


def annotate(frame, res, shown, unsupported=False, margin=40):
    """Draw box, reading and band names on a copy of the frame."""
    view = frame.copy()
    h, w = view.shape[:2]
    cv2.rectangle(view, (margin, margin), (w - margin, h - margin), (255, 255, 0), 1)
    if res is not None and res["box"] is not None:
        color = (0, 255, 0) if shown else (0, 165, 255)
        cv2.polylines(view, [res["box"]], True, color, 2)
        names = [b["name"] for b in res["bands"] if b["name"] != "?"]
        if names:
            cv2.putText(view, "-".join(names), (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if shown:
        text = rc.fmt(rc.OHMS[shown]) + " ohm"
        cv2.putText(view, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 0), 3)
    elif unsupported and res is not None and res["box"] is not None:
        cv2.putText(view, "not supported", (10, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    elif res is not None and res["box"] is not None:
        cv2.putText(view, "reading...", (10, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 165, 255), 2)
    else:
        cv2.putText(view, "place a resistor", (10, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 200, 200), 2)
    return view


class Decoder(threading.Thread):
    """Decodes the newest frame in the background so the video never stalls."""

    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model
        self.frame = None
        self.result = None
        self.seq = 0
        self.history = collections.deque(maxlen=HISTORY)
        self.reject_since = None  # when the current unbroken run of rejections began
        self.running = True
        self.lock = threading.Lock()

    def is_unsupported(self):
        """True once a resistor has been rejected continuously for UNSUPPORTED_AFTER seconds.
        Call with self.lock held."""
        return self.reject_since is not None and time.time() - self.reject_since >= UNSUPPORTED_AFTER

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def run(self):
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            res = rc.read_resistor(frame, self.model)
            with self.lock:
                if res["ohms"] is not None:
                    self.history.append(res["label"])
                    self.reject_since = None
                elif res["box"] is not None:
                    self.history.append(REJECT)
                    if self.reject_since is None:
                        self.reject_since = time.time()
                else:
                    self.history.append(None)
                    self.reject_since = None  # nothing in view (e.g. hand still moving)
                self.result = res
                self.seq += 1


def main():
    model = rc.load_model()
    if model is None:
        raise SystemExit("color_model.json not found - run main/calibrate.py first")
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow(WIN)
    dec = Decoder(model)
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
                res, shown, unsupported = dec.result, stable_value(dec.history), dec.is_unsupported()
            state = rc.fmt(rc.OHMS[shown]) + " ohm" if shown else ("not supported" if unsupported else "-")
            if state != last_printed:
                print(state)
                last_printed = state
            cv2.imshow(WIN, annotate(frame, res, shown, unsupported))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        dec.running = False
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
