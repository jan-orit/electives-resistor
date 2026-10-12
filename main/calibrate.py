"""Fit the band-colour model from labelled captures and report honest accuracy.

    python3 main/calibrate.py                                   # fit on captures/<label>_NNN.jpg
    python3 main/calibrate.py --dir captures/white_bg_captures_b2 captures/white_bg_captures_b3   # one or more folders

The label in each filename gives the expected band sequence (resistor_cv.VALUES), so no hand
labelling of bands is needed. Each colour is modelled as an independent Gaussian over
(hue, saturation, value); the fit alternates "assign each band to its expected colour" and
"re-estimate each colour" (hard-assignment EM), starting from rough priors.

Accuracy is leave-one-image-out: every image is classified by a model fitted WITHOUT it.
"""
import argparse
import collections
import glob
import json
import os
import re

import cv2
import numpy as np

import resistor_cv as rc

# Rough starting points, (mu_h, mu_s, mu_v), (sd_h, sd_s, sd_v). Only seeds the EM.
PRIOR = {
    "Black": ((120, 255, 4), (40, 40, 6)),
    "Brown": ((0, 255, 11), (12, 60, 25)),
    "Red": ((0, 255, 78), (8, 40, 30)),
    "Orange": ((7, 247, 180), (6, 40, 30)),
    "Yellow": ((22, 215, 130), (6, 40, 40)),
    "Violet": ((123, 255, 22), (15, 40, 10)),
    "Gold": ((15, 122, 150), (5, 30, 30)),
    "Green": ((70, 200, 50), (20, 60, 40)),
    "Blue": ((105, 230, 60), (20, 60, 40)),
    "Grey": ((110, 60, 140), (40, 50, 40)),
}


def parse(path):
    m = re.match(r"^(\w+?)_(\d+)\.jpg$", os.path.basename(path))
    return m.group(1) if m and m.group(1) in rc.ALL_SEQ else None


def best_for_label(segs, model, label):
    return rc.decode_segments(segs, model, only=label, values=rc.ALL_SEQ)


def fit(data, iters=4):
    """data: list of (label, segs). Returns model dict colour -> (mu, sd)."""
    model = dict(PRIOR)
    for _ in range(iters):
        samples = collections.defaultdict(list)
        for label, segs in data:
            dec = best_for_label(segs, model, label)
            if dec is None:
                continue
            for i, c in zip(dec["idx"], rc.ALL_SEQ[label]):
                samples[c].append(segs[i]["hsv"])
        new = {}
        for c, (pmu, psd) in PRIOR.items():
            xs = np.array(samples[c], dtype=float) if samples[c] else None
            if xs is None or len(xs) < 3:
                new[c] = model[c]
                continue
            # circular mean for hue
            ang = xs[:, 0] * 2 * np.pi / 180
            mh = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2) % 180
            dh = np.array([rc._hue_d(h, mh) for h in xs[:, 0]])
            mu = (mh, xs[:, 1].mean(), xs[:, 2].mean())
            sd = tuple(max(f, s) for f, s in zip(rc.SD_FLOOR, (np.sqrt((dh ** 2).mean()), xs[:, 1].std(), xs[:, 2].std())))
            new[c] = (mu, sd)
        model = new
    return model


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--dir", nargs="+", default=[os.path.join(here, "..", "captures")], help="one or more folders of <label>_NNN.jpg")
    ap.add_argument("--out", default=os.path.join(here, "color_model.json"))
    args = ap.parse_args()

    files = sorted(p for d in args.dir for p in glob.glob(os.path.join(d, "*.jpg")) if parse(p))
    if not files:
        raise SystemExit("no labelled captures found")
    names, data, skipped = [], [], []
    for p in files:
        frame = cv2.imread(p)
        r = rc.read_resistor(frame)
        if r.get("profile") is None:
            skipped.append((p, "no body"))
            continue
        segs = rc.band_segments(r["profile"])
        if len(segs) < 3:
            skipped.append((p, f"{len(segs)} bands found"))
            continue
        names.append(p)
        data.append((parse(p), segs))
    print(f"{len(data)} usable images, {len(skipped)} skipped")
    for p, why in skipped:
        print("  skipped", os.path.basename(p), why)
    counts = collections.Counter(l for l, _ in data)
    print("per label:", dict(sorted(counts.items())))

    # leave-one-image-out: each image is decoded by a model fitted without it
    folds = []
    for i, (label, segs) in enumerate(data):
        model = fit(data[:i] + data[i + 1:])
        dec = rc.decode_segments(segs, model)
        foreign = rc.foreign_band(segs, dec, model) if dec else None
        folds.append((names[i], label, dec, foreign, rc.max_band_z(segs, dec, model) if dec else None))

    in_set = [f for f in folds if f[1] in rc.VALUES]
    out_set = [f for f in folds if f[1] not in rc.VALUES]
    correct_scores = [f[2]["score"] for f in in_set if f[2] and f[2]["label"] == f[1] and not f[3]]
    min_score = (min(correct_scores) - 2.0) if correct_scores else -1e9
    min_margin = 3.0

    def outcome(f):
        _, label, dec, foreign, z = f
        if dec is None or foreign or (z is not None and z > rc.BAND_Z) or dec["score"] < min_score or dec["margin"] < min_margin:
            return "reject"
        return dec["label"]

    print("\n== In-set values (should be read correctly) ==")
    conf = collections.defaultdict(collections.Counter)
    for f in in_set:
        conf[f[1]][outcome(f)] += 1
    labels = [l for l in rc.VALUES if l in conf]
    cols = labels + ["reject"] + sorted({o for c in conf.values() for o in c} - set(labels) - {"reject"})
    print("%6s" % "" + "".join("%7s" % c for c in cols))
    for l in labels:
        print("%6s" % l + "".join("%7d" % conf[l][c] for c in cols))
    ok = sum(1 for f in in_set if outcome(f) == f[1])
    wrong = [f for f in in_set if outcome(f) not in (f[1], "reject")]
    rej = [f for f in in_set if outcome(f) == "reject"]
    print(f"correct {ok}/{len(in_set)} ({100 * ok / max(1, len(in_set)):.1f}%), wrong bin {len(wrong)}, rejected {len(rej)}")
    for f in wrong:
        print(f"  WRONG BIN: {os.path.basename(f[0])} true {f[1]} -> {outcome(f)}")
    for f in rej:
        why = f"foreign {f[3]}" if f[3] else (f"off-colour band z={f[4]:.1f}" if f[4] and f[4] > rc.BAND_Z else f"score {f[2]['score']:.1f} margin {f[2]['margin']:.1f}" if f[2] else "too few bands")
        print(f"  rejected: {os.path.basename(f[0])} true {f[1]} ({why})")

    if out_set:
        print("\n== Out-of-set values (should be REJECTED) ==")
        oconf = collections.defaultdict(collections.Counter)
        for f in out_set:
            oconf[f[1]][outcome(f)] += 1
        for l, c in oconf.items():
            print(f"  {l}: " + ", ".join(f"{k} x{v}" for k, v in c.most_common()))
        bad = [f for f in out_set if outcome(f) != "reject"]
        print(f"rejected {len(out_set) - len(bad)}/{len(out_set)}; forced into a wrong bin: {len(bad)}")
        for f in bad:
            print(f"  WRONG BIN: {os.path.basename(f[0])} true {f[1]} -> {outcome(f)}")

    ms = np.array([f[2]["margin"] for f in in_set if f[2] and f[2]["label"] == f[1]])
    if len(ms):
        print(f"\nmargin of correct in-set reads: min {ms.min():.1f}, median {np.median(ms):.1f}")

    model = fit(data)
    out = {"colors": {c: {"mu": [float(x) for x in m], "sd": [float(x) for x in s]} for c, (m, s) in model.items()},
           "reject": {"min_score": float(min_score), "min_margin": min_margin},
           "trained_on": dict(counts)}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    print("\nFitted colours (hue, sat, val)  +/- sd")
    for c, (m, s) in model.items():
        print(f"  {c:7s} mu=({m[0]:5.1f},{m[1]:5.1f},{m[2]:5.1f})  sd=({s[0]:4.1f},{s[1]:4.1f},{s[2]:4.1f})")


if __name__ == "__main__":
    main()
