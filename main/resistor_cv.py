"""Classical (no-ML) resistor reader: one frame in, resistance + debug info out.

Pipeline: background-distance segmentation -> body contour -> straighten ->
per-column median colour -> band segments -> colour names -> value.
"""
import cv2
import numpy as np

DIGIT = {"Black": 0, "Brown": 1, "Red": 2, "Orange": 3, "Yellow": 4,
         "Green": 5, "Blue": 6, "Violet": 7, "Grey": 8, "White": 9}

# Tunables (pixels at 640x480 with the microscope framing used so far)
LEAD_OPEN = 45       # ellipse that erases leads (thinner than this) but keeps the body
MASK_DE = 40         # body-vs-background chroma distance threshold (bg <=~25, body >=~60)
BAND_DE = 30         # min distance from body colour (Lab, or 0.5*saturation gap) to count as "band"
MIN_BAND_W = 5       # narrowest band, in columns
TIP = 0.10           # fraction of body length ignored at each rounded tip
MERGE_GAP = 3        # columns of gap that still merge two pieces of one band
ROWS = (0.25, 0.75)   # rows of the straightened crop to sample
GLARE_S, GLARE_V = 70, 180  # pixels below this saturation and above this value are specular glare


def _lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def find_body(frame):
    """Return (contour, mask) of the most plausible resistor body, or (None, None)."""
    h, w = frame.shape[:2]
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    lab = _lab(blur)
    ring = np.concatenate([lab[:6].reshape(-1, 3), lab[-6:].reshape(-1, 3),
                           lab[:, :6].reshape(-1, 3), lab[:, -6:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    # Chroma-only distance: the beige body is ~10-19 Lab-chroma units from a grey/black/white
    # background (<=~4). Brightness is ignored on purpose: auto-exposure and paper speckle swing it.
    # Directional: the resistor is always shifted toward red/yellow (a*, b* up); a cool/bluish
    # patch of paper is shifted the other way and must not count.
    shift = np.maximum(lab[..., 1:] - bg[1:], 0)
    de = 6 * np.linalg.norm(shift, axis=2)
    de8 = np.clip(de, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(de8, MASK_DE, 255, cv2.THRESH_BINARY)
    # A black band is as dark as black paper and punches a gap in the mask: bridge it along x.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 3)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LEAD_OPEN, LEAD_OPEN))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 4000:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        score = area * (0.15 if touches else 1.0)  # prefer fully visible parts
        if score > best_score:
            best, best_score = c, score
    if best is None:
        return None, mask
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [best], -1, 255, -1)
    return best, clean


def straighten(frame, cnt, mask):
    """Rotate so the body's long axis is horizontal; return (crop, crop_mask, angle)."""
    m = cv2.moments(cnt)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    theta = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    h, w = frame.shape[:2]
    d = int(np.hypot(h, w))
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    M[0, 2] += d / 2 - cx
    M[1, 2] += d / 2 - cy
    rot = cv2.warpAffine(frame, M, (d, d), flags=cv2.INTER_CUBIC)
    rmask = cv2.warpAffine(mask, M, (d, d), flags=cv2.INTER_NEAREST)
    x, y, bw, bh = cv2.boundingRect(rmask)
    return rot[y:y + bh, x:x + bw], rmask[y:y + bh, x:x + bw], theta


def column_profile(crop, cmask):
    """Per-column median colour (BGR) over the middle rows, ignoring specular glare pixels.

    NaN where the body isn't present. Glare = low saturation + high brightness; dropping those
    pixels (instead of picking an edge strip) keeps the lit colour of the band, not its shadow.
    """
    ch, cw = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    glare = (hsv[..., 1] < GLARE_S) & (hsv[..., 2] > GLARE_V)
    r0, r1 = int(ch * ROWS[0]), int(ch * ROWS[1])
    prof = np.full((cw, 3), np.nan, np.float32)
    for x in range(cw):
        body = cmask[r0:r1, x] > 0
        if body.sum() < max(3, 0.5 * (r1 - r0)):
            continue
        keep = body & ~glare[r0:r1, x]
        sel = keep if keep.sum() >= 0.2 * (r1 - r0) else body
        prof[x] = np.median(crop[r0:r1, x][sel], axis=0)
    return prof


def name_color(h, s, v):
    """HSV (OpenCV scale) -> colour name, or 'Glare'/'Unknown'. Rules are first-pass; see calibration."""
    if v < 70:
        return "Black"
    if s < 60 and v > 190:
        return "Glare"
    if s < 70:
        return "Grey" if v < 170 else "Glare"
    if h <= 8 or h >= 165:
        return "Red" if v >= 200 else "Brown"
    if h <= 13:
        return "Brown" if v < 160 else "Orange"
    if h <= 24:
        return "Gold" if s < 135 else "Orange"
    if h <= 35:
        return "Yellow"
    if h <= 85:
        return "Green"
    if h <= 130:
        return "Blue"
    return "Violet"


def band_segments(prof):
    """Segment the column profile into colour bands (colour-name free).

    A column is "band" if it differs from the body colour in Lab (dark/red/violet bands) or in
    saturation (yellow is nearly the same Lab colour as cream but much more saturated).
    """
    valid = ~np.isnan(prof[:, 0])
    if valid.sum() < 50:
        return []
    cols = np.where(valid)[0]
    u8 = np.nan_to_num(prof).reshape(1, -1, 3).astype(np.uint8)
    lab = _lab(u8)[0]
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)[0].astype(np.float32)
    de = np.linalg.norm(lab - np.median(lab[cols], axis=0), axis=1)
    ds = np.abs(hsv[:, 1] - np.median(hsv[cols, 1]))
    score = np.maximum(de, 0.5 * ds)
    score[~valid] = 0
    lo = cols[0] + int(TIP * len(cols))
    hi = cols[-1] - int(TIP * len(cols))
    is_band = score > BAND_DE
    is_band[:lo] = False
    is_band[hi:] = False

    segs, start = [], None
    for x in range(len(is_band) + 1):
        on = x < len(is_band) and is_band[x]
        if on and start is None:
            start = x
        elif not on and start is not None:
            segs.append([start, x - 1])
            start = None
    merged = []
    for s in segs:  # join fragments of one band; gold shows up as several thin lines
        if merged:
            gap = s[0] - merged[-1][1]
            thin = (s[1] - s[0] < 12) and (merged[-1][1] - merged[-1][0] < 12)
            if gap <= MERGE_GAP or (thin and gap <= 10):
                merged[-1][1] = s[1]
                continue
        merged.append(s)
    merged = [m for m in merged if m[1] - m[0] + 1 >= MIN_BAND_W]
    # drop thin dark edge lines hugging a wide band
    keep = []
    for i, (a, b) in enumerate(merged):
        w = b - a + 1
        near_wide = any((o[1] - o[0] + 1) >= 14 and min(abs(a - o[1]), abs(o[0] - b)) <= 8
                        for j, o in enumerate(merged) if j != i)
        if w < 7 and near_wide:
            continue
        keep.append((a, b))

    out = []
    for a, b in keep:
        w = b - a + 1
        core = prof[a + w // 4: b - w // 4 + 1] if w >= 8 else prof[a:b + 1]
        bgr = np.nanmedian(core, axis=0).astype(np.uint8).reshape(1, 1, 3)
        h, s_, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
        out.append({"x0": int(a), "x1": int(b), "hsv": (int(h), int(s_), int(v)),
                    "strength": float(score[a:b + 1].max())})
    return out


# ---- colour model + nine-value constrained decoding -------------------------------------
VALUES = {  # supported resistances -> band colours (digit, digit, multiplier, tolerance)
    "100": ("Brown", "Black", "Brown", "Gold"),
    "220": ("Red", "Red", "Brown", "Gold"),
    "1k": ("Brown", "Black", "Red", "Gold"),
    "2k2": ("Red", "Red", "Red", "Gold"),
    "4k7": ("Yellow", "Violet", "Red", "Gold"),
    "10k": ("Brown", "Black", "Orange", "Gold"),
    "47k": ("Yellow", "Violet", "Orange", "Gold"),
    "100k": ("Brown", "Black", "Yellow", "Gold"),
    "5k6": ("Green", "Blue", "Red", "Gold"),
    "680": ("Blue", "Grey", "Brown", "Gold"),
}
# Resistors that are NOT sortable but are used to teach the model what other colours look like
# (so a band that is really e.g. green is recognised as foreign and rejected). Currently empty.
TRAIN_ONLY = {}
ALL_SEQ = {**VALUES, **TRAIN_ONLY}
SUPPORTED_COLORS = {c for cols in VALUES.values() for c in cols}
OHMS = {"100": 100, "220": 220, "1k": 1e3, "2k2": 2.2e3, "4k7": 4.7e3,
        "10k": 10e3, "47k": 47e3, "100k": 100e3, "5k6": 5.6e3, "680": 680}
SD_FLOOR = (3.0, 12.0, 3.0)  # hue, saturation, value


def _hue_d(a, b):
    d = abs(a - b)
    return min(d, 180 - d)


def band_loglik(hsv, m):
    """Gaussian log-likelihood of a band colour under colour model m = (mu(3), sd(3))."""
    mu, sd = m
    z = (_hue_d(hsv[0], mu[0]) / sd[0], (hsv[1] - mu[1]) / sd[1], (hsv[2] - mu[2]) / sd[2])
    return -0.5 * sum(t * t for t in z) - sum(np.log(sd))


FOREIGN_FIT = -14.0  # absolute log-likelihood a band must reach under a foreign colour
FOREIGN_GAP = 4.0    # ...and beat the best supported colour by this much
BAND_Z = 3.5         # reject if any chosen band is further than this many sd from its assigned colour
MISS_GOLD = 4.0  # log-likelihood penalty for reading a value without seeing its gold band


def band_z(seg, m):
    """Distance of a band's colour from colour model m, in standard deviations (3-D)."""
    mu, sd = m
    h, s_, v = seg["hsv"]
    return float(np.sqrt((_hue_d(h, mu[0]) / sd[0]) ** 2 + ((s_ - mu[1]) / sd[1]) ** 2 + ((v - mu[2]) / sd[2]) ** 2))


def max_band_z(segs, dec, model, values=None):
    """Worst per-band colour mismatch among the bands the decoder chose. The average score hides
    one clearly wrong band (e.g. a 100 ohm forced to read as 1k); this does not."""
    cols = (values or VALUES)[dec["label"]]
    return max(band_z(segs[i], model[c]) for i, c in zip(dec["idx"], cols))


def decode_segments(segs, model, only=None, values=None):
    """Best (label, ordered segment indices, score, margin) over orientation, subset, value.

    Position-aware by construction: e.g. a very dark first band can only be brown. The gold
    tolerance band is optional (it is faint on some shots); without it the first three bands
    are decoded and the nine-value constraint resolves the reading direction.
    """
    import itertools
    values = values or VALUES
    if len(segs) < 3:
        return None
    cand = {}
    for k in (4, 3):
        if len(segs) < k:
            continue
        for combo in itertools.combinations(range(len(segs)), k):
            for idx in (combo, combo[::-1]):
                for lab, cols in values.items():
                    if only and lab != only:
                        continue
                    ll = sum(band_loglik(segs[i]["hsv"], model[c]) for i, c in zip(idx, cols))
                    if k == 3:
                        ll -= MISS_GOLD
                    if lab not in cand or ll > cand[lab][0]:
                        cand[lab] = (ll, idx)
    ranked = sorted(cand.items(), key=lambda kv: -kv[1][0])
    best_lab, (best_ll, best_idx) = ranked[0]
    margin = best_ll - ranked[1][1][0] if len(ranked) > 1 else float("inf")
    return {"label": best_lab, "idx": best_idx, "score": best_ll / len(best_idx), "margin": margin}


def foreign_band(segs, dec, model, values=None):
    """Return a colour name if ANY detected band clearly looks like a colour outside the
    supported set (green/blue/grey), whether or not the decoder used that band.

    A band counts as foreign when it fits a foreign colour well in absolute terms and clearly
    better than every supported colour. (Checking only the chosen bands misses the case where
    the decoder skips the green/blue bands and picks edge artefacts instead.)
    """
    for sg in segs:
        hsv = sg["hsv"]
        lls = {c: band_loglik(hsv, m) for c, m in model.items()}
        best = max(lls, key=lls.get)
        if best in SUPPORTED_COLORS:
            continue
        best_supported = max(v for c, v in lls.items() if c in SUPPORTED_COLORS)
        if lls[best] > FOREIGN_FIT and lls[best] - best_supported > FOREIGN_GAP:
            return best
    return None


def fmt(ohms):
    if ohms is None:
        return "reject"
    if ohms >= 1e6:
        return f"{ohms / 1e6:g}M"
    if ohms >= 1e3:
        return f"{ohms / 1e3:g}k"
    return f"{ohms:g}"


def load_model(path=None):
    import json
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_model.json")
    if not os.path.exists(path):
        return None
    raw = json.load(open(path))
    return {c: (tuple(v["mu"]), tuple(v["sd"])) for c, v in raw["colors"].items()}, raw.get("reject", {})


def read_resistor(frame, model=None):
    """Full pipeline. Always returns a dict; ohms is None when it can't be read confidently."""
    out = {"ohms": None, "label": None, "bands": [], "reason": "", "crop": None, "box": None}
    cnt, mask = find_body(frame)
    if cnt is None:
        out["reason"] = "no-body"
        return out
    out["box"] = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.int32)
    crop, cmask, theta = straighten(frame, cnt, mask)
    out["crop"], out["crop_mask"] = crop, cmask
    prof = column_profile(crop, cmask)
    out["profile"] = prof
    segs = band_segments(prof)
    for sgm in segs:
        sgm["name"] = "?"
    out["bands"] = segs
    if model is None:
        out["reason"] = "no colour model"
        return out
    colors, rej = model
    dec = decode_segments(segs, colors)
    if dec is None:
        out["reason"] = f"{len(segs)} bands"
        return out
    for i, c in zip(dec["idx"], VALUES[dec["label"]]):  # zip stops at 3 when gold is missing
        segs[i]["name"] = c
    out.update(decoded=dec, label=dec["label"])
    foreign = foreign_band(segs, dec, colors)
    if foreign:
        out["reason"] = f"foreign colour ({foreign}) - not a supported value"
    elif max_band_z(segs, dec, colors) > BAND_Z:
        out["reason"] = f"band off-colour (z {max_band_z(segs, dec, colors):.1f}) - not a supported value"
    elif dec["score"] < rej.get("min_score", -1e9) or dec["margin"] < rej.get("min_margin", 0):
        out["reason"] = f"low confidence (score {dec['score']:.1f}, margin {dec['margin']:.1f})"
    else:
        out["ohms"] = OHMS[dec["label"]]
    return out
