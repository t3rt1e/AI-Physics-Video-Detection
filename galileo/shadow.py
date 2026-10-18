"""The ball's shadow on the floor, frame by frame.

A shadow is the backdrop, only darker. Each frame is divided by the still
background (the median of the clip) and by the camera's overall exposure;
patches clearly darker than the backdrop, around where the ball lands and
outside the ball itself, are the candidates, and the one closest to below
the ball is taken. Whether it really is the ball's shadow (it moves with the
ball) is for the checks to say (galileo/checks.py).

Light travels in straight lines and arrives at once, which gives two tests:
  * at a bounce the ball touches the floor, so it touches its own shadow;
  * a far light (a lamp across the room, a window, the sun) puts the
    shadow's centre at the ball's foot point plus a fixed multiple of the
    ball's height: the shadow moves in step with the ball, with no lag.
"""
from __future__ import annotations

import cv2
import numpy as np

DARK = 0.07          # darker than the backdrop by this share: a shadow candidate
SPAN_X = 4.0         # how far to the side to look, ball widths
SPAN_Y = (1.5, 1.5)  # how far above and below the floor line to look, ball widths


def gains(frames, bg_gray) -> list[float]:
    """Each frame's overall brightness against the background (auto exposure)."""
    out = []
    b = bg_gray[::8, ::8]
    for im in frames:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        r = g[::8, ::8].astype(np.float32) / b
        out.append(float(np.median(r)) if r.size else 1.0)
    return out


def observe(frames, bg, dets, floor_y) -> dict:
    """The shadow candidate next to each detection of the ball.

    frames: the steadied frames (BGR); bg: their background (BGR); dets: the
    ball's detections (f, x, y, d in the frames' pixels); floor_y: per
    detection, the floor line (where the ball's bottom touches at the nearest
    bounce), NaN where unknown. Returns arrays, one entry per detection:
    found, sx, sy (darkness-weighted centre), area (in ball widths squared),
    mass (darkness x area, same units), dark (peak darkening), gap (from the
    ball's edge to the nearest shadow pixel, ball widths; 0.2 is touching).
    """
    n = len(dets)
    out = {k: np.full(n, np.nan) for k in ("sx", "sy", "area", "mass", "dark", "gap")}
    out["found"] = np.zeros(n, bool)
    if bg is None or n == 0:
        return out
    bg_g = (cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) if bg.ndim == 3 else bg).astype(np.float32) + 1.0
    H, W = bg_g.shape[:2]
    gain_cache: dict = {}
    for i, dt in enumerate(dets):
        fy = floor_y[i]
        if not np.isfinite(fy) or not np.isfinite(dt.d) or dt.d < 4:
            continue
        d = float(dt.d)
        x0 = int(max(0, dt.x - SPAN_X * d))
        x1 = int(min(W, dt.x + SPAN_X * d + 1))
        y0 = int(max(0, fy - SPAN_Y[0] * d))
        y1 = int(min(H, fy + SPAN_Y[1] * d + 1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        im = frames[dt.f]
        g = (cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) if im.ndim == 3
             else im[y0:y1, x0:x1]).astype(np.float32)
        if dt.f not in gain_cache:
            gain_cache[dt.f] = gains([im], bg_g)[0]
        ratio = g / (bg_g[y0:y1, x0:x1] * max(gain_cache[dt.f], 1e-3))
        ratio = cv2.GaussianBlur(ratio, (0, 0), max(1.0, 0.08 * d))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx - dt.x) ** 2 + (yy - dt.y) ** 2 <= (0.7 * d) ** 2
        darkness = np.where(inside, 0.0, 1.0 - ratio)
        mask = (darkness > DARK).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_score = None, 0.0
        for k in range(1, n_lab):
            if stats[k, cv2.CC_STAT_AREA] < max(4, 0.03 * d * d):
                continue
            sel = lab == k
            wgt = darkness[sel]
            m = float(wgt.sum())
            cx = float((xx[sel] * wgt).sum() / m)
            cy = float((yy[sel] * wgt).sum() / m)
            dist = np.hypot(cx - dt.x, cy - fy) / d
            score = m * np.exp(-0.5 * (dist / 1.5) ** 2)
            if score > best_score:
                best, best_score = (k, sel, wgt, m, cx, cy), score
        if best is None:
            continue
        k, sel, wgt, m, cx, cy = best
        # From the ball's edge to the shadow's nearest pixel, in ball widths (at
        # least 0.2: the ball itself, and a rim around it, are left out).
        gap = float(np.min(np.hypot(xx[sel] - dt.x, yy[sel] - dt.y))) / d - 0.5
        out["found"][i] = True
        out["sx"][i], out["sy"][i] = cx, cy
        out["area"][i] = float(sel.sum()) / (d * d)
        out["mass"][i] = m / (d * d)
        out["dark"][i] = float(wgt.max())
        out["gap"][i] = gap
    return out
