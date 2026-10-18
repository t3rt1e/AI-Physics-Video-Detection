"""The room's straight lines: how the camera is tipped, read off the scene.

Door frames, wall corners and shelf uprights are vertical. A level camera
keeps them parallel in the picture; tip it down and they lean together
toward a point far below the picture (tip it up: above), the vertical
vanishing point. How quickly they converge, kappa = 1 / (distance from the
picture's centre to that point), gives the camera's pitch once the focal
length f is known:

    tan(pitch) = f * kappa

Horizontal lines that run away from the camera (floor seams, the foot of a
side wall) meet on the horizon, which sits f * tan(pitch) above the centre
for a camera looking down. The two together fix f as well:

    f^2 = horizon distance / kappa

Everything here is measured on the still background (the median of the
steadied frames), so people and the ball don't add lines. Results are in
units of the picture's longer side L: K = kappa * L, h = horizon distance /
L, and f as a fraction of L.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def segments(img, min_frac: float = 0.035, max_n: int = 600) -> np.ndarray:
    """Straight edges in a still picture, (n, 4) as x1, y1, x2, y2, longest first."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape[:2]
    segs = None
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        segs = lsd.detect(g)[0]
    except Exception:
        segs = None
    if segs is None:
        edges = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 50, 150)
        segs = cv2.HoughLinesP(edges, 1, np.pi / 720, 40, minLineLength=int(min_frac * math.hypot(W, H)),
                               maxLineGap=4)
    if segs is None:
        return np.zeros((0, 4))
    segs = np.asarray(segs, float).reshape(-1, 4)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    keep = L >= max(min_frac * math.hypot(W, H), 12.0)
    # The picture's own border (and black bars) isn't part of the room.
    b = 4.0
    xs, ys = segs[:, [0, 2]], segs[:, [1, 3]]
    on_edge = ((xs <= b).all(1) | (xs >= W - 1 - b).all(1) | (ys <= b).all(1)
               | (ys >= H - 1 - b).all(1))
    keep &= ~on_edge
    segs, L = segs[keep], L[keep]
    order = np.argsort(-L)[:max_n]
    return segs[order]


def _geom(segs, c, S):
    p1, p2 = segs[:, :2], segs[:, 2:]
    m = (0.5 * (p1 + p2) - c) / S
    e = p2 - p1
    L = np.hypot(e[:, 0], e[:, 1])
    e = e / np.maximum(L, 1e-9)[:, None]
    return m, e, L


def _resid(q, m, e):
    """|sin| of the angle between each segment and the line from its middle
    to the vanishing point q (homogeneous, normalized picture coordinates)."""
    d = q[None, :2] - q[2] * m
    nd = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)
    return np.abs(e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]) / nd


def _lines(m, e):
    return np.column_stack([e[:, 1], -e[:, 0], e[:, 0] * m[:, 1] - e[:, 1] * m[:, 0]])


def _refine(q, m, e, L):
    """Length-weighted least squares on the segments' angles, q on the unit sphere."""
    from scipy.optimize import least_squares

    def to_q(a):
        return np.array([math.cos(a[0]) * math.cos(a[1]), math.sin(a[0]) * math.cos(a[1]),
                         math.sin(a[1])])
    a0 = np.array([math.atan2(q[1], q[0]), math.asin(float(np.clip(q[2], -1, 1)))])
    sw = np.sqrt(L / L.sum())

    def res(a):
        qq = to_q(a)
        d = qq[None, :2] - qq[2] * m
        nd = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)
        return sw * (e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]) / nd
    try:
        sol = least_squares(res, a0, max_nfev=200)
        return to_q(sol.x)
    except Exception:
        return q


def _ransac(m, e, L, idx, rng, iters=400, thr_deg=1.0, accept=None):
    if len(idx) < 3:
        return None
    lines = _lines(m[idx], e[idx])
    w = L[idx] / L[idx].sum()
    thr = math.sin(math.radians(thr_deg))
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(idx), 2, replace=False, p=w)
        q = np.cross(lines[i], lines[j])
        nq = float(np.linalg.norm(q))
        if nq < 1e-12:
            continue
        q = q / nq
        if accept is not None and not accept(q):
            continue
        r = _resid(q, m[idx], e[idx])
        inl = r < thr
        score = float(L[idx][inl].sum())
        if best is None or score > best[0]:
            best = (score, q, inl)
    if best is None:
        return None
    score, q, inl = best
    for _ in range(2):
        sel = idx[inl]
        if len(sel) < 3:
            return None
        q = _refine(q, m[sel], e[sel], L[sel])
        inl = _resid(q, m[idx], e[idx]) < 1.5 * thr
    return q, idx[inl]


def _orient(q, down):
    if q[0] * down[0] + q[1] * down[1] < 0:
        q = -q
    return q


def vertical(segs, W, H, down, *, K_window=(-0.6, 0.6), max_dev: float = 30.0,
             seed: int = 0, c=None) -> dict | None:
    """The vertical vanishing point, from lines running roughly along `down`
    (the picture's down for the falling object, a unit 2-vector).

    K_window: the convergence values worth considering. Lines running away
    from the camera across the floor also meet on the down axis (on the
    horizon), so without other evidence only a moderate tilt is believed;
    the ball's own fall narrows the window (novex/camera.py).
    Returns {"K", "K_sigma", "roll", "roll_sigma", "lines", "rms_deg",
    "inliers"}, or None without enough well-spread vertical lines.
    """
    if segs is None or len(segs) < 3:
        return None
    rng = np.random.default_rng(seed)
    c = np.array([W / 2.0, H / 2.0]) if c is None else np.asarray(c, float)
    S = 0.5 * math.hypot(W, H)
    Lmax = float(max(W, H))
    m, e, L = _geom(segs, c, S)
    down = np.asarray(down, float) / np.linalg.norm(down)
    cosang = np.abs(e @ down)
    idx = np.flatnonzero(cosang >= math.cos(math.radians(max_dev)))

    def K_of(qq):
        qq = _orient(qq, down)
        nxy = math.hypot(qq[0], qq[1])
        return qq[2] / max(nxy, 1e-12) * Lmax / S, math.degrees(math.atan2(qq[0], qq[1]))

    def accept(q):
        qq = _orient(q, down)
        n2 = math.hypot(qq[0], qq[1])
        if n2 < 1e-9 or (qq[0] * down[0] + qq[1] * down[1]) / n2 < math.cos(math.radians(40)):
            return False                  # off to the side: not where gravity's point can be
        return K_window[0] <= K_of(qq)[0] <= K_window[1]
    got = _ransac(m, e, L, idx, rng, accept=accept)
    if got is None:
        return None
    q, inl = got
    q = _orient(q, down)
    # Separate uprights (a door frame's two edges, a wall seam) at different
    # places across the picture: convergence is read between them. Pieces
    # of one upright count once. A single slanted thing (a person's leg)
    # must not be able to tip the answer, so it takes three uprights, or two
    # long ones far apart.
    perp = np.array([-down[1], down[0]])
    pos = (m[inl] @ perp) * S
    order = np.argsort(pos)
    groups, cur = [], [inl[order[0]]] if len(inl) else []
    for a, b in zip(order, order[1:]):
        if pos[b] - pos[a] > 0.02 * W:
            groups.append(cur)
            cur = []
        cur.append(inl[b])
    if cur:
        groups.append(cur)
    glen = [float(L[g].sum()) for g in groups]
    gpos = [float(np.average((m[g] @ perp) * S, weights=L[g])) for g in groups]
    long_pair = (len(groups) == 2 and min(glen) >= 0.45 * H and abs(gpos[1] - gpos[0]) >= 0.3 * W)
    if not (len(groups) >= 3 or long_pair) or L[inl].sum() < 0.35 * min(W, H):
        return None
    K, roll = K_of(q)
    if not K_window[0] - 0.05 <= K <= K_window[1] + 0.05:
        return None
    # Error bars: resampling the lines, leaving out one upright at a time,
    # and what a quarter of a degree per line allows at this spread.
    Ks, Rs = [], []
    for _ in range(30):
        b = rng.choice(inl, len(inl), replace=True)
        if len(np.unique(b)) < 3:
            continue
        qb = _refine(q, m[b], e[b], L[b])
        kb, rb = K_of(qb)
        Ks.append(kb)
        Rs.append(rb)
    if len(Ks) < 10:
        return None
    jack = []
    if len(groups) >= 3:
        for g in groups:
            rest = np.setdiff1d(inl, g)
            jack.append(K_of(_refine(q, m[rest], e[rest], L[rest]))[0])
    ng = len(groups)
    s_jack = float(np.sqrt((ng - 1) / ng * np.sum((np.array(jack) - np.mean(jack)) ** 2))) \
        if jack else 0.0
    rms = math.degrees(math.asin(min(1.0, float(np.sqrt(np.average(
        _resid(q, m[inl], e[inl]) ** 2, weights=L[inl]))))))
    spread = float(np.std(gpos)) if ng > 1 else 1.0
    s_line = math.radians(max(rms, 0.25)) / max(spread * math.sqrt(ng), 1.0) * Lmax
    return {"K": float(K), "K_sigma": float(max(np.std(Ks), s_jack, s_line, 0.002)),
            "roll": float(roll), "roll_sigma": float(max(np.std(Rs), 0.1)),
            "lines": int(len(inl)), "uprights": int(ng), "rms_deg": rms, "inliers": inl}


def horizon(segs, W, H, down, exclude=(), *, side: float = 0.0, seed: int = 1,
            c=None) -> dict | None:
    """Where the horizon crosses the picture's down axis: the strongest
    vanishing point of the other lines (floor seams, the foot of a wall),
    as a signed distance along `down` over L (negative: above the centre).
    side: -1 or +1 to keep only that side (a camera looking down has its
    horizon above the centre), 0 for either. Lines nearly parallel to the
    horizon say little about its height and are left out.
    Returns {"h", "h_sigma", "lines"} or None."""
    if segs is None or len(segs) < 3:
        return None
    rng = np.random.default_rng(seed)
    c = np.array([W / 2.0, H / 2.0]) if c is None else np.asarray(c, float)
    S = 0.5 * math.hypot(W, H)
    Lmax = float(max(W, H))
    m, e, L = _geom(segs, c, S)
    down = np.asarray(down, float) / np.linalg.norm(down)
    along = np.abs(e @ down)
    mask = np.ones(len(segs), bool)
    mask[list(exclude)] = False
    mask &= along > math.sin(math.radians(6))
    idx = np.flatnonzero(mask)

    def h_of(q):
        p = q[:2] / q[2]
        return float(p @ down) * S / Lmax

    def accept(q):
        if abs(q[2]) < 1e-6 or math.hypot(q[0], q[1]) / abs(q[2]) > 12.0:
            return False
        return side == 0.0 or h_of(q) * side > 0
    got = _ransac(m, e, L, idx, rng, accept=accept)
    if got is None:
        return None
    q, inl = got
    if len(inl) < 4 or abs(q[2]) < 1e-9 or not accept(q):
        return None
    h = h_of(q)
    hs = []
    for _ in range(30):
        b = rng.choice(inl, len(inl), replace=True)
        if len(np.unique(b)) < 3:
            continue
        qb = _refine(q, m[b], e[b], L[b])
        if abs(qb[2]) < 1e-9:
            continue
        hs.append(h_of(qb))
    if len(hs) < 10:
        return None
    return {"h": h, "h_sigma": float(max(np.std(hs), 0.003)), "lines": int(len(inl))}


def read(img, down, K_window=(-0.6, 0.6), frame_size=None, centre=None) -> dict:
    """Uprights and horizon from one still picture. frame_size (W, H) and
    centre: the camera's own picture size and principal point, when img is a
    canvas bigger than one frame."""
    H, W = img.shape[:2]
    if frame_size is not None:
        W, H = frame_size
    segs = segments(img)
    out = {"segments": int(len(segs)), "vertical": None, "horizon": None}
    v = vertical(segs, W, H, down, K_window=K_window, c=centre)
    side = 0.0
    if v is not None:
        inl = v.pop("inliers")
        out["vertical"] = v
        if abs(v["K"]) > 3 * v["K_sigma"]:
            side = -1.0 if v["K"] > 0 else 1.0
        out["horizon"] = horizon(segs, W, H, down, exclude=inl, side=side, c=centre)
    else:
        out["horizon"] = horizon(segs, W, H, down, c=centre)
    return out
