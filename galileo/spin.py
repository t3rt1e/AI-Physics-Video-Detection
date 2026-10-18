"""Spin: how the ball turns, read from its markings.

For each pair of close-ups (the ball centred and scaled the same), the turn
that carries the first onto the second is found directly on the sphere:
every point of the second close-up is looked up where the same spot of the
ball was in the first, and the turn (3D), the centring error and the change
of size are adjusted until they match best (see turn3d). The turn about the
line of sight is read well, from the markings sweeping round the centre; the
turn about the picture's own axes moves the markings across the face, which
looks a lot like a centring error, so it is read loosely. The friction of a
bounce across the picture turns the ball about the line of sight, which is
the part the checks use.

Angles are clockwise as seen in the picture (y down); spins in rad/s.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

SIDE = 64            # close-up size, px
FILL = 1.6           # close-up side over the ball's width
N_ANG = 180          # 2 degrees a step around the ring
MAX_TURN = 45.0      # the most it can turn between two frames and still be read, degrees
MIN_TEXTURE = 2.5    # grey-level spread of the markings below which there is nothing to follow
MIN_MATCH = 0.45     # correlation of two rings below which the reading is a guess


def close_up(img, cx, cy, d) -> np.ndarray | None:
    """The ball's close-up: SIDE x SIDE grey, the ball centred, SIDE / FILL across."""
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(d)) or d < 8:
        return None
    n = int(math.ceil(FILL * d)) | 1
    h, w = img.shape[:2]
    if cx < -0.25 * d or cy < -0.25 * d or cx > w + 0.25 * d or cy > h + 0.25 * d:
        return None
    crop = cv2.getRectSubPix(img, (n, n), (float(cx), float(cy)))
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if n > SIDE else cv2.INTER_LINEAR
    return cv2.resize(crop, (SIDE, SIDE), interpolation=interp)


def ring(p: np.ndarray | None) -> np.ndarray | None:
    """The close-up unrolled: rows are angle (N_ANG), columns distance from the
    centre (a quarter to 0.85 of the radius), shading and the outline taken out."""
    if p is None:
        return None
    p = p.astype(np.float32)
    hp = p - cv2.GaussianBlur(p, (0, 0), SIDE / 12.0)
    R = 0.5 * SIDE / FILL
    n_r = 24
    c = (SIDE - 1) / 2.0
    pol = cv2.warpPolar(hp, (n_r, N_ANG), (c, c), 0.85 * R,
                        cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)
    r = pol[:, int(0.3 * n_r):]
    r = r - r.mean(axis=0, keepdims=True)      # what is the same all round can't show a turn
    return r


def turn(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """How far ring b is turned from ring a, degrees clockwise, and how well
    they match once turned (correlation, 0 to 1)."""
    fa = np.fft.fft(a, axis=0)
    fb = np.fft.fft(b, axis=0)
    cc = np.fft.ifft(np.conj(fa) * fb, axis=0).real.sum(axis=1)
    norm = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if norm <= 0:
        return 0.0, 0.0
    cc = cc / norm
    kmax = int(MAX_TURN / (360.0 / N_ANG))
    ks = np.r_[0:kmax + 1, N_ANG - kmax:N_ANG]
    j = int(ks[int(np.argmax(cc[ks]))])
    c0, c1, c2 = cc[(j - 1) % N_ANG], cc[j], cc[(j + 1) % N_ANG]
    den = c0 - 2.0 * c1 + c2
    frac = 0.5 * (c0 - c2) / den if den < 0 else 0.0
    k = j + float(np.clip(frac, -0.5, 0.5))
    if k > N_ANG / 2:
        k -= N_ANG
    return k * 360.0 / N_ANG, float(c1)


_R = 0.5 * SIDE / FILL
_C = (SIDE - 1) / 2.0
_yy, _xx = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
_U, _V = (_xx - _C) / _R, (_yy - _C) / _R
_RHO2 = _U ** 2 + _V ** 2
_INNER = _RHO2 <= 0.8 ** 2
_W = np.sqrt(np.clip(1.0 - _RHO2, 0.0, 1.0))


def _rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _hp(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32)
    return p - cv2.GaussianBlur(p, (0, 0), SIDE / 12.0)


def _flow_guess(a8, b8):
    """A first guess at the turn from optical flow (rad per frame): fit of
    the flow of the markings to the turning sphere's flow, linearized."""
    flow = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 3, 9, 4, 5, 1.1, 0)
    gx = cv2.Sobel(a8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a8, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    sel = _INNER & (grad > max(8.0, float(np.percentile(grad[_INNER], 40))))
    n = int(sel.sum())
    if n < 60:
        return None
    u, v, w = _U[sel], _V[sel], _W[sel]
    z0, o1 = np.zeros(n, np.float32), np.ones(n, np.float32)
    A = np.vstack([np.stack([z0, -_R * w, -_R * v, o1, z0, u], axis=1),
                   np.stack([_R * w, z0, _R * u, z0, o1, v], axis=1)]).astype(np.float64)
    y = np.concatenate([flow[:, :, 0][sel], flow[:, :, 1][sel]]).astype(np.float64)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def turn3d(pa: np.ndarray, pb: np.ndarray, iters: int = 8):
    """The ball's turn between two close-ups as a 3D vector (rad per frame).

    Picture x right, y down, z away from the camera; the visible face is the
    half of the sphere toward the camera, so the point of it seen at (u, v)
    (in radii from the centre) is p = (u, v, -sqrt(1 - u^2 - v^2)). For a
    turn w, every point of the second close-up is looked up where the same
    spot of the ball was in the first (rotated back by w), and w, the
    centring error and the change of size are adjusted until the two match
    best (Gauss-Newton on the brightness). A ball spinning about a tilted
    axis reads correctly, where a ring correlation would mix the parts in.
    wz is the turn about the line of sight, clockwise in the picture.
    Returns (w, match 0..1, pixels used) or None.
    """
    A_, B_ = _hp(pa), _hp(pb)
    a8 = np.clip(128.0 + 3.0 * A_, 0, 255).astype(np.uint8)
    b8 = np.clip(128.0 + 3.0 * B_, 0, 255).astype(np.uint8)
    g0 = _flow_guess(a8, b8)
    if g0 is None:
        return None
    gx = cv2.Sobel(B_, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(B_, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    sel = _INNER & (grad > max(2.0, float(np.percentile(grad[_INNER], 30))))
    if sel.sum() < 60:
        return None
    u, v, w = _U[sel], _V[sel], _W[sel]
    pB = np.stack([u, v, -w], axis=1).astype(np.float64)
    bvals = B_[sel].astype(np.float64)
    # theta: wx, wy, wz (rad), tx, ty (px), log scale
    th = np.array([g0[0], g0[1], g0[2], g0[3], g0[4], 0.0], float)

    def resid(t_):
        pA = pB @ _rodrigues(-t_[:3]).T
        k = np.exp(-t_[5])
        mx = (_C + _R * k * pA[:, 0] - t_[3]).astype(np.float32)
        my = (_C + _R * k * pA[:, 1] - t_[4]).astype(np.float32)
        av = cv2.remap(A_, mx.reshape(1, -1), my.reshape(1, -1), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE).ravel().astype(np.float64)
        ok = (pA[:, 2] < -0.15)                        # still on the face it was seen on
        return np.where(ok, av - bvals, 0.0), ok
    steps = np.array([2e-3, 2e-3, 2e-3, 0.05, 0.05, 2e-3])
    lam = 1e-3
    r0, ok = resid(th)
    cost = float(np.sum(r0 ** 2))
    for _ in range(iters):
        J = np.empty((len(r0), 6))
        for k_ in range(6):
            tt = th.copy()
            tt[k_] += steps[k_]
            J[:, k_] = (resid(tt)[0] - r0) / steps[k_]
        H = J.T @ J
        g = J.T @ r0
        try:
            dth = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), g)
        except np.linalg.LinAlgError:
            break
        r1, ok1 = resid(th + dth)
        c1 = float(np.sum(r1 ** 2))
        if c1 < cost:
            th, r0, ok, cost = th + dth, r1, ok1, c1
            lam = max(lam / 3, 1e-6)
            if np.max(np.abs(dth[:3])) < 1e-5:
                break
        else:
            lam *= 5
    av = r0 + bvals
    m = ok
    if m.sum() < 40:
        return None
    x1, x2 = av[m] - av[m].mean(), bvals[m] - bvals[m].mean()
    match = float(np.sum(x1 * x2) / np.sqrt(np.sum(x1 ** 2) * np.sum(x2 ** 2) + 1e-12))
    return th[:3], match, int(m.sum())


def texture(r: np.ndarray | None) -> float:
    return 0.0 if r is None else float(np.std(r))


MAX_PAIRS = 220      # spin is read on at most about this many pairs of close-ups


def series(patches: list, t) -> dict:
    """The ball's spin between close-ups. patches: one close-up (or None) per
    tracked point, in track order; t: their times. Long clips are read on
    every second (third...) point, which is plenty for a spin that only
    changes at bounces. Returns arrays over the readable pairs: i, j (the
    pair's points), t (its middle), w3 (rad/s: about the picture's x, y and
    the line of sight), omega (the last, clockwise), match; and the share of
    pairs that could be read."""
    t = np.asarray(t, float)
    n = len(patches)
    have = [p is not None for p in patches]
    stride = max(1, int(math.ceil(sum(have) / MAX_PAIRS)))
    dts = np.diff(t)
    dt_med = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 0.0
    rings = {}
    out = {"i": [], "j": [], "t": [], "w3": [], "match": []}
    tried = 0
    tex_all = []
    for i in range(0, n - stride, stride):
        j = i + stride
        if not (have[i] and have[j]):
            continue
        dt = t[j] - t[i]
        if dt <= 0 or (dt_med and dt > 2.5 * stride * dt_med):
            continue
        tried += 1
        for k in (i, j):
            if k not in rings:
                rings[k] = texture(ring(patches[k]))
                tex_all.append(rings[k])
        if min(rings[i], rings[j]) < MIN_TEXTURE:
            continue
        res = turn3d(patches[i], patches[j])
        if res is None:
            continue
        w, m, _ = res
        if m < MIN_MATCH or np.degrees(np.max(np.abs(w))) > MAX_TURN:
            continue
        out["i"].append(i)
        out["j"].append(j)
        out["t"].append(0.5 * (t[i] + t[j]))
        out["w3"].append(np.asarray(w, float) / dt)
        out["match"].append(m)
    w3 = np.asarray(out["w3"], float).reshape(-1, 3)
    return {"i": np.asarray(out["i"], int), "j": np.asarray(out["j"], int),
            "t": np.asarray(out["t"], float), "w3": w3, "omega": w3[:, 2].copy(),
            "match": np.asarray(out["match"], float),
            "readable": len(out["i"]) / tried if tried else 0.0, "tried": tried,
            "stride": stride,
            "texture": float(np.median(tex_all)) if tex_all else 0.0}


def rate(sp: dict, idx, comp: int = 2) -> tuple[float, float, int] | None:
    """Steady spin over a stretch of tracked points (a flight), about one axis
    (2: the line of sight): a robust mean of the pairs inside it, its
    standard error, and how many pairs."""
    idx = np.asarray(idx, int)
    if len(idx) < 2 or len(sp["i"]) == 0:
        return None
    lo, hi = int(idx.min()), int(idx.max())
    sel = (sp["i"] >= lo) & (sp["j"] <= hi)
    w = sp["w3"][sel, comp]
    if len(w) < 4:
        return None
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med))) * 1.4826
    keep = np.abs(w - med) <= max(3.0 * mad, 1e-9) if mad > 0 else np.ones(len(w), bool)
    w = w[keep]
    if len(w) < 4:
        return None
    s = float(np.std(w, ddof=1)) if len(w) > 1 else float("inf")
    return float(np.mean(w)), s / math.sqrt(len(w)), int(len(w))


def noise(sp: dict, flights) -> float:
    """How much the spin readings scatter from one pair to the next within a
    flight (rad/s, median over the flights). A plainly marked ball turning
    about the line of sight reads to a few tenths; faint markings, or a ball
    tumbling fast about an axis across the picture, scatter by several."""
    vals = []
    for idx in flights:
        idx = np.asarray(idx, int)
        if len(idx) < 2:
            continue
        sel = (sp["i"] >= idx.min()) & (sp["j"] <= idx.max())
        wz = sp["w3"][sel, 2]
        if len(wz) >= 6:
            vals.append(1.4826 * float(np.median(np.abs(np.diff(wz)))) / math.sqrt(2.0))
    return float(np.median(vals)) if vals else float("inf")


def readable(sp: dict | None, flights) -> tuple[bool, str]:
    """Can the checks trust this ball's spin? (yes/no, and why not)."""
    if sp is None or sp.get("tried", 0) == 0:
        return False, "The ball wasn't seen closely enough to read its spin."
    if sp["readable"] < 0.3 or len(sp["i"]) < 8:
        return False, "No markings to follow (a plain ball, or too blurred), so its spin can't be read."
    nz = noise(sp, flights)
    big = float(np.median(np.abs(sp["w3"][:, 2]))) if len(sp["i"]) else 0.0
    if nz > max(1.2, 0.12 * big):
        return False, ("Its spin reads unsteadily (faint markings, or it tumbles about an axis "
                       "across the picture), so it can't be judged.")
    return True, ""
