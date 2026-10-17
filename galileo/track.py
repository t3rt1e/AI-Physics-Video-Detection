"""Classical ball tracker: background subtraction, blob measurement, linking.

Built for the Galileo Tower (bright matte ball, black backdrop, fixed
camera) and good enough for most clean clips of a falling object. For
cluttered footage, galileo.sam2_tracker produces the same Track output.

Width is measured from the blob's intensity profile projected across the
direction of motion. Motion blur smears a ball along its path but leaves
that projection unchanged, so the width stays honest on fast frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d as _gf1d
except Exception:  # pragma: no cover
    _gf1d = None


@dataclass
class Det:
    f: int            # index into the analysed frame list
    x: float          # working px
    y: float
    d: float          # width across the motion direction, working px
    area: float
    elong: float
    contrast: float


@dataclass
class Track:
    dets: list = field(default_factory=list)
    score: float = 0.0
    kind: str = "moving"

    @property
    def f(self):
        return np.array([d.f for d in self.dets])

    @property
    def x(self):
        return np.array([d.x for d in self.dets], float)

    @property
    def y(self):
        return np.array([d.y for d in self.dets], float)

    @property
    def d(self):
        return np.array([d.d for d in self.dets], float)

    @property
    def area(self):
        return np.array([d.area for d in self.dets], float)


# ---------------------------------------------------------------------------
def background(frames, n_sample: int = 41) -> np.ndarray:
    idx = np.unique(np.linspace(0, len(frames) - 1, min(n_sample, len(frames))).astype(int))
    stack = np.stack([frames[i] for i in idx])
    return np.median(stack, axis=0).astype(np.uint8)


def _marginal(vals, w, lo, n):
    pos = w - lo
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    P = np.zeros(n + 1)
    np.add.at(P, i0, vals * (1 - fr))
    np.add.at(P, np.minimum(i0 + 1, n), vals * fr)
    return P


_SIGMAS = (0.0, 0.7, 1.2, 1.8)
_FINE = 4  # sub-steps per px for the model profile


def _gauss_rows(F, sigma_fine):
    """Gaussian-smooth each row of F (sigma in fine steps)."""
    if sigma_fine <= 0:
        return F
    if _gf1d is not None:
        return _gf1d(F, sigma_fine, axis=1, mode="constant", truncate=3.5)
    r = int(np.ceil(3.5 * sigma_fine))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_fine) ** 2)
    k /= k.sum()
    pad = np.pad(F, ((0, 0), (r, r)))
    out = np.zeros_like(F)
    for j, kv in enumerate(k):
        out += kv * pad[:, j:j + F.shape[1]]
    return out


def _width_along(D, M, X, Y, cx, cy, e, quick=False) -> float:
    """Ball diameter from the image projected across unit vector e.

    Projecting along the motion direction undoes motion blur. The projected
    profile is fitted as  a*flat_disc + b*shaded_sphere (ambient + diffuse
    light, a,b >= 0), blurred by a Gaussian sigma for lens and sensor, plus
    c*mask_profile for the noise floor. Returns 2R in px.
    """
    w = (X - cx) * e[0] + (Y - cy) * e[1]
    lo = int(np.floor(w.min())) - 1
    hi = int(np.ceil(w.max())) + 1
    n = hi - lo + 1
    P = _marginal(D, w, lo, n)
    Mm = _marginal(M, w, lo, n)
    c = np.arange(n + 1) + lo
    Pn = np.clip(P - np.percentile(P, 10), 0, None)
    if Pn.sum() <= 0:
        return float("nan")
    sw = float(np.sqrt(np.sum(Pn * c**2) / Pn.sum()))
    if sw <= 0.3:
        return float("nan")
    if quick:
        return float(2 * np.sqrt(4.5) * sw)
    Rs = np.linspace(1.3 * sw, 3.4 * sw, 43)
    fine = lo + np.arange((n + 1) * _FINE) / _FINE
    sample = np.arange(n + 1) * _FINE
    base = np.clip(1.0 - (fine[None, :] / Rs[:, None]) ** 2, 0, None)
    shapes0 = [np.sqrt(base), base]          # flat disc chord, shaded sphere
    PP = float(P @ P)
    best = (np.inf, float("nan"))
    for sg in _SIGMAS:
        F = [(_gauss_rows(S, sg * _FINE))[:, sample] for S in shapes0]
        cols = [F[0], F[1], np.broadcast_to(Mm, F[0].shape)]
        for use in ((0, 1, 2), (0, 2), (1, 2)):
            X3 = np.stack([cols[u] for u in use], axis=2)          # (nR, nb, m)
            G = np.einsum("rbi,rbj->rij", X3, X3)
            rhs = np.einsum("rbi,b->ri", X3, P)
            G = G + np.eye(len(use))[None] * 1e-9
            try:
                coef = np.linalg.solve(G, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                continue
            ssr = PP - 2 * np.einsum("ri,ri->r", coef, rhs) + np.einsum("ri,rij,rj->r", coef, G, coef)
            valid = np.all(coef[:, :-1] > 0, axis=1)
            ssr = np.where(valid, ssr, np.inf)
            j = int(np.argmin(ssr))
            if ssr[j] < best[0]:
                R = Rs[j]
                if 0 < j < len(Rs) - 1 and np.isfinite(ssr[j - 1]) and np.isfinite(ssr[j + 1]):
                    den = ssr[j - 1] - 2 * ssr[j] + ssr[j + 1]
                    if den > 0:
                        R = Rs[j] + 0.5 * (ssr[j - 1] - ssr[j + 1]) / den * (Rs[1] - Rs[0])
                best = (float(ssr[j]), float(2 * R))
    return best[1]


def measure_width(frame, bg, x, y, d_guess, axis=None) -> float:
    """Accurate width of the blob near (x, y): used on the chosen track only."""
    H, W = frame.shape[:2]
    r = int(max(d_guess, 4) * 1.6 + 6)
    xa, xb = int(max(x - r, 0)), int(min(x + r + 1, W))
    ya, yb = int(max(y - r, 0)), int(min(y + r + 1, H))
    if xb - xa < 5 or yb - ya < 5:
        return float("nan")
    fr, bgr = frame[ya:yb, xa:xb], bg[ya:yb, xa:xb]
    if fr.ndim == 3:
        # Video stores colour at half resolution (4:2:0), so colour edges are
        # soft and read wide. Measure on brightness, which keeps full
        # resolution, unless the ball barely differs from the backdrop in it.
        Dl = cv2.absdiff(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        Dc = np.max(cv2.absdiff(fr, bgr), axis=2)
        Dr = Dl if float(Dl.max()) >= max(25.0, 0.35 * float(Dc.max())) else Dc
    else:
        Dr = cv2.absdiff(fr, bgr)
    Dr = Dr.astype(np.float64)
    Db = cv2.GaussianBlur(Dr, (3, 3), 0)
    med = float(np.median(Db))
    pk = float(Db.max())
    if pk - med < 8:
        return float("nan")
    core = (Db > med + 0.25 * (pk - med)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n < 2:
        return float("nan")
    # Component nearest the expected position.
    j = 1 + int(np.argmin([np.hypot(cen[i][0] + xa - x, cen[i][1] + ya - y) for i in range(1, n)]))
    cm = cv2.dilate((lab == j).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cm[(lab > 0) & (lab != j)] = 0
    yy, xx = np.mgrid[ya:yb, xa:xb]
    sel = cm > 0
    Dm, Xm, Ym = Dr[sel], xx[sel].astype(np.float64), yy[sel].astype(np.float64)
    wts = np.clip(Dm - med, 0, None)
    if wts.sum() <= 0:
        return float("nan")
    cx = float(np.sum(wts * Xm) / wts.sum())
    cy = float(np.sum(wts * Ym) / wts.sum())
    ones = np.ones_like(Dm)
    if axis is None:
        sxx = np.sum(wts * (Xm - cx) ** 2)
        syy = np.sum(wts * (Ym - cy) ** 2)
        sxy = np.sum(wts * (Xm - cx) * (Ym - cy))
        ev, evec = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
        if ev[1] / max(ev[0], 1e-9) > 1.3:
            axis = evec[:, 0]
    if axis is not None:
        return _width_along(Dm, ones, Xm, Ym, cx, cy, axis)
    return float(np.nanmean([_width_along(Dm, ones, Xm, Ym, cx, cy, (1.0, 0.0)),
                             _width_along(Dm, ones, Xm, Ym, cx, cy, (0.0, 1.0))]))


def detect(frame, bg, *, roi=None, min_area=8, max_frac=0.2, max_blobs=12,
           min_thr=14.0) -> list[Det]:
    if frame.ndim == 3:
        D_raw = np.max(cv2.absdiff(frame, bg), axis=2)
    else:
        D_raw = cv2.absdiff(frame, bg)
    D = cv2.GaussianBlur(D_raw, (3, 3), 0)
    if roi is not None:
        x0, y0, x1, y1 = roi
        m = np.zeros_like(D)
        m[y0:y1, x0:x1] = 1
        D = D * m
    sub = D[::4, ::4].astype(np.float32)
    med = float(np.median(sub))
    mad = float(1.4826 * np.median(np.abs(sub - med))) + 0.5
    thr = max(min_thr, med + 6 * mad)
    floor = med + 2 * mad
    mask = (D > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = D.shape
    max_area = max_frac * H * W
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[: max_blobs * 2]
    out = []
    for i in order:
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue  # cut off by the frame edge: centroid and width would be wrong
        pad = 5
        xa, ya = max(x - pad, 0), max(y - pad, 0)
        xb, yb = min(x + w + pad, W), min(y + h + pad, H)
        lab = labels[ya:yb, xa:xb]
        core = (lab == i).astype(np.uint8)
        cm = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cm[(lab > 0) & (lab != i)] = 0
        Dr = D_raw[ya:yb, xa:xb].astype(np.float64)
        Wt = np.clip(Dr - floor, 0, None) * cm
        s = Wt.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        sel = Wt > 0
        Wv, Xv, Yv = Wt[sel], xx[sel].astype(np.float64), yy[sel].astype(np.float64)
        cx = float(np.sum(Wv * Xv) / s)
        cy = float(np.sum(Wv * Yv) / s)
        sxx = np.sum(Wv * (Xv - cx) ** 2) / s
        syy = np.sum(Wv * (Yv - cy) ** 2) / s
        sxy = np.sum(Wv * (Xv - cx) * (Yv - cy)) / s
        ev, evec = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
        lam_min, lam_max = max(ev[0], 1e-6), max(ev[1], 1e-6)
        elong = float(np.sqrt(lam_max / lam_min))
        msel = cm > 0
        Dm, Xm, Ym = Dr[msel], xx[msel].astype(np.float64), yy[msel].astype(np.float64)
        ones = np.ones_like(Dm)
        width = _width_along(Dm, ones, Xm, Ym, cx, cy, evec[:, 0], quick=True)
        peak = float(D[ya:yb, xa:xb][core > 0].max())
        if peak < 1.3 * thr:
            continue
        out.append(Det(f=-1, x=cx, y=cy, d=width, area=float(area), elong=elong,
                       contrast=peak))
        if len(out) >= max_blobs:
            break
    return out


# ---------------------------------------------------------------------------
def _predict(tr: list[Det], t_of, t_new):
    last = tr[-1]
    if len(tr) >= 3:
        pts = tr[-3:]
        ts = np.array([t_of[p.f] for p in pts])
        if np.ptp(ts) > 0:
            tt = ts - ts[-1]
            cx = np.polyfit(tt, [p.x for p in pts], 2)
            cy = np.polyfit(tt, [p.y for p in pts], 2)
            dt = t_new - ts[-1]
            v = np.hypot(np.polyval(np.polyder(cx), 0), np.polyval(np.polyder(cy), 0))
            return np.polyval(cx, dt), np.polyval(cy, dt), v * abs(dt)
    if len(tr) == 2:
        a, b = tr[-2], tr[-1]
        dt0 = t_of[b.f] - t_of[a.f]
        if dt0 > 0:
            dt = t_new - t_of[b.f]
            vx, vy = (b.x - a.x) / dt0, (b.y - a.y) / dt0
            return b.x + vx * dt, b.y + vy * dt, np.hypot(vx, vy) * abs(dt)
    return last.x, last.y, 0.0


def link(dets_per_frame, times, max_missing=3) -> list[Track]:
    active: list[dict] = []
    closed: list[Track] = []
    for fi, dets in enumerate(dets_per_frame):
        for d in dets:
            d.f = fi
        pairs = []
        for ti, tr in enumerate(active):
            px, py, mv = _predict(tr["dets"], times, times[fi])
            last = tr["dets"][-1]
            gate = 2.0 * last.d + 0.6 * mv + 8.0
            for di, d in enumerate(dets):
                dist = np.hypot(d.x - px, d.y - py)
                size_ok = 0.5 < d.d / max(last.d, 1e-6) < 2.0
                if dist < gate and size_ok:
                    pairs.append((dist / gate, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            active[ti]["dets"].append(dets[di])
            active[ti]["miss"] = 0
            used_t.add(ti)
            used_d.add(di)
        keep = []
        for ti, tr in enumerate(active):
            if ti not in used_t:
                tr["miss"] += 1
            if tr["miss"] > max_missing:
                closed.append(Track(dets=tr["dets"]))
            else:
                keep.append(tr)
        active = keep
        for di, d in enumerate(dets):
            if di not in used_d:
                active.append({"dets": [d], "miss": 0})
    closed.extend(Track(dets=tr["dets"]) for tr in active)
    return [t for t in closed if len(t.dets) >= 3]


def merge_fragments(tracks: list[Track], times, max_gap_frames=4) -> list[Track]:
    """Join pieces of one object's path broken at bounces or brief dropouts."""
    tracks = sorted(tracks, key=lambda t: t.dets[0].f)
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(tracks):
            if a.kind == "static":
                continue
            la = a.dets[-1]
            best = None
            for j, b in enumerate(tracks):
                if i == j or b.kind == "static":
                    continue
                fb = b.dets[0]
                gap = fb.f - la.f
                if gap < 1 or gap > max_gap_frames:
                    continue
                if not 0.6 < fb.d / max(la.d, 1e-6) < 1.6:
                    continue
                dist = np.hypot(fb.x - la.x, fb.y - la.y)
                lim = 3.0 * la.d + 10.0 + _speed(a, times) * (times[fb.f] - times[la.f]) * 1.5
                if dist < lim and (best is None or dist < best[0]):
                    best = (dist, j)
            if best is not None:
                b = tracks[best[1]]
                a.dets = a.dets + b.dets
                tracks.pop(best[1])
                merged = True
                break
    return tracks


def _speed(tr: Track, times) -> float:
    if len(tr.dets) < 2:
        return 0.0
    a, b = tr.dets[-2], tr.dets[-1]
    dt = times[b.f] - times[a.f]
    return float(np.hypot(b.x - a.x, b.y - a.y) / dt) if dt > 0 else 0.0


def classify_and_score(tracks: list[Track], times) -> list[Track]:
    for tr in tracks:
        x, y, d = tr.x, tr.y, tr.d
        dm = float(np.median(d)) if len(d) else 1.0
        extent = float(np.hypot(np.ptp(x), np.ptp(y)))
        if extent < 1.0 * dm:
            tr.kind = "static"
            tr.score = 0.0
            continue
        tr.kind = "moving"
        t = np.array([times[f] for f in tr.f])
        n = len(t)
        rms = dm
        ay = 0.0
        if n >= 4 and np.ptp(t) > 0:
            tt = t - t.mean()
            cy = np.polyfit(tt, y, 2)
            cx = np.polyfit(tt, x, 2)
            rms = float(np.sqrt(np.mean((y - np.polyval(cy, tt)) ** 2 +
                                        (x - np.polyval(cx, tt)) ** 2)))
            ay = 2 * cy[0]
        score = n * min(extent / dm, 25.0) * np.exp(-rms / (0.5 * dm + 1.0))
        if ay <= 0:
            score *= 0.2
        tr.score = float(score)
    return sorted(tracks, key=lambda t: -t.score)


@dataclass
class TrackResult:
    main: Track | None
    others: list
    rest: Det | None          # where the object sat before release, if seen
    rest_d: float | None
    n_candidates: int
    d_fine: np.ndarray | None = None   # accurate widths for a subset of main (NaN elsewhere)


def refine_widths(frames, bg, tr: Track, times, max_n: int = 40) -> np.ndarray:
    """Accurate per-frame width for up to max_n detections spread along the track."""
    n = len(tr.dets)
    out = np.full(n, np.nan)
    pick = np.unique(np.linspace(0, n - 1, min(max_n, n)).astype(int))
    for j in pick:
        dt = tr.dets[j]
        a = tr.dets[max(j - 1, 0)]
        b = tr.dets[min(j + 1, n - 1)]
        vx, vy = b.x - a.x, b.y - a.y
        sp = float(np.hypot(vx, vy))
        axis = (-vy / sp, vx / sp) if sp > 0.6 * max(dt.d, 1.0) else None
        out[j] = measure_width(frames[dt.f], bg, dt.x, dt.y, dt.d, axis=axis)
    return out


def track_falling_object(frames, times, *, roi=None, bg=None) -> TrackResult:
    bg = background(frames) if bg is None else bg
    dets = [detect(f, bg, roi=roi) for f in frames]
    tracks = link(dets, times)
    tracks = classify_and_score(tracks, times)
    tracks = merge_fragments(tracks, times)
    tracks = classify_and_score(tracks, times)
    moving = [t for t in tracks if t.kind == "moving"]
    if not moving:
        return TrackResult(None, [], None, None, sum(len(d) for d in dets))
    main = moving[0]
    others = [t for t in moving[1:] if t.score > 0.3 * main.score
              and len(set(t.f) & set(main.f)) >= 6]
    d_fine = refine_widths(frames, bg, main, times)
    # A static blob where the fall begins = the object's resting spot (ghost).
    rest, rest_d = None, None
    f0 = main.dets[0]
    for t in tracks:
        if t.kind != "static":
            continue
        sx, sy = float(np.median(t.x)), float(np.median(t.y))
        if np.hypot(sx - f0.x, sy - f0.y) < 2.5 * f0.d and abs(t.dets[0].f - f0.f) < 30:
            mid = t.dets[len(t.dets) // 2]
            wd = measure_width(frames[mid.f], bg, sx, sy, float(np.median(t.d)))
            rest = Det(f=t.dets[0].f, x=sx, y=sy, d=wd, area=0, elong=1, contrast=0)
            rest_d = wd if np.isfinite(wd) else None
            break
    return TrackResult(main, others, rest, rest_d, sum(len(d) for d in dets), d_fine)
