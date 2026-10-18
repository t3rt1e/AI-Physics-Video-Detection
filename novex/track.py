"""Classical ball tracker: background subtraction, blob measurement, linking.

Built for the Novex Tower (bright matte ball, black backdrop, fixed
camera) and good enough for most clean clips of a falling object. For
cluttered footage, novex.sam2_tracker produces the same Track output.

Width is measured from the blob's intensity profile projected across the
direction of motion. Motion blur smears a ball along its path but leaves
that projection unchanged, so the width stays honest on fast frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import math

import cv2
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d as _gf1d
except Exception:  # pragma: no cover
    _gf1d = None


@dataclass
class Det:
    f: int            # index into the analysed frame list
    x: float          # working px (brightness-weighted centre)
    y: float
    d: float          # width across the motion direction, working px
    area: float
    elong: float
    contrast: float
    ax: float = float("nan")   # area centroid of the silhouette: the centre of mass
    ay: float = float("nan")   # of a uniform flat object, whatever its shape
    tx: float = float("nan")   # thickness-weighted centroid: each pixel counts by its
    ty: float = float("nan")   # depth into the outline, a stand-in for how thick it is
    shadow: bool = False       # looks like a shadow (the backdrop, only darker)
    hull: np.ndarray | None = None   # silhouette outline (convex hull), working px
    col: np.ndarray | None = None    # mean colour (BGR), to keep two objects apart


@dataclass
class Track:
    dets: list = field(default_factory=list)
    score: float = 0.0
    kind: str = "moving"
    falls: bool = False        # has at least one clean flight that speeds up downward
    lone: bool = False         # shadow-coloured, but nothing solid moves with it

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

    @property
    def ax(self):
        return np.array([d.ax if np.isfinite(d.ax) else d.x for d in self.dets], float)

    @property
    def ay(self):
        return np.array([d.ay if np.isfinite(d.ay) else d.y for d in self.dets], float)

    @property
    def tx(self):
        return np.array([d.tx if np.isfinite(d.tx) else d.x for d in self.dets], float)

    @property
    def ty(self):
        return np.array([d.ty if np.isfinite(d.ty) else d.y for d in self.dets], float)

    @property
    def shadow_frac(self) -> float:
        return float(np.mean([d.shadow for d in self.dets])) if self.dets else 0.0

    @property
    def is_shadow(self) -> bool:
        """Shadow-coloured, and something solid moves along with it."""
        return self.shadow_frac > 0.5 and not self.lone

    def extent_across(self, times) -> np.ndarray:
        """Outline length across the direction of motion, per detection (px).

        Motion blur stretches a blob along its path, never across it, so this
        is blur-free. A tumbling stick or phone shows its full length across
        the path whenever its long side lines up that way.
        """
        n = len(self.dets)
        out = np.full(n, np.nan)
        for j, dt_ in enumerate(self.dets):
            if dt_.hull is None or len(dt_.hull) < 3:
                continue
            a = self.dets[max(j - 1, 0)]
            b = self.dets[min(j + 1, n - 1)]
            vx, vy = b.x - a.x, b.y - a.y
            sp = float(np.hypot(vx, vy))
            if sp < 1e-6:
                ext = [np.ptp(dt_.hull[:, 0]), np.ptp(dt_.hull[:, 1])]
                out[j] = float(max(ext)) + 1.0
                continue
            e = np.array([-vy / sp, vx / sp])
            out[j] = float(np.ptp(dt_.hull @ e)) + 1.0
        return out


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
        # resolution, unless the ball barely differs from the backdrop in it,
        # or only part of it does (a yellow ball half in front of a pale wall,
        # half in front of a dark shelf: brightness sees half a ball).
        Dl = cv2.absdiff(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        Dc = np.max(cv2.absdiff(fr, bgr), axis=2)
        # The ball's own shadow on the wall or floor right beside it is the
        # backdrop, only darker: every colour channel drops by about the same
        # factor. It is no part of the ball, so it doesn't count.
        # A grey ball (steel) or a grey panel looks like that too, so only
        # shadow-like pixels outside the outline of the clearly coloured part
        # are dropped, and only when there is a clearly coloured part.
        rat = (fr.astype(np.float32) + 4.0) / (bgr.astype(np.float32) + 4.0)
        rmean = rat.mean(axis=2)
        shadow = ((rmean < 0.93) & (rmean > 0.25)
                  & (np.abs(rat - rmean[:, :, None]).max(axis=2) < 0.06 + 0.1 * (1 - rmean)))
        if shadow.any():
            Dcb0 = cv2.GaussianBlur(Dc.astype(np.float32), (3, 3), 0)
            lit = (Dcb0 > float(np.median(Dcb0)) + 0.35 * (float(Dcb0.max()) - float(np.median(Dcb0))))
            body = lit & ~shadow
            if body.sum() >= max(12, 0.3 * lit.sum()):
                ys_, xs_ = np.nonzero(body)
                hull0 = cv2.convexHull(np.column_stack([xs_, ys_]).astype(np.int32))
                inside = np.zeros(shadow.shape, np.uint8)
                cv2.fillConvexPoly(inside, hull0, 1)
                inside = cv2.dilate(inside, np.ones((3, 3), np.uint8)) > 0
                drop = shadow & ~inside
                Dl = np.where(drop, 0, Dl).astype(Dl.dtype)
                Dc = np.where(drop, 0, Dc).astype(Dc.dtype)
        use_l = float(Dl.max()) >= max(25.0, 0.35 * float(Dc.max()))
        partial = False
        if use_l:
            yy0, xx0 = np.mgrid[ya:yb, xa:xb]
            near0 = (xx0 - x) ** 2 + (yy0 - y) ** 2 <= (0.75 * max(d_guess, 4.0)) ** 2
            Dcb = cv2.GaussianBlur(Dc.astype(np.float64), (3, 3), 0)
            Dlb = cv2.GaussianBlur(Dl.astype(np.float64), (3, 3), 0)
            mc, pc = float(np.median(Dcb)), float(Dcb[near0].max()) if near0.any() else 0.0
            ml, pl = float(np.median(Dlb)), float(Dlb[near0].max()) if near0.any() else 0.0
            col = near0 & (Dcb > mc + 0.35 * (pc - mc))
            lum = near0 & (Dlb > ml + 0.35 * (pl - ml))
            partial = bool(col.sum() >= 12 and (lum & col).sum() < 0.85 * col.sum())
        # Brightness seeing only part of the ball reads it small: colour then.
        maps = [Dl if (use_l and not partial) else Dc]
    else:
        maps = [cv2.absdiff(fr, bgr)]
        Dc = Dl = None
    yy, xx = np.mgrid[ya:yb, xa:xb]
    # Every piece of the blob near the expected spot. A multi-coloured ball (a
    # volleyball's panels) differs from the backdrop by a different amount in
    # each panel and can come apart into several pieces; keeping only the
    # nearest piece measured half a volleyball.
    near = (xx - x) ** 2 + (yy - y) ** 2 <= (0.6 * max(d_guess, 4.0)) ** 2

    def model_width(Dr):
        Dr = Dr.astype(np.float64)
        Db = cv2.GaussianBlur(Dr, (3, 3), 0)
        med = float(np.median(Db))
        pk = float(Db.max())
        if pk - med < 8:
            return float("nan"), None, axis
        core = (Db > med + 0.25 * (pk - med)).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(core, connectivity=8)
        if n < 2:
            return float("nan"), None, axis
        ids = np.unique(lab[near & (lab > 0)])
        if len(ids) == 0:
            ids = np.array([1 + int(np.argmin([np.hypot(cen[i][0] + xa - x, cen[i][1] + ya - y)
                                               for i in range(1, n)]))])
        comp = np.isin(lab, ids)
        cm = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cm[(lab > 0) & ~comp] = 0
        sel = cm > 0
        Dm, Xm, Ym = Dr[sel], xx[sel].astype(np.float64), yy[sel].astype(np.float64)
        wts = np.clip(Dm - med, 0, None)
        if wts.sum() <= 0:
            return float("nan"), comp, axis
        cx = float(np.sum(wts * Xm) / wts.sum())
        cy = float(np.sum(wts * Ym) / wts.sum())
        ones = np.ones_like(Dm)
        ax_ = axis
        if ax_ is None:
            sxx = np.sum(wts * (Xm - cx) ** 2)
            syy = np.sum(wts * (Ym - cy) ** 2)
            sxy = np.sum(wts * (Xm - cx) * (Ym - cy))
            ev, evec = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
            if ev[1] / max(ev[0], 1e-9) > 1.3:
                ax_ = evec[:, 0]
        axes_ = [ax_] if ax_ is not None else [(1.0, 0.0), (0.0, 1.0)]
        ws = [_width_along(Dm, ones, Xm, Ym, cx, cy, a) for a in axes_]
        ws = [w_ for w_ in ws if np.isfinite(w_)]
        return (float(np.mean(ws)) if ws else float("nan")), comp, ax_
    got = [model_width(D) for D in maps]
    got = [g_ for g_ in got if g_[1] is not None]
    if not got:
        return float("nan")
    fin = [g_ for g_ in got if np.isfinite(g_[0])]
    w_model, comp, ax_used = max(fin, key=lambda g_: g_[0]) if fin else got[0]
    axes = [ax_used] if ax_used is not None else [(1.0, 0.0), (0.0, 1.0)]
    # The profile model assumes the ball differs from the backdrop by about
    # the same amount everywhere. Panels break that, so for a big ball also
    # take the outline's width (convex hull of the blob, across the motion)
    # and trust it when the two disagree.
    # The outline is cut at half the contrast, where a blurred edge really is,
    # in brightness and in colour (a yellow panel can match a grey wall in
    # brightness alone).
    sil = np.zeros_like(comp)
    for Dmap in ([Dl, Dc] if Dc is not None else maps):
        Ds = cv2.GaussianBlur(Dmap.astype(np.float64), (3, 3), 0)
        med_s, pk_s = float(np.median(Ds)), float(Ds.max())
        if pk_s - med_s < 8:
            continue
        n2, lab2, _, _ = cv2.connectedComponentsWithStats(
            (Ds > med_s + 0.5 * (pk_s - med_s)).astype(np.uint8), connectivity=8)
        ids2 = np.unique(lab2[near & (lab2 > 0)])
        if len(ids2):
            sil |= np.isin(lab2, ids2)
    pts = np.column_stack([xx[sil], yy[sil]]).astype(np.float32)
    if len(pts) >= 12:
        hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
        if axis is not None:
            w_sil = float(np.ptp(hull @ np.asarray(axis, float)) + 1.0)   # across the motion
        else:
            # Still ball: the typical width over many directions, so a panel
            # that matches the backdrop (a missing wedge) or a little blur
            # each spoil only a few of them.
            dirs = [(np.cos(a), np.sin(a)) for a in np.radians(np.arange(0, 180, 15))]
            w_sil = float(np.median([np.ptp(hull @ np.asarray(a)) + 1.0 for a in dirs]))
        # A part of the ball only ever reads narrow, so a narrower model gives
        # way sooner than a wider one.
        if w_sil >= 24 and (not np.isfinite(w_model) or w_model < 0.94 * w_sil
                            or w_model > 1.12 * w_sil):
            return w_sil
    return w_model


def detect(frame, bg, *, roi=None, min_area=8, max_frac=0.2, max_blobs=12,
           min_thr=14.0, footprint=None) -> list[Det]:
    """Moving blobs in one steadied frame. footprint: the frame's outline on a
    canvas bigger than it (a panning camera); blobs reaching past it are cut
    off by the edge of the picture and are skipped, like those at the
    border."""
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
        if footprint is not None and min(
                cv2.pointPolygonTest(footprint, (float(px), float(py)), True)
                for px, py in ((x - 1, y - 1), (x + w + 1, y - 1), (x - 1, y + h + 1),
                               (x + w + 1, y + h + 1))) < 3.0:
            continue  # reaches past this frame's own picture on the canvas
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
        # The silhouette: holes filled, so a dark logo or seam doesn't move the
        # centre. Its area centroid is the centre of mass of a uniform object.
        sil = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ax_ = ay_ = tx_ = ty_ = float("nan")
        hull = None
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            mom = cv2.moments(cnt)
            if mom["m00"] > 0:
                ax_ = float(mom["m10"] / mom["m00"]) + xa
                ay_ = float(mom["m01"] / mom["m00"]) + ya
            hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32) + np.float32([xa, ya])
            # A solid seen from the side is thickest where its outline is widest:
            # weighting each pixel by its distance to the edge weighs a round
            # lump by its volume (r^3), not its area (r^2). For a lumpy 3D
            # object that lands much nearer its centre of mass.
            fill = np.zeros_like(sil)
            cv2.drawContours(fill, [cnt], -1, 1, -1)
            dtm = cv2.distanceTransform(fill, cv2.DIST_L2, 3)
            sw = float(dtm.sum())
            if sw > 0:
                tx_ = float((dtm * xx).sum() / sw)
                ty_ = float((dtm * yy).sum() / sw)
        col = (frame[ya:yb, xa:xb][core > 0].reshape(-1, 3).mean(axis=0).astype(np.float32)
               if frame.ndim == 3 else None)
        out.append(Det(f=-1, x=cx, y=cy, d=width, area=float(area), elong=elong,
                       contrast=peak, ax=ax_, ay=ay_, tx=tx_, ty=ty_,
                       shadow=_shadow_like(frame, bg, ya, yb, xa, xb, core),
                       hull=hull, col=col))
        if len(out) >= max_blobs:
            break
    return out


def _shadow_like(frame, bg, ya, yb, xa, xb, core) -> bool:
    """A shadow is the backdrop, only darker: every colour channel drops by
    about the same factor. A ball has its own colour, or is much darker."""
    if frame.ndim != 3:
        return False
    fr = frame[ya:yb, xa:xb][core > 0].astype(np.float32)
    bg_ = bg[ya:yb, xa:xb][core > 0].astype(np.float32)
    if len(fr) < 6:
        return False
    ratio = np.median((fr + 1.0) / (bg_ + 1.0), axis=0)
    return bool(np.all((ratio > 0.45) & (ratio < 0.93)) and np.ptp(ratio) < 0.08)


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
                ratio = d.d / max(last.d, 1e-6)
                # A tumbling stick or phone changes width as it turns.
                size_ok = 0.5 < ratio < 2.0 or (min(d.elong, last.elong) > 1.5
                                                 and 0.3 < ratio < 3.3)
                if dist < gate and size_ok:
                    cost = dist / gate
                    if d.col is not None and last.col is not None:
                        # Colour keeps two objects apart where their paths cross.
                        cost += 0.4 * min(float(np.linalg.norm(d.col - last.col)) / 60.0, 2.0)
                    pairs.append((cost, ti, di))
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


def _steadiness(t, v) -> float:
    """Relative scatter of v around a straight-line trend in time."""
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 4 or np.ptp(t[ok]) <= 0:
        return 0.5
    c = np.polyfit(t[ok] - t[ok].mean(), v[ok], 1)
    res = v[ok] - np.polyval(c, t[ok] - t[ok].mean())
    return float(1.4826 * np.median(np.abs(res)) / max(np.median(v[ok]), 1e-6))


def _flight_score(t, x, y, d, dm, area=None, down=(0.0, 1.0),
                  dn=None) -> tuple[float, bool]:
    """How much one flight looks like a dropped object: long, smooth, steady in
    size, and speeding up along `down`. Also says whether it is a clean fall.

    dn: the size its travel is counted in (default dm). A big ball close to
    the camera has no room to travel many of its own widths, so the caller
    caps it at a share of the picture."""
    from .physics import robust_fit_2d
    dn = dm if dn is None else dn
    n = len(t)
    fit = robust_fit_2d(t, x, y)
    a_down, s_down = fit.accel_along(down)
    a_mag = float(np.hypot(*fit.accel_vec()))
    ext = float(np.hypot(np.ptp(x), np.ptp(y)))
    # Speeds up downward, clearly (a hand pulling away or a slowing arm doesn't).
    f_acc = 1.0 if (a_down > 0 and a_down > 3.0 * s_down) else 0.15
    # Acceleration points down, not sideways or up.
    cosang = a_down / a_mag if a_mag > 0 else 0.0
    f_dir = 1.0 if cosang > 0.87 else (0.5 if cosang > 0.64 else 0.15)
    # A ball keeps its size; hands, arms and shadows change shape. A slow trend
    # (ball moving toward the camera) is allowed. Something long that tumbles
    # changes width but keeps its area, so the steadier of the two counts.
    cv = _steadiness(t, d)
    if area is not None:
        cv = min(cv, _steadiness(t, np.sqrt(np.asarray(area, float))) * 1.5)
    f_size = 1.0 / (1.0 + (cv / 0.15) ** 2)
    # Smooth path: the fit residual is small next to the ball's size.
    f_fit = float(np.exp(-fit.rms / (0.35 * dm + 0.7)))
    clean = (f_acc == 1.0 and cosang > 0.8 and n >= 8 and float(np.ptp(t)) >= 0.1
             and ext >= 1.5 * dn and fit.rms < 0.5 * dm + 1.0)
    return float(n * min(ext / dn, 25.0) * f_acc * f_dir * f_size * f_fit), bool(clean)


def _own_down(t, x, y, dm):
    """The way this track accelerates, if it clearly does so steadily (a camera
    held sideways makes things fall sideways); None otherwise."""
    from .camera import gravity_2d
    g, n_ok, spread = gravity_2d(t, x, y, dm)
    if n_ok < 4 or spread > 15:
        return None
    if g[1] > math.cos(math.radians(30)):
        return None                      # close enough to the picture's own down
    return g


def classify_and_score(tracks: list[Track], times, size_cap: float | None = None) -> list[Track]:
    """Score every moving track by its ballistic flights.

    The track is cut at bounces first and each flight is judged on its own,
    so a ball that bounces five times collects five flights of evidence
    instead of failing one parabola fitted through all of them. Falling
    means speeding up toward the picture's bottom. Only when nothing in the
    clip does that cleanly does a track that steadily speeds up some other
    way (a camera held sideways) count, at a small discount: a leg or a
    hand moving sideways mustn't outscore a ball falling down.

    size_cap: travel is counted in the object's own widths, but at most in
    this many pixels each (a share of the picture), so a big ball close to
    the camera isn't judged to barely move.
    """
    from .physics import split_flights
    own_scores = {}
    for tr in tracks:
        x, y, d = tr.x, tr.y, tr.d
        dm = float(np.nanmedian(d)) if np.isfinite(d).any() else 1.0
        dm = max(dm, 1.0)
        dn = min(dm, size_cap) if size_cap else dm
        extent = float(np.hypot(np.ptp(x), np.ptp(y)))
        if extent < 1.0 * dn:
            tr.kind = "static"
            tr.score = 0.0
            continue
        tr.kind = "moving"
        t = np.array([times[f] for f in tr.f], float)
        score = 0.0
        tr.falls = False
        area = tr.area
        if len(t) >= 5 and np.ptp(t) > 0:
            for idx in split_flights(t, x, y, (0.0, 1.0), diam_px=dm, min_pts=5):
                sc_, clean = _flight_score(t[idx], x[idx], y[idx], d[idx], dm, area[idx], dn=dn)
                score += sc_
                tr.falls = tr.falls or clean
            own = _own_down(t, x, y, dm) if len(t) >= 12 else None
            if own is not None:
                s2, f2 = 0.0, False
                for idx in split_flights(t, x, y, own, diam_px=dm, min_pts=5):
                    sc_, clean = _flight_score(t[idx], x[idx], y[idx], d[idx], dm, area[idx],
                                               down=own, dn=dn)
                    s2 += sc_
                    f2 = f2 or clean
                if 0.8 * s2 > score:
                    own_scores[id(tr)] = (0.8 * s2, f2)
        if score <= 0:
            # Moves, but no flight worth fitting: keep it as a weak fallback.
            score = 0.02 * len(t) * min(extent / dn, 25.0)
        tr.score = float(score)
    if own_scores and not any(tr.falls for tr in tracks if tr.kind == "moving"):
        for tr in tracks:
            if id(tr) in own_scores:
                tr.score, tr.falls = own_scores[id(tr)]
    return sorted(tracks, key=lambda t: -t.score)


def pick_seeded(tracks: list[Track], times, seed) -> Track | None:
    """The moving track that passes through the seed point(s).

    seed: list of (t, x, y) in the same pixels as the tracks. One point is a
    click on the ball; many points are an earlier track to follow again.
    """
    if not seed or not tracks:
        return None
    times = np.asarray(times, float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30
    tol_t = max(2.5 * dt, 0.04)
    best, best_key = None, None
    for tr in tracks:
        if tr.kind != "moving" or not tr.dets:
            continue
        tt = times[tr.f]
        dm = max(float(np.nanmedian(tr.d)) if np.isfinite(tr.d).any() else 8.0, 2.0)
        lim = max(3.0 * dm, 25.0)
        hits, dists = 0, []
        for (ts, xs, ys) in seed:
            j = np.flatnonzero(np.abs(tt - ts) <= tol_t)
            if len(j) == 0:
                continue
            dd = float(np.min(np.hypot(tr.x[j] - xs, tr.y[j] - ys)))
            if dd <= lim:
                hits += 1
                dists.append(dd)
        if hits == 0:
            continue
        key = (hits, -float(np.median(dists)))
        if best_key is None or key > best_key:
            best, best_key = tr, key
    if best is not None:
        return best
    if len(seed) == 1:
        # The click missed every detection in time (the ball was still held,
        # or the click fell between frames): take the track passing closest.
        ts, xs, ys = seed[0]
        for tr in tracks:
            if tr.kind != "moving" or not tr.dets:
                continue
            dm = max(float(np.nanmedian(tr.d)) if np.isfinite(tr.d).any() else 8.0, 2.0)
            cost = float(np.min(np.hypot(tr.x - xs, tr.y - ys)
                                + 200.0 * np.abs(times[tr.f] - ts)))
            if cost < 6.0 * dm + 60.0 and (best_key is None or cost < best_key):
                best, best_key = tr, cost
    return best


@dataclass
class TrackResult:
    main: Track | None
    others: list
    rest: Det | None          # where the object sat before release, if seen
    rest_d: float | None
    n_candidates: int
    d_fine: np.ndarray | None = None   # accurate widths for a subset of main (NaN elsewhere)
    seeded: bool = False               # main was chosen by a seed (a click, or a first pass)
    dets: list | None = None           # every candidate blob, per frame (for the debug video)
    tracks: list | None = None         # every linked track, best score first
    objects: list | None = None        # one TrackResult per falling object, this one first
    L_fine: np.ndarray | None = None   # accurate long-side lengths (non-round objects), NaN elsewhere


def measure_length(frame, bg, hull) -> float:
    """Length of an elongated object along its long axis, to a fraction of a pixel.

    The outline found by thresholding overshoots each end by a pixel or so
    (a faint edge pixel still passes the threshold). Instead, the picture is
    turned so the long axis lies flat, the strongest contrast is taken down
    each column, and each end is placed where that contrast falls to half of
    its level near that end (each end its own level, so a two-tone object
    like a hammer is measured right).
    """
    if hull is None or len(hull) < 3:
        return float("nan")
    h = np.asarray(hull, np.float64).reshape(-1, 2)
    m = cv2.moments(h.astype(np.float32).reshape(-1, 1, 2))
    if m["m00"] <= 0:
        return float("nan")
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    phi = 0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    e1 = np.array([np.cos(phi), np.sin(phi)])
    e2 = np.array([-np.sin(phi), np.cos(phi)])
    u = (h - [cx, cy]) @ e1
    v = (h - [cx, cy]) @ e2
    pad = 7
    u0, u1 = float(u.min()) - pad, float(u.max()) + pad
    v0, v1 = float(v.min()) - pad, float(v.max()) + pad
    Wn, Hn = int(np.ceil(u1 - u0)) + 1, int(np.ceil(v1 - v0)) + 1
    if Wn < 8 or Hn < 5 or Wn * Hn > 400_000:
        return float("nan")
    # Output pixel (i, j) shows the point c + (u0 + i) e1 + (v0 + j) e2.
    M = np.array([[e1[0], e2[0], cx + u0 * e1[0] + v0 * e2[0]],
                  [e1[1], e2[1], cy + u0 * e1[1] + v0 * e2[1]]])
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    a = cv2.warpAffine(frame, M, (Wn, Hn), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    b = cv2.warpAffine(bg, M, (Wn, Hn), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    D = cv2.absdiff(a, b).astype(np.float64)
    if D.ndim == 3:
        D = D.max(axis=2)
    band = slice(max(int(pad - 2), 0), min(int(Hn - pad + 2), Hn))
    prof = D[band].max(axis=0)
    base = float(np.median(np.concatenate([D[:3].ravel(), D[-3:].ravel()])))
    top = float(prof.max())
    if top - base < 12:
        return float("nan")
    inside = np.flatnonzero(prof > base + 0.6 * (top - base))
    if len(inside) < 4:
        return float("nan")
    i_a, i_b = int(inside[0]), int(inside[-1])
    span = i_b - i_a
    k = max(2, span // 6)
    edges = []
    for side in (0, 1):
        seg = prof[i_a:i_a + k + 1] if side == 0 else prof[i_b - k:i_b + 1]
        lvl = base + 0.5 * (float(np.percentile(seg, 75)) - base)
        if side == 0:
            j = i_a
            while j > 0 and prof[j - 1] > lvl:
                j -= 1
            if j == 0:
                return float("nan")
            p0, p1 = prof[j - 1], prof[j]
            edges.append(j - 1 + (lvl - p0) / max(p1 - p0, 1e-9))
        else:
            j = i_b
            while j < len(prof) - 1 and prof[j + 1] > lvl:
                j += 1
            if j == len(prof) - 1:
                return float("nan")
            p0, p1 = prof[j], prof[j + 1]
            edges.append(j + (p0 - lvl) / max(p0 - p1, 1e-9))
    return float(edges[1] - edges[0])


def refine_lengths(frames, bg, tr: Track, max_n: int = 120) -> np.ndarray | None:
    """measure_length for up to max_n detections, if the object looks elongated."""
    el = [d.elong for d in tr.dets]
    if not tr.dets or np.median(el) < 1.25:
        return None
    n = len(tr.dets)
    out = np.full(n, np.nan)
    for j in np.unique(np.linspace(0, n - 1, min(max_n, n)).astype(int)):
        d = tr.dets[j]
        out[j] = measure_length(frames[d.f], bg, d.hull)
    return out


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


def _same_object(a: Track, b: Track, times) -> bool:
    """Is b a piece of a (a panel, a highlight) or the rest of a's own path
    after a dropout, rather than a second object?"""
    da = max(float(np.nanmedian(a.d)) if np.isfinite(a.d).any() else 4.0, 2.0)
    db = max(float(np.nanmedian(b.d)) if np.isfinite(b.d).any() else 4.0, 2.0)
    ia = {int(f): j for j, f in enumerate(a.f)}
    common = [int(f) for f in b.f if int(f) in ia]
    if len(common) >= 3:
        ib = {int(f): j for j, f in enumerate(b.f)}
        sep = np.array([np.hypot(a.x[ia[f]] - b.x[ib[f]], a.y[ia[f]] - b.y[ib[f]])
                        for f in common])
        # Overlapping pieces that move as one. Two balls dropped side by side
        # also move together, but they sit more than a ball apart.
        return bool(np.median(sep) < 0.6 * (da + db) and np.ptp(sep) < 0.6 * max(da, db))
    if len(common) > 0:
        return False
    # No shared frames: is b where a would be, a moment later (or earlier)?
    first, second = (a, b) if a.f[-1] < b.f[0] else (b, a)
    last, nxt = first.dets[-1], second.dets[0]
    gap = float(times[nxt.f] - times[last.f])
    if gap > 0.35:
        return False                      # a separate drop later in the clip
    ratio = nxt.d / max(last.d, 1e-6)
    if not 0.65 < ratio < 1.55:
        return False
    # Allow for the speed it had, and the speed a fall adds over the gap.
    sp = _speed(first, times)
    acc = 2.0 * sp / max(float(times[last.f] - times[first.dets[0].f]), 0.05)
    dist = float(np.hypot(nxt.x - last.x, nxt.y - last.y))
    return dist < 4.0 * max(da, db) + 1.5 * sp * gap + acc * gap * gap


def _moves_with(a: Track, b: Track, times) -> bool:
    """Does b move along with a, the way whatever casts a shadow moves with it?"""
    ia = {int(f): j for j, f in enumerate(a.f)}
    common = [(ia[int(f)], j) for j, f in enumerate(b.f) if int(f) in ia]
    if len(common) < max(8, 0.3 * len(a.dets)):
        return False
    ja, jb = (np.asarray(c) for c in zip(*common))
    t = np.asarray(times, float)[a.f[ja]]
    if np.any(np.diff(t) <= 0):
        return False
    va = np.concatenate([np.gradient(a.x[ja], t), np.gradient(a.y[ja], t)])
    vb = np.concatenate([np.gradient(b.x[jb], t), np.gradient(b.y[jb], t)])
    if np.std(va) <= 0 or np.std(vb) <= 0:
        return False
    return float(np.corrcoef(va, vb)[0, 1]) > 0.7


def _mark_lone_shadows(moving: list[Track], times, min_d: float) -> None:
    """A shadow is cast by something that moves with it. A pale ball on a
    bright, plain backdrop looks like a shadow by its colour alone (the
    backdrop, only darker), but nothing solid moves along with it. Only
    something sizeable (min_d px across) counts: a faint speck is more
    likely the shadow of something out of the picture."""
    solid = [t for t in moving if t.shadow_frac <= 0.5]
    for tr in moving:
        d = float(np.nanmedian(tr.d)) if np.isfinite(tr.d).any() else 0.0
        tr.lone = (tr.shadow_frac > 0.5 and d >= min_d
                   and not any(_moves_with(tr, s, times) for s in solid))


def select_objects(moving: list[Track], times, primary: Track, max_n: int = 4) -> list[Track]:
    """The distinct falling objects, primary first.

    Every moving track with a clean fall of its own counts, except shadows
    (the backdrop, only darker) and pieces or continuations of an object
    already chosen. A hand pulling away doesn't speed up downward, so it
    never has a clean fall.
    """
    out = [primary]
    best = max(primary.score, 1e-9)
    for tr in moving:
        if len(out) >= max_n:
            break
        if tr is primary or not tr.falls or len(tr.dets) < 12:
            continue
        if tr.score < 0.01 * best or tr.is_shadow:
            continue
        if any(_same_object(o, tr, times) for o in out):
            continue
        out.append(tr)
    return out


def _rest_spot(frames, bg, tracks, main: Track, times):
    """A static blob where the fall begins = the object's resting spot (ghost)."""
    f0 = main.dets[0]
    t0 = float(times[f0.f])
    for t in tracks:
        if t.kind != "static":
            continue
        sx, sy = float(np.median(t.x)), float(np.median(t.y))
        ts0, ts1 = float(times[t.dets[0].f]), float(times[t.dets[-1].f])
        if np.hypot(sx - f0.x, sy - f0.y) < 2.5 * f0.d and ts0 - 0.1 <= t0 <= ts1 + 0.6:
            mid = t.dets[len(t.dets) // 2]
            wd = measure_width(frames[mid.f], bg, sx, sy, float(np.median(t.d)))
            rest = Det(f=t.dets[0].f, x=sx, y=sy, d=wd, area=0, elong=1, contrast=0)
            return rest, (wd if np.isfinite(wd) else None)
    return None, None


def track_falling_object(frames, times, *, roi=None, bg=None, seed=None,
                         max_objects: int = 4, footprints=None) -> TrackResult:
    """Find the falling objects. seed: optional list of (t, x, y) points on the
    one to treat as primary. The result's main is the primary object; every
    falling object (primary first) is in .objects, each as its own result."""
    bg = background(frames) if bg is None else bg
    times = np.asarray(times, float)
    dets = [detect(f, bg, roi=roi,
                   footprint=None if footprints is None else
                   np.asarray(footprints[i], np.float32).reshape(-1, 1, 2))
            for i, f in enumerate(frames)]
    # Travel counts in the object's widths, up to a sixth of the picture each.
    cap = min(frames[0].shape[:2]) / 6.0
    tracks = link(dets, times)
    tracks = classify_and_score(tracks, times, size_cap=cap)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30
    # Bridge short dropouts (bounce contact, a frame lost behind a hand),
    # judged in time so 240 fps clips get the same 40 ms as 60 fps ones.
    gap = max(4, int(round(0.04 / max(dt, 1e-6))))
    tracks = merge_fragments(tracks, times, max_gap_frames=gap)
    tracks = classify_and_score(tracks, times, size_cap=cap)
    moving = [t for t in tracks if t.kind == "moving"]
    if not moving:
        return TrackResult(None, [], None, None, sum(len(d) for d in dets), dets=dets,
                           tracks=tracks)
    _mark_lone_shadows(moving, times, min_d=min(frames[0].shape[:2]) / 20.0)
    main = pick_seeded(moving, times, seed) if seed else None
    seeded = main is not None
    if main is None:
        # Prefer a track with a clean fall over a busier one without.
        falling = [t for t in moving if t.falls and not t.is_shadow]
        main = falling[0] if falling and falling[0].score >= 0.25 * moving[0].score else moving[0]
    objs = select_objects(moving, times, main, max_n=max_objects) if max_objects > 1 else [main]
    n_cand = sum(len(d) for d in dets)
    results = []
    for k, ob in enumerate(objs):
        d_fine = refine_widths(frames, bg, ob, times, max_n=120)
        rest, rest_d = _rest_spot(frames, bg, tracks, ob, times)
        results.append(TrackResult(ob, [o for o in objs if o is not ob], rest, rest_d, n_cand,
                                   d_fine, seeded=seeded and k == 0, dets=dets, tracks=tracks,
                                   L_fine=refine_lengths(frames, bg, ob)))
    for r in results:
        r.objects = results
    return results[0]
