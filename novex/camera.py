"""The camera: which way gravity points as seen by it, at any angle.

A round object of known size is a 3D probe. Its apparent width d gives its
distance along the camera's axis (Z = f * D / d, f the focal length in
pixels), and its spot in the picture gives the rest:

    X = (u - cx) * D / d,   Y = (v - cy) * D / d,   Z = f * D / d.

So each fall can be followed in 3D, and its acceleration points along
gravity whatever way the camera faces: tilted down at a floor, up at a
balcony, rolled sideways. The parts of that direction that matter:

  pitch (beta)  how far gravity leans toward or away from the camera. A
                camera looking down 45 degrees sees only cos(45) = 71% of a
                fall in the picture; the rest is the ball shrinking as it
                moves away.
  roll          which way "down" points in the picture.

Once gravity's direction is known, the picture is turned (a homography,
exact for a pinhole camera) into the view of a level, upright camera at the
same spot. In that view gravity lies in the picture and points straight
down, the ball only moves toward or away from the camera at a steady speed,
and the usual measurement applies unchanged.

The weak link is f. Novex reads it from the file when it can, takes a
field of view you give it, and otherwise assumes a typical phone lens with
a wide error bar. Only the pitch feels it: a wrong f stretches the depth
axis, which changes g by about (error in f) * sin(beta)^2.
"""
from __future__ import annotations

import math
import re

import numpy as np

# Focal length as a fraction of the picture's longer side. Phone main
# cameras film at about 0.7 to 0.9, action cameras and ultra-wides at 0.3 to
# 0.55, zoom lenses and cinema framing at 1 to 2.5.
F_FRAC_DEFAULT = 0.80
F_REL_DEFAULT = 0.30

# Gravity counts as tilted toward or away from the camera only when it
# clearly is. The room's uprights measure that well; the ball's width less
# so: a constant error in it doesn't move the direction, but one that creeps
# (a shadow joining the ball at the floor) can, by several degrees. Below
# 10 degrees, leaving a tilt out costs under 1.5% of g.
PITCH_MIN_DEG = 4.0          # with the room's uprights
PITCH_MIN_DEG_BALL = 12.0    # from the ball alone
PITCH_MIN_Z = 4.0
PITCH_MIN_Z_BALL = 5.0
# A width that reads a little small at speed (blur) looks, to a single fall,
# exactly like the ball moving away faster and faster: both bend 1 / width
# into a parabola. So the ball alone counts only over two or more flights
# that agree, and only for a clear tilt.
# Steeper than this, the fall runs mostly along the line of sight.
PITCH_MAX_DEG = 72.0


# ---------------------------------------------------------------------------
# Focal length

_F35_KEYS = ("com.apple.quicktime.camera.focal_length.35mm_equivalent",
             "focal_length_35mm", "focallengthin35mmformat", "focal_length.35mm_equivalent")


def _num(v):
    try:
        return float(str(v).strip().split()[0].rstrip("m"))
    except (ValueError, IndexError):
        return None


def focal_prior(info=None, hfov_deg: float | None = None) -> dict:
    """Focal length as a fraction of the longer side, its relative error, and
    where it came from."""
    if hfov_deg:
        h = float(np.clip(hfov_deg, 5.0, 175.0))
        return {"f_frac": 0.5 / math.tan(math.radians(h / 2)), "rel": 0.03,
                "source": f"field of view you gave ({h:.0f}°)"}
    tags = {str(k).lower().split(":")[-1]: v for k, v in (getattr(info, "tags", None) or {}).items()}
    for k in _F35_KEYS:
        if k in tags:
            f35 = _num(tags[k])
            if f35 and 8 <= f35 <= 600:
                # 35 mm film is 36 mm wide; video crops a little more than photos.
                return {"f_frac": f35 / 36.0 * 1.05, "rel": 0.12,
                        "source": f"file says a {f35:.0f} mm (35 mm equivalent) lens"}
    for k, v in tags.items():
        if "lens" in k and isinstance(v, str):
            m = re.search(r"(\d+(?:\.\d+)?)\s*mm", v)
            if m and "35mm" in k:
                f35 = float(m.group(1))
                return {"f_frac": f35 / 36.0 * 1.05, "rel": 0.12,
                        "source": f"file says a {f35:.0f} mm (35 mm equivalent) lens"}
    return {"f_frac": F_FRAC_DEFAULT, "rel": F_REL_DEFAULT,
            "source": "assumed a typical phone lens (pass --hfov to set it)"}


# ---------------------------------------------------------------------------
# Off-axis widths

def minor_width(u, v, d, f, cx, cy, ax=None, ay=None):
    """The width that sets the ruler, from a measured width.

    Away from the middle of the picture a ball looks stretched along the line
    from the centre (by 1 / cos of its angle off the axis); across that line
    its width is exactly f * D / Z. A width measured across unit direction
    (ax, ay) is brought back to that across-the-line width. Where ax is NaN
    the width was the average of the horizontal and vertical ones (a ball
    too slow to blur); with no direction at all, nothing changes.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    d = np.asarray(d, float)
    if ax is None or ay is None:
        return d
    rx, ry = u - cx, v - cy
    r = np.hypot(rx, ry)
    tan2 = (r / f) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.where(r > 0, rx / np.maximum(r, 1e-9), 0.0)
        sr = np.where(r > 0, ry / np.maximum(r, 1e-9), 0.0)
        ax = np.asarray(ax, float)
        ay = np.asarray(ay, float)
        er = ax * cr + ay * sr
        across = np.sqrt(1.0 + er**2 * tan2)
        both = 0.5 * (np.sqrt(1.0 + cr**2 * tan2) + np.sqrt(1.0 + sr**2 * tan2))
    return d / np.where(np.isfinite(ax), across, both)


def sensor_axes(x, y, d):
    """Across which direction the tracker measured each width: across the
    motion when the ball moved more than 0.6 of its width between the
    neighbouring frames (as novex.track.refine_widths does), else NaN (the
    average of both axes)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    ax = np.full(n, np.nan)
    ay = np.full(n, np.nan)
    if n < 2:
        return ax, ay
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    vx, vy = x[i1] - x[i0], y[i1] - y[i0]
    sp = np.hypot(vx, vy)
    dd = np.where(np.isfinite(d), d, np.nanmedian(d) if np.isfinite(d).any() else 1.0)
    fast = sp > 0.6 * np.maximum(dd, 1.0)
    ax[fast] = -vy[fast] / sp[fast]
    ay[fast] = vx[fast] / sp[fast]
    return ax, ay


# ---------------------------------------------------------------------------
# Down in the picture, from the motion alone

def gravity_2d(t, x, y, diam_px=None, win_s=0.08):
    """Which way the falling object accelerates in the picture.

    Short stretches of the track (about win_s long) each get a parabola; a
    bounce or a hand spoils only the stretches around it, and those fit
    badly or disagree with the rest. Returns (unit vector, number of
    stretches that agree, their spread in degrees); (0, 1) if unclear.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(t)
    down = np.array([0.0, 1.0])
    if n < 9:
        return down, 0, 180.0
    dt = float(np.median(np.diff(t)))
    h = max(4, int(round(0.5 * win_s / max(dt, 1e-6))))
    noise = max(np.hypot(_noise(x) if n >= 7 else 0.3, _noise(y) if n >= 7 else 0.3), 0.1)
    acc, sig = [], []
    step = max(1, h // 2)
    for j in range(h, n - h, step):
        sel = slice(j - h, j + h + 1)
        tt = t[sel] - t[j]
        if np.ptp(tt) > 2.6 * h * dt:
            continue                                   # a gap in the track
        A = np.vstack([np.ones_like(tt), tt, 0.5 * tt * tt]).T
        cx, *_ = np.linalg.lstsq(A, x[sel], rcond=None)
        cy, *_ = np.linalg.lstsq(A, y[sel], rcond=None)
        res = np.hypot(x[sel] - A @ cx, y[sel] - A @ cy)
        rms = float(np.sqrt(np.mean(res**2)))
        if rms > 3.0 * noise + 0.02 * (diam_px or 0.0):
            continue                                   # a bounce or a grab inside
        a = np.array([cx[2], cy[2]])
        try:
            se = rms * math.sqrt(float(np.linalg.pinv(A.T @ A)[2, 2]))
        except np.linalg.LinAlgError:
            continue
        acc.append(a)
        sig.append(max(se, 1e-9))
    if not acc:
        return down, 0, 180.0
    acc = np.array(acc)
    sig = np.array(sig)
    mag = np.linalg.norm(acc, axis=1)
    strong = mag > 4 * sig
    if strong.sum() < 3:
        return down, int(strong.sum()), 180.0
    acc, mag, sig = acc[strong], mag[strong], sig[strong]
    m_med = float(np.median(mag))
    ok = (mag > 0.6 * m_med) & (mag < 1.6 * m_med)
    acc, mag, sig = acc[ok], mag[ok], sig[ok]
    ang = np.arctan2(acc[:, 0], acc[:, 1])
    a0 = float(np.angle(np.sum(np.exp(1j * ang) * (mag / sig) ** 2)))
    dev = np.angle(np.exp(1j * (ang - a0)))
    near = np.abs(dev) < math.radians(25)
    if near.sum() < 3:
        return down, int(near.sum()), 180.0
    a1 = a0 + float(np.average(dev[near], weights=(mag[near] / sig[near]) ** 2))
    spread = float(np.degrees(np.sqrt(np.average(np.angle(np.exp(1j * (ang[near] - a1))) ** 2))))
    return np.array([math.sin(a1), math.cos(a1)]), int(near.sum()), spread


# ---------------------------------------------------------------------------
# 3D fit of one flight

def _noise(u):
    if len(u) < 7:
        return float("nan")
    d3 = np.diff(u, 3)
    return float(1.4826 * np.median(np.abs(d3 - np.median(d3))) / np.sqrt(20))


def _quad(t, z, w):
    A = np.vstack([np.ones_like(t), t, 0.5 * t * t]).T
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
    return c


def _robust_quad(tau, z, w=None, iters=4):
    """Huber-weighted parabola z = c0 + c1 tau + c2 tau^2 / 2. Returns (c, cov,
    rms of the kept residuals, kept mask)."""
    A = np.vstack([np.ones_like(tau), tau, 0.5 * tau * tau]).T
    w0 = np.ones(len(tau)) if w is None else np.asarray(w, float)
    wt = w0.copy()
    for _ in range(iters):
        sw = np.sqrt(wt)
        c, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
        r = z - A @ c
        sc = max(1.4826 * float(np.median(np.abs(r[w0 > 0]))) if (w0 > 0).any() else 1.0, 1e-9)
        zz = np.abs(r) / sc
        wt = w0 * np.where(zz <= 2.5, 1.0, 2.5 / np.maximum(zz, 1e-9))
        wt[zz > 6] = 0.0
    keep = wt > 0
    sw = np.sqrt(wt)
    c, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
    r = z - A @ c
    # The error bar comes from the plain scatter of every point kept (the
    # down-weighting is for the fit, not a licence to look more precise).
    dof = max(float(keep.sum()) - 3, 1.0)
    s2 = float(np.sum((w0 * r * r)[keep]) / max(float(np.sum(w0[keep])) - 3, 1.0)) \
        if keep.sum() > 3 else float(np.sum(r * r) / dof)
    cov = np.linalg.pinv((A * wt[:, None]).T @ A) * s2
    return c, cov, float(np.sqrt(s2)), keep


def fit_flight_3d(t, u, v, d, f, cx, cy, d_ref, w=None):
    """A flight in 3D: depth from the widths, the rest from the positions.

    Depth along the camera's axis is Z = f * d_ref / d (units: pixels at the
    reference distance, i.e. multiples of D / d_ref meters, so no ruler is
    needed for the direction). Its acceleration comes from the widths alone:
    precise positions must not be allowed to explain a bend in the path (lens
    distortion, air drag, a tracking bias) as motion in depth. The sideways
    and up-down parts then come from the positions at that smoothed depth,
    X = (u - cx) * Z / f. The widths' own scatter about a smooth trend sets
    the depth error, and it carries into the sideways parts too.
    Returns {"G": (3,), "cov": (3, 3), ...} or None.
    """
    t = np.asarray(t, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    d = np.asarray(d, float)
    n = len(t)
    okd = np.isfinite(d) & (d > 2)
    if n < 10 or okd.sum() < 8:
        return None
    t0 = float(t.mean())
    tau = t - t0
    w = np.ones(n) if w is None else np.asarray(w, float)
    Zd = f * d_ref / d[okd]
    cz, cov_z, sz, keep = _robust_quad(tau[okd], Zd)
    if keep.sum() < 8 or not np.isfinite(cz).all():
        return None

    def lateral(c):
        Zs = c[0] + c[1] * tau + 0.5 * c[2] * tau * tau
        if np.any(Zs <= 0):
            return None
        X, Y = (u - cx) * Zs / f, (v - cy) * Zs / f
        cx_, cvx, sx, _ = _robust_quad(tau, X, w)
        cy_, cvy, sy, _ = _robust_quad(tau, Y, w)
        return cx_, cvx, cy_, cvy
    lat = lateral(cz)
    if lat is None:
        return None
    cxq, cvx, cyq, cvy = lat
    G = np.array([cxq[2], cyq[2], cz[2]])
    # How the depth fit's error moves the sideways parts.
    Jxy = np.zeros((2, 3))
    ev, evec = np.linalg.eigh(cov_z)
    for k in range(3):
        step = np.zeros(3)
        step[k] = max(np.sqrt(max(cov_z[k, k], 0.0)), 1e-6)
        lp, lm = lateral(cz + step), lateral(cz - step)
        if lp is None or lm is None:
            return None
        Jxy[:, k] = [(lp[0][2] - lm[0][2]) / (2 * step[k]), (lp[2][2] - lm[2][2]) / (2 * step[k])]
    cov = np.zeros((3, 3))
    cov[0, 0], cov[1, 1] = cvx[2, 2], cvy[2, 2]
    cov[:2, :2] += Jxy @ cov_z @ Jxy.T
    cov[:2, 2] = cov[2, :2] = Jxy @ cov_z[:, 2]
    cov[2, 2] = cov_z[2, 2]
    rms = float(np.hypot(np.sqrt(cvx[2, 2]), np.sqrt(cvy[2, 2])))
    return {"G": G, "cov": cov, "t0": t0, "n": n, "nd": int(keep.sum()),
            "sig_d_rel": float(sz / max(np.median(Zd), 1e-9)), "span_s": float(np.ptp(t)),
            "rms_px": rms}


def direction(G, cov):
    """Unit gravity direction, pitch (deg, + = away from the camera) and roll
    (deg, + = toward the picture's right), with their errors (deg)."""
    G = np.asarray(G, float)
    g = float(np.linalg.norm(G))
    nvec = G / g
    beta = math.degrees(math.asin(np.clip(nvec[2], -1, 1)))
    roll = math.degrees(math.atan2(nvec[0], nvec[1]))
    # Errors by linearization.
    Jn = (np.eye(3) - np.outer(nvec, nvec)) / g
    cn = Jn @ cov @ Jn.T
    sb = math.degrees(math.sqrt(max(cn[2, 2], 0.0)) / max(math.sqrt(1 - nvec[2] ** 2), 1e-6))
    q = nvec[0] ** 2 + nvec[1] ** 2
    jr = np.array([nvec[1], -nvec[0]]) / max(q, 1e-9)
    sr = math.degrees(math.sqrt(max(float(jr @ cn[:2, :2] @ jr), 0.0)))
    return nvec, beta, sb, roll, sr, cn


def combine(fits):
    """Inverse-variance mean of several flights' gravity vectors.

    Flights should agree: one real camera, one gravity. When they scatter
    more than their error bars allow, the error bar of the mean grows by that
    much (the widths are less trustworthy than they look). With three or
    more flights, one that is wildly off is left out. Returns (G, cov, used
    indices, reduced chi-square)."""
    fits = [(i, ft) for i, ft in enumerate(fits) if ft is not None]
    if not fits:
        return None, None, [], 0.0

    def mean(sel):
        Wsum = sum(np.linalg.pinv(ft["cov"]) for ft in sel)
        cov = np.linalg.pinv(Wsum)
        return cov @ sum(np.linalg.pinv(ft["cov"]) @ ft["G"] for ft in sel), cov
    use = [i for i, _ in fits]
    G, cov = mean([ft for _, ft in fits])
    if len(fits) >= 3:
        chis = [(i, float((ft["G"] - G) @ np.linalg.pinv(ft["cov"] + cov) @ (ft["G"] - G)))
                for i, ft in fits]
        i_worst, c_worst = max(chis, key=lambda q: q[1])
        others = sorted(c for _, c in chis if _ != i_worst)
        if c_worst > 25.0 and c_worst > 8 * max(np.median(others), 1.0):
            use = [i for i, _ in fits if i != i_worst]
            G, cov = mean([ft for i, ft in fits if i in use])
    sel = [ft for i, ft in fits if i in use]
    red = 0.0
    if len(sel) > 1:
        chi = sum(float((ft["G"] - G) @ np.linalg.pinv(ft["cov"]) @ (ft["G"] - G)) for ft in sel)
        red = chi / (3 * (len(sel) - 1))
        if red > 1:
            cov = cov * red
    return G, cov, use, float(red)


# ---------------------------------------------------------------------------
# Turning the picture upright and level

def level_rotation(nvec):
    """Smallest rotation taking gravity's direction to the picture's down."""
    a = np.asarray(nvec, float)
    a = a / np.linalg.norm(a)
    b = np.array([0.0, 1.0, 0.0])
    c = float(a @ b)
    ax = np.cross(a, b)
    s = float(np.linalg.norm(ax))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        return np.diag([-1.0, -1.0, 1.0])          # upside down: half a turn about the axis
    k = ax / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


class Leveler:
    """Maps picture points to the view of a level, upright camera at the same
    spot (and back). f, cx, cy in the pixels being mapped."""

    def __init__(self, f, cx, cy, nvec=None, R=None):
        self.f, self.cx, self.cy = float(f), float(cx), float(cy)
        self.R = np.asarray(R, float) if R is not None else (
            level_rotation(nvec) if nvec is not None else np.eye(3))
        self.identity = bool(np.allclose(self.R, np.eye(3)))

    def scaled(self, ratio, cx, cy):
        return Leveler(self.f * ratio, cx, cy, R=self.R)

    def _rays(self, u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        return np.stack([(u - self.cx) / self.f, (v - self.cy) / self.f, np.ones_like(u)], -1)

    def forward(self, u, v):
        """Picture -> level view. Also returns the third ray coordinate (>0 when
        the point is in front of the level camera)."""
        if self.identity:
            return np.asarray(u, float), np.asarray(v, float), np.ones(np.shape(u))
        r = self._rays(u, v) @ self.R.T
        z = r[..., 2]
        zs = np.where(np.abs(z) > 1e-9, z, 1e-9)
        return self.cx + self.f * r[..., 0] / zs, self.cy + self.f * r[..., 1] / zs, z

    def backward(self, uv_u, uv_v):
        """Level view -> picture."""
        if self.identity:
            return np.asarray(uv_u, float), np.asarray(uv_v, float)
        r = self._rays(uv_u, uv_v) @ self.R
        z = np.where(np.abs(r[..., 2]) > 1e-9, r[..., 2], 1e-9)
        return self.cx + self.f * r[..., 0] / z, self.cy + self.f * r[..., 1] / z

    def widths(self, u, v, d_minor):
        """A ball's width (across the line to the centre) as the level camera
        would see it: the same angle, divided by the depth ratio."""
        if self.identity:
            return np.asarray(d_minor, float)
        r = self._rays(u, v) @ self.R.T
        return np.asarray(d_minor, float) / np.where(r[..., 2] > 1e-6, r[..., 2], np.nan)

    def homography(self):
        K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])
        return K @ self.R @ np.linalg.inv(K)
