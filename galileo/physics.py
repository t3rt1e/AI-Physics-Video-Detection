"""Trajectory fitting: the core measurement.

Positions come in as pixels with timestamps in video seconds. Everything in
this module is scale-free and clock-free: it measures acceleration in
pixels per video-second squared (a_px). Scale (m/px) and the slow-motion
factor k are applied later:  g = s * a_px * k**2.

Main pieces
  robust_fit_2d   quadratic fit of x(t), y(t) with outlier down-weighting
  fit_flights     joint fit of all flights, including a rolling-shutter term
  split_flights   cut a track into ballistic flights at bounces
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 2D residual norm of isotropic Gaussian noise has median sigma*sqrt(2 ln 2).
_RAYLEIGH_MEDIAN = 1.1774


@dataclass
class Fit2D:
    """x(tau) = sum cx[k] tau^k, y(tau) = sum cy[k] tau^k, tau = t - t0 + alpha*(r - rc)."""

    t0: float
    cx: np.ndarray
    cy: np.ndarray
    cov_x: np.ndarray
    cov_y: np.ndarray
    sigma: float          # robust per-axis residual scale, px
    rms: float            # weighted rms of 2D residuals, px
    n: int
    w: np.ndarray         # final robust weights (0..1)
    alpha: float = 0.0    # rolling shutter, video-seconds per px along rs_axis
    rs_axis: str = "y"
    rc: float = 0.0
    ssr: float = 0.0
    resid: np.ndarray | None = None   # 2D residual norm per point, px

    # --- derived quantities -------------------------------------------
    def accel_vec(self) -> np.ndarray:
        return np.array([2.0 * self.cx[2], 2.0 * self.cy[2]])

    def accel_cov(self) -> np.ndarray:
        return np.diag([4.0 * self.cov_x[2, 2], 4.0 * self.cov_y[2, 2]])

    def accel_along(self, u) -> tuple[float, float]:
        u = np.asarray(u, float)
        a = float(self.accel_vec() @ u)
        var = float(u @ self.accel_cov() @ u)
        return a, float(np.sqrt(max(var, 0.0)))

    def accel_mag(self) -> tuple[float, float]:
        a = self.accel_vec()
        m = float(np.hypot(*a))
        if m == 0:
            return 0.0, float("inf")
        u = a / m
        return m, float(np.sqrt(max(float(u @ self.accel_cov() @ u), 0.0)))

    def pos(self, tau) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(tau, float)
        return np.polyval(self.cx[::-1], tau), np.polyval(self.cy[::-1], tau)

    def vel(self, tau) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(tau, float)
        dcx = np.polyder(self.cx[::-1])
        dcy = np.polyder(self.cy[::-1])
        return np.polyval(dcx, tau), np.polyval(dcy, tau)


def _wls(A: np.ndarray, z: np.ndarray, w: np.ndarray):
    sw = np.sqrt(np.clip(w, 0, None))
    Aw = A * sw[:, None]
    coef, *_ = np.linalg.lstsq(Aw, z * sw, rcond=None)
    cov_u = np.linalg.pinv(Aw.T @ Aw)
    return coef, cov_u


def robust_fit_2d(t, x, y, w0=None, *, deg=2, alpha=0.0, r=None, rc=0.0,
                  t0=None, iters=8, huber_k=2.5, fixed_w=None) -> Fit2D:
    """Huber-weighted polynomial fit of a 2D trajectory.

    r/rc/alpha implement the rolling-shutter time shift: a point imaged at
    readout coordinate r was really captured at t + alpha*(r - rc).
    If fixed_w is given, no reweighting is done (used to compare alphas fairly).
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(t)
    w0 = np.ones(n) if w0 is None else np.asarray(w0, float)
    if t0 is None:
        t0 = float(np.average(t, weights=np.clip(w0, 1e-9, None)))
    tau = t - t0
    if alpha and r is not None:
        tau = tau + alpha * (np.asarray(r, float) - rc)
    A = np.vander(tau, deg + 1, increasing=True)

    w = w0.copy() if fixed_w is None else np.asarray(fixed_w, float)
    sigma = 1.0
    n_iter = 1 if fixed_w is not None else iters
    for _ in range(n_iter):
        cx, cux = _wls(A, x, w)
        cy, cuy = _wls(A, y, w)
        rx = x - A @ cx
        ry = y - A @ cy
        rr = np.hypot(rx, ry)
        good = w0 > 0
        med = np.median(rr[good]) if good.any() else 1.0
        sigma = max(med / _RAYLEIGH_MEDIAN, 0.03)
        if fixed_w is not None:
            break
        z = rr / sigma
        w_new = w0 * np.where(z <= huber_k, 1.0, huber_k / np.maximum(z, 1e-12))
        if np.allclose(w_new, w, atol=1e-4):
            w = w_new
            break
        w = w_new

    cx, cux = _wls(A, x, w)
    cy, cuy = _wls(A, y, w)
    rx = x - A @ cx
    ry = y - A @ cy
    neff = float(np.sum(w))
    dof = max(neff - (deg + 1), 1.0)
    s2x = float(np.sum(w * rx**2) / dof)
    s2y = float(np.sum(w * ry**2) / dof)
    ssr = float(np.sum(w * (rx**2 + ry**2)))
    rms = float(np.sqrt(np.sum(w * (rx**2 + ry**2)) / max(neff, 1.0)))
    return Fit2D(t0=float(t0), cx=cx, cy=cy, cov_x=cux * s2x, cov_y=cuy * s2y,
                 sigma=float(sigma), rms=rms, n=n, w=w, alpha=float(alpha),
                 rs_axis="y", rc=float(rc), ssr=ssr, resid=np.hypot(rx, ry))


# ---------------------------------------------------------------------------
# Flights and bounces
# ---------------------------------------------------------------------------

def _noise_px(u: np.ndarray) -> float:
    """Per-point noise from third differences (kills a parabola exactly)."""
    if len(u) < 6:
        return 0.3
    d3 = np.diff(u, 3)
    # var(d3) = 20 sigma^2 for white noise.
    return float(max(1.4826 * np.median(np.abs(d3 - np.median(d3))) / np.sqrt(20), 0.05))


def glitch_mask(t, x, y, diam_px=None) -> np.ndarray:
    """True for points that are fine, False for one-frame tracking glitches.

    A glitch jumps away from the path and comes straight back, e.g. the
    tracker grabbing a hand or a shadow for a single frame. Smooth motion
    barely deviates from the midpoint of its neighbours; a glitch deviates by
    much more than the noise. Bounce frames also stand out, and dropping them
    is harmless because contact frames are never fitted anyway.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(t)
    ok = np.ones(n, bool)
    if n < 5:
        return ok
    noise = float(np.hypot(_noise_px(x), _noise_px(y)))
    lim = max(6.0 * noise, 0.35 * (diam_px or 0.0), 1.5)
    for i in range(1, n - 1):
        t0, t1, t2 = t[i - 1], t[i], t[i + 1]
        if t2 <= t0:
            continue
        a = (t1 - t0) / (t2 - t0)
        mx = x[i - 1] + a * (x[i + 1] - x[i - 1])
        my = y[i - 1] + a * (y[i + 1] - y[i - 1])
        dev = np.hypot(x[i] - mx, y[i] - my)
        span = np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1])
        if dev > lim and dev > 0.6 * span:
            ok[i] = False
    return ok


def split_flights(t, x, y, gdir=(0.0, 1.0), *, min_pts=5, max_gap_s=0.2,
                  diam_px=None, smooth_s=1.0 / 60.0) -> list[np.ndarray]:
    """Cut a track into ballistic flights.

    Splits at bounces (the object stops moving down and starts moving up)
    and at long gaps, and trims still frames at either end (ball held before
    release, ball at rest after). One-frame glitches are left out.

    Direction changes are judged over smooth_s (1/60 s) rather than frame to
    frame: at 120-240 fps the ball moves under a pixel per frame early in a
    fall, so frame-to-frame noise would otherwise look like bounces.
    Returns a list of index arrays into the inputs.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(t)
    if n < 3:
        return []
    g = np.asarray(gdir, float)
    g = g / np.linalg.norm(g)
    good = np.flatnonzero(glitch_mask(t, x, y, diam_px))
    u = x * g[0] + y * g[1]
    noise = _noise_px(u[good])
    eps = max(3.0 * noise, 0.25)
    if diam_px:
        eps = max(eps, 0.02 * diam_px)

    # Break at gaps first.
    pieces = []
    start = 0
    for i in range(1, len(good)):
        if t[good[i]] - t[good[i - 1]] > max_gap_s:
            pieces.append(good[start:i])
            start = i
    pieces.append(good[start:])

    flights: list[np.ndarray] = []
    for idx in pieces:
        if len(idx) < 3:
            continue
        uu = u[idx]
        dt_med = float(np.median(np.diff(t[idx]))) if len(idx) > 1 else 1.0
        step = max(1, int(round(smooth_s / max(dt_med, 1e-6))))
        step = min(step, max(1, (len(idx) - 1) // 4))
        du = uu[step:] - uu[:-step]
        # Signs of real changes only; repeated frames give zero steps.
        sig_steps = [(k, np.sign(d)) for k, d in enumerate(du) if abs(d) > eps]
        bounces = []
        for (k1, s1), (k2, s2) in zip(sig_steps, sig_steps[1:]):
            if s1 > 0 and s2 < 0:
                lo, hi = k1, min(k2 + step, len(uu) - 1)
                b = lo + int(np.argmax(uu[lo:hi + 1]))
                if not bounces or b > bounces[-1] + 1:
                    bounces.append(b)
        # A bounce is where the object is lowest. A tracking hiccup near the
        # top of a hop (the blob meeting something on the wall for a few
        # frames) can also turn down-then-up there; that isn't one.
        keep = []
        for j, b in enumerate(bounces):
            a0 = bounces[j - 1] if j > 0 else 0
            b1 = bounces[j + 1] if j + 1 < len(bounces) else len(uu) - 1
            seg = uu[a0:b1 + 1]
            span = float(np.ptp(seg)) if len(seg) else 0.0
            if uu[b] >= float(np.max(seg)) - max(0.25 * span, 2.0 * eps):
                keep.append(b)
        bounces = keep
        cuts = [0] + bounces + [len(idx) - 1]
        for j in range(len(cuts) - 1):
            a, b = cuts[j], cuts[j + 1]
            # Exclude the contact frame itself from both neighbours.
            a = a + (1 if j > 0 else 0)
            b = b - (1 if j < len(cuts) - 2 else 0)
            seg = idx[a:b + 1]
            seg = _trim_stationary(seg, u, eps, step)
            if len(seg) >= min_pts:
                flights.append(seg)
    return flights


def _trim_stationary(seg: np.ndarray, u: np.ndarray, eps: float, step: int = 1) -> np.ndarray:
    """Drop frames where the object sits still at either end of a segment.

    A held ball before release, or a ball resting after landing, is not on
    the parabola. Movement is judged over `step` frames so slow starts at high
    frame rates still count as moving once they add up.
    """
    if len(seg) < 3:
        return seg
    uu = u[seg]
    step = max(1, min(step, len(seg) - 1))
    moving = np.abs(uu[step:] - uu[:-step]) > eps
    if not moving.any():
        return seg[:0]
    ks = np.flatnonzero(moving)
    first = 0 if moving[0] else int(ks[0]) + 1
    last = len(seg) - 1 if moving[-1] else int(ks[-1]) + step - 1
    last = min(last, len(seg) - 1)
    if last <= first:
        return seg[:0]
    return seg[first:last + 1]


# ---------------------------------------------------------------------------
# Joint fit with rolling shutter
# ---------------------------------------------------------------------------

@dataclass
class FlightResult:
    idx: np.ndarray
    fit: Fit2D
    a_along: float = 0.0       # px / s^2 along gravity
    a_sigma: float = 0.0
    duration: float = 0.0


@dataclass
class JointFit:
    flights: list[FlightResult]
    alpha: float
    alpha_max: float
    rs_axis: str
    rs_significance: float
    gdir: np.ndarray
    a_px: float = 0.0
    a_sigma: float = 0.0
    main: int = 0
    notes: list[str] = field(default_factory=list)
    used: list[int] = field(default_factory=list)   # flights that went into a_px
    excluded: tuple = ()                            # flights left out of a_px (e.g. a throw)
    rcoord: np.ndarray | None = None                # sensor readout coordinate per point


def _fit_all(flights, t, x, y, w0, alpha, r, rc, fixed=None):
    fits = []
    for j, idx in enumerate(flights):
        fw = None if fixed is None else fixed[j]
        fits.append(robust_fit_2d(t[idx], x[idx], y[idx], w0[idx], alpha=alpha,
                                  r=None if r is None else r[idx], rc=rc, fixed_w=fw))
    return fits


def fit_flights(t, x, y, w0, flights, *, fps, n_px_axis, rs_axis="y",
                rs_mode="auto", gdir=(0.0, 1.0), exclude=(), r=None) -> JointFit:
    """Fit every flight; optionally estimate one rolling-shutter alpha for all.

    The rolling-shutter term can only be as large as one full frame of
    readout (alpha_max = 1/(fps * lines)), which keeps it from absorbing an
    arbitrary trajectory. It is kept only if it clearly improves the fit.
    exclude: positions in `flights` to leave out of the combined a_px (they
    are still fitted, e.g. a thrown first flight when bounces measure g).
    r: where each point sits along the sensor's readout direction, when x, y
    aren't the sensor's own pixels (a levelled view); default y (or x).
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w0 = np.asarray(w0, float)
    r = np.asarray(r, float) if r is not None else (y if rs_axis == "y" else x)
    rc = float(n_px_axis) / 2.0
    alpha_max = 1.0 / (max(fps, 1e-6) * max(n_px_axis, 1.0))

    fits0 = _fit_all(flights, t, x, y, w0, 0.0, None, rc)
    alpha = 0.0
    signif = 0.0
    notes = []
    if rs_mode == "auto" and sum(len(f) for f in flights) >= 12:
        # Robust (Huber) objective with one fixed noise scale, so every alpha is
        # judged by the same yardstick; each alpha gets its own reweighting.
        sig0 = float(np.median([f.sigma for f in fits0]))
        sig0 = max(sig0, 0.03)

        def objective(a):
            J = 0.0
            for f in _fit_all(flights, t, x, y, w0, a, r, rc):
                z = f.resid / sig0
                k = 2.5
                J += float(np.sum(np.where(z <= k, 0.5 * z * z, k * z - 0.5 * k * k)))
            return J

        grid = np.linspace(-alpha_max, alpha_max, 33)
        J = np.array([objective(a) for a in grid])
        k = int(np.argmin(J))
        a_best = grid[k]
        if 0 < k < len(grid) - 1:
            y0, y1, y2 = J[k - 1], J[k], J[k + 1]
            den = y0 - 2 * y1 + y2
            if den > 0:
                a_best = grid[k] + 0.5 * (y0 - y2) / den * (grid[1] - grid[0])
        J0 = objective(0.0)
        J1 = objective(a_best)
        signif = float(2.0 * (J0 - J1))   # ~ delta chi-square, 1 extra parameter
        if signif > 25.0:
            alpha = float(a_best)
    elif rs_mode not in ("auto", "off"):
        alpha = float(rs_mode)

    fits = _fit_all(flights, t, x, y, w0, alpha, r if alpha else None, rc) if alpha else fits0
    for f in fits:
        f.rs_axis = rs_axis

    g = np.asarray(gdir, float)
    g = g / np.linalg.norm(g)
    out = []
    for idx, f in zip(flights, fits):
        a, s = f.accel_along(g)
        out.append(FlightResult(idx=np.asarray(idx), fit=f, a_along=a, a_sigma=s,
                                duration=float(t[idx][-1] - t[idx][0])))
    jf = JointFit(flights=out, alpha=alpha, alpha_max=alpha_max, rs_axis=rs_axis,
                  rs_significance=signif, gdir=g, notes=notes, excluded=tuple(exclude), rcoord=r)
    _combine(jf)
    return jf


def _combine(jf: JointFit):
    """Main flight = most points. a_px = weighted mean of flights that agree with it.

    Excluded flights are left out unless nothing usable would remain.
    """
    if not jf.flights:
        return
    cand = [j for j, f in enumerate(jf.flights) if j not in jf.excluded and len(f.idx) >= 6]
    if not cand:
        cand = list(range(len(jf.flights)))
    # The main flight is the longest real fall: a stretch with no clear
    # downward acceleration (a ball rolling across the floor at the end) is
    # no free flight, however long it is.
    falling = [j for j in cand if jf.flights[j].a_along > 3 * jf.flights[j].a_sigma > 0]
    jf.main = int(max(falling or cand, key=lambda j: len(jf.flights[j].idx)))
    m = jf.flights[jf.main]
    vals, wts, used = [], [], []
    for j, f in enumerate(jf.flights):
        if j not in cand:
            continue
        if len(f.idx) < 6 or not np.isfinite(f.a_sigma) or f.a_sigma <= 0:
            continue
        # Hops are weighed in only if they agree with the main flight to 25%.
        if not m.a_along or abs(f.a_along / m.a_along - 1.0) > 0.25:
            continue
        vals.append(f.a_along)
        wts.append(1.0 / f.a_sigma**2)
        used.append(j)
    if not vals:
        jf.a_px, jf.a_sigma = m.a_along, m.a_sigma
        jf.used = [jf.main]
        return
    jf.used = used
    wts = np.asarray(wts)
    jf.a_px = float(np.sum(np.asarray(vals) * wts) / np.sum(wts))
    jf.a_sigma = float(1.0 / np.sqrt(np.sum(wts)))


def rolling_shutter_effect(jf: JointFit, t, x, y, w0) -> float:
    """Relative change in a_px caused by the rolling-shutter correction."""
    if not jf.alpha:
        return 0.0
    plain = fit_flights(t, x, y, w0, [f.idx for f in jf.flights], fps=1.0, n_px_axis=1.0,
                        rs_mode="off", gdir=jf.gdir)
    if plain.a_px == 0:
        return 0.0
    return float(jf.a_px / plain.a_px - 1.0)


def trim_release(t, x, y, idx, gdir=(0.0, 1.0), a_ref=None, s_ref=None, *,
                 max_s: float = 0.2, max_frac: float = 0.4):
    """Cut the start of a flight that the hand was still pushing.

    A thrown or tossed ball is in free fall only once it leaves the hand.
    While the hand pushes, the ball's acceleration is the hand's, far from
    gravity's. The local acceleration (a parabola through about 60 ms of
    frames, with its own error bar) is compared with a_ref, the bounces' acceleration when there are
    any, otherwise the flight's own later part. Leading frames that are way
    off (by half of a_ref or more) are cut, plus the window around them. Air
    drag only bends the path gently, so it never trips this. If the trouble
    runs past max_s seconds or max_frac of the flight, nothing is cut: a
    whole flight that misbehaves is evidence, not a hand. Returns (idx, n_cut).
    """
    idx = np.asarray(idx)
    n = len(idx)
    if n < 15:
        return idx, 0
    g = np.asarray(gdir, float)
    g = g / np.linalg.norm(g)
    tt = t[idx]
    u = x[idx] * g[0] + y[idx] * g[1]
    dt = float(np.median(np.diff(tt))) if n > 1 else 1.0
    k_max = min(int(max_s / max(dt, 1e-6)), int(max_frac * n))
    if k_max < 2:
        return idx, 0
    if a_ref is None:
        tail = idx[int(0.4 * n):]
        a_ref, _ = robust_fit_2d(t[tail], x[tail], y[tail]).accel_along(g)
    if not (np.isfinite(a_ref) and a_ref > 0):
        return idx, 0
    w = max(3, int(round(0.03 / max(dt, 1e-6))))
    off = np.zeros(k_max + 1, bool)
    for j in range(k_max + 1):
        sel = slice(max(0, j - w), min(n, j + w + 1))
        if sel.stop - sel.start < 6:
            continue
        try:
            c, cov = np.polyfit(tt[sel] - tt[j], u[sel], 2, cov=True)
        except (ValueError, np.linalg.LinAlgError):
            continue
        s_loc = 2 * float(np.sqrt(max(cov[0, 0], 0.0)))
        # Way off, and not just noise (a handful of frames at 240 fps barely curve).
        off[j] = abs(2 * c[0] - a_ref) > max(0.5 * a_ref, 4 * s_loc)
    if not off.any():
        return idx, 0
    last = int(np.flatnonzero(off)[-1])
    if last >= k_max - 1:
        return idx, 0
    k = min(last + w, k_max)
    return idx[k:], k


def launch_velocity(fit: Fit2D, t_first: float, gdir=(0.0, 1.0)) -> tuple[float, float]:
    """(speed along gravity, time since the vertex) at the first tracked point.

    A ball dropped from rest has its parabola's vertex at, or just before,
    the first tracked frame. A thrown one is already moving fast when first
    seen: its vertex lies well before (or it has none).
    """
    g = np.asarray(gdir, float)
    g = g / np.linalg.norm(g)
    pu = fit.cx * g[0] + fit.cy * g[1]
    tau = t_first - fit.t0
    v = float(pu[1] + 2 * pu[2] * tau)
    if pu[2] <= 0:
        return v, float("inf")
    tau_v = -pu[1] / (2 * pu[2])
    return v, float(tau - tau_v)


def drag_path(fit: Fit2D, ta: float, tb: float, v_scale: float, vt: float, g: float,
              k: float = 1.0, v_extra=None, n: int | None = None):
    """How far air drag has pushed a flight off its drag-free path, from ta on.

    Quadratic drag pulls with -(g / vt^2) |v| v (m/s^2). Integrating that
    twice along the flight gives the displacement it caused; subtracting it
    from the tracked positions leaves the path the ball would have taken in a
    vacuum, which a plain parabola (and the rolling-shutter term) can then fit
    without mistaking drag for something else.

    fit gives the velocity in px per video second; v_scale turns that into
    real m/s (m per px times the slow-motion factor k). v_extra adds a
    velocity on the same grid (the drag velocity from a previous pass, when
    fit is already drag-free). Returns (grid, (dx, dy), (vx_d, vy_d)) in px
    and px per video second.
    """
    n = n or 96
    grid = np.linspace(ta, tb, n)
    vx, vy = fit.vel(grid - fit.t0)
    vx = np.asarray(vx, float).copy()
    vy = np.asarray(vy, float).copy()
    if v_extra is not None:
        vx += v_extra[0]
        vy += v_extra[1]
    sp = np.hypot(vx, vy) * v_scale                      # real m/s
    kd = g / max(vt, 1e-6) ** 2
    ax = -kd * sp * vx / max(k, 1e-9)                     # px per video second^2
    ay = -kd * sp * vy / max(k, 1e-9)

    def integ(f):
        out = np.zeros_like(f)
        out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(grid))
        return out
    vxd, vyd = integ(ax), integ(ay)
    return grid, (integ(vxd), integ(vyd)), (vxd, vyd)


def drag_factor(fit: Fit2D, tau, gdir, v_scale: float, vt: float) -> tuple[float, float]:
    """How much air drag lowered a flight's fitted acceleration, to first order.

    Quadratic drag adds -(g / vt^2) |v| v to the acceleration, so along
    gravity the ball feels g * (1 - |v| v_down / vt^2). A parabola fitted to
    positions measures a weighted average of the acceleration, with weight
    (1 - s^2)^2 across the flight (s from -1 at the first frame to 1 at the
    last), so the fit reads low by q = that same average of
    |v| v_down / vt^2, and g = a_fit / (1 - q). v comes from the fit itself,
    converted to m/s by v_scale (m per px times the slow-motion factor k). A
    hop that rises and falls has q near 0, because drag adds on the way up
    and subtracts on the way down. Returns (factor, q).
    """
    tau = np.asarray(tau, float)
    vx, vy = fit.vel(tau)
    g = np.asarray(gdir, float)
    g = g / np.linalg.norm(g)
    vx = np.asarray(vx, float) * v_scale
    vy = np.asarray(vy, float) * v_scale
    v_down = vx * g[0] + vy * g[1]
    span = float(np.ptp(tau)) if len(tau) > 1 else 0.0
    if span > 0:
        sn = 2.0 * (tau - tau.min()) / span - 1.0
        w = (1.0 - sn**2) ** 2
    else:
        w = np.ones(len(tau))
    if w.sum() <= 0:
        w = np.ones(len(tau))
    q = float(np.sum(w * np.hypot(vx, vy) * v_down) / np.sum(w)) / max(vt, 1e-6) ** 2
    q = float(np.clip(q, -0.5, 0.6))
    return 1.0 / (1.0 - q), q


def gravity_direction(jf: JointFit) -> np.ndarray:
    m = jf.flights[jf.main].fit
    a = m.accel_vec()
    n = np.linalg.norm(a)
    return a / n if n > 0 else np.array([0.0, 1.0])
