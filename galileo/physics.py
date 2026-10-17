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


def split_flights(t, x, y, gdir=(0.0, 1.0), *, min_pts=5, max_gap_s=0.2,
                  diam_px=None) -> list[np.ndarray]:
    """Cut a track into ballistic flights.

    Splits at bounces (the object stops moving down and starts moving up
    between neighbouring frames) and at long gaps, and trims stationary
    frames at either end (ball held before release, ball at rest after).
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
    u = x * g[0] + y * g[1]
    noise = _noise_px(u)
    eps = max(3.0 * noise, 0.25)
    if diam_px:
        eps = max(eps, 0.02 * diam_px)

    # Break at gaps first.
    pieces = []
    start = 0
    for i in range(1, n):
        if t[i] - t[i - 1] > max_gap_s:
            pieces.append(np.arange(start, i))
            start = i
    pieces.append(np.arange(start, n))

    flights: list[np.ndarray] = []
    for idx in pieces:
        if len(idx) < 3:
            continue
        uu = u[idx]
        du = np.diff(uu)
        # Signs of real steps only; duplicate frames give zero steps.
        sig_steps = [(k, np.sign(d)) for k, d in enumerate(du) if abs(d) > eps]
        bounces = []
        for (k1, s1), (k2, s2) in zip(sig_steps, sig_steps[1:]):
            if s1 > 0 and s2 < 0:
                # Lowest point between the last downward and first upward step.
                lo, hi = k1 + 1, k2
                b = lo + int(np.argmax(uu[lo:hi + 1]))
                bounces.append(b)
        cuts = [0] + bounces + [len(idx) - 1]
        for j in range(len(cuts) - 1):
            a, b = cuts[j], cuts[j + 1]
            # Exclude the contact frame itself from both neighbours.
            a = a + (1 if j > 0 else 0)
            b = b - (1 if j < len(cuts) - 2 else 0)
            seg = idx[a:b + 1]
            seg = _trim_stationary(seg, u, eps)
            if len(seg) >= min_pts:
                flights.append(seg)
    return flights


def _trim_stationary(seg: np.ndarray, u: np.ndarray, eps: float) -> np.ndarray:
    """Drop frames where the object sits still at either end of a segment.

    A held ball before release, or a ball resting after landing, is not on
    the parabola. The first kept frame is the first one that has moved; the
    last kept frame is the last one before the object stops.
    """
    if len(seg) < 3:
        return seg
    uu = u[seg]
    moving = np.abs(np.diff(uu)) > eps
    if not moving.any():
        return seg[:0]
    ks = np.flatnonzero(moving)
    first = 0 if moving[0] else int(ks[0]) + 1
    last = len(seg) - 1 if moving[-1] else int(ks[-1])
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


def _fit_all(flights, t, x, y, w0, alpha, r, rc, fixed=None):
    fits = []
    for j, idx in enumerate(flights):
        fw = None if fixed is None else fixed[j]
        fits.append(robust_fit_2d(t[idx], x[idx], y[idx], w0[idx], alpha=alpha,
                                  r=None if r is None else r[idx], rc=rc, fixed_w=fw))
    return fits


def fit_flights(t, x, y, w0, flights, *, fps, n_px_axis, rs_axis="y",
                rs_mode="auto", gdir=(0.0, 1.0)) -> JointFit:
    """Fit every flight; optionally estimate one rolling-shutter alpha for all.

    The rolling-shutter term can only be as large as one full frame of
    readout (alpha_max = 1/(fps * lines)), which keeps it from absorbing an
    arbitrary trajectory. It is kept only if it clearly improves the fit.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w0 = np.asarray(w0, float)
    r = y if rs_axis == "y" else x
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
                  rs_significance=signif, gdir=g, notes=notes)
    _combine(jf)
    return jf


def _combine(jf: JointFit):
    """Main flight = most points. a_px = weighted mean of flights that agree with it."""
    if not jf.flights:
        return
    sizes = [len(f.idx) for f in jf.flights]
    jf.main = int(np.argmax(sizes))
    m = jf.flights[jf.main]
    vals, wts = [], []
    for f in jf.flights:
        if len(f.idx) < 6 or not np.isfinite(f.a_sigma) or f.a_sigma <= 0:
            continue
        # Hops are weighed in only if they agree with the main flight to 25%.
        if abs(f.a_along / m.a_along - 1.0) > 0.25:
            continue
        vals.append(f.a_along)
        wts.append(1.0 / f.a_sigma**2)
    if not vals:
        jf.a_px, jf.a_sigma = m.a_along, m.a_sigma
        return
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


def gravity_direction(jf: JointFit) -> np.ndarray:
    m = jf.flights[jf.main].fit
    a = m.accel_vec()
    n = np.linalg.norm(a)
    return a / n if n > 0 else np.array([0.0, 1.0])
