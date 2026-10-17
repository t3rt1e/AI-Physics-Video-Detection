"""Layer 1: tests that need neither a ruler nor a clock.

Slowing down a parabola gives another parabola, and scaling pixels to
meters doesn't change a ratio, so none of these can be fooled by slow
motion or a wrong object size. Each check returns a plain dict:

  {"name", "status": pass|warn|fail|skip, "summary", "detail"}
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .physics import JointFit, robust_fit_2d


def _result(name, status, summary, **detail):
    return {"name": name, "status": status, "summary": summary, "detail": detail}


def parabola_check(t, x, y, w0, jf: JointFit, drag_corrected: bool = False) -> dict:
    """Constant acceleration within one fall: compare first and second halves.

    Only a clear, significant change fails. When the halves are too noisy to
    compare, the check says so instead of passing. drag_corrected says the
    positions already had air drag's push taken out (light balls), which only
    changes the wording.
    """
    name = "Constant acceleration"
    if not jf.flights:
        return _result(name, "skip", "No clean fall to test.")
    fl = jf.flights[jf.main]
    idx = fl.idx
    n = len(idx)
    if n < 12:
        return _result(name, "skip", f"Only {n} frames in the fall; need 12 to split it in half.")
    h = n // 2
    parts = [idx[: h + 1], idx[h:]]
    r = np.asarray(y if jf.rs_axis == "y" else x, float)
    accs = []
    for p in parts:
        f = robust_fit_2d(t[p], x[p], y[p], w0[p], alpha=jf.alpha, r=r[p], rc=fl.fit.rc)
        accs.append(f.accel_along(jf.gdir))
    (a1, s1), (a2, s2) = accs
    detail = dict(a_first_px=a1, a_second_px=a2, sigma_first=s1, sigma_second=s2,
                  drag_corrected=bool(drag_corrected))
    # A half that clearly runs the wrong way (decelerating fall, rising "drop").
    if (a1 + 2.5 * s1 < 0 and a2 - 2.5 * s2 > 0) or (a2 + 2.5 * s2 < 0 and a1 - 2.5 * s1 > 0):
        return _result(name, "fail",
                       "The object stops accelerating downward partway through the fall.",
                       **detail)
    a_ref = max(abs(a1), abs(a2), 1e-9)
    if a1 <= 0 or a2 <= 0 or s1 / a_ref > 0.35 or s2 / a_ref > 0.35:
        return _result(name, "skip", "Too noisy to compare the two halves of the fall.",
                       **detail)
    ratio = a2 / a1
    z = (a2 - a1) / max(np.hypot(s1, s2), 1e-9)
    detail.update(ratio=ratio, z=float(z))
    lo, hi = C.PARABOLA_RATIO_FAIL
    wlo, whi = C.PARABOLA_RATIO_WARN
    pct = (ratio - 1.0) * 100.0
    if (ratio < lo or ratio > hi) and abs(z) > 4:
        return _result(name, "fail",
                       f"Acceleration changes by {pct:+.0f}% between the first and second half "
                       "of the fall. Real gravity is constant.", **detail)
    if (ratio < wlo or ratio > whi) and abs(z) > 3:
        why = ("lens distortion, or a sign of generated motion" if drag_corrected else
               "air drag, lens distortion, or a sign of generated motion")
        return _result(name, "warn",
                       f"Acceleration drifts {pct:+.0f}% during the fall ({why}).", **detail)
    extra = " after taking out air drag" if drag_corrected else ""
    return _result(name, "pass",
                   f"Acceleration holds steady through the fall ({pct:+.0f}% half to half"
                   f"{extra}).", **detail)


def _u_poly(fit, g):
    """Coefficients (increasing powers of tau) of the position along gravity."""
    return fit.cx * g[0] + fit.cy * g[1]


def bounce_check(t, jf: JointFit, frame_dt: float) -> dict:
    """Bounce law: same gravity every hop, and hops never gain energy."""
    name = "Bounce law"
    fl = jf.flights
    if len(fl) < 2:
        return _result(name, "skip", "No bounces to test.")
    g = jf.gdir
    pairs = []
    for a, b in zip(fl, fl[1:]):
        gap = t[b.idx[0]] - t[a.idx[-1]]
        if gap > 6 * frame_dt + 1e-9:
            continue  # not neighbours across a single bounce
        pa, pb = _u_poly(a.fit, g), _u_poly(b.fit, g)
        # u_a(t) - u_b(t) = 0 in absolute time.
        ta0, tb0 = a.fit.t0, b.fit.t0
        tm = 0.5 * (t[a.idx[-1]] + t[b.idx[0]])
        # Solve numerically on a fine grid around the gap (robust to edge cases).
        grid = np.linspace(t[a.idx[-1]] - 2 * frame_dt, t[b.idx[0]] + 2 * frame_dt, 401)
        da = np.polyval(pa[::-1], grid - ta0) - np.polyval(pb[::-1], grid - tb0)
        sgn = np.flatnonzero(np.diff(np.sign(da)))
        tb = grid[sgn[np.argmin(np.abs(grid[sgn] - tm))]] if len(sgn) else tm
        va = np.polyval(np.polyder(pa[::-1]), tb - ta0)
        vb = np.polyval(np.polyder(pb[::-1]), tb - tb0)
        if not (va > 0 and vb < 0):
            continue  # not a bounce (coming in down, going out up): a gap or a glitch
        e = -vb / va
        pairs.append({"t_bounce": float(tb), "v_in": float(va), "v_out": float(vb),
                      "restitution": float(e),
                      "a_ratio": float(b.a_along / a.a_along) if a.a_along else float("nan"),
                      "a_ratio_z": float((b.a_along - a.a_along) /
                                         max(np.hypot(a.a_sigma, b.a_sigma), 1e-9)),
                      "n_after": int(len(b.idx))})
    if not pairs:
        return _result(name, "skip", "Bounces found but the hops around them weren't tracked.")
    es = [p["restitution"] for p in pairs if np.isfinite(p["restitution"])]
    detail = {"bounces": pairs}
    gain = [e for e in es if e > C.RESTITUTION_FAIL]
    if gain:
        return _result(name, "fail",
                       f"The ball leaves the floor faster than it arrived (restitution "
                       f"{max(gain):.2f} > 1). Real bounces lose energy; this looks reversed "
                       "or generated.", **detail)
    lo, hi = C.PARABOLA_RATIO_FAIL
    bad = [p for p in pairs if p["n_after"] >= 8 and (p["a_ratio"] < lo or p["a_ratio"] > hi)
           and abs(p["a_ratio_z"]) > 4]
    if bad:
        r = bad[0]["a_ratio"]
        return _result(name, "fail",
                       f"Gravity changes between hops (next hop accelerates {r:.2f}x as hard). "
                       "Every hop should share one g.", **detail)
    if es:
        return _result(name, "pass",
                       f"{len(pairs)} bounce(s) lose energy normally (restitution "
                       f"{min(es):.2f}-{max(es):.2f}) and hops share one g.", **detail)
    return _result(name, "skip", "Couldn't measure the bounce speeds.", **detail)


def equal_falls_check(a_main: tuple[float, float], others: list[tuple[float, float]]) -> dict:
    """Two objects falling side by side must fall together (same distance from camera)."""
    name = "Equal falls"
    if not others:
        return _result(name, "skip", "Only one falling object.")
    a0, s0 = a_main
    worst = None
    for a1, s1 in others:
        if a0 <= 0 or a1 <= 0:
            continue
        r = a1 / a0
        z = (a1 - a0) / max(np.hypot(s0, s1), 1e-9)
        if worst is None or abs(np.log(r)) > abs(np.log(worst[0])):
            worst = (r, z)
    if worst is None:
        return _result(name, "skip", "Second object not usable.")
    r, z = worst
    detail = {"ratio": float(r), "z": float(z)}
    if (r < 0.7 or r > 1.4) and abs(z) > 4:
        return _result(name, "warn",
                       f"Two falling objects accelerate differently ({r:.2f}x). Fine if one is "
                       "much closer to the camera; otherwise a red flag.", **detail)
    return _result(name, "pass", f"Falling objects keep pace ({r:.2f}x).", **detail)


def release_check(jf: JointFit, rest_xy, diam_px) -> dict:
    """A dropped object starts from rest at the spot where it was held."""
    name = "Starts from rest"
    if rest_xy is None or not jf.flights:
        return _result(name, "skip", "Didn't see the object held still before the drop.")
    fl = jf.flights[0]
    f = fl.fit
    g = jf.gdir
    pu = _u_poly(f, g)
    if pu[2] <= 0:
        return _result(name, "skip", "Fall direction unclear.")
    tau_v = -pu[1] / (2 * pu[2])
    xv, yv = f.pos(tau_v)
    dist = float(np.hypot(xv - rest_xy[0], yv - rest_xy[1]))
    ratio = dist / max(diam_px, 1e-6)
    detail = {"vertex_px": [float(xv), float(yv)], "rest_px": [float(rest_xy[0]), float(rest_xy[1])],
              "offset_diameters": ratio}
    if ratio > 10:
        return _result(name, "warn",
                       "The motion doesn't trace back to the spot where the object was held.",
                       **detail)
    if ratio > 1.5:
        return _result(name, "warn",
                       f"The fall's starting point sits {ratio:.1f} ball-widths from where the "
                       "object was held.", **detail)
    return _result(name, "pass", "The fall starts from rest where the object was held.", **detail)
