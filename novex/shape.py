"""Ball or not? And where is the centre of mass of something that isn't round?

A ball looks the same from every side: its width is a ruler on every frame
and its centre is its centre of mass. Anything else (a phone, a book, a
rotisserie chicken) changes outline as it tumbles. For those:

  * Position: the area centroid of the silhouette. For a uniform object with
    a symmetric shape (a box, a can, a phone) that is the centre of mass as
    the camera sees it, whichever way the object faces.
  * Nothing about the object needs to be known to find its real centre of
    mass: in free fall only that point moves on a parabola, and the body
    turns about it. So whatever the outline's centre does in step with the
    object's turning is the gap between the two, and taking it out leaves
    the centre of mass (spin_fit). That works for a hammer's heavy head, a
    half-full bottle, or a lumpy chicken tumbling end over end.
  * Ruler: the longest side, measured at half contrast at each end and read
    near the top of its range over the fall (a tumbling object is often seen
    foreshortened), against the object's known length.
"""
from __future__ import annotations

import cv2
import numpy as np


def outline_stats(dets) -> dict:
    """Per detection, from its silhouette outline (convex hull):
    circ  4 pi area / perimeter^2 (1 for a disc, 0.785 for a square)
    long, short  sides of the tightest rotated box (px)
    phi   direction of the long axis (radians, defined up to half a turn)
    elong ratio of the long to the short axis of the outline's spread
    """
    n = len(dets)
    out = {k: np.full(n, np.nan) for k in ("circ", "long", "short", "phi", "elong", "area")}
    for j, d in enumerate(dets):
        h = getattr(d, "hull", None)
        if h is None or len(h) < 3:
            continue
        h = np.asarray(h, np.float32).reshape(-1, 1, 2)
        a = float(cv2.contourArea(h))
        p = float(cv2.arcLength(h, True))
        if a <= 0 or p <= 0:
            continue
        out["area"][j] = a
        out["circ"][j] = min(1.0, 4.0 * np.pi * a / (p * p))
        _, (w, hh), _ = cv2.minAreaRect(h)
        # The outline runs through the edge pixels' centres: add a pixel.
        out["long"][j] = max(w, hh) + 1.0
        out["short"][j] = min(w, hh) + 1.0
        m = cv2.moments(h)
        if m["m00"] > 0:
            mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
            out["phi"][j] = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
            tr = mu20 + mu02
            disc = np.sqrt(max(tr * tr / 4.0 - (mu20 * mu02 - mu11 * mu11), 0.0))
            out["elong"][j] = np.sqrt((tr / 2 + disc) / max(tr / 2 - disc, 1e-9))
    return out


def width_scatter(w, t, flights) -> float:
    """Relative scatter of a width around each flight's own smooth trend (a
    ball moving toward or away from the camera changes size smoothly; a
    tumbler flickers). The trend is a parabola in 1 / width: the distance
    along the camera's axis of anything in free flight, whichever way the
    camera faces."""
    res = []
    for fl in flights:
        fl = np.asarray(fl)
        v, tt = np.asarray(w, float)[fl], np.asarray(t, float)[fl]
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() < 5:
            continue
        tc = tt[ok] - tt[ok].mean()
        deg = 2 if ok.sum() >= 10 else 1
        c = np.polyfit(tc, 1.0 / v[ok], deg)
        fit = 1.0 / np.maximum(np.polyval(c, tc), 1e-9)
        res.extend(((v[ok] - fit) / np.median(v[ok])).tolist())
    if len(res) < 5:
        return float("nan")
    return float(1.4826 * np.median(np.abs(res)))


def classify(stats, across, t, flights, speed, dt) -> dict:
    """Round or not, judged on the frames with the least motion blur.

    across: outline length across the path per frame (blur-free).
    speed: px per second per frame (NaN outside flights); dt: frame interval.
    Returns {"shape": "round" | "rigid", "sure": bool, ...}.
    """
    t = np.asarray(t, float)
    idx = np.concatenate([np.asarray(f) for f in flights]) if flights else np.arange(len(t))
    size = float(np.nanmedian(across[idx])) if np.isfinite(across[idx]).any() else float("nan")
    if not np.isfinite(size):
        size = float(np.nanmedian(stats["long"][idx])) if np.isfinite(stats["long"][idx]).any() \
            else float("nan")
    info = {"shape": "round", "sure": False, "size_px": size, "circularity": None,
            "aspect": None, "width_scatter": None, "frames": 0}
    if not np.isfinite(size) or size < 7:
        info["why"] = "too small in the picture to see its outline"
        return info
    blur = np.asarray(speed, float) * dt
    b = blur[idx]
    lo = idx[np.isfinite(b) & (b < 0.25 * size)]
    if len(lo) < 4:
        order = idx[np.argsort(np.where(np.isfinite(b), b, np.inf))]
        lo = order[:max(4, len(idx) // 4)]
    ok = lo[np.isfinite(stats["circ"][lo]) & np.isfinite(stats["short"][lo])]
    if len(ok) < 3:
        info["why"] = "outline not seen clearly"
        return info
    circ = float(np.median(stats["circ"][ok]))
    aspect = float(np.median(stats["long"][ok] / stats["short"][ok]))
    bf = float(np.clip(np.nanmedian(blur[ok] / size), 0.0, 1.0)) if np.isfinite(blur[ok]).any() \
        else 0.0
    cv = width_scatter(across, t, flights)
    cv_lim = 0.10 + 1.0 / size                 # a pixel of edge noise on a small ball
    round_ = aspect < 1.15 + bf and circ > 0.84 and (not np.isfinite(cv) or cv < cv_lim)
    clear_round = aspect < 1.08 + bf and circ > 0.9 and (not np.isfinite(cv) or cv < 0.6 * cv_lim)
    clear_rigid = aspect > 1.45 + bf or circ < 0.76 or (np.isfinite(cv) and cv > 2.0 * cv_lim)
    info.update(shape="round" if round_ else "rigid",
                sure=bool(clear_round if round_ else clear_rigid),
                circularity=circ, aspect=aspect,
                width_scatter=cv if np.isfinite(cv) else None, frames=int(len(ok)),
                blur_frac=bf)
    if round_:
        info["why"] = (f"round outline (aspect {aspect:.2f}, roundness {circ:.2f}) and a steady "
                       "width")
    else:
        bits = []
        if aspect >= 1.15 + bf:
            bits.append(f"{aspect:.1f}x longer than wide")
        if circ <= 0.84:
            bits.append(f"corners (roundness {circ:.2f})")
        if np.isfinite(cv) and cv >= cv_lim:
            bits.append(f"its width changes {100 * cv:.0f}% as it turns")
        info["why"] = "not round: " + ", ".join(bits) if bits else "not round"
    return info


def rigid_length(stats, t, flights, speed, vel_angle, dt):
    """Pixel length of a non-round object. Returns (samples, info).

    Motion blur stretches the outline along the path by speed x exposure,
    which reaches the long side as speed x exposure x |cos(angle between the
    long side and the path)|. The exposure isn't known, so it is fitted: the
    long side against that blur length over the frames of the fall (a
    tumbling object shows its length at every angle to its path). With the
    blur taken out, the length is read near the top of its range (a 3D
    tumble often shows the object foreshortened). The samples returned sit
    around the 90th percentile.
    """
    idx = np.concatenate([np.asarray(f) for f in flights])
    L = np.asarray(stats["long"], float)[idx]
    cosang = np.abs(np.cos(np.asarray(stats["phi"], float)[idx]
                           - np.asarray(vel_angle, float)[idx]))
    b = np.asarray(speed, float)[idx] * dt * cosang     # blur along the long side, full exposure
    ok = np.isfinite(L) & np.isfinite(b)
    if ok.sum() < 3:
        ok = np.isfinite(L)
        if ok.sum() < 3:
            return [], {"frames": 0}
        b = np.zeros_like(L)
    Lk, bk = L[ok], b[ok]
    expo, note = 0.0, None
    if ok.sum() >= 8 and np.ptp(bk) > 0.05 * np.median(Lk):
        # Robust straight line L = L0 + expo * b, expo between 0 and 1.
        keep = np.ones(len(Lk), bool)
        for _ in range(3):
            if keep.sum() < 6:
                break
            A = np.vstack([np.ones(keep.sum()), bk[keep]]).T
            (l0, e), *_ = np.linalg.lstsq(A, Lk[keep], rcond=None)
            res = Lk - (l0 + e * bk)
            sc = 1.4826 * np.median(np.abs(res[keep])) + 0.3
            keep = np.abs(res) < 3.0 * sc
        expo = float(np.clip(e, 0.0, 1.0))
    elif float(np.nanmedian(bk)) > 0.05 * np.median(Lk):
        note = "motion blur may stretch the outline a little"
    Lc = Lk - expo * bk
    p90 = float(np.percentile(Lc, 90))
    top = Lc[Lc >= np.percentile(Lc, 80)]
    return [float(v) for v in top], {"length_px": p90, "median_px": float(np.median(Lc)),
                                     "frames": int(ok.sum()), "exposure_frac": expo,
                                     "note": note}


def _line(tau, th, w=None):
    """Robust straight line th = a + b tau. Returns (a, b, rms, keep)."""
    keep = np.ones(len(tau), bool) if w is None else w.copy()
    a = b = 0.0
    rms = np.inf
    for _ in range(3):
        if keep.sum() < 4:
            break
        b, a = np.polyfit(tau[keep], th[keep], 1)
        res = th - (a + b * tau)
        rms = float(1.4826 * np.median(np.abs(res[keep]))) + 1e-4
        keep = np.abs(res) < max(3.0 * rms, 0.05)
    return float(a), float(b), rms, keep


def _smooth(v, k=2):
    """Moving average over 2k+1 samples (ends use what they have)."""
    v = np.asarray(v, float)
    if len(v) < 2 * k + 1:
        return v
    c = np.cumsum(np.concatenate([[0.0], v]))
    out = np.empty_like(v)
    for i in range(len(v)):
        a, b = max(0, i - k), min(len(v), i + k + 1)
        out[i] = (c[b] - c[a]) / (b - a)
    return out


def _periodicity(t, sigs, fmin, fmax, n=160):
    """Strongest common rhythm in the shape signals: (fraction of their
    variation explained, frequency in Hz)."""
    best = (0.0, None)
    if fmax <= fmin:
        return best
    for f in np.linspace(fmin, fmax, n):
        num = den = 0.0
        for sg in sigs:
            ok = np.isfinite(sg)
            if ok.sum() < 10:
                continue
            tt, z = t[ok] - t[ok].mean(), sg[ok]
            base = np.vstack([np.ones(ok.sum()), tt]).T
            r0 = z - base @ np.linalg.lstsq(base, z, rcond=None)[0]
            v0 = float(r0 @ r0)
            if v0 <= 0:
                continue
            A = np.hstack([base, np.vstack([np.cos(2 * np.pi * f * tt),
                                             np.sin(2 * np.pi * f * tt)]).T])
            r1 = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
            num += v0 - float(r1 @ r1)
            den += v0
        if den > 0 and num / den > best[0]:
            best = (num / den, float(f))
    return best


def _fit_offset(tau, x, y, cols_x, cols_y):
    """Parabolas in x and y plus extra columns for the outline's offset from the
    centre of mass. Robust (Huber) weights. Returns (coef, rms with, rms
    without, F statistic, covariance, how much the offset blurs the
    acceleration: its error with the offset over its error without)."""
    n = len(tau)
    P = np.vstack([np.ones(n), tau, tau * tau]).T
    Z = np.zeros_like(P)
    A = np.block([[P, Z, cols_x], [Z, P, cols_y]])
    A0 = np.block([[P, Z], [Z, P]])
    bvec = np.concatenate([x, y])
    w = np.ones(2 * n)
    for _ in range(3):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], bvec * sw, rcond=None)
        res = bvec - A @ coef
        sc = 1.4826 * np.median(np.abs(res)) + 1e-6
        w = 1.0 / np.maximum(1.0, np.abs(res) / (2.0 * sc))
    sw = np.sqrt(w)
    coef0, *_ = np.linalg.lstsq(A0 * sw[:, None], bvec * sw, rcond=None)
    ssr1 = float(np.sum(w * (bvec - A @ coef) ** 2))
    ssr0 = float(np.sum(w * (bvec - A0 @ coef0) ** 2))
    k = A.shape[1] - A0.shape[1]
    dof = max(2 * n - A.shape[1], 1)
    F = ((ssr0 - ssr1) / max(k, 1)) / max(ssr1 / dof, 1e-12)
    cov = np.linalg.pinv((A * w[:, None]).T @ A) * (ssr1 / dof)
    # Same noise level for both, so the ratio is only about the columns: an
    # offset that looks like a parabola (less than a turn) makes the
    # acceleration much less certain.
    cov0 = np.linalg.pinv((A0 * w[:, None]).T @ A0) * (ssr1 / dof)
    infl = float(np.sqrt(max(cov[2, 2] / max(cov0[2, 2], 1e-30),
                             cov[5, 5] / max(cov0[5, 5], 1e-30))))
    W = float(np.sum(w))
    return coef, float(np.sqrt(ssr1 / W)), float(np.sqrt(ssr0 / W)), float(F), cov, infl


# How much a centre-of-mass correction may blur the acceleration (its error
# with the correction over its error without). Beyond this the offset looks
# too much like part of the fall (less than about half a turn) to separate.
_MAX_INFL = 4.0


def spin_fit(t, x, y, stats, flights, size_px, speed=None, dt=None, min_turn=1.5):
    """Centre of mass of a turning object of any shape, flight by flight.

    The outline's centre c(t) sits off the centre of mass by an amount that
    depends only on which way the object faces. A free body turns steadily,
    so that offset changes in step with the turning, while the centre of
    mass follows a parabola. Three ways to know "in step with the turning":

      spin     the outline's long axis turns at a steady rate (an object
               turning in the picture's plane): the offset is a fixed vector
               in the body, c = parabola - R(theta) r
      tumble   the axis turns unevenly (it tumbles toward the camera too):
               the offset is any ellipse traced in step with the measured
               axis angle, c = parabola - (a cos theta + b sin theta)
      rhythm   no clear axis (a lumpy, roundish outline): the outline's area
               and stretch repeat as it turns, and that rhythm sets the
               phase (and its half, for a shape that repeats twice a turn)

    A correction is kept only when it clearly fits better (F test), is
    smaller than the object, and the object turned far enough for the
    offset not to hide in the parabola's start point and speed. Returns
    (x_cm, y_cm, info) or None when no flight could be checked.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    phi, elong = stats["phi"], stats["elong"]
    ok_axis = np.isfinite(phi) & np.isfinite(elong) & (elong > 1.15)
    xc, yc = x.copy(), y.copy()
    per = []
    for j, fl in enumerate(flights):
        fl = np.asarray(fl)
        n = len(fl)
        if n < 12:
            continue
        tau = t[fl] - t[fl].mean()
        dur = float(np.ptp(t[fl]))
        m = fl[ok_axis[fl]]
        if speed is not None and dt is not None and len(m) >= 10:
            # Motion blur pulls the outline's axis toward the direction of travel.
            slow = m[np.asarray(speed, float)[m] * dt < 0.3 * size_px]
            if len(slow) >= 10:
                m = slow
        cands = []
        if len(m) >= 10:
            th = np.unwrap(2.0 * phi[m]) / 2.0
            tc = float(t[m].mean())
            a0, om, rms_th, _ = _line(t[m] - tc, th)
            turn = abs(om) * dur
            span = float(np.ptp(_smooth(th)))
            if rms_th <= 0.15 and turn >= min_turn:
                theta = a0 + om * (t[fl] - tc)
                c, s_ = np.cos(theta), np.sin(theta)
                cands.append(("spin", np.vstack([-c, s_]).T, np.vstack([-s_, -c]).T,
                              {"omega": om, "turn_rad": turn}, (c, s_)))
            if span >= 2.5 and len(m) >= 0.6 * n:
                thf = np.interp(t[fl], t[m], _smooth(th))
                c, s_ = np.cos(thf), np.sin(thf)
                cols = np.vstack([-c, -s_]).T
                cands.append(("tumble", cols, None, {"omega": span / max(dur, 1e-9),
                                                     "turn_rad": span}, None))
        if not cands and dur > 0:
            # No usable axis: listen for the outline's own rhythm.
            area = np.asarray(stats["area"], float)[fl]
            e = np.asarray(elong, float)[fl] - 1.0
            ph2 = 2.0 * np.asarray(phi, float)[fl]
            sigs = [np.sqrt(area) / max(np.nanmedian(np.sqrt(area)), 1e-9), e,
                    e * np.cos(ph2), e * np.sin(ph2)]
            fmax = min(0.25 / dt if dt else 15.0, 15.0)
            frac, f = _periodicity(t[fl], sigs, 1.25 / dur, fmax)
            if f is not None and frac >= 0.5:
                for ff in (f, 0.5 * f):
                    if ff * dur < 0.75:
                        continue
                    ps = 2 * np.pi * ff * tau
                    cols = [-np.cos(ps), -np.sin(ps)]
                    if 2 * ff * dur >= 1.5:
                        cols += [-np.cos(2 * ps), -np.sin(2 * ps)]
                    cols = np.vstack(cols).T
                    cands.append(("rhythm", cols, None, {"omega": 2 * np.pi * ff,
                                                         "turn_rad": 2 * np.pi * ff * dur,
                                                         "rhythm": frac}, None))
        best = None
        for kind, cx_, cy_, meta, rot in cands:
            if cy_ is None:
                # An ellipse: its own coefficients for x and for y.
                cols = cx_
                Zc = np.zeros_like(cols)
                cx_, cy_ = np.hstack([cols, Zc]), np.hstack([Zc, cols])
            coef, rms1, rms0, F, cov, infl = _fit_offset(tau, x[fl], y[fl], cx_, cy_)
            k = cx_.shape[1]
            off = coef[6:]
            if kind == "spin":
                amp = float(np.hypot(*off))
            else:
                h = k // 2
                amp = 0.0
                for q in range(0, h, 2):      # each harmonic traces an ellipse
                    M = np.array([[off[q], off[q + 1]], [off[h + q], off[h + q + 1]]])
                    amp += float(np.linalg.svd(M, compute_uv=False)[0])
            sig = float(np.sqrt(max(np.mean(np.diag(cov[6:, 6:])), 1e-12)))
            # Kept only if it clearly fits better, is smaller than the object,
            # and can be told apart from the fall itself.
            ok = (F > 12.0 and rms1 < 0.85 * rms0 and amp > 3.0 * sig
                  and amp < 0.6 * size_px and infl < _MAX_INFL)
            rec = {"flight": j, "method": kind, "offset_px": amp, "offset_sigma_px": sig,
                   "rms_before": rms0, "rms_after": rms1, "F": F, "inflation": infl,
                   "used": bool(ok), "params": k, **meta}
            if best is None or (ok and (not best[0]["used"] or rms1 < best[0]["rms_after"])):
                best = (rec, coef, cx_, cy_, kind, rot)
        if best is None:
            continue
        rec, coef, cx_, cy_, kind, rot = best
        per.append(rec)
        if rec["used"]:
            if kind == "spin":
                c, s_ = rot
                r = coef[6:]
                xc[fl] = x[fl] + c * r[0] - s_ * r[1]
                yc[fl] = y[fl] + s_ * r[0] + c * r[1]
            else:
                xc[fl] = x[fl] - cx_ @ coef[6:]
                yc[fl] = y[fl] - cy_ @ coef[6:]
    if not per:
        return None
    used = [p for p in per if p["used"]]
    info = {"flights": per, "applied": bool(used),
            "method": used[0]["method"] if used else None,
            "offset_px": float(np.median([p["offset_px"] for p in used])) if used else 0.0,
            "spin_rev_s": float(np.median([abs(p["omega"]) for p in per]) / (2 * np.pi))}
    return xc, yc, info
