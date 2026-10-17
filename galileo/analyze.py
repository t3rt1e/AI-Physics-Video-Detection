"""The pipeline: video in, verdict out.

    frames -> stabilize -> track -> flights -> fit (with rolling shutter)
           -> clocks (k) -> rulers (s) -> scale-free checks -> verdict

g = s * a_px * k**2, where a_px is the measured acceleration in pixels per
video-second squared, s is meters per pixel at the object, and k is real
seconds per video second.
"""
from __future__ import annotations

import time
from typing import Callable

import cv2
import numpy as np

from . import checks, clocks
from . import config as C
from .physics import fit_flights, gravity_direction, rolling_shutter_effect, split_flights
from .scale import fuse, ruler_from_size, ruler_manual
from .stabilize import stabilize
from .track import track_falling_object
from .verdict import decide
from .video import motion_window, probe, read_frames, work_scale


def _depth_factors(t, d, flight_idx):
    """Relative scale per frame if the object moves toward or away from the camera."""
    m = np.ones(len(t))
    if len(flight_idx) < 8:
        return m, 0.0
    tt, dd = t[flight_idx], d[flight_idx]
    ok = np.isfinite(dd) & (dd > 2)
    if ok.sum() < 8:
        return m, 0.0
    inv = 1.0 / dd[ok]
    A = np.vstack([np.ones(ok.sum()), tt[ok] - tt[ok].mean()]).T
    coef, *_ = np.linalg.lstsq(A, inv, rcond=None)
    res = inv - A @ coef
    s = 1.4826 * np.median(np.abs(res)) + 1e-12
    se = s / np.sqrt(np.sum((tt[ok] - tt[ok].mean()) ** 2))
    change = abs(coef[1]) * np.ptp(tt) / coef[0]
    # Width estimates wobble a few percent with blur, so only a clear trend counts.
    if change > 0.12 and abs(coef[1]) / se > 6:
        fit_inv = coef[0] + coef[1] * (t - tt[ok].mean())
        d_ref = 1.0 / coef[0]
        m = d_ref * fit_inv  # d_ref / d_fit
        return m, float(change)
    return m, float(change)


def analyze(path: str, *, preset: str | None = None, size_m: float | None = None,
            size_rel_unc: float = 0.01, label: str | None = None, roi=None,
            t_start: float | None = None, t_end: float | None = None,
            manual_ruler=None, use_gemini: bool = False, use_depth: bool = False,
            tracker: str = "classical", sam2_box=None, stabilize_mode: str = "auto",
            rs_mode: str = "auto", g_ref: float = C.G_EARTH, scale_correction: float = 1.0,
            progress: Callable[[str], None] | None = None) -> dict:
    t_wall = time.time()
    say = progress or (lambda s: None)
    info = probe(path)
    result: dict = {"file": info.to_dict(), "notes": [], "version": 1}

    # ---- frames ------------------------------------------------------------
    n_est = info.n_frames or int(info.fps * info.duration) or C.MAX_FRAMES
    if t_start is None and t_end is None and n_est > C.MAX_FRAMES:
        say("finding the motion")
        t_start, t_end = motion_window(path, info, C.MAX_FRAMES)
        if t_start is not None:
            result["notes"].append(f"Long clip: analysed {t_start:.2f}-{t_end:.2f} s, where "
                                   "the motion is.")
    n_use = min(n_est, C.MAX_FRAMES)
    sc = work_scale(info, n_use, C.WORK_MAX_SIDE, C.FRAME_BUDGET_BYTES)
    say("decoding")
    frames, times, nums = read_frames(path, info, scale=sc, t_start=t_start, t_end=t_end,
                                      max_frames=C.MAX_FRAMES)
    if len(frames) < 6:
        return _unsure(result, "The clip is too short to measure.", t_wall)
    Hw, Ww = frames[0].shape[:2]
    dt_med = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / (info.fps or 30)
    fps_file = 1.0 / dt_med if dt_med > 0 else (info.fps or 30.0)
    result["work"] = {"scale": sc, "width": Ww, "height": Hw,
                      "frames": len(frames), "fps_measured": fps_file}

    # ---- stabilize + track ---------------------------------------------------
    say("stabilizing")
    frames_s, T, stab = stabilize(frames, stabilize_mode)
    result["stabilization"] = stab
    roi_px = None
    if roi is not None:
        roi_px = (int(roi[0] * Ww), int(roi[1] * Hw), int(roi[2] * Ww), int(roi[3] * Hw))
    say("tracking")
    if tracker == "sam2":
        from .sam2_tracker import track_with_sam2
        tr = track_with_sam2(frames_s, times, box=sam2_box)
    else:
        tr = track_falling_object(frames_s, times, roi=roi_px)
    if tr.main is None:
        return _unsure(result, "No falling object found. Try a region (ROI) around the drop, "
                               "or the SAM 2 tracker.", t_wall)
    main = tr.main
    f = main.f
    t = times[f]
    x, y, d_quick = main.x, main.y, main.d
    d = tr.d_fine if tr.d_fine is not None else d_quick
    area = main.area
    w0 = np.where(area > 1.8 * np.median(area), 0.3, 1.0)

    # ---- flights + fit --------------------------------------------------------
    say("fitting")
    gdir = np.array([0.0, 1.0])
    d_ref = float(np.nanmedian(d)) if np.isfinite(d).any() else float(np.median(d_quick))
    fl = split_flights(t, x, y, gdir, diam_px=d_ref)
    if not fl:
        return _unsure(result, "The object moves, but no clean ballistic flight was found.",
                       t_wall, track=_track_payload(main, times, Ww, Hw, T))
    m_fac, depth_change = _depth_factors(t, d, np.concatenate(fl))
    cxp, cyp = Ww / 2.0, Hw / 2.0
    xs = cxp + (x - cxp) * m_fac
    ys = cyp + (y - cyp) * m_fac
    rs_axis = "x" if info.rotation in (90, 270) else "y"
    n_axis = Ww if rs_axis == "x" else Hw
    jf = fit_flights(t, xs, ys, w0, fl, fps=fps_file, n_px_axis=n_axis, rs_axis=rs_axis,
                     rs_mode=rs_mode, gdir=gdir)
    gd = gravity_direction(jf)
    if np.degrees(np.arccos(np.clip(gd @ gdir, -1, 1))) > 10:
        gdir = gd
        fl = split_flights(t, xs, ys, gdir, diam_px=d_ref) or fl
        jf = fit_flights(t, xs, ys, w0, fl, fps=fps_file, n_px_axis=n_axis,
                         rs_axis=rs_axis, rs_mode=rs_mode, gdir=gdir)
    rs_rel = rolling_shutter_effect(jf, t, xs, ys, w0)
    naive = {"a_px": jf.a_px, "a_sigma": jf.a_sigma}

    # ---- clocks -------------------------------------------------------------
    say("reading clocks")
    ev = clocks.metadata_clock({"fps": fps_file, "capture_fps": info.capture_fps})
    mf = jf.flights[jf.main]
    mi = mf.idx
    vx, vy = mf.fit.vel(t[mi] - mf.fit.t0)
    # Expected motion per frame, only where the fit is valid (inside the flight).
    disp = {int(fi): float(np.hypot(a, b) * dt_med) for fi, a, b in zip(f[mi], vx, vy)}
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in frames_s]
    ev.append(clocks.duplicate_clock(gray, f, x, y, d_ref, disp))
    u_all = xs * gdir[0] + ys * gdir[1]
    main_idx = jf.flights[jf.main].idx
    ev.append(clocks.interpolation_clock(f[main_idx], u_all[main_idx]))
    bright = [float(np.median(g[::4, ::4])) for g in gray]
    ev.append(clocks.flicker_clock(bright, fps_file))
    clock = clocks.combine([e for e in ev if e], fps_file)
    del gray

    refit_note = None
    keep = clock.get("keep_frames")
    if keep and clock["confidence"] >= C.CLOCK_MIN_CONFIDENCE:
        sel = np.isin(f, np.asarray(keep))
        if sel.sum() >= 6:
            ii = np.flatnonzero(sel)
            fl2 = split_flights(t[ii], xs[ii], ys[ii], gdir, diam_px=d_ref)
            if fl2:
                fl2 = [ii[s] for s in fl2]
                jf = fit_flights(t, xs, ys, w0, fl2, fps=fps_file / clock["k"],
                                 n_px_axis=n_axis, rs_axis=rs_axis, rs_mode=rs_mode,
                                 gdir=gdir)
                fl = fl2
                refit_note = (f"Refit on the {len(ii)} original frames "
                              f"({clock['source']}).")

    # ---- rulers ---------------------------------------------------------------
    say("measuring scale")
    rulers = []
    obj_label = label
    used = np.concatenate([fl_i for fl_i in fl])
    d_samples = [v for v in d[used] if np.isfinite(v)]
    if len(d_samples) < 5:
        d_samples = [v for v in d if np.isfinite(v)]
    if tr.rest_d:
        d_samples += [tr.rest_d] * 3
    if preset and preset in C.OBJECT_PRESETS:
        p = C.OBJECT_PRESETS[preset]
        size_m, size_rel_unc, obj_label = p["size_m"], p["rel_unc"], p["label"]
    gem = None
    if size_m is None and use_gemini:
        try:
            from . import gemini
            k_mid = int(used[len(used) // 2])
            gem = gemini.identify_object(frames[int(f[k_mid])], (x[k_mid], y[k_mid], d_ref))
            if gem and gem.get("size_m"):
                size_m = float(gem["size_m"])
                size_rel_unc = float(gem.get("size_rel_uncertainty", 0.1))
                obj_label = gem.get("object", "object")
        except Exception as e:  # network, quota, missing key
            result["notes"].append(f"Gemini object ID unavailable: {e}")
    if size_m:
        r = ruler_from_size(d_samples, size_m, size_rel_unc, obj_label or "object")
        if r:
            rulers.append(r)
    if manual_ruler:
        (p1, p2, L) = manual_ruler
        rulers.append(ruler_manual((p1[0] * sc, p1[1] * sc), (p2[0] * sc, p2[1] * sc), L))
    if use_depth:
        try:
            from .depth import depth_pro_ruler
            k_mid = int(used[len(used) // 2])
            dr = depth_pro_ruler(frames[int(f[k_mid])], (x[k_mid], y[k_mid]), d_ref)
            if dr:
                rulers.append(dr)
        except Exception as e:
            result["notes"].append(f"Depth Pro unavailable: {e}")
    scale = fuse(rulers)
    if scale and scale_correction != 1.0:
        scale["s"] *= scale_correction
        scale["correction"] = scale_correction
    if scale and not scale["agree"]:
        result["notes"].append(f"Rulers disagree by {scale['spread'] * 100:.0f}%; error bars "
                               "widened.")

    # ---- scale-free checks ------------------------------------------------------
    say("testing physics")
    tests = [checks.parabola_check(t, xs, ys, w0, jf),
             checks.bounce_check(t, jf, dt_med)]
    others = []
    for o in tr.others:
        to = times[o.f]
        flo = split_flights(to, o.x, o.y, gdir, diam_px=float(np.median(o.d)))
        if flo:
            jo = fit_flights(to, o.x, o.y, np.ones(len(to)), flo, fps=fps_file,
                             n_px_axis=n_axis, rs_mode="off", gdir=gdir)
            others.append((jo.a_px, jo.a_sigma))
    tests.append(checks.equal_falls_check((jf.a_px, jf.a_sigma), others))
    rest_xy = (tr.rest.x, tr.rest.y) if tr.rest is not None else None
    lead = np.arange(0, int(jf.flights[0].idx[0]))
    if len(lead) >= 3:
        lx, ly = xs[lead][-6:], ys[lead][-6:]
        if np.ptp(lx) < 0.3 * d_ref and np.ptp(ly) < 0.3 * d_ref:
            rest_xy = (float(np.median(lx)), float(np.median(ly)))
    tests.append(checks.release_check(jf, rest_xy, d_ref))

    mf = jf.flights[jf.main]
    # Tracking quality is judged on jitter around a smooth path (third
    # differences), not on the parabola fit, so a smooth but non-physical
    # path still gets a physics verdict.
    ui = (xs * gdir[0] + ys * gdir[1])[mf.idx]
    wi = (-xs * gdir[1] + ys * gdir[0])[mf.idx]
    from .physics import _noise_px
    jitter = float(np.hypot(_noise_px(ui), _noise_px(wi)))
    tracking_ok = jitter < 0.25 * d_ref + 0.5
    if not tracking_ok:
        result["notes"].append(f"Tracking is jittery ({jitter:.1f} px on a {d_ref:.0f} px "
                               "object).")

    v = decide(a_px=jf.a_px, a_sigma=jf.a_sigma, scale=scale, clock=clock, tests=tests,
               n_points=len(mf.idx), duration=mf.duration,
               g_ref=g_ref, tracking_ok=tracking_ok, rs_rel=rs_rel)

    # ---- payload ----------------------------------------------------------------
    k_used = v.get("k", 1.0)
    result.update({
        "verdict": v,
        "tests": tests,
        "clock": _clean(clock),
        "scale": scale,
        "gemini_object": gem,
        "fit": {
            "a_px": jf.a_px, "a_sigma": jf.a_sigma, "a_px_naive": naive["a_px"],
            "gdir": [float(gdir[0]), float(gdir[1])],
            "rolling_shutter": {"alpha": jf.alpha, "alpha_max": jf.alpha_max,
                                "axis": jf.rs_axis, "effect_rel": rs_rel,
                                "F": jf.rs_significance},
            "depth_change_rel": depth_change,
            "flights": [{"n": int(len(fr_.idx)), "t0": float(t[fr_.idx[0]]),
                         "t1": float(t[fr_.idx[-1]]), "a_px": fr_.a_along,
                         "a_sigma": fr_.a_sigma, "rms_px": fr_.fit.rms}
                        for fr_ in jf.flights],
            "main": jf.main, "diam_px": d_ref, "refit": refit_note,
        },
        "track": _track_payload(main, times, Ww, Hw, T),
        "plot": _plot_payload(jf, t, xs, ys, gdir, scale, k_used, g_ref, d_ref),
        "ghost": _ghost_payload(jf, times, gdir, scale, k_used, g_ref, Ww, Hw, T, t, xs, ys),
        "elapsed_s": time.time() - t_wall,
    })
    if use_gemini and v["verdict"] != "UNSURE":
        try:
            from . import gemini
            text = gemini.explain(result)
            if text:
                v["explanation"] = text
        except Exception as e:
            result["notes"].append(f"Gemini explanation unavailable: {e}")
    say("done")
    return result


# ---------------------------------------------------------------------------
def _unsure(result, why, t_wall, **extra):
    result.update({"verdict": {"verdict": "UNSURE", "confidence": 0.0, "headline": why,
                               "reasons": [why], "g": None, "sigma": None},
                   "tests": [], "elapsed_s": time.time() - t_wall})
    result.update(extra)
    return result


def _clean(d):
    out = {}
    for k, v in d.items():
        if k == "keep_frames":
            continue
        if k == "evidence":
            out[k] = [{kk: vv for kk, vv in e.items() if kk != "keep_frames"} for e in v]
            continue
        out[k] = v
    return out


def _to_frame_coords(T, fi, x, y):
    if T is None:
        return x, y
    A = cv2.invertAffineTransform(np.asarray(T[fi], np.float64))
    return (A[0, 0] * x + A[0, 1] * y + A[0, 2], A[1, 0] * x + A[1, 1] * y + A[1, 2])


def _track_payload(main, times, W, H, T):
    pts = []
    for dt in main.dets:
        px, py = _to_frame_coords(T, dt.f, dt.x, dt.y)
        pts.append({"t": float(times[dt.f]), "x": float(px / W), "y": float(py / H),
                    "r": float(0.5 * dt.d / W)})
    return {"points": pts}


def _origin(jf, t, xs, ys, gdir, scale, k):
    """Where the fall starts: the release (parabola vertex) if it is plausible,
    otherwise the first tracked point, with the clip's speed there."""
    f0 = jf.flights[0]
    pu = f0.fit.cx * gdir[0] + f0.fit.cy * gdir[1]
    t_first = float(t[f0.idx[0]])
    dur = float(t[f0.idx[-1]] - t_first)
    if pu[2] > 0:
        tau_v = -pu[1] / (2 * pu[2])
        t_v = f0.fit.t0 + tau_v
        if t_first - max(0.35, 0.5 * dur) <= t_v <= t_first + 0.02:
            u_v = pu[0] + pu[1] * tau_v + pu[2] * tau_v**2
            return {"kind": "release", "t0": float(t_v), "u0": float(u_v), "v0": 0.0, "pu": pu}
    tau = t_first - f0.fit.t0
    u_f = pu[0] + pu[1] * tau + pu[2] * tau * tau
    v_f = pu[1] + 2 * pu[2] * tau            # px per video second
    v0 = float(v_f * scale["s"] * k) if scale else 0.0   # m per real second
    return {"kind": "first_point", "t0": t_first, "u0": float(u_f), "v0": v0, "pu": pu}


def _plot_payload(jf, t, xs, ys, gdir, scale, k, g_ref, d_ref):
    """Fall distance vs real time, measured against Earth's gravity from the same start."""
    if not jf.flights or scale is None:
        return None
    s = scale["s"]
    o = _origin(jf, t, xs, ys, gdir, scale, k)
    f0 = jf.flights[0]
    u = xs * gdir[0] + ys * gdir[1]
    pts = []
    for j, fr_ in enumerate(jf.flights):
        for i in fr_.idx:
            pts.append([float((t[i] - o["t0"]) / k), float((u[i] - o["u0"]) * s), j])
    t_end = max(p[0] for p in pts if p[2] == 0)
    # Fit of the first flight, as a polynomial in real time from the origin.
    pu = o["pu"]
    dt0 = o["t0"] - f0.fit.t0          # origin in the fit's own time variable
    c1 = (pu[1] + 2 * pu[2] * dt0) * s * k
    c2 = pu[2] * s * k * k
    c0 = (pu[0] + pu[1] * dt0 + pu[2] * dt0 * dt0 - o["u0"]) * s
    fit_coef = [float(c0), float(c1), float(c2)]
    earth_coef = [0.0, float(o["v0"]), 0.5 * g_ref]
    grid = np.linspace(0, t_end * 1.05, 60)
    poly = lambda c, x: c[0] + c[1] * x + c[2] * x * x  # noqa: E731
    return {"points": pts, "fit": [[float(g), float(poly(fit_coef, g))] for g in grid],
            "earth": [[float(g), float(poly(earth_coef, g))] for g in grid],
            "fit_coef": fit_coef, "earth_coef": earth_coef, "origin": o["kind"],
            "g_fit": float(2 * c2), "g_ref": g_ref, "k": k, "diam_m": float(d_ref * s),
            "t_first_end": float(t_end)}


def _ghost_payload(jf, times, gdir, scale, k, g_ref, W, H, T, t=None, xs=None, ys=None):
    """Where a real 9.8 m/s^2 fall from the same start would be, frame by frame."""
    if not jf.flights or scale is None or t is None:
        return None
    o = _origin(jf, t, xs, ys, gdir, scale, k)
    f0 = jf.flights[0]
    tau0 = o["t0"] - f0.fit.t0
    x0, y0 = f0.fit.pos(tau0)
    # Start from the origin point, moving along gravity with the clip's own speed.
    a_px = g_ref / scale["s"] / (k * k)          # px per video-second^2
    v_px = o["v0"] / scale["s"] / k              # px per video-second
    out = []
    for fi, tt in enumerate(times):
        if tt < o["t0"]:
            continue
        dt = tt - o["t0"]
        du = v_px * dt + 0.5 * a_px * dt * dt
        gx, gy = x0 + du * gdir[0], y0 + du * gdir[1]
        if gy > H * 1.2 or gy < -0.2 * H:
            break
        px, py = _to_frame_coords(T, fi, gx, gy)
        out.append({"t": float(tt), "x": float(px / W), "y": float(py / H)})
    return {"points": out}
