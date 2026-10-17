"""The pipeline: video in, verdict out.

    frames -> stabilize -> track -> flights -> zoom in if the ball is small
           -> fit (with rolling shutter) -> clocks (k) -> rulers (s)
           -> air drag -> scale-free checks -> verdict

g = s * a_px * k**2 (times an air-drag factor for light balls), where a_px is
the measured acceleration in pixels per video-second squared, s is meters
per pixel at the object, and k is real seconds per video second.

Two passes. The first looks at the whole picture, scaled down so the clip
fits in memory, and finds the ball. If the ball came out small there (high
frame rate clips get scaled down the most), the second pass decodes only the
stretch of time and the patch of picture where the ball flew, at up to full
resolution, and measures it again there.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from . import checks, clocks
from . import config as C
from .physics import (_noise_px, drag_factor, drag_path, fit_flights, gravity_direction,
                      launch_velocity, robust_fit_2d, rolling_shutter_effect, split_flights,
                      trim_release)
from .scale import Ruler, fuse, ruler_from_size, ruler_manual
from .stabilize import rescale_transforms, stabilize, warp
from .track import track_falling_object
from .verdict import decide
from .video import motion_window, probe, read_frames, work_scale

DOWN = np.array([0.0, 1.0])
PING_PONG_VT = 8.4


@dataclass
class Geom:
    """Where the analysed pixels sit in the picture.

    Positions in the pipeline are working pixels: the whole upright frame
    scaled by `scale` (W x H). The frames that were tracked may be a crop of
    it with its top-left corner at (ox, oy); T are their stabilization
    transforms, in the crop's own pixels.
    """
    W: int
    H: int
    scale: float
    ox: float = 0.0
    oy: float = 0.0
    T: list | None = None

    def to_frame(self, fi, x, y):
        """Stabilized working px -> where that point shows in frame fi."""
        if self.T is None:
            return x, y
        lx, ly = x - self.ox, y - self.oy
        A = cv2.invertAffineTransform(np.asarray(self.T[fi], np.float64))
        return (A[0, 0] * lx + A[0, 1] * ly + A[0, 2] + self.ox,
                A[1, 0] * lx + A[1, 1] * ly + A[1, 2] + self.oy)

    def from_frame(self, fi, x, y):
        """A point in frame fi (working px) -> stabilized working px."""
        if self.T is None:
            return x, y
        lx, ly = x - self.ox, y - self.oy
        A = np.asarray(self.T[fi], np.float64)
        return (A[0, 0] * lx + A[0, 1] * ly + A[0, 2] + self.ox,
                A[1, 0] * lx + A[1, 1] * ly + A[1, 2] + self.oy)


def _depth_factors(t, d, flights):
    """Scale per frame (distance now / reference distance) from the ball's apparent size.

    A ball that moves toward or away from the camera grows or shrinks in the
    picture, and its pixel motion with it. A bounce can kick it toward or
    away from the camera, so each flight gets its own distance, and a
    distance that changes steadily during the flight when its size clearly
    trends. Positions times this factor are pixels at one reference
    distance, where the ruler applies. Uses the careful widths (the quick
    blob width swells when the ball meets its own shadow near the floor), and
    only clear differences count. Returns (m, biggest change).
    """
    n = len(t)
    m = np.ones(n)
    d = np.asarray(d, float)
    inv = np.where(np.isfinite(d) & (d > 2), 1.0 / np.where(d > 0, d, 1.0), np.nan)
    models = []
    for idx in flights:
        tt, vv = t[idx], inv[idx]
        ok = np.isfinite(vv)
        if ok.sum() < 6:
            models.append(None)
            continue
        tc = float(tt[ok].mean())
        A = np.vstack([np.ones(ok.sum()), tt[ok] - tc]).T
        keep = np.ones(ok.sum(), bool)
        for _ in range(2):
            coef, *_ = np.linalg.lstsq(A[keep], vv[ok][keep], rcond=None)
            res = vv[ok] - A @ coef
            sc = 1.4826 * np.median(np.abs(res[keep])) + 1e-12
            keep = np.abs(res) < 3.5 * sc
        nk = int(keep.sum())
        se_a = sc / np.sqrt(max(nk, 1))
        se_b = sc / np.sqrt(max(np.sum((tt[ok][keep] - tc) ** 2), 1e-12))
        change = abs(coef[1]) * float(np.ptp(tt)) / coef[0]
        slope = float(coef[1]) if (change > 0.08 and abs(coef[1]) / se_b > 6) else 0.0
        models.append((float(coef[0]), slope, tc, float(se_a), len(idx)))
    good = [md for md in models if md]
    if not good:
        return m, 0.0
    # Reference distance: the flights' typical size, weighted by their length.
    levels = np.concatenate([np.full(md[4], md[0]) for md in good])
    inv_ref = float(np.median(levels))
    in_flight = np.zeros(n, bool)
    for idx, md in zip(flights, models):
        if not md:
            continue
        a, b, tc, se_a, _ = md
        lvl = a / inv_ref
        if b == 0.0 and (abs(lvl - 1) < 0.04 or abs(a - inv_ref) < 5 * se_a):
            continue                     # same distance as the reference, within the wobble
        m[idx] = (a + b * (t[idx] - tc)) / inv_ref
        in_flight[idx] = True
    if in_flight.any():
        # Frames between flights (held, bounce contact) take the nearest flight's value.
        allf = np.concatenate([np.asarray(i) for i in flights])
        allf = allf[np.argsort(t[allf])]
        out = np.setdiff1d(np.arange(n), allf)
        if len(out):
            m[out] = np.interp(t[out], t[allf], m[allf])
    return m, float(np.max(np.abs(m - 1.0)))


def _arrays(tr, times, geo, notes=None):
    """Track as arrays in (global) working px."""
    main = tr.main
    f = main.f
    t = np.asarray(times, float)[f]
    x = main.x + geo.ox
    y = main.y + geo.oy
    d_quick = main.d
    d = tr.d_fine if tr.d_fine is not None else d_quick
    d = np.where(np.isfinite(d), d, np.nan)
    area = main.area
    w0 = np.where(area > 1.8 * np.median(area), 0.3, 1.0)
    d_ref = float(np.nanmedian(d)) if np.isfinite(d).any() else float(np.median(d_quick))
    # Safety net: the careful width and the blob's outline should roughly
    # agree. If the careful one is far smaller on a big ball, it measured a
    # piece of it (a panel), so use the outline.
    dq = float(np.nanmedian(d_quick)) if np.isfinite(d_quick).any() else float("nan")
    if np.isfinite(dq) and dq >= 16 and d_ref < 0.7 * dq:
        if notes is not None:
            notes.append(f"The ball's width came out {d_ref:.0f} px one way and {dq:.0f} px the "
                         "other; used the outline width.")
        d = np.where(np.isfinite(d_quick), d_quick, np.nan)
        d_ref = dq
    return f, t, x, y, d, w0, d_ref


def _seed_points(seed, info, times, geo):
    """A click on the ball (t, x, y) -> [(t, x, y)] in stabilized working px.

    x, y are fractions of the picture (0 to 1), or pixels of the original
    video when larger than 1.
    """
    if seed is None:
        return None
    ts, sx, sy = (float(v) for v in seed)
    if sx > 1.0 or sy > 1.0:
        sx, sy = sx / max(info.width, 1), sy / max(info.height, 1)
    if not (0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0) or len(times) == 0:
        return None
    fi = int(np.argmin(np.abs(np.asarray(times) - ts)))
    if abs(times[fi] - ts) > 0.5:
        return None
    xs, ys = geo.from_frame(fi, sx * geo.W, sy * geo.H)
    return [(float(times[fi]), float(xs), float(ys))]


def _track(tracker, frames, times, roi_px, seed_pts, sam2_box):
    if tracker == "sam2":
        from .sam2_tracker import track_with_sam2
        seed1 = None
        if seed_pts:
            ts, xs, ys = seed_pts[0]
            seed1 = (int(np.argmin(np.abs(np.asarray(times) - ts))), xs, ys)
        tr = track_with_sam2(frames, times, box=sam2_box, seed=seed1)
        tr.seeded = seed1 is not None or sam2_box is not None
        return tr
    return track_falling_object(frames, times, roi=roi_px, seed=seed_pts)


# ---------------------------------------------------------------------------
def analyze(path: str, *, debug_video: str | None = None, debug_speed: float = 1.0,
            progress: Callable[[str], None] | None = None, **kw) -> dict:
    """Measure one clip.

    Keyword options: preset, size_m, size_rel_unc, label, roi, t_start, t_end,
    manual_ruler, use_gemini, use_depth, tracker, sam2_box, stabilize_mode,
    rs_mode, g_ref, scale_correction, and:

    seed: (t, x, y), a point on the ball at video time t; x, y as fractions of
    the picture (or original pixels). Use it when Galileo follows the wrong
    thing. zoom: allow the full-resolution second pass. drag_vt: the ball's
    terminal speed in m/s for the air-drag correction (presets carry one).
    debug_video: also write an MP4 showing what Galileo saw, frame by frame
    (debug_speed < 1 slows it down).
    """
    dbg: dict | None = {} if debug_video else None
    result = _run(path, dbg=dbg, progress=progress, **kw)
    if debug_video:
        say = progress or (lambda s_: None)
        say("drawing the debug video")
        try:
            from .debugvideo import render_debug_video
            render_debug_video(path, dbg, result, debug_video, speed=debug_speed)
            result["debug_video"] = debug_video
        except Exception as e:  # never lose the measurement over the picture
            result.setdefault("notes", []).append(f"Debug video failed: {e}")
    return result


def _run(path: str, *, dbg: dict | None = None, preset: str | None = None,
         size_m: float | None = None, size_rel_unc: float = 0.01, label: str | None = None,
         roi=None, t_start: float | None = None, t_end: float | None = None,
         manual_ruler=None, use_gemini: bool = False, use_depth: bool = False,
         tracker: str = "classical", sam2_box=None, stabilize_mode: str = "auto",
         rs_mode: str = "auto", g_ref: float = C.G_EARTH, scale_correction: float = 1.0,
         seed=None, zoom: bool = True, drag_vt: float | None = None,
         progress: Callable[[str], None] | None = None) -> dict:
    t_wall = time.time()
    say = progress or (lambda s: None)
    info = probe(path)
    result: dict = {"file": info.to_dict(), "notes": [], "version": 2}

    # ---- first pass: the whole picture ------------------------------------------
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
        return _unsure(result, "The clip is too short to measure.", t_wall,
                       stage="decode: too short")
    Hw, Ww = frames[0].shape[:2]
    dt_med = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / (info.fps or 30)
    fps_file = 1.0 / dt_med if dt_med > 0 else (info.fps or 30.0)
    result["work"] = {"scale": sc, "width": Ww, "height": Hw,
                      "frames": len(frames), "fps_measured": fps_file}

    say("stabilizing")
    frames_s, T, stab = stabilize(frames, stabilize_mode)
    del frames
    result["stabilization"] = stab
    geo = Geom(W=Ww, H=Hw, scale=sc, T=T)
    roi_px = None
    if roi is not None:
        roi_px = (int(roi[0] * Ww), int(roi[1] * Hw), int(roi[2] * Ww), int(roi[3] * Hw))
    seed_pts = _seed_points(seed, info, times, geo)
    if seed is not None and not seed_pts:
        result["notes"].append("The ball pick is outside the part of the clip that was "
                               "measured, so it was ignored.")
    say("tracking")
    tr = _track(tracker, frames_s, times, roi_px, seed_pts, sam2_box)
    if dbg is not None:
        dbg.update(info=info, seed=seed_pts,
                   pass1={"geo": geo, "times": times, "nums": nums, "tr": tr})
    if tr.main is None:
        return _unsure(result, "No falling object found. Pick the ball (click it in the web "
                               "app, or --seed on the command line), or try a region (ROI).",
                       t_wall, stage="tracking: nothing moving")
    if seed_pts and not tr.seeded:
        result["notes"].append("Nothing moving passed through the ball pick, so Galileo "
                               "chose the object itself.")
    f, t, x, y, d, w0, d_ref = _arrays(tr, times, geo, result["notes"])
    fl = split_flights(t, x, y, DOWN, diam_px=d_ref)
    if not fl:
        return _unsure(result, "The object moves, but no clean ballistic flight was found.",
                       t_wall, stage="tracking: no flight",
                       track=_track_payload(tr.main, times, geo))

    # What only the whole picture can give: a full view for Gemini and Depth
    # Pro, the light level for the flicker clock, and other falling objects.
    used0 = np.concatenate(fl)
    k_mid = int(used0[len(used0) // 2])
    view = {"img": frames_s[int(f[k_mid])].copy(), "x": float(x[k_mid]), "y": float(y[k_mid]),
            "d": d_ref, "scale": sc}
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in frames_s]
    bright = [float(np.median(g[::4, ::4])) for g in gray]
    rs_axis = "x" if info.rotation in (90, 270) else "y"
    others_acc = []
    for o in tr.others:
        to = times[o.f]
        flo = split_flights(to, o.x, o.y, DOWN, diam_px=float(np.nanmedian(o.d)))
        if flo:
            jo = fit_flights(to, o.x, o.y, np.ones(len(to)), flo, fps=fps_file,
                             n_px_axis=Ww if rs_axis == "x" else Hw, rs_mode="off", gdir=DOWN)
            others_acc.append((jo.a_px, jo.a_sigma))
    del frames_s

    # ---- second pass: zoom in on the ball ---------------------------------------
    if zoom:
        plan = _zoom_plan(info, geo, times, t, x, y, f, fl, d_ref)
        if plan:
            say("zooming in on the ball")
            z = _zoom_pass(path, info, plan, geo, times, tr, tracker)
            if z is None:
                result["notes"].append(f"Tried to zoom in on the {d_ref:.0f} px ball but lost "
                                       "it, so it was measured in the first pass.")
            else:
                gray_z, times_z, geo_z, tr_z, nums_z = z
                f2, t2, x2, y2, d2, w02, d_ref2 = _arrays(tr_z, times_z, geo_z, result["notes"])
                fl2 = split_flights(t2, x2, y2, DOWN, diam_px=d_ref2)
                if fl2:
                    ratio = geo_z.W / geo.W
                    result["zoom"] = {
                        "scale": geo_z.scale, "frames": len(times_z),
                        "crop_px": [int(round(v / geo_z.scale)) for v in plan["crop"]],
                        "t0": float(times_z[0]), "t1": float(times_z[-1]),
                        "ball_px_before": d_ref, "ball_px": d_ref2}
                    others_acc = [(a * ratio, s_ * ratio) for a, s_ in others_acc]
                    if dbg is not None:
                        dbg["pass2"] = {"geo": geo_z, "times": times_z, "nums": nums_z,
                                        "tr": tr_z, "crop": plan["crop"]}
                    gray, times, geo, tr = gray_z, times_z, geo_z, tr_z
                    f, t, x, y, d, w0, d_ref, fl = f2, t2, x2, y2, d2, w02, d_ref2, fl2
                else:
                    result["notes"].append("The zoomed-in pass found no clean flight, so the "
                                           "first pass was used.")

    # ---- fit ----------------------------------------------------------------------
    say("fitting")
    gdir = DOWN.copy()
    m_fac, depth_change = _depth_factors(t, d, fl)
    cxp, cyp = geo.W / 2.0, geo.H / 2.0
    xs = cxp + (x - cxp) * m_fac
    ys = cyp + (y - cyp) * m_fac
    n_axis = geo.W if rs_axis == "x" else geo.H
    fit_kw = dict(fps=fps_file, n_px_axis=n_axis, rs_axis=rs_axis, rs_mode=rs_mode, gdir=gdir)
    jf = fit_flights(t, xs, ys, w0, fl, **fit_kw)
    gdir, tilt_note = _aim_gravity(jf)
    if tilt_note:
        result["notes"].append(tilt_note)
    if gdir is not DOWN and not np.allclose(gdir, DOWN):
        fl = split_flights(t, xs, ys, gdir, diam_px=d_ref) or fl
        fit_kw["gdir"] = gdir
        jf = fit_flights(t, xs, ys, w0, fl, **fit_kw)
    # The first flight starts in the hand: cut the push, and if the ball was
    # thrown, let the bounces measure gravity instead.
    fl, launch = _launch(t, xs, ys, fl, jf, gdir)
    fit_kw["exclude"] = launch["exclude"]
    jf = fit_flights(t, xs, ys, w0, fl, **fit_kw)
    rs_rel = rolling_shutter_effect(jf, t, xs, ys, w0)
    naive = {"a_px": jf.a_px, "a_sigma": jf.a_sigma}

    # ---- clocks -------------------------------------------------------------------
    say("reading clocks")
    ev = clocks.metadata_clock({"fps": fps_file, "capture_fps": info.capture_fps})
    mf = jf.flights[jf.main]
    mi = mf.idx
    vx, vy = mf.fit.vel(t[mi] - mf.fit.t0)
    # Expected motion per frame, only where the fit is valid (inside the flight).
    disp = {int(fi): float(np.hypot(a, b) * dt_med) for fi, a, b in zip(f[mi], vx, vy)}
    ev.append(clocks.duplicate_clock(gray, f, x - geo.ox, y - geo.oy, d_ref, disp))
    u_all = xs * gdir[0] + ys * gdir[1]
    ev.append(clocks.interpolation_clock(f[mi], u_all[mi]))
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
                fl2 = [ii[s_] for s_ in fl2]
                fit_kw["fps"] = fps_file / clock["k"]
                jf2 = fit_flights(t, xs, ys, w0, fl2, **{**fit_kw, "exclude": ()})
                fl2, launch = _launch(t, xs, ys, fl2, jf2, gdir)
                fit_kw["exclude"] = launch["exclude"]
                jf = fit_flights(t, xs, ys, w0, fl2, **fit_kw)
                fl = fl2
                refit_note = (f"Refit on the {len(ii)} original frames "
                              f"({clock['source']}).")

    # ---- rulers ---------------------------------------------------------------------
    say("measuring scale")
    rulers = []
    obj_label = label
    used = np.concatenate([fl_i for fl_i in fl])
    d_norm = d * m_fac                 # widths at the reference distance
    d_samples = [v for v in d_norm[used] if np.isfinite(v)]
    if len(d_samples) < 5:
        d_samples = [v for v in d_norm if np.isfinite(v)]
    if tr.rest_d:
        d_samples += [tr.rest_d] * 3
    vt = drag_vt
    if preset and preset in C.OBJECT_PRESETS:
        p = C.OBJECT_PRESETS[preset]
        size_m, size_rel_unc, obj_label = p["size_m"], p["rel_unc"], p["label"]
        vt = vt or p.get("vt")
    gem = None
    if size_m is None and use_gemini:
        try:
            from . import gemini
            gem = gemini.identify_object(view["img"], (view["x"], view["y"], view["d"]))
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
        g_s = geo.scale
        rulers.append(ruler_manual((p1[0] * g_s, p1[1] * g_s), (p2[0] * g_s, p2[1] * g_s), L))
    if use_depth:
        try:
            from .depth import depth_pro_ruler
            dr = depth_pro_ruler(view["img"], (view["x"], view["y"]), view["d"])
            if dr:
                # Measured on the first pass; re-express per analysed pixel.
                rulers.append(Ruler(name=dr.name, s=dr.s * view["scale"] / geo.scale,
                                    rel_sigma=dr.rel_sigma, detail=dr.detail))
        except Exception as e:
            result["notes"].append(f"Depth Pro unavailable: {e}")
    scale = fuse(rulers)
    if scale and scale_correction != 1.0:
        scale["s"] *= scale_correction
        scale["correction"] = scale_correction
    if scale and not scale["agree"]:
        result["notes"].append(f"Rulers disagree by {scale['spread'] * 100:.0f}%; error bars "
                               "widened.")

    # ---- air drag ----------------------------------------------------------------------
    # For a light ball, work out how far drag pushed it off a vacuum path at
    # each frame (from its measured speed), take that out of the positions, and
    # fit again. The parabola and the rolling-shutter term then see pure
    # gravity, instead of bending to absorb drag.
    k_clock = 1.0
    if clock and clock.get("confidence", 0) >= C.CLOCK_MIN_CONFIDENCE:
        k_clock = float(clock["k"])
    jf_obs = jf                                  # fit to the positions as filmed
    a_raw, s_raw = jf.a_px, jf.a_sigma
    xs_f, ys_f = xs, ys
    D = np.zeros((len(t), 2))
    drag = None
    if vt and scale and a_raw > 0:
        v_scale = scale["s"] * k_clock          # px per video-s -> m per real s
        jf_free, vd = jf_obs, [None] * len(jf_obs.flights)
        for _ in range(2):
            D = np.zeros((len(t), 2))
            new_vd = []
            for q, fr_ in enumerate(jf_free.flights):
                idx = fr_.idx
                grid, (dx_, dy_), vdq = drag_path(fr_.fit, t[idx[0]], t[idx[-1]], v_scale, vt,
                                                  g_ref, k_clock, v_extra=vd[q])
                D[idx, 0] = np.interp(t[idx], grid, dx_)
                D[idx, 1] = np.interp(t[idx], grid, dy_)
                new_vd.append(vdq)
            vd = new_vd
            xs_f, ys_f = xs - D[:, 0], ys - D[:, 1]
            jf_free = fit_flights(t, xs_f, ys_f, w0, fl, **fit_kw)
        factor = jf_free.a_px / a_raw
        if 0.8 < factor < 2.0:
            jf = jf_free
            rs_rel = rolling_shutter_effect(jf, t, xs_f, ys_f, w0)
            drag = {"factor": float(factor), "vt": float(vt), "v_scale": float(v_scale),
                    "flights": [float(ff.a_along / fo.a_along) if fo.a_along > 0 else 1.0
                                for fo, ff in zip(jf_obs.flights, jf.flights)],
                    "max_shift_px": float(np.abs(D).max())}
        else:
            xs_f, ys_f = xs, ys
            D = np.zeros((len(t), 2))
    elif scale and size_m and 0.036 <= size_m <= 0.044:
        mfl = jf.flights[jf.main]
        fac, _ = drag_factor(mfl.fit, t[mfl.idx] - mfl.fit.t0, gdir, scale["s"] * k_clock,
                             PING_PONG_VT)
        if fac > 1.03:
            result["notes"].append(
                f"If this is a ping-pong ball, choose the ping-pong preset (--object ping-pong): "
                f"air drag slows this fall by about {100 * (1 - 1 / fac):.0f}%, and Galileo "
                "corrects for it when it knows the ball.")

    # ---- scale-free checks --------------------------------------------------------------
    say("testing physics")
    # Mention drag only where it matters (light balls), not for a steel ball's 0.05%.
    drag_on = bool(drag) and max(abs(v_ - 1) for v_ in drag["flights"]) > 0.005
    tests = [checks.parabola_check(t, xs_f, ys_f, w0, jf, drag_corrected=drag_on),
             checks.bounce_check(t, jf, dt_med)]
    tests.append(checks.equal_falls_check((a_raw, s_raw), others_acc))
    rest_xy = None
    if tr.rest is not None:
        rest_xy = (tr.rest.x + geo.ox, tr.rest.y + geo.oy)
    lead = np.arange(0, int(jf.flights[0].idx[0]))
    if len(lead) >= 3:
        lx, ly = xs[lead][-6:], ys[lead][-6:]
        if np.ptp(lx) < 0.3 * d_ref and np.ptp(ly) < 0.3 * d_ref:
            rest_xy = (float(np.median(lx)), float(np.median(ly)))
    tests.append(checks.release_check(jf, rest_xy, d_ref, thrown=launch["thrown"]))
    if launch["thrown"]:
        v0 = abs(launch["v0_px"]) * scale["s"] * k_clock if scale else None
        n_after = len(jf.flights) - 1
        if v0 is not None and v0 >= 2.5:
            first = (f"Thrown, not dropped: the ball was already moving at {v0:.1f} m/s when "
                     "first seen")
        else:
            first = "The ball was already falling fast when it was first tracked"
        if launch["exclude"]:
            result["notes"].append(f"{first}, so that first flight is left out and the "
                                   f"{n_after} flight(s) after the first bounce measure gravity.")
        else:
            result["notes"].append(f"{first}. That's fine for measuring g; it just doesn't "
                                   "start from rest.")
    if launch["trim_s"] > 0:
        result["notes"].append(f"Left out the first {1000 * launch['trim_s']:.0f} ms of the first "
                               "flight, where the hand was still pushing the ball.")

    mf = jf.flights[jf.main]
    # Tracking quality is judged on jitter around a smooth path (third
    # differences), not on the parabola fit, so a smooth but non-physical
    # path still gets a physics verdict.
    ui = (xs * gdir[0] + ys * gdir[1])[mf.idx]
    wi = (-xs * gdir[1] + ys * gdir[0])[mf.idx]
    jitter = float(np.hypot(_noise_px(ui), _noise_px(wi)))
    tracking_ok = jitter < 0.25 * d_ref + 0.5
    if not tracking_ok:
        result["notes"].append(f"Tracking is jittery ({jitter:.1f} px on a {d_ref:.0f} px "
                               "object).")
    if d_ref < 8:
        where = " even after zooming in" if result.get("zoom") else ""
        result["notes"].append(f"The ball is only {d_ref:.0f} px across{where}; move the camera "
                               "closer or use a bigger ball for a tighter answer.")

    # Does the tracked thing fall at all? A dropped ball ends its flight moving
    # down, having come down a few of its own widths. A hand pulling away doesn't.
    pu_m = mf.fit.cx * gdir[0] + mf.fit.cy * gdir[1]
    tau_m = t[mf.idx] - mf.fit.t0
    u_m = pu_m[0] + pu_m[1] * tau_m + pu_m[2] * tau_m**2
    v_end = float(pu_m[1] + 2 * pu_m[2] * tau_m[-1])
    travel = float(u_m[-1] - np.min(u_m))
    falls = travel >= 2.0 * d_ref and v_end > 0

    v = decide(a_px=a_raw, a_sigma=jf.a_sigma / drag["factor"] if drag else s_raw, scale=scale,
               clock=clock, tests=tests, n_points=len(mf.idx), duration=mf.duration,
               g_ref=g_ref, tracking_ok=tracking_ok, rs_rel=rs_rel, drag=drag, falls=falls)

    # ---- payload ----------------------------------------------------------------------
    k_used = v.get("k", 1.0)
    vt_plot = vt if drag else None
    if dbg is not None:
        dbg["final"] = {"geo": geo, "times": times, "tr": tr, "f": f, "t": t, "x": x, "y": y,
                        "d": d, "xs": xs, "ys": ys, "m_fac": m_fac, "jf": jf_obs,
                        "jf_free": jf, "drag_shift": D, "gdir": gdir,
                        "scale": scale, "k": k_used, "drag": drag, "g_ref": g_ref,
                        "d_ref": d_ref, "flights": [fr_.idx for fr_ in jf.flights],
                        "excluded": tuple(jf.excluded), "launch": launch,
                        "ghost": _ghost_path(jf_obs, times, gdir, scale, k_used, g_ref, geo.H,
                                             t, xs, ys, vt_plot)}
    result.update({
        "verdict": v,
        "tests": tests,
        "clock": _clean(clock),
        "scale": scale,
        "drag": drag,
        "gemini_object": gem,
        "fit": {
            "a_px": a_raw, "a_sigma": s_raw, "a_px_naive": naive["a_px"],
            "a_px_drag_corrected": jf.a_px if drag else None,
            "gdir": [float(gdir[0]), float(gdir[1])],
            "rolling_shutter": {"alpha": jf.alpha, "alpha_max": jf.alpha_max,
                                "axis": jf.rs_axis, "effect_rel": rs_rel,
                                "F": jf.rs_significance},
            "depth_change_rel": depth_change,
            "flights": [{"n": int(len(fr_.idx)), "t0": float(t[fr_.idx[0]]),
                         "t1": float(t[fr_.idx[-1]]), "a_px": fr_.a_along,
                         "a_sigma": fr_.a_sigma, "rms_px": fr_.fit.rms,
                         "a_px_observed": fo.a_along}
                        for fr_, fo in zip(jf.flights, jf_obs.flights)],
            "main": jf.main, "diam_px": d_ref, "refit": refit_note,
            "jitter_px": jitter, "fall_diameters": travel / d_ref if d_ref else None,
            "launch": {k_: v_ for k_, v_ in launch.items() if k_ != "exclude"},
            "excluded_flights": [int(j) for j in jf.excluded],
        },
        "track": _track_payload(tr.main, times, geo),
        "plot": _plot_payload(jf_obs, t, xs, ys, gdir, scale, k_used, g_ref, d_ref, vt_plot,
                              drag["flights"][0] if drag else 1.0),
        "ghost": _ghost_payload(jf_obs, times, gdir, scale, k_used, g_ref, geo, t, xs, ys,
                                vt_plot),
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
def _launch(t, xs, ys, fl, jf, gdir):
    """The first flight begins in the hand. Cut the push; spot a throw.

    Flights that start at a bounce are clean free flights: nothing but the
    floor touched the ball. The first flight (and any that starts after the
    ball was lost) may begin with the hand still pushing, so those frames are
    cut until the rest agrees with the bounces' acceleration. A throw adds
    speed, not acceleration, so a thrown ball is in free fall once it leaves
    the hand; but a throw is fast, short and often near the edge of the
    picture, so when bounces exist they measure gravity and the throw's own
    acceleration isn't held against the clip.
    """
    info = {"thrown": False, "trim_s": 0.0, "v0_px": 0.0, "since_vertex_s": 0.0,
            "exclude": ()}
    if not fl:
        return fl, info
    hop = [False] + [float(t[b[0]] - t[a[-1]]) <= 0.12 for a, b in zip(fl, fl[1:])]
    hops = [j for j in range(len(fl)) if hop[j] and len(fl[j]) >= 8
            and jf.flights[j].a_along > 0 and np.isfinite(jf.flights[j].a_sigma)
            and jf.flights[j].a_sigma > 0]
    a_ref = s_ref = None
    if hops:
        w = np.array([1.0 / jf.flights[j].a_sigma ** 2 for j in hops])
        a_ref = float(np.sum(w * np.array([jf.flights[j].a_along for j in hops])) / w.sum())
        s_ref = float(1.0 / np.sqrt(w.sum()))
    fl = list(fl)
    for j in range(len(fl)):
        if hop[j]:
            continue
        idx2, k = trim_release(t, xs, ys, fl[j], gdir, a_ref, s_ref)
        if k:
            if j == 0:
                info["trim_s"] = float(t[idx2[0]] - t[fl[j][0]])
            fl[j] = idx2
    f0 = fl[0]
    if len(f0) >= 5:
        fit0 = robust_fit_2d(t[f0], xs[f0], ys[f0])
        v0, since = launch_velocity(fit0, float(t[f0[0]]), gdir)
        dt = float(np.median(np.diff(t[f0]))) if len(f0) > 1 else 0.0
        info["v0_px"], info["since_vertex_s"] = float(v0), float(min(since, 99.0))
        # Already moving down fast when first seen: a drop that tracking picks
        # up a moment late has fallen for ~0.1 s; a throw looks like a quarter
        # second or more of falling at the moment it's first seen.
        info["thrown"] = bool(v0 > 0 and since > max(0.25, 6 * dt))
    if info["thrown"] and hops:
        info["exclude"] = (0,)
    return fl, info


def _aim_gravity(jf):
    """Which way is down? The picture's own down, unless the camera is clearly tilted.

    Never flips: if the measured acceleration points up or sideways, the
    tracked thing isn't falling, and measuring along it would turn a hand or
    a rising ball into a fake g.
    """
    gd = gravity_direction(jf)
    a_mag, a_sig = jf.flights[jf.main].fit.accel_mag()
    cosang = float(np.clip(gd @ DOWN, -1, 1))
    ang = float(np.degrees(np.arccos(cosang)))
    if cosang <= 0 or not (a_mag > 5 * a_sig) or ang <= 10:
        return DOWN, None
    if ang <= C.MAX_TILT_DEG:
        return gd, (f"The camera looks tilted by about {ang:.0f}°, so gravity was measured "
                    "along the tilt.")
    return DOWN, (f"The fall's acceleration points {ang:.0f}° away from straight down in the "
                  "picture, so only its downward part counts. If the video is sideways, "
                  "rotate it upright and measure again.")


def _zoom_plan(info, geo, times, t, x, y, f, fl, d_ref):
    """Where and when to look again, at what resolution. None if it wouldn't help."""
    if d_ref >= C.ZOOM_MIN_PX or info.width <= 0:
        return None
    r_c = geo.W / info.width                 # first-pass px per original px
    d_full = d_ref / r_c
    z = min(1.0, C.ZOOM_TARGET_PX / max(d_full, 1e-6))
    if z < 1.25 * r_c:
        return None                          # the first pass was already near full size
    times = np.asarray(times, float)
    lead = 0.3

    def window(i_a, i_b):
        t0 = max(float(t[i_a]) - lead, float(times[0]))
        t1 = min(float(t[i_b]) + 0.1, float(times[-1]))
        sel = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
        fx, fy = [], []
        for fi, xi, yi in zip(f[sel], x[sel], y[sel]):
            a, b = geo.to_frame(int(fi), xi, yi)
            fx += [a, xi]                     # where it shows, and where it is stabilized
            fy += [b, yi]
        m = 2.5 * d_ref + 6.0
        box = (max(min(fx) - m, 0.0) / r_c, max(min(fy) - m, 0.0) / r_c,
               min(max(fx) + m, geo.W) / r_c, min(max(fy) + m, geo.H) / r_c)
        n = int(np.sum((times >= t0 - 1e-9) & (times <= t1 + 1e-9)))
        return t0, t1, box, n

    def cost(zz, box, n):
        return (box[2] - box[0]) * zz * (box[3] - box[1]) * zz * 3 * n

    longest = max(fl, key=len)
    for i_a, i_b in ((fl[0][0], fl[-1][-1]), (longest[0], longest[-1])):
        t0, t1, box, n = window(i_a, i_b)
        n = min(n, C.MAX_FRAMES)
        zz = z
        c = cost(zz, box, n)
        if c > C.ZOOM_BUDGET_BYTES:
            zz = z * float(np.sqrt(C.ZOOM_BUDGET_BYTES / c))
        if zz >= 1.25 * r_c:
            Wz, Hz = max(2, round(info.width * zz)), max(2, round(info.height * zz))
            crop = (int(np.floor(box[0] * zz)), int(np.floor(box[1] * zz)),
                    min(int(np.ceil(box[2] * zz)), Wz), min(int(np.ceil(box[3] * zz)), Hz))
            if crop[2] - crop[0] >= 16 and crop[3] - crop[1] >= 16:
                return {"z": zz, "t0": t0, "t1": t1, "crop": crop, "W": Wz, "H": Hz}
    return None


def _zoom_pass(path, info, plan, geo, times, tr, tracker):
    """Decode the ball's patch at high resolution and track it there again.

    Returns (gray frames, times, geometry, track result, frame numbers) or None.
    """
    z = plan["z"]
    x0, y0, x1, y1 = plan["crop"]
    frames, tz, nz = read_frames(path, info, scale=z, t_start=plan["t0"], t_end=plan["t1"],
                                 max_frames=C.MAX_FRAMES, crop=plan["crop"])
    if len(frames) < 8:
        return None
    ratio = plan["W"] / geo.W
    times = np.asarray(times, float)
    Tz = None
    if geo.T is not None:
        # Reuse the first pass's camera motion, re-expressed for this crop.
        idx = np.clip(np.searchsorted(times, tz), 1, len(times) - 1)
        idx = np.where(np.abs(times[idx - 1] - tz) < np.abs(times[idx] - tz), idx - 1, idx)
        Tz = rescale_transforms([geo.T[i] for i in idx], ratio, (x0, y0))
        frames = warp(frames, Tz)
    # Follow the same object as the first pass.
    seeds = [(float(times[dt.f]), (dt.x + geo.ox) * ratio - x0, (dt.y + geo.oy) * ratio - y0)
             for dt in tr.main.dets if plan["t0"] <= times[dt.f] <= plan["t1"]]
    if tracker == "sam2":
        from .sam2_tracker import track_with_sam2
        ts, sx, sy = seeds[len(seeds) // 4]
        fi = int(np.argmin(np.abs(tz - ts)))
        trz = track_with_sam2(frames, tz, seed=(fi, sx, sy))
        trz.seeded = True
    else:
        trz = track_falling_object(frames, tz, seed=seeds)
    if trz.main is None or not trz.seeded:
        return None
    d_z = float(np.nanmedian(trz.main.d))
    d_c = float(np.nanmedian(tr.main.d)) * ratio
    if not (0.6 < d_z / max(d_c, 1e-6) < 1.7):
        return None
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in frames]
    del frames
    geo_z = Geom(W=plan["W"], H=plan["H"], scale=z, ox=float(x0), oy=float(y0), T=Tz)
    return gray, tz, geo_z, trz, nz


def _unsure(result, why, t_wall, stage=None, **extra):
    result.update({"verdict": {"verdict": "UNSURE", "confidence": 0.0, "headline": why,
                               "reasons": [why], "g": None, "sigma": None, "stage": stage},
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


def _track_payload(main, times, geo):
    pts = []
    for dt in main.dets:
        px, py = geo.to_frame(dt.f, dt.x + geo.ox, dt.y + geo.oy)
        pts.append({"t": float(times[dt.f]), "x": float(px / geo.W), "y": float(py / geo.H),
                    "r": float(0.5 * dt.d / geo.W)})
    return {"points": pts}


def _earth_fall(tau, v0, g, vt=None):
    """Distance (m) fallen after real time tau from speed v0 (m/s, + down).

    With vt (terminal speed), quadratic air drag is included, so a light
    ball is compared with what a real light ball does.
    """
    tau = np.asarray(tau, float)
    if not vt:
        return v0 * tau + 0.5 * g * tau**2
    t_max = float(np.max(tau)) if tau.size else 0.0
    if t_max <= 0:
        return np.zeros_like(tau)
    n = int(np.ceil(t_max / 1e-3)) + 2
    h = t_max / (n - 1)
    kd = g / vt**2
    v, dist = float(v0), 0.0
    ds = np.empty(n)
    ds[0] = 0.0
    for i in range(1, n):
        vm = v + 0.5 * h * (g - kd * abs(v) * v)
        dist += h * vm
        v += h * (g - kd * abs(vm) * vm)
        ds[i] = dist
    return np.interp(tau, np.linspace(0.0, t_max, n), ds)


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


def _plot_payload(jf, t, xs, ys, gdir, scale, k, g_ref, d_ref, vt=None, drag_first=1.0):
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
    grid = np.linspace(0, t_end * 1.05, 60)
    earth = _earth_fall(grid, o["v0"], g_ref, vt)
    # Quadratic stand-in for the Earth curve (exact without drag).
    A = np.vstack([np.ones_like(grid), grid, grid**2]).T
    earth_coef = ([0.0, float(o["v0"]), 0.5 * g_ref] if not vt else
                  [float(c_) for c_ in np.linalg.lstsq(A, earth, rcond=None)[0]])
    poly = lambda c, x: c[0] + c[1] * x + c[2] * x * x  # noqa: E731
    return {"points": pts, "fit": [[float(g), float(poly(fit_coef, g))] for g in grid],
            "earth": [[float(g), float(e)] for g, e in zip(grid, earth)],
            "fit_coef": fit_coef, "earth_coef": earth_coef, "origin": o["kind"],
            "g_fit": float(2 * c2), "g_ref": g_ref, "k": k, "diam_m": float(d_ref * s),
            "t_first_end": float(t_end), "drag_vt": float(vt) if vt else None,
            "g_fit_drag_corrected": float(2 * c2 * drag_first) if vt else None}


def _ghost_path(jf, times, gdir, scale, k, g_ref, H, t=None, xs=None, ys=None, vt=None):
    """Where a real ball falling from the same start would be: (times, x, y) in
    stabilized working px, one entry per frame from the start of the fall."""
    if not jf.flights or scale is None or t is None:
        return None
    o = _origin(jf, t, xs, ys, gdir, scale, k)
    f0 = jf.flights[0]
    tau0 = o["t0"] - f0.fit.t0
    x0, y0 = f0.fit.pos(tau0)
    times = np.asarray(times, float)
    sel = np.flatnonzero(times >= o["t0"])
    if len(sel) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    # Start from the origin point, moving along gravity with the clip's own speed.
    du = _earth_fall((times[sel] - o["t0"]) / k, o["v0"], g_ref, vt) / scale["s"]
    gx = x0 + du * gdir[0]
    gy = y0 + du * gdir[1]
    inside = (gy <= H * 1.2) & (gy >= -0.2 * H)
    n = int(np.argmin(inside)) if not inside.all() else len(inside)
    return times[sel][:n], gx[:n], gy[:n]


def _ghost_payload(jf, times, gdir, scale, k, g_ref, geo, t=None, xs=None, ys=None, vt=None):
    """The same, as fractions of the picture for the web overlay."""
    gp = _ghost_path(jf, times, gdir, scale, k, g_ref, geo.H, t, xs, ys, vt)
    if gp is None:
        return None
    times = np.asarray(times, float)
    out = []
    for tt, gx, gy in zip(*gp):
        fi = int(np.argmin(np.abs(times - tt)))
        px, py = geo.to_frame(fi, gx, gy)
        out.append({"t": float(tt), "x": float(px / geo.W), "y": float(py / geo.H)})
    return {"points": out}
