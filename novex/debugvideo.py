"""Debug video: what Novex saw, frame by frame.

    python -m novex analyze clip.mp4 --object ping-pong --debug-video clip_debug.mp4

Drawn on every frame of the measured stretch:

  grey rings      every candidate blob the detector found in this frame
  thin grey lines other moving things it linked into tracks, with their score
  coloured dots   the tracked ball, one colour per flight; grey dots are frames
                  left out of the fit (held, bounce contact, a glitch)
  thin curves     the parabola fitted to each flight
  gold ring       where a real ball dropped from the same start would be
  white arrow     velocity from the fit (how far it goes in 0.1 s)
  red arrow       acceleration from the fit; the gold line beside it is 9.8 m/s^2
  cyan box        the patch decoded again at full resolution (zoom pass)
  white cross     your --seed pick
  other colours   every other falling object Novex measured, with its number,
                  name and g; non-round objects also show their outline and a
                  cross on the centre of mass Novex followed
  readout         time, state, position, speed, fitted and local acceleration
  charts below    height and acceleration against time, with a cursor at this frame
                  (for object 1, the primary)

Text is plain ASCII (m/s^2, +/-) because OpenCV's fonts have nothing else.
"""
from __future__ import annotations

import cv2
import numpy as np

from .physics import glitch_mask
from .video import Writer, iter_frames, probe

# BGR
FLIGHT_COLORS = [(255, 150, 40), (40, 180, 255), (90, 210, 90), (220, 100, 230),
                 (60, 220, 230), (255, 230, 120), (150, 150, 255), (120, 255, 200)]
GREY = (165, 165, 165)
DIM = (125, 125, 125)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (76, 166, 226)
TRACK = (242, 168, 127)
RED = (70, 70, 255)
CYAN = (230, 210, 60)
PANEL_BG = (24, 24, 24)
OBJECT_COLORS = [(255, 120, 255), (120, 255, 120), (255, 255, 120), (120, 200, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pt(x, y):
    return int(round(float(x))), int(round(float(y)))


def _text(img, s, org, fs, color=WHITE, thick=1):
    # Outline by shifting, not by a thicker stroke: thicker Hershey text is also
    # wider, which would leave a dark copy poking out past the end of the line.
    x, y = _pt(*org)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(img, s, (x + dx, y + dy), FONT, fs, BLACK, thick, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, fs, color, thick, cv2.LINE_AA)


def _dashed(img, pts, color, thick=1, on=6, off=5):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return
    seg = np.hypot(*np.diff(pts, axis=0).T)
    pos = np.concatenate([[0], np.cumsum(seg)])
    total = pos[-1]
    s = 0.0
    while s < total:
        e = min(s + on, total)
        a = np.array([np.interp(s, pos, pts[:, 0]), np.interp(s, pos, pts[:, 1])])
        b = np.array([np.interp(e, pos, pts[:, 0]), np.interp(e, pos, pts[:, 1])])
        cv2.line(img, _pt(*a), _pt(*b), color, thick, cv2.LINE_AA)
        s += on + off


def _nice_step(span, target):
    raw = max(span, 1e-9) / max(target, 1)
    p = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * p:
            return m * p
    return 10 * p


class _Chart:
    def __init__(self, img, box, t_range, v_range, title, fs):
        self.img = img
        self.x0, self.y0, self.x1, self.y1 = box
        self.t0, self.t1 = t_range
        self.v0, self.v1 = v_range
        self.title = title
        self.fs = fs

    def X(self, t):
        return self.x0 + (np.asarray(t, float) - self.t0) / max(self.t1 - self.t0, 1e-9) * (
            self.x1 - self.x0)

    def Y(self, v):
        v = np.clip(np.asarray(v, float), self.v0, self.v1)
        return self.y1 - (v - self.v0) / max(self.v1 - self.v0, 1e-9) * (self.y1 - self.y0)

    def axes(self):
        img, fs = self.img, self.fs
        cv2.rectangle(img, (self.x0, self.y0), (self.x1, self.y1), (70, 70, 70), 1)
        step = _nice_step(self.v1 - self.v0, 4)
        v = np.ceil(self.v0 / step) * step
        while v <= self.v1 + 1e-9:
            y = int(self.Y(v))
            cv2.line(img, (self.x0, y), (self.x1, y), (50, 50, 50), 1)
            lab = f"{v:g}"
            tw = cv2.getTextSize(lab, FONT, fs * 0.8, 1)[0][0]
            _text(img, lab, (self.x0 - 8 - tw, y + 4), fs * 0.8, (180, 180, 180))
            v += step
        step = _nice_step(self.t1 - self.t0, 6)
        t = np.ceil(self.t0 / step) * step
        while t <= self.t1 + 1e-9:
            x = int(self.X(t))
            cv2.line(img, (x, self.y1), (x, self.y1 + 4), (120, 120, 120), 1)
            _text(img, f"{t:g}", (x - 8, self.y1 + 16), fs * 0.8, (180, 180, 180))
            t += step
        _text(img, self.title, (self.x0, self.y0 - 7), fs, (220, 220, 220))


def _fit_curve(fit, ta, tb, n=40):
    tt = np.linspace(ta, tb, n)
    px, py = fit.pos(tt - fit.t0)
    return tt, np.asarray(px, float), np.asarray(py, float)


def _object_overlay(fin, g1, meta):
    """Another object's track, fits and outline, in first-pass stabilized px."""
    gf = fin["geo"]
    r = gf.W / g1.W
    t = np.asarray(fin["t"], float)
    m_fac = np.asarray(fin["m_fac"], float)
    cxp, cyp = gf.W / 2.0, gf.H / 2.0
    lev = fin.get("lev")

    def raw(tt, px, py):
        m = np.interp(tt, t, m_fac)
        qx, qy = cxp + (px - cxp) / m, cyp + (py - cyp) / m
        if lev is not None and not lev.identity:
            qx, qy = lev.backward(qx, qy)
        return qx, qy
    flights = fin["flights"]
    flight_of = np.full(len(t), -1)
    for q, idx in enumerate(flights):
        flight_of[idx] = q
    curves = []
    for fr_ in fin["jf"].flights:
        idx = fr_.idx
        tt, px, py = _fit_curve(fr_.fit, t[idx[0]], t[idx[-1]])
        px, py = raw(tt, px, py)
        curves.append((tt, px / r, py / r))
    xs, ys = raw(t, np.asarray(fin["xs"], float), np.asarray(fin["ys"], float))
    dets = fin["tr"].main.dets
    hulls = []
    for dd in dets:
        if dd.hull is None:
            hulls.append(None)
        else:
            hulls.append((np.asarray(dd.hull, float) + [gf.ox, gf.oy]) / r)
    v = meta.get("verdict") or {}
    lab = f"{meta['n']}" + (f": {meta['label']}" if meta.get("label") else "")
    if v.get("g") is not None:
        lab += f"  g = {v['g']:.2f} ({v['verdict']})"
    elif v.get("verdict"):
        lab += f"  ({v['verdict']})"
    return {"t": t, "X": np.asarray(fin["x"], float) / r, "Y": np.asarray(fin["y"], float) / r,
            "CX": xs / r, "CY": ys / r, "D": np.asarray(fin["d"], float) / r,
            "flight_of": flight_of, "curves": curves, "hulls": hulls,
            "rigid": bool(fin.get("rigid")), "label": lab, "n": meta["n"]}


def _draw_object(frame, ov, t_cur, dt, to_out, kx, fs, col, primary=False):
    """Trail, fitted paths, outline and label of one object at time t_cur."""
    if not primary:
        for tt, px, py in ov["curves"]:
            ox_, oy_ = to_out(px, py)
            cv2.polylines(frame, [np.stack([ox_, oy_], 1).astype(np.int32)], False, col, 1,
                          cv2.LINE_AA)
        past = ov["t"] <= t_cur + 1e-9
        if past.any():
            ox_, oy_ = to_out(ov["CX"][past], ov["CY"][past])
            for xo, yo, q in zip(ox_, oy_, ov["flight_of"][past]):
                cv2.circle(frame, _pt(xo, yo), 2, col if q >= 0 else GREY, -1, cv2.LINE_AA)
    jj = np.flatnonzero(np.abs(ov["t"] - t_cur) <= 0.5 * dt)
    if not len(jj):
        return
    j = int(jj[0])
    h = ov["hulls"][j] if ov["rigid"] else None
    if h is not None and len(h) >= 3:
        hx, hy = to_out(h[:, 0], h[:, 1])
        cv2.polylines(frame, [np.stack([hx, hy], 1).astype(np.int32)], True, col, 1,
                      cv2.LINE_AA)
    cx, cy = to_out(ov["CX"][j], ov["CY"][j])
    c = _pt(cx, cy)
    if ov["rigid"]:
        for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(frame, c, (c[0] + 7 * dx_, c[1] + 7 * dy_), col, 2, cv2.LINE_AA)
    elif not primary:
        rad = max(5, int(0.5 * ov["D"][j] * kx) + 3) if np.isfinite(ov["D"][j]) else 8
        cv2.circle(frame, c, rad, col, 2, cv2.LINE_AA)
    rad = max(8, int(0.5 * np.nanmedian(ov["D"]) * kx) + 6) if np.isfinite(ov["D"]).any() else 12
    _text(frame, ov["label"], (c[0] + rad, c[1] - rad), fs * 0.8, col)


def render_debug_video(path, dbg, result, out_path, *, speed=1.0, max_side=1280):
    """Write the annotated MP4. dbg is what analyze() collected along the way."""
    if not dbg or "pass1" not in dbg:
        raise RuntimeError("nothing was decoded, so there is nothing to draw")
    info = dbg.get("info") or probe(path)
    p1, p2, fin = dbg["pass1"], dbg.get("pass2"), dbg.get("final")
    g1 = p1["geo"]
    times1 = np.asarray(p1["times"], float)
    nums1 = np.asarray(p1["nums"], int)
    idx1 = {int(n): i for i, n in enumerate(nums1)}
    idx2 = {int(n): i for i, n in enumerate(p2["nums"])} if p2 else {}
    tr1 = p1["tr"]
    verdict = (result or {}).get("verdict") or {}
    dt = float(np.median(np.diff(times1))) if len(times1) > 1 else 1 / 30

    # ---- the final track, in pass-1 stabilized px ------------------------------
    T = None
    if fin is not None:
        gf = fin["geo"]
        r = gf.W / g1.W                         # final px per pass-1 px
        t = np.asarray(fin["t"], float)
        x, y = np.asarray(fin["x"], float), np.asarray(fin["y"], float)
        xs, ys = np.asarray(fin["xs"], float), np.asarray(fin["ys"], float)
        d_q = np.asarray(fin["tr"].main.d, float)
        m_fac = np.asarray(fin["m_fac"], float)
        gdir = np.asarray(fin["gdir"], float)
        jf = fin["jf"]
        flights = fin["flights"]
        scale, k = fin["scale"], float(fin["k"] or 1.0)
        s_m = scale["s"] if scale else None       # m per final px
        drag = fin["drag"]
        g_ref = float(fin["g_ref"])
        flight_of = np.full(len(t), -1)
        for q, idx in enumerate(flights):
            flight_of[idx] = q
        good = glitch_mask(t, x, y, fin["d_ref"])
        cxp, cyp = gf.W / 2.0, gf.H / 2.0
        lev = fin.get("lev")

        def raw(tt, px, py):
            # Fit space (level view, scaled to one distance) -> stabilized px.
            m = np.interp(tt, t, m_fac)
            qx, qy = cxp + (px - cxp) / m, cyp + (py - cyp) / m
            if lev is not None and not lev.identity:
                qx, qy = lev.backward(qx, qy)
            return qx, qy

        u = xs * gdir[0] + ys * gdir[1]
        # Local acceleration: a parabola through about +-40 ms around each point.
        w = max(3, int(round(0.04 / max(dt, 1e-6))))
        a_loc = np.full(len(t), np.nan)
        a_loc_s = np.full(len(t), np.nan)
        for q, idx in enumerate(flights):
            for jj, j in enumerate(idx):
                sel = idx[max(0, jj - w): jj + w + 1]
                if len(sel) < 6:
                    continue
                tt = t[sel] - t[j]
                try:
                    c, cov = np.polyfit(tt, u[sel], 2, cov=True)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                a_loc[j] = 2 * c[0]
                a_loc_s[j] = 2 * np.sqrt(max(cov[0, 0], 0))
        a_unit = (s_m * k * k) if s_m else 1.0  # px/s^2 -> m/s^2
        v_unit = (s_m * k) if s_m else 1.0
        unit_a = "m/s^2" if s_m else "px/s^2"
        unit_v = "m/s" if s_m else "px/s"
        ghost = fin.get("ghost")
        if ghost is not None and len(ghost[0]):
            gtt, ggx, ggy = (np.asarray(v_, float) for v_ in ghost)
            ggx, ggy = raw(gtt, ggx, ggy)
            ghost1 = (gtt, ggx / r, ggy / r)
        else:
            ghost1 = None
        curves = []
        for q, fr_ in enumerate(jf.flights):
            idx = fr_.idx
            tt, px, py = _fit_curve(fr_.fit, t[idx[0]], t[idx[-1]])
            px, py = raw(tt, px, py)
            curves.append((tt, px / r, py / r))
        launch = fin.get("launch") or {}
        T = dict(t=t, X=x / r, Y=y / r, D=d_q / r, flight_of=flight_of, good=good, u=u,
                 a_loc=a_loc, a_loc_s=a_loc_s, curves=curves, ghost=ghost1, r=r,
                 excluded=set(fin.get("excluded") or ()),
                 push_from=(t[flights[0][0]] - launch.get("trim_s", 0.0) - 0.5 * dt)
                 if flights and launch.get("trim_s") else None)

    # ---- the other objects --------------------------------------------------------
    objs_meta = dbg.get("objects") or []
    extras, prim_ov = [], None
    for k, om in enumerate(objs_meta):
        if not om.get("final"):
            continue
        ov = _object_overlay(om["final"], g1, om)
        if k == 0:
            prim_ov = ov
        else:
            extras.append(ov)
    show_prim_label = prim_ov is not None and (extras or prim_ov["rigid"])
    obj_tracks = {id(o.main) for o in (tr1.objects or [])}

    # ---- decode and draw ---------------------------------------------------------
    out_scale = min(1.0, max_side / max(info.width, info.height, 1))
    step = int(np.median(np.diff(nums1))) if len(nums1) > 1 else 1   # a long clip: 1 in N
    frames = iter_frames(path, info, scale=out_scale, first_num=int(nums1[0]),
                         last_num=int(nums1[-1]), step=max(1, step))
    writer = None
    chart_bg = None
    n_done = 0
    for img, _t_pts, num in frames:
        fi1 = idx1.get(int(num))
        if fi1 is None:
            continue
        if writer is None:
            H, W = img.shape[:2]
            kx, ky = W / g1.W, H / g1.H
            fs = max(0.38, min(0.75, H / 1250.0))
            P = int(max(170, 0.30 * H)) // 2 * 2
            chart_bg, charts = _chart_panel(W, P, fs, times1, T, fin, verdict)
            fps_out = max(1.0, (1.0 / dt) * speed)
            writer = Writer(out_path, fps_out, (W, H + P))
        t_cur = float(times1[fi1])
        frame = img.copy()

        def to_out(X1, Y1, fi=fi1):
            fx, fy = g1.to_frame(fi, X1, Y1)
            return np.asarray(fx, float) * kx, np.asarray(fy, float) * ky

        # Zoom box and the pass that supplied this frame's candidates.
        fiz = idx2.get(int(num))
        if p2 is not None and fiz is not None:
            cx0, cy0, cx1, cy1 = p2["crop"]
            zx = W / p2["geo"].W
            zy = H / p2["geo"].H
            cv2.rectangle(frame, _pt(cx0 * zx, cy0 * zy), _pt(cx1 * zx, cy1 * zy), CYAN, 1,
                          cv2.LINE_AA)
            _text(frame, f"zoom pass: {100 * p2['geo'].scale:.0f}% resolution",
                  (cx0 * zx + 4, max(cy0 * zy - 6, 14)), fs * 0.8, CYAN)
            dets = p2["tr"].dets[fiz] if p2["tr"].dets else None
            dgeo = p2["geo"]
        else:
            dets = tr1.dets[fi1] if tr1.dets else None
            dgeo = g1
        n_cand = 0
        if dets:
            rr = dgeo.W / g1.W
            for dd in dets:
                # The tracker saw stabilized frames; bring each blob back to this frame.
                ox_, oy_ = to_out((dd.x + dgeo.ox) / rr, (dd.y + dgeo.oy) / rr)
                rad = max(3.0, 0.5 * (dd.d if np.isfinite(dd.d) else 6.0) / rr * kx + 2)
                cv2.circle(frame, _pt(ox_, oy_), int(rad), GREY, 1, cv2.LINE_AA)
                n_cand += 1

        # Other tracks from the first pass, while they are active.
        main1 = tr1.main
        for o in (tr1.tracks or [])[:12]:
            if o is main1 or o.kind != "moving" or (extras and id(o) in obj_tracks):
                continue
            tt = times1[o.f]
            if tt[0] > t_cur or tt[-1] < t_cur - 0.3:
                continue
            sel = tt <= t_cur
            ox_, oy_ = to_out(o.x[sel], o.y[sel])
            pts = np.stack([ox_, oy_], 1).astype(np.int32)
            cv2.polylines(frame, [pts], False, DIM, 1, cv2.LINE_AA)
            _text(frame, f"score {o.score:.0f}", (pts[-1][0] + 6, pts[-1][1] - 6), fs * 0.7, DIM)
        if main1 is not None and (fin is None or T["r"] != 1.0):
            tt = times1[main1.f]
            sel = tt <= t_cur
            if sel.any():
                ox_, oy_ = to_out(main1.x[sel], main1.y[sel])
                pts = np.stack([ox_, oy_], 1).astype(np.int32)
                cv2.polylines(frame, [pts], False, TRACK, 1, cv2.LINE_AA)

        hud = [f"t = {t_cur:.3f} s   frame {fi1 + 1}/{len(times1)}"]
        state = "no falling object found" if main1 is None else "ball not found in this frame"
        if T is not None:
            # Fitted paths, then the trail, then this frame's ball.
            for q, (tt, px, py) in enumerate(T["curves"]):
                ox_, oy_ = to_out(px, py)
                col = FLIGHT_COLORS[q % len(FLIGHT_COLORS)]
                active = tt[0] - dt / 2 <= t_cur <= tt[-1] + dt / 2
                pts = np.stack([ox_, oy_], 1).astype(np.int32)
                if q in T["excluded"]:
                    _dashed(frame, pts, col, 2 if active else 1)   # fitted, but not counted
                else:
                    cv2.polylines(frame, [pts], False, col, 2 if active else 1, cv2.LINE_AA)
            past = T["t"] <= t_cur + 1e-9
            if past.any():
                ox_, oy_ = to_out(T["X"][past], T["Y"][past])
                for xo, yo, q, okp in zip(ox_, oy_, T["flight_of"][past], T["good"][past]):
                    col = FLIGHT_COLORS[q % len(FLIGHT_COLORS)] if q >= 0 else GREY
                    cv2.circle(frame, _pt(xo, yo), 2 if q >= 0 else 3, col,
                               -1 if okp else 1, cv2.LINE_AA)
            if T["ghost"] is not None:
                gtt, gx1, gy1 = T["ghost"]
                jg = int(np.argmin(np.abs(gtt - t_cur))) if len(gtt) else -1
                if jg >= 0 and abs(gtt[jg] - t_cur) <= 0.6 * dt:
                    ox_, oy_ = to_out(gx1[jg], gy1[jg])
                    rad = max(6, int(0.5 * np.nanmedian(T["D"]) * kx) + 3)
                    cv2.circle(frame, _pt(ox_, oy_), rad, GOLD, 2, cv2.LINE_AA)
            jj = np.flatnonzero(np.abs(T["t"] - t_cur) <= 0.5 * dt)
            if len(jj):
                j = int(jj[0])
                q = int(T["flight_of"][j])
                ox_, oy_ = to_out(T["X"][j], T["Y"][j])
                rad = max(5, int(0.5 * T["D"][j] * kx) + 3) if np.isfinite(T["D"][j]) else 8
                col = FLIGHT_COLORS[q % len(FLIGHT_COLORS)] if q >= 0 else GREY
                cv2.circle(frame, _pt(ox_, oy_), rad, col, 2, cv2.LINE_AA)
                gf = fin["geo"]
                fxo, fyo = g1.to_frame(fi1, T["X"][j], T["Y"][j])
                what = "object" if fin.get("rigid") else "ball"
                hud.append(f"{what} at x = {fxo / g1.W:.3f}, y = {fyo / g1.H:.3f} of the frame, "
                           f"{T['D'][j] * T['r']:.1f} px wide")
                if q >= 0 and q in T["excluded"]:
                    state = (f"flight {q + 1} of {len(T['curves'])} (thrown: fitted, but the "
                             "bounces measure g)")
                elif q >= 0:
                    state = f"flight {q + 1} of {len(T['curves'])} (fitted)"
                elif not T["good"][j]:
                    state = "glitch: jumped off the path, left out"
                elif flights and j < flights[0][0] and T["push_from"] is not None \
                        and T["t"][j] >= T["push_from"]:
                    state = "hand still pushing the ball, left out"
                elif flights and j < flights[0][0]:
                    state = "before the first flight (held or just let go)"
                elif flights and j > flights[-1][-1]:
                    state = "after the last flight"
                else:
                    state = "bounce or contact, left out of the fit"
            else:
                j, q = None, -1
            # Velocity and acceleration of the flight this frame belongs to.
            qa = q
            if qa < 0:
                for q2, (tt, _, _) in enumerate(T["curves"]):
                    if tt[0] - dt / 2 <= t_cur <= tt[-1] + dt / 2:
                        qa = q2
            if qa >= 0:
                fr_ = jf.flights[qa]
                vx, vy = fr_.fit.vel(t_cur - fr_.fit.t0)
                fx_, fy_ = fr_.fit.pos(t_cur - fr_.fit.t0)

                def out_pt(qx, qy):
                    (qx,), (qy,) = raw(np.array([t_cur]), np.array([qx]), np.array([qy]))
                    return to_out(qx / T["r"], qy / T["r"])
                bx, by = out_pt(fx_, fy_)
                v_down = vx * gdir[0] + vy * gdir[1]
                v_side = -vx * gdir[1] + vy * gdir[0]
                sp = float(np.hypot(vx, vy))
                # Arrow: where the fit says it goes in 0.1 s of real time.
                ex_, ey_ = out_pt(fx_ + vx * 0.1 * k, fy_ + vy * 0.1 * k)
                ax_, ay_ = float(ex_ - bx), float(ey_ - by)
                L = float(np.hypot(ax_, ay_))
                cap = 0.35 * H
                if L > cap:
                    ax_, ay_ = ax_ * cap / L, ay_ * cap / L
                if L > 2:
                    cv2.arrowedLine(frame, _pt(bx, by), _pt(bx + ax_, by + ay_), WHITE, 2,
                                    cv2.LINE_AA, tipLength=0.2)
                a_vec = fr_.fit.accel_vec()
                a_along = float(a_vec @ gdir)
                a_raw = a_along
                fac = 1.0
                if drag:
                    fac = float(drag["flights"][qa])
                a_corr = a_raw * fac
                ref = 60.0 * fs / 0.6
                if s_m:
                    la = ref * abs(a_corr * a_unit) / g_ref
                    am = float(np.hypot(*a_vec))
                    ua = (a_vec / am) if am > 0 else np.array([0.0, 1.0])

                    def out_dir(u_):
                        # A direction in fit space, as it shows in the picture here.
                        e = 0.25 * max(float(np.nanmedian(T["D"])) if np.isfinite(T["D"]).any()
                                       else 4.0, 2.0) * T["r"]
                        qx, qy = out_pt(fx_ + u_[0] * e, fy_ + u_[1] * e)
                        dx_, dy_ = float(qx - bx), float(qy - by)
                        nn = float(np.hypot(dx_, dy_))
                        return (dx_ / nn, dy_ / nn) if nn > 0 else (float(u_[0]), float(u_[1]))
                    ux, uy = out_dir(ua)
                    gx_, gy_ = out_dir(gdir)
                    off = rad + 4 if len(jj) else 10
                    sx_, sy_ = bx + off, by
                    cv2.arrowedLine(frame, _pt(sx_, sy_), _pt(sx_ + ux * la, sy_ + uy * la),
                                    RED, 2, cv2.LINE_AA, tipLength=0.2)
                    cv2.line(frame, _pt(sx_ + 6, sy_), _pt(sx_ + 6 + gx_ * ref,
                                                           sy_ + gy_ * ref), GOLD, 1,
                             cv2.LINE_AA)
                hud.append(f"v = {sp * v_unit:.2f} {unit_v}   (down {v_down * v_unit:+.2f}, "
                           f"sideways {v_side * v_unit:+.2f})")
                line = f"a fit, flight {qa + 1} = {a_raw * a_unit:.2f} {unit_a}"
                if drag and abs(fac - 1) > 0.005:
                    line += f" raw, {a_corr * a_unit:.2f} after air drag"
                hud.append(line)
                if j is not None and np.isfinite(T["a_loc"][j]):
                    hud.append(f"a local (+/-{1000 * w * dt:.0f} ms) = {T['a_loc'][j] * a_unit:.2f}"
                               f" +/- {T['a_loc_s'][j] * a_unit:.2f} {unit_a}")
        for q, ov in enumerate(extras):
            _draw_object(frame, ov, t_cur, dt, to_out, kx, fs,
                         OBJECT_COLORS[q % len(OBJECT_COLORS)])
        if show_prim_label:
            _draw_object(frame, prim_ov, t_cur, dt, to_out, kx, fs, WHITE, primary=True)
        hud.insert(1, f"state: {state}")
        hud.append(f"candidates in this frame: {n_cand}")
        if extras:
            hud.append("objects: " + "   ".join(
                ov["label"] for ov in ([prim_ov] if prim_ov else []) + extras))

        # Your pick.
        if dbg.get("seed"):
            ts_, sx1, sy1 = dbg["seed"][0]
            if abs(ts_ - t_cur) <= max(0.1, dt):
                ox_, oy_ = to_out(sx1 + g1.ox, sy1 + g1.oy)
                c = _pt(ox_, oy_)
                for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cv2.line(frame, (c[0] + 4 * dx_, c[1] + 4 * dy_),
                             (c[0] + 12 * dx_, c[1] + 12 * dy_), WHITE, 2, cv2.LINE_AA)
                _text(frame, "your pick", (c[0] + 14, c[1] + 4), fs * 0.8)

        _hud_box(frame, hud, fs)
        v = verdict
        g_txt = f"g = {v['g']:.2f} +/- {v['sigma']:.2f} m/s^2" if v.get("g") is not None else "no g"
        foot = f"{v.get('verdict', '?')}   {g_txt}   stage: {v.get('stage') or '-'}"
        if (result or {}).get("zoom"):
            z = result["zoom"]
            foot += f"   zoom: ball {z['ball_px_before']:.0f} -> {z['ball_px']:.0f} px"
        _text(frame, foot, (10, H - 12), fs, WHITE, 1)

        panel = chart_bg.copy()
        for ch in charts:
            xc = int(ch.X(t_cur))
            cv2.line(panel, (xc, ch.y0), (xc, ch.y1), WHITE, 1)
        if T is not None and len(jj):
            j = int(jj[0])
            for ch, val in zip(charts, (_height(T, j, fin), _accel(T, j, fin))):
                if val is not None and np.isfinite(val):
                    cv2.circle(panel, _pt(ch.X(T["t"][j]), ch.Y(val)), 5, WHITE, 1, cv2.LINE_AA)
        writer.write(np.vstack([frame, panel]), (t_cur - times1[0]) / speed)
        n_done += 1
    if writer is None:
        raise RuntimeError("no frames to draw")
    writer.close()
    return {"path": out_path, "frames": n_done}


def _height(T, j, fin):
    s_m = fin["scale"]["s"] if fin["scale"] else None
    h = T["u_ref"] - T["u"][j]
    return h * s_m if s_m else h


def _accel(T, j, fin):
    s_m = fin["scale"]["s"] if fin["scale"] else None
    k = float(fin["k"] or 1.0)
    a = T["a_loc"][j]
    return a * s_m * k * k if s_m else a


def _hud_box(img, lines, fs):
    pad = 8
    lh = int(26 * fs / 0.6)
    wmax = max(cv2.getTextSize(s, FONT, fs, 1)[0][0] for s in lines)
    x1, y1 = pad + wmax + 2 * pad, pad + lh * len(lines) + pad
    roi = img[0:min(y1, img.shape[0]), 0:min(x1, img.shape[1])]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, s in enumerate(lines):
        _text(img, s, (2 * pad, pad + lh * (i + 1) - 6), fs, WHITE, 1)


def _chart_panel(W, P, fs, times1, T, fin, verdict):
    """The static part of the bottom panel: axes, points, fits, Earth lines."""
    panel = np.zeros((P, W, 3), np.uint8)
    panel[:] = PANEL_BG
    legend_h = int(22 * fs / 0.6)
    top, bot = int(30 * fs / 0.6), P - legend_h - 26
    left_w = W // 2
    t_range = (float(times1[0]), float(times1[-1]) + 1e-6)
    charts = []
    if T is None:
        _text(panel, "No flight was fitted, so there is nothing to chart.", (16, P // 2), fs,
              (200, 200, 200))
        return panel, charts
    s_m = fin["scale"]["s"] if fin["scale"] else None
    k = float(fin["k"] or 1.0)
    unit_h = "m" if s_m else "px"
    unit_a = "m/s^2" if s_m else "px/s^2"
    us = s_m if s_m else 1.0
    au = s_m * k * k if s_m else 1.0
    g_ref = float(fin["g_ref"])
    u = T["u"]
    T["u_ref"] = float(np.nanmax(u))
    h = (T["u_ref"] - u) * us
    h_hi = max(float(np.nanmax(h)) * 1.1, 1e-6)

    c1 = _Chart(panel, (56, top, left_w - 16, bot), t_range, (0.0, h_hi),
                f"height above the lowest point ({unit_h})", fs)
    a_loc = T["a_loc"] * au
    fin_a = [fr_.a_along * au for fr_ in (fin.get("jf_free") or fin["jf"]).flights]
    vals = np.concatenate([a_loc[np.isfinite(a_loc)], fin_a, [g_ref if s_m else 0.0, 0.0]])
    lo = float(np.percentile(vals, 3)) if len(vals) else 0.0
    hi = float(np.percentile(vals, 97)) if len(vals) else 1.0
    if s_m:
        lo, hi = min(lo, -0.2 * g_ref), max(hi, 1.3 * g_ref)
        lo, hi = max(lo, -2.5 * g_ref), min(hi, 3.5 * g_ref)
    span = max(hi - lo, 1e-6)
    c2 = _Chart(panel, (left_w + 56, top, W - 16, bot), t_range, (lo - 0.05 * span,
                                                                   hi + 0.05 * span),
                f"acceleration along gravity ({unit_a})", fs)
    charts = [c1, c2]
    for ch in charts:
        ch.axes()
    # Height: fitted parabolas, Earth's fall, tracked points.
    gdir = np.asarray(fin["gdir"], float)
    jf_bars = fin.get("jf_free") or fin["jf"]
    for q, (fr_, fb) in enumerate(zip(fin["jf"].flights, jf_bars.flights)):
        col = FLIGHT_COLORS[q % len(FLIGHT_COLORS)]
        tt = np.linspace(T["t"][fr_.idx[0]], T["t"][fr_.idx[-1]], 50)
        px, py = fr_.fit.pos(tt - fr_.fit.t0)
        uu = np.asarray(px) * gdir[0] + np.asarray(py) * gdir[1]
        pts = np.stack([c1.X(tt), c1.Y((T["u_ref"] - uu) * us)], 1).astype(np.int32)
        cv2.polylines(panel, [pts], False, col, 1, cv2.LINE_AA)
        a_show = fb.a_along * au
        bar = [(c2.X(tt[0]), c2.Y(a_show)), (c2.X(tt[-1]), c2.Y(a_show))]
        if q in set(fin.get("excluded") or ()):
            _dashed(panel, bar, col, 2)          # a throw: shown, not counted
        else:
            cv2.line(panel, _pt(*bar[0]), _pt(*bar[1]), col, 2, cv2.LINE_AA)
    gh = fin.get("ghost")
    if gh is not None and len(gh[0]) > 1:
        gtt, gx, gy = (np.asarray(v_, float) for v_ in gh)
        uu = gx * gdir[0] + gy * gdir[1]
        _dashed(panel, np.stack([c1.X(gtt), c1.Y((T["u_ref"] - uu) * us)], 1), GOLD, 1)
    if s_m:
        _dashed(panel, [(c2.x0, c2.Y(g_ref)), (c2.x1, c2.Y(g_ref))], GOLD, 1)
        _text(panel, f"Earth {g_ref:.2f}", (c2.x1 - 92, c2.Y(g_ref) - 5), fs * 0.75, GOLD)
    cv2.line(panel, _pt(c2.x0, c2.Y(0.0)), _pt(c2.x1, c2.Y(0.0)), (90, 90, 90), 1)
    for j in range(len(T["t"])):
        q = int(T["flight_of"][j])
        col = FLIGHT_COLORS[q % len(FLIGHT_COLORS)] if q >= 0 else GREY
        cv2.circle(panel, _pt(c1.X(T["t"][j]), c1.Y(h[j])), 2, col, -1, cv2.LINE_AA)
        if np.isfinite(a_loc[j]):
            cv2.circle(panel, _pt(c2.X(T["t"][j]), c2.Y(a_loc[j])), 2, col, -1, cv2.LINE_AA)
    _text(panel, "time (s)", (W // 2 - 30, P - legend_h - 4), fs * 0.75, (180, 180, 180))
    drag = fin.get("drag")
    bars = "bars: each flight's fit" + (" after air drag" if drag else "")
    if fin.get("excluded"):
        bars += " (dashed: thrown, not counted)"
    legend = (f"dots: tracked ball (colour = flight, grey = left out)   lines: fitted path   "
              f"{bars}   gold: Earth   right dots: local fit over a few frames")
    lf = fs * 0.72
    while lf > 0.3 and cv2.getTextSize(legend, FONT, lf, 1)[0][0] > W - 20:
        lf -= 0.02
    _text(panel, legend, (10, P - 8), lf, (200, 200, 200))
    return panel, charts
