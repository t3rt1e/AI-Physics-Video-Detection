"""Synthetic test clips with known physics.

Renders a ball against a dark backdrop with motion blur, optional rolling
shutter, sensor noise, camera drift, bounces, and every slow-motion trick
Galileo has to see through (retimed high-speed capture, repeated frames,
interpolated frames). Used by the tests and for rehearsing the demo before
the tower exists. Synthetic clips are labelled as such; never present them
as real footage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .video import Writer


@dataclass
class Scene:
    width: int = 1280
    height: int = 720
    px_per_m: float = 520.0        # scale at the ball's distance
    ball_d_m: float = 0.038
    ball_bgr: tuple = (40, 150, 255)   # orange
    shading: float = 0.35          # 0 = flat disc, 1 = strong limb darkening
    bg_level: int = 22
    grid: bool = False             # draw a 2 cm grid (calibration shots)
    texture: bool = True           # a few static shapes so stabilization has features
    noise: float = 2.0
    psf_sigma: float = 0.7         # optics blur, px
    x0: float | None = None        # ball x (px); default centre
    y0: float = 70.0               # rest position, ball centre (px)
    floor_y: float | None = None   # ball centre at contact (px); None = no floor
    seed: int = 0


@dataclass
class Motion:
    g: float = 9.81                 # m/s^2 (use ~2.3 for "AI floaty")
    t_release: float = 0.25         # s
    restitution: float | None = None
    accel_fn: object = None         # optional a(t_since_release) in m/s^2 (AI-like)
    vx: float = 0.0                 # m/s sideways drift
    pan: tuple = (0.0, 0.0)         # camera drift, px/s
    zoom_rate: float = 0.0          # fractional zoom per second


def _trajectory(m: Motion, sc: Scene, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ball centre (x, y) in px at times t (s)."""
    x0 = sc.width / 2 if sc.x0 is None else sc.x0
    tr = np.clip(t - m.t_release, 0, None)
    if m.accel_fn is not None:
        dt = 1e-4
        grid = np.arange(0, max(float(tr.max()), 0) + 2 * dt, dt)
        a = np.array([m.accel_fn(s) for s in grid])
        v = np.concatenate([[0], np.cumsum(0.5 * (a[1:] + a[:-1]) * dt)])
        s = np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
        yy = sc.y0 + np.interp(tr, grid, s) * sc.px_per_m
    else:
        yy = np.array([_y_bounce(ti, m, sc) for ti in tr])
    xx = x0 + m.vx * tr * sc.px_per_m
    return xx, yy


def _y_bounce(tr: float, m: Motion, sc: Scene) -> float:
    a = m.g * sc.px_per_m
    y, v, tc = sc.y0, 0.0, 0.0
    if sc.floor_y is None or m.restitution is None:
        return sc.y0 + 0.5 * a * tr * tr
    for _ in range(60):
        disc = v * v + 2 * a * (sc.floor_y - y)
        t_hit = (-v + math.sqrt(max(disc, 0.0))) / a
        if tc + t_hit >= tr:
            dt = tr - tc
            return y + v * dt + 0.5 * a * dt * dt
        v_imp = v + a * t_hit
        tc += t_hit
        y = sc.floor_y
        v = -m.restitution * v_imp
        if abs(v) < 2.0:
            return sc.floor_y
    return sc.floor_y


def _backdrop(sc: Scene, rng) -> np.ndarray:
    h, w = sc.height, sc.width
    img = np.full((h, w, 3), sc.bg_level, np.float32)
    img += np.linspace(0, 10, h, dtype=np.float32)[:, None, None]
    if sc.texture:
        for _ in range(14):
            cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
            if abs(cx - (sc.x0 or w / 2)) < 90:
                cx = int(cx + 220 * np.sign(cx - w / 2 + 1e-3))
            col = tuple(float(c) for c in rng.uniform(40, 120, 3))
            cv2.rectangle(img, (cx, cy), (cx + int(rng.uniform(15, 60)),
                                          cy + int(rng.uniform(15, 60))), col, -1)
    if sc.grid:
        step = 0.02 * sc.px_per_m
        for gx in np.arange(0, w, step):
            cv2.line(img, (int(gx), 0), (int(gx), h), (70, 70, 70), 1)
        for gy in np.arange(0, h, step):
            cv2.line(img, (0, int(gy)), (w, int(gy)), (70, 70, 70), 1)
    return img


def render_frame(t_frame: float, m: Motion, sc: Scene, bg: np.ndarray, rng, *,
                 exposure: float, readout: float, subsamples: int = 16,
                 override_xy=None) -> np.ndarray:
    """One frame: exposure-integrated ball over the backdrop, per-row readout time."""
    h, w = sc.height, sc.width
    R = 0.5 * sc.ball_d_m * sc.px_per_m
    rows = np.arange(h)
    # Bounding rows of the ball over this exposure.
    ts_probe = np.linspace(t_frame, t_frame + readout + exposure, 9)
    if override_xy is None:
        px, py = _trajectory(m, sc, ts_probe)
    else:
        px = np.full(9, override_xy[0])
        py = np.full(9, override_xy[1])
    r0 = int(max(np.floor(py.min() - R - 3), 0))
    r1 = int(min(np.ceil(py.max() + R + 3), h - 1))
    c0 = int(max(np.floor(px.min() - R - 3), 0))
    c1 = int(min(np.ceil(px.max() + R + 3), w - 1))
    img = bg.copy()
    if r1 > r0 and c1 > c0:
        rr = rows[r0:r1 + 1].astype(np.float64) + 0.5
        cc = np.arange(c0, c1 + 1, dtype=np.float64) + 0.5
        cov = np.zeros((len(rr), len(cc)))
        shade = np.zeros_like(cov)
        for s in range(subsamples):
            tau = t_frame + readout * (rr - 0.5) / h + exposure * (s + 0.5) / subsamples
            if override_xy is None:
                bx, by = _trajectory(m, sc, tau)
            else:
                bx = np.full_like(tau, override_xy[0])
                by = np.full_like(tau, override_xy[1])
            dy = rr - by
            hw = np.sqrt(np.clip(R * R - dy * dy, 0, None))
            dx = np.abs(cc[None, :] - bx[:, None])
            c = np.clip(hw[:, None] - dx + 0.5, 0.0, 1.0)
            rho2 = np.clip((dx**2 + dy[:, None] ** 2) / (R * R), 0, 1)
            cov += c
            shade += c * (1.0 - sc.shading + sc.shading * np.sqrt(1.0 - rho2))
        cov /= subsamples
        lum = np.where(cov > 0, shade / np.maximum(cov * subsamples, 1e-9), 0)
        ball = np.asarray(sc.ball_bgr, np.float32)[None, None, :] * lum[:, :, None]
        reg = img[r0:r1 + 1, c0:c1 + 1]
        img[r0:r1 + 1, c0:c1 + 1] = reg * (1 - cov[:, :, None]) + ball * cov[:, :, None]
    if sc.psf_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sc.psf_sigma)
    if m.pan != (0.0, 0.0) or m.zoom_rate:
        z = 1.0 + m.zoom_rate * t_frame
        A = np.array([[z, 0, (1 - z) * w / 2 + m.pan[0] * t_frame],
                      [0, z, (1 - z) * h / 2 + m.pan[1] * t_frame]], np.float32)
        img = cv2.warpAffine(img, A, (w, h), borderMode=cv2.BORDER_REPLICATE)
    img += rng.normal(0, sc.noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_clip(path: str, *, scene: Scene | None = None, motion: Motion | None = None,
              capture_fps: float = 60.0, duration: float = 1.2, exposure_frac: float = 0.5,
              rolling_shutter: float = 0.0, slowmo: dict | None = None,
              tag_capture_fps: bool = False) -> dict:
    """Render and save a clip. Returns the ground truth.

    rolling_shutter: readout time as a fraction of the frame period.
    slowmo: None, or {"k": 8, "method": "retime"|"dup"|"interp"|"blend"}.
      retime: frames captured at capture_fps, written at capture_fps / k.
      dup:    frames captured at capture_fps, each written k times (same fps).
      interp: k-1 straight-line in-betweens inserted between captured frames.
      blend:  k-1 cross-faded in-betweens.
    """
    sc = scene or Scene()
    m = motion or Motion()
    rng = np.random.default_rng(sc.seed)
    bg = _backdrop(sc, rng)
    period = 1.0 / capture_fps
    exposure = exposure_frac * period
    readout = rolling_shutter * period
    n = int(round(duration * capture_fps))
    t_cap = np.arange(n) * period
    frames = [render_frame(t, m, sc, bg, rng, exposure=exposure, readout=readout)
              for t in t_cap]
    k = int(slowmo["k"]) if slowmo else 1
    method = slowmo["method"] if slowmo else None
    out_fps = capture_fps
    seq: list[np.ndarray] = frames
    if method == "retime":
        out_fps = capture_fps / k
    elif method == "dup":
        seq = [f for f in frames for _ in range(k)]
    elif method in ("interp", "blend"):
        seq = []
        for i in range(n - 1):
            seq.append(frames[i])
            for j in range(1, k):
                a = j / k
                if method == "blend":
                    seq.append(cv2.addWeighted(frames[i], 1 - a, frames[i + 1], a, 0))
                else:
                    tm = t_cap[i] + a * period
                    xa, ya = _trajectory(m, sc, np.array([t_cap[i] + 0.5 * exposure]))
                    xb, yb = _trajectory(m, sc, np.array([t_cap[i + 1] + 0.5 * exposure]))
                    xy = (xa[0] + a * (xb[0] - xa[0]), ya[0] + a * (yb[0] - ya[0]))
                    seq.append(render_frame(tm, m, sc, bg, rng, exposure=0.0, readout=0.0,
                                            subsamples=1, override_xy=xy))
        seq.append(frames[-1])
    tags = {}
    if tag_capture_fps:
        tags["com.android.capture.fps"] = f"{capture_fps:.6f}"
    wr = Writer(path, out_fps, (sc.width, sc.height), tags=tags)
    for i, f in enumerate(seq):
        wr.write(f, i / out_fps)
    wr.close()
    return {"path": path, "g": m.g, "px_per_m": sc.px_per_m, "ball_d_m": sc.ball_d_m,
            "capture_fps": capture_fps, "file_fps": out_fps, "k": k if slowmo else 1,
            "method": method, "frames": len(seq)}


def ease_in_out(total_drop_m: float, duration_s: float):
    """AI-style animation curve: accelerates, then brakes before landing."""
    def a(t):
        if t > duration_s:
            return 0.0
        u = t / duration_s
        return total_drop_m * (6 - 12 * u) / duration_s**2
    return a
