"""3D synthetic clips: a ball in a room, filmed by a camera that can sit at
any angle and move while it films (turn, tilt, roll, zoom, walk closer).

Ground truth for Novex's camera handling. World axes: x right, y up, z away
from where the camera starts; the floor is y = 0. The camera is a pinhole
with square pixels and its principal point at the picture's centre. The ball
is ray traced with motion blur (and rolling shutter if asked), over a room
whose floor, walls and ceiling carry fixed textures, so stabilization has
something to hold on to. Synthetic clips are labelled as such; never present
them as real footage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .video import Writer


@dataclass
class Camera3D:
    """Where the camera is and how it moves while filming.

    Angles in degrees: yaw turns right, pitch looks down, roll turns the
    camera clockwise as seen from behind it (the picture turns the other
    way). hfov is the horizontal field of view at the start. Rates are per
    second from t = 0; zoom_rate grows the focal length by that fraction per
    second. shake_deg / shake_m add a handheld wobble.
    """
    pos: tuple = (0.0, 1.0, 0.0)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    hfov: float = 65.0
    width: int = 960
    height: int = 540
    vel: tuple = (0.0, 0.0, 0.0)
    yaw_rate: float = 0.0
    pitch_rate: float = 0.0
    roll_rate: float = 0.0
    zoom_rate: float = 0.0
    shake_deg: float = 0.0
    shake_m: float = 0.0
    seed: int = 0

    @property
    def f0(self) -> float:
        return 0.5 * self.width / math.tan(math.radians(0.5 * self.hfov))


@dataclass
class Ball3D:
    """A ball held still, let go at t_release, falling with gravity g (and
    quadratic air drag if vt is set), bouncing off the floor with
    `restitution` (None: never touches it). accel_fn(t since release) replaces
    gravity's pull with any downward acceleration (m/s^2), to fake AI motion."""
    d_m: float = 0.074
    pos: tuple = (0.0, 1.5, 2.5)
    vel: tuple = (0.0, 0.0, 0.0)
    t_release: float = 0.25
    g: float = 9.81
    restitution: float | None = 0.6
    vt: float | None = None
    bgr: tuple = (60, 225, 215)
    kick: float = 0.0
    accel_fn: object = None
    seed: int = 0


@dataclass
class Room:
    wall_z: float = 4.0            # back wall at z = wall_z
    side_x: float | None = 3.0     # side walls at x = +-side_x (None: open)
    ceiling_y: float = 2.8
    shadow: float = 0.35           # how dark the ball's shadow on the floor is
    light: tuple = (0.25, -0.7, 1.0)    # direction the light travels (from behind the camera, above)
    seed: int = 0


# ---------------------------------------------------------------------------
# Camera geometry

def _basis(yaw, pitch, roll):
    """Camera -> world rotation (columns: right, down, forward)."""
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    fwd = np.array([math.cos(p) * math.sin(y), -math.sin(p), math.cos(p) * math.cos(y)])
    right = np.array([math.cos(y), 0.0, -math.sin(y)])
    down = np.array([-math.sin(p) * math.sin(y), -math.cos(p), -math.sin(p) * math.cos(y)])
    r2 = math.cos(r) * right + math.sin(r) * down
    d2 = -math.sin(r) * right + math.cos(r) * down
    return np.column_stack([r2, d2, fwd])


def _wobble(t, amp, seed, k):
    rng = np.random.default_rng(seed * 31 + k)
    f = rng.uniform([0.5, 1.1, 2.3, 4.7], [0.9, 1.9, 3.6, 7.5])
    ph = rng.uniform(0, 2 * np.pi, 4)
    a = amp * np.array([0.6, 0.35, 0.2, 0.1])
    return float(np.sum(a * np.sin(2 * np.pi * f * t + ph)))


def camera_pose(cam: Camera3D, t: float):
    """(centre (3,), camera->world rotation (3, 3), focal length px) at time t."""
    sh = cam.shake_deg
    yaw = cam.yaw + cam.yaw_rate * t + (_wobble(t, sh, cam.seed, 1) if sh else 0.0)
    pitch = cam.pitch + cam.pitch_rate * t + (_wobble(t, sh, cam.seed, 2) if sh else 0.0)
    roll = cam.roll + cam.roll_rate * t + (_wobble(t, 0.5 * sh, cam.seed, 3) if sh else 0.0)
    C = np.asarray(cam.pos, float) + np.asarray(cam.vel, float) * t
    if cam.shake_m:
        C = C + np.array([_wobble(t, cam.shake_m, cam.seed, 4 + i) for i in range(3)])
    f = cam.f0 * (1.0 + cam.zoom_rate * t)
    return C, _basis(yaw, pitch, roll), f


def project(cam: Camera3D, t: float, P):
    """World points (n, 3) -> pixel (u, v) and depth along the camera's axis."""
    C, R, f = camera_pose(cam, t)
    pc = (np.atleast_2d(P) - C) @ R
    u = 0.5 * cam.width + f * pc[:, 0] / pc[:, 2]
    v = 0.5 * cam.height + f * pc[:, 1] / pc[:, 2]
    return u, v, pc[:, 2]


# ---------------------------------------------------------------------------
# Ball motion

def ball_path(b: Ball3D, t_end: float, dt: float = 2e-4):
    """(times, centres (n, 3)) on a fine grid from t = 0."""
    n = int(t_end / dt) + 2
    ts = np.arange(n) * dt
    P = np.empty((n, 3))
    p = np.asarray(b.pos, float).copy()
    v = np.asarray(b.vel, float).copy()
    R = 0.5 * b.d_m
    kd = b.g / b.vt**2 if b.vt else 0.0
    rng = np.random.default_rng(b.seed + 7)
    resting = False
    for i, t in enumerate(ts):
        P[i] = p
        if t < b.t_release or resting:
            continue
        a_down = b.accel_fn(t - b.t_release) if b.accel_fn is not None else b.g
        sp = float(np.linalg.norm(v))
        acc = np.array([0.0, -a_down, 0.0]) - kd * sp * v
        v = v + acc * dt
        p = p + v * dt
        if b.restitution is not None and p[1] - R <= 0.0 and v[1] < 0:
            p[1] = 2 * R - p[1]
            v[1] = -b.restitution * v[1]
            if b.kick:
                v[0] += rng.normal(0, b.kick)
                v[2] += rng.normal(0, b.kick)
            if v[1] < 0.15:
                resting = True
                p[1] = R
    return ts, P


# ---------------------------------------------------------------------------
# The room

_TEXEL = 0.006   # m per texel


def _noise_tex(h, w, rng, scale, amp):
    small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def _floor_tex(rng, w, h):
    """Wooden planks along x, 0.15 m wide, with grain and a few marks."""
    img = np.zeros((h, w, 3), np.float32)
    plank = int(0.15 / _TEXEL)
    for j in range(0, h, plank):
        base = rng.uniform(55, 95)
        tint = np.array([0.85, 0.95, 1.12]) * base
        img[j:j + plank] = tint
        # Plank ends at random places.
        for _ in range(int(w * _TEXEL / 1.2)):
            cut = int(rng.uniform(0, w))
            img[j:j + plank, max(cut - 1, 0):cut + 2] *= 0.6
    img += _noise_tex(h, w, rng, 6, 6.0)[:, :, None]
    img += _noise_tex(h, w, rng, 40, 10.0)[:, :, None]
    for j in range(0, h, plank):
        img[j:j + 2] *= 0.55
    for _ in range(25):
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        cv2.circle(img, (cx, cy), int(rng.uniform(3, 9)), tuple(float(c) for c in rng.uniform(30, 60, 3)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _wall_tex(rng, w, h, door=True):
    """Off-white wall with panel seams, a baseboard, outlets, a door, shelves
    and a few posters: the uprights and edges a real room has."""
    img = np.full((h, w, 3), 0, np.float32)
    img[:] = np.array([196, 202, 208], np.float32)
    img += _noise_tex(h, w, rng, 60, 5.0)[:, :, None]
    img += _noise_tex(h, w, rng, 4, 1.5)[:, :, None]
    bb = int(0.09 / _TEXEL)                      # baseboard, at the bottom (y = 0)
    # Panel seams, about every metre and a bit.
    xs = 0.0
    while True:
        xs += rng.uniform(0.9, 1.4)
        xp = int(xs / _TEXEL)
        if xp >= w:
            break
        img[:, max(xp - 1, 0):xp + 1] *= 0.72
    for _ in range(int(w * _TEXEL / 1.5)):
        cx = int(rng.uniform(0.02, 0.98) * w)
        cy = h - int(rng.uniform(0.25, 1.2) / _TEXEL)
        cv2.rectangle(img, (cx, cy), (cx + int(0.07 / _TEXEL), cy + int(0.11 / _TEXEL)),
                      (225, 228, 230), -1)
        cv2.rectangle(img, (cx, cy), (cx + int(0.07 / _TEXEL), cy + int(0.11 / _TEXEL)),
                      (120, 120, 120), 1)
    # Bookshelves: tall boxes with shelves.
    for _ in range(max(1, int(w * _TEXEL / 4))):
        x0 = int(rng.uniform(0.0, 0.9) * w)
        sw, sh = int(rng.uniform(0.6, 1.0) / _TEXEL), int(rng.uniform(1.4, 2.0) / _TEXEL)
        col = tuple(float(c) for c in rng.uniform(50, 110, 3))
        cv2.rectangle(img, (x0, h - sh), (x0 + sw, h), col, -1)
        cv2.rectangle(img, (x0, h - sh), (x0 + sw, h), (30, 30, 30), 4)
        for k in range(1, 5):
            yk = h - int(k * sh / 5)
            cv2.line(img, (x0, yk), (x0 + sw, yk), (35, 35, 35), 4)
            # books
            bx = x0 + 6
            while bx < x0 + sw - 10:
                bw_ = int(rng.uniform(5, 14))
                bh_ = int(rng.uniform(0.6, 0.9) * sh / 5)
                cv2.rectangle(img, (bx, yk - bh_), (bx + bw_, yk - 3),
                              tuple(float(c) for c in rng.uniform(40, 220, 3)), -1)
                bx += bw_ + int(rng.uniform(0, 4))
    for _ in range(3):
        x0 = int(rng.uniform(0.05, 0.85) * w)
        y0 = h - int(rng.uniform(1.4, 2.3) / _TEXEL)
        pw, ph = int(rng.uniform(0.4, 0.8) / _TEXEL), int(rng.uniform(0.3, 0.6) / _TEXEL)
        col = tuple(float(c) for c in rng.uniform(40, 170, 3))
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), col, -1)
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (40, 40, 40), 3)
        for _ in range(5):
            a, b_ = int(rng.uniform(x0, x0 + pw)), int(rng.uniform(y0, y0 + ph))
            cv2.circle(img, (a, b_), int(rng.uniform(5, 25)),
                       tuple(float(c) for c in rng.uniform(20, 230, 3)), -1)
    if door:
        dx = int(rng.uniform(0.35, 0.6) * w)
        dw, dh = int(0.9 / _TEXEL), int(2.05 / _TEXEL)
        cv2.rectangle(img, (dx, h - dh), (dx + dw, h), (150, 165, 185), -1)
        cv2.rectangle(img, (dx, h - dh), (dx + dw, h), (90, 95, 105), 6)
        cv2.circle(img, (dx + dw - 20, h - int(1.0 / _TEXEL)), 6, (60, 170, 200), -1)
    img[h - bb:h] = np.array([232, 236, 238], np.float32)
    img[h - bb - 2:h - bb] = 150
    return np.clip(img, 0, 255).astype(np.uint8)


def _ceiling_tex(rng, w, h):
    img = np.full((h, w, 3), 210, np.float32)
    img += _noise_tex(h, w, rng, 50, 4.0)[:, :, None]
    for _ in range(6):
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        cv2.rectangle(img, (cx, cy), (cx + int(0.6 / _TEXEL), cy + int(0.6 / _TEXEL)),
                      (245, 245, 245), -1)
        cv2.rectangle(img, (cx, cy), (cx + int(0.6 / _TEXEL), cy + int(0.6 / _TEXEL)),
                      (150, 150, 150), 2)
    return np.clip(img, 0, 255).astype(np.uint8)


class _Planes:
    """The room's planes with textures, for fast per-pixel lookups."""

    X0, X1, Z0, Z1 = -6.0, 6.0, -6.0, 12.0

    def __init__(self, room: Room):
        rng = np.random.default_rng(room.seed + 1)
        self.room = room
        X0, X1, Z0, Z1 = self.X0, self.X1, self.Z0, self.Z1
        H = room.ceiling_y
        nx, nz, ny = int((X1 - X0) / _TEXEL), int((Z1 - Z0) / _TEXEL), int(H / _TEXEL)
        self.planes = []
        # (point, normal, origin, axis u, axis v, texture)
        self.planes.append(("floor", np.array([0, 0, 0.0]), np.array([0, 1, 0.0]),
                            np.array([X0, 0, Z1]), np.array([1, 0, 0.0]), np.array([0, 0, -1.0]),
                            _floor_tex(rng, nx, nz)))
        self.planes.append(("ceiling", np.array([0, H, 0.0]), np.array([0, -1, 0.0]),
                            np.array([X0, H, Z0]), np.array([1, 0, 0.0]), np.array([0, 0, 1.0]),
                            _ceiling_tex(rng, nx, nz)))
        self.planes.append(("back", np.array([0, 0, room.wall_z]), np.array([0, 0, -1.0]),
                            np.array([X0, H, room.wall_z]), np.array([1, 0, 0.0]),
                            np.array([0, -1, 0.0]), _wall_tex(rng, nx, ny)))
        self.planes.append(("front", np.array([0, 0, Z0]), np.array([0, 0, 1.0]),
                            np.array([X1, H, Z0]), np.array([-1, 0, 0.0]),
                            np.array([0, -1, 0.0]), _wall_tex(rng, nx, ny, door=False)))
        if room.side_x:
            for s in (-1.0, 1.0):
                self.planes.append((f"side{int(s)}", np.array([s * room.side_x, 0, 0]),
                                    np.array([-s, 0, 0.0]),
                                    np.array([s * room.side_x, H, Z0 if s > 0 else Z1]),
                                    np.array([0, 0, s]), np.array([0, -1, 0.0]),
                                    _wall_tex(rng, nz, ny)))

    def render(self, C, R, f, W, H, dirs, ball_xyz=None, ball_r=0.0):
        """What the camera (centre C, camera->world R, focal f) sees of the room.

        Each plane is warped straight from its texture (a plane maps to the
        picture by a homography); every pixel then takes the nearest plane its
        ray meets. dirs: (H, W, 3) world directions of the pixel rays.
        """
        K = np.array([[f, 0, 0.5 * W], [0, f, 0.5 * H], [0, 0, 1.0]])
        best = np.full((H, W), np.inf, np.float32)
        label = np.full((H, W), -1, np.int8)
        for j, (name, p0, n, o, eu, ev, tex) in enumerate(self.planes):
            den = dirs @ n.astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.float32((p0 - C) @ n) / den
            ok = (den < -1e-6) & (s > 1e-3) & (s < best)
            best = np.where(ok, s, best)
            label[ok] = j
        out = np.zeros((H, W, 3), np.float32)
        for j, (name, p0, n, o, eu, ev, tex) in enumerate(self.planes):
            sel = label == j
            if not sel.any():
                continue
            Hm = K @ R.T @ np.column_stack([_TEXEL * eu, _TEXEL * ev, o - C])
            # Only the part of the picture this plane fills.
            rows = np.flatnonzero(sel.any(axis=1))
            cols = np.flatnonzero(sel.any(axis=0))
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            Tm = np.array([[1, 0, -c0], [0, 1, -r0], [0, 0, 1.0]])
            img = cv2.warpPerspective(tex, Tm @ Hm, (int(c1 - c0), int(r1 - r0)),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            sub = sel[r0:r1, c0:c1]
            out[r0:r1, c0:c1][sub] = img[sub]
            if name == "floor" and ball_xyz is not None and self.room.shadow > 0:
                P = C + dirs[sel] * best[sel][:, None]
                L = np.asarray(self.room.light, float)
                L = L / np.linalg.norm(L)
                # The ball's shadow: where the light ray through the ball meets the floor.
                k = ball_xyz[1] / max(-L[1], 1e-6)
                sx, sz = ball_xyz[0] + k * L[0], ball_xyz[2] + k * L[2]
                rad = ball_r * (1.0 + 0.6 * ball_xyz[1])
                d2 = (P[:, 0] - sx) ** 2 + (P[:, 2] - sz) ** 2
                dark = self.room.shadow / (1.0 + 0.8 * ball_xyz[1]) * np.exp(-0.5 * d2 / rad**2)
                out[sel] *= (1.0 - dark)[:, None].astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Rendering

def _rays(cam: Camera3D, R, f, xs, ys):
    """World directions for pixel coordinates xs, ys (any shape)."""
    cx, cy = 0.5 * cam.width, 0.5 * cam.height
    d = np.stack([(xs - cx) / f, (ys - cy) / f, np.ones_like(xs)], axis=-1)
    return d @ R.T


def _draw_ball(img, cam, C, R, f, b: Ball3D, path, t0, exposure, readout, light, ss=3, nt=8):
    """Exposure-integrated, anti-aliased ball (rows read out over `readout`)."""
    ts, P = path
    H, W = img.shape[:2]
    rad = 0.5 * b.d_m

    def at(tt):
        return np.stack([np.interp(tt, ts, P[:, i]) for i in range(3)], axis=-1)

    # Where the ball can be during this frame (all rows): its projected box.
    probe = at(np.linspace(t0, t0 + readout + exposure, 9))
    pc = (probe - C) @ R
    if np.any(pc[:, 2] <= rad):
        return
    u = 0.5 * W + f * pc[:, 0] / pc[:, 2]
    v = 0.5 * H + f * pc[:, 1] / pc[:, 2]
    rp = f * rad / pc[:, 2] * 1.6 + 3
    x0 = int(max(np.floor((u - rp).min()), 0))
    x1 = int(min(np.ceil((u + rp).max()), W - 1))
    y0 = int(max(np.floor((v - rp).min()), 0))
    y1 = int(min(np.ceil((v + rp).max()), H - 1))
    if x1 < x0 or y1 < y0:
        return
    off = (np.arange(ss) + 0.5) / ss
    gx = (np.arange(x0, x1 + 1)[:, None] + off[None, :]).ravel()
    gy = (np.arange(y0, y1 + 1)[:, None] + off[None, :]).ravel()
    XS, YS = np.meshgrid(gx, gy)
    D = _rays(cam, R, f, XS, YS)                         # (ny*ss, nx*ss, 3)
    Dn = D / np.linalg.norm(D, axis=-1, keepdims=True)
    row_t = readout * (YS / H)                           # rolling shutter delay per sample row
    L = np.asarray(light, float)
    L = L / np.linalg.norm(L)
    cov = np.zeros(XS.shape, np.float32)
    shade = np.zeros(XS.shape, np.float32)
    for k in range(nt):
        tk = t0 + exposure * (k + 0.5) / nt + (row_t if readout else 0.0)
        c = at(tk)                                        # (rows, cols, 3) or (3,)
        oc = c - C
        bq = np.sum(Dn * oc, axis=-1)
        disc = bq * bq - (np.sum(oc * oc, axis=-1) - rad * rad)
        hit = disc >= 0
        s = bq - np.sqrt(np.clip(disc, 0, None))
        Ph = C + Dn * s[..., None]
        nrm = (Ph - c) / rad
        lam = np.clip(-(nrm @ L), 0, None)
        cov += hit
        shade += hit * (0.55 + 0.5 * lam)
    cov /= nt
    shade = np.where(cov > 0, shade / np.maximum(cov * nt, 1e-9), 0)
    # Back to pixels.
    ny, nx = y1 - y0 + 1, x1 - x0 + 1
    covp = cov.reshape(ny, ss, nx, ss).mean(axis=(1, 3))
    shp = (shade * cov).reshape(ny, ss, nx, ss).mean(axis=(1, 3)) / np.maximum(covp, 1e-9)
    col = np.asarray(b.bgr, np.float32)[None, None, :] * shp[:, :, None]
    reg = img[y0:y1 + 1, x0:x1 + 1]
    img[y0:y1 + 1, x0:x1 + 1] = reg * (1 - covp[:, :, None]) + col * covp[:, :, None]


def render_clip(path: str, *, cam: Camera3D | None = None, ball: Ball3D | None = None,
                room: Room | None = None, fps: float = 120.0, duration: float = 1.6,
                exposure_frac: float = 0.5, rolling_shutter: float = 0.0, noise: float = 1.5,
                psf: float = 0.6, balls: tuple = ()) -> dict:
    """Render and save a clip; returns the ground truth.

    balls: more Ball3D objects falling in the same room.
    """
    cam = cam or Camera3D()
    ball = ball or Ball3D()
    room = room or Room()
    planes = _Planes(room)
    rng = np.random.default_rng(cam.seed + 5)
    W, H = cam.width, cam.height
    period = 1.0 / fps
    n = int(round(duration * fps))
    all_balls = (ball,) + tuple(balls)
    paths = [ball_path(b, duration + 0.1) for b in all_balls]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    wr = Writer(path, fps, (W, H))
    truth = {"frames": [], "g": ball.g, "d_m": ball.d_m, "fps": fps, "width": W, "height": H,
             "f0": cam.f0, "hfov": cam.hfov}
    for i in range(n):
        t = i * period
        C, R, f = camera_pose(cam, t)
        dirs = _rays(cam, R.astype(np.float32), np.float32(f), xs, ys)
        bc = np.array([np.interp(t, paths[0][0], paths[0][1][:, j]) for j in range(3)])
        img = planes.render(C, R, f, W, H, dirs, bc, 0.5 * ball.d_m)
        for b, pth in zip(all_balls, paths):
            _draw_ball(img, cam, C, R, f, b, pth, t, exposure_frac * period,
                       rolling_shutter * period, room.light)
        if psf > 0:
            img = cv2.GaussianBlur(img, (0, 0), psf)
        img += rng.normal(0, noise, img.shape).astype(np.float32)
        wr.write(np.clip(img, 0, 255).astype(np.uint8), t)
        tm = t + 0.5 * exposure_frac * period
        u, v, z = project(cam, t, np.array([[np.interp(tm, paths[0][0], paths[0][1][:, j])
                                              for j in range(3)]]))
        truth["frames"].append({"t": t, "u": float(u[0]), "v": float(v[0]), "z": float(z[0]),
                                "d_px": float(f * ball.d_m / z[0]), "f": float(f)})
    wr.close()
    truth["path"] = path
    return truth


def render_still(cam: Camera3D, room: Room | None = None, t: float = 0.0) -> np.ndarray:
    """The empty room as the camera sees it at time t (no ball, no noise)."""
    planes = _Planes(room or Room())
    C, R, f = camera_pose(cam, t)
    ys, xs = np.mgrid[0:cam.height, 0:cam.width].astype(np.float32) + 0.5
    dirs = _rays(cam, R.astype(np.float32), np.float32(f), xs, ys)
    img = planes.render(C, R, f, cam.width, cam.height, dirs)
    return np.clip(img, 0, 255).astype(np.uint8)


def with_camera(cam: Camera3D, **kw) -> Camera3D:
    return replace(cam, **kw)
