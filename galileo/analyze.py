"""The pipeline: video in, verdict out.

    frames -> stabilize -> track every falling object -> zoom in on small ones
           -> what is each one, and is it round (you, Gemini, CLIP, a look-alike)
           -> per object: fit (with rolling shutter) -> clocks (k, shared)
              -> rulers (s) -> air drag -> scale-free checks -> verdict
           -> one verdict for the clip

g = s * a_px * k**2 (times an air-drag factor for light objects), where a_px
is the measured acceleration in pixels per video-second squared, s is meters
per pixel at the object, and k is real seconds per video second.

Two passes. The first looks at the whole picture, scaled down so the clip
fits in memory, and finds the falling objects. If one came out small there
(high frame rate clips get scaled down the most), a second pass decodes only
the stretch of time and the patch of picture where it flew, at up to full
resolution, and measures it again there.

Round objects are measured by their width and brightness centre. Anything
else is followed by its outline's area centroid, corrected for spin when its
mass sits off-centre, and measured by its longest side (galileo/shape.py).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from . import camera as cam_
from . import checks, clocks, identify, scene
from . import config as C
from . import shape as shp
from .physics import (_noise_px, drag_factor, drag_path, fit_flights, gravity_direction,
                      launch_velocity, robust_fit_2d, rolling_shutter_effect, split_flights,
                      trim_release)
from .scale import Ruler, fuse, ruler_from_size, ruler_manual
from .stabilize import rescale_transforms, stabilize, warp
from .track import background, track_falling_object
from .verdict import combine_objects, decide
from .video import motion_window, probe, read_frames, work_scale

DOWN = np.array([0.0, 1.0])
PING_PONG_VT = 8.4


@dataclass
class Geom:
    """Where the analysed pixels sit in the picture.

    Positions in the pipeline are working pixels of the reference view: the
    whole upright frame scaled by `scale` (W x H), as the camera saw it at
    the reference moment (its principal point at W/2, H/2; a moving camera
    can put points off its edges, at negative or large coordinates). The
    tracked (steadied) frames are a window of that with its top-left corner
    at (ox, oy): a crop for the zoom pass, a canvas bigger than the frame
    for a panning camera. T are the stabilization transforms (3x3), from
    each frame as decoded (its own top-left at (fox, foy) of the whole
    frame, default the same as the window's) to the window's pixels.
    """
    W: int
    H: int
    scale: float
    ox: float = 0.0
    oy: float = 0.0
    T: list | None = None
    fox: float | None = None
    foy: float | None = None

    def _foff(self):
        return (self.ox if self.fox is None else self.fox,
                self.oy if self.foy is None else self.foy)

    def to_frame(self, fi, x, y):
        """Stabilized working px -> where that point shows in frame fi."""
        if self.T is None:
            return x, y
        fx0, fy0 = self._foff()
        T = np.asarray(self.T[fi], np.float64)
        A = np.linalg.inv(T if T.shape == (3, 3) else np.vstack([T, [0, 0, 1.0]]))
        lx, ly = np.asarray(x, float) - self.ox, np.asarray(y, float) - self.oy
        w_ = A[2, 0] * lx + A[2, 1] * ly + A[2, 2]
        return ((A[0, 0] * lx + A[0, 1] * ly + A[0, 2]) / w_ + fx0,
                (A[1, 0] * lx + A[1, 1] * ly + A[1, 2]) / w_ + fy0)

    def from_frame(self, fi, x, y):
        """A point in frame fi (working px) -> stabilized working px."""
        if self.T is None:
            return x, y
        fx0, fy0 = self._foff()
        A = np.asarray(self.T[fi], np.float64)
        if A.shape != (3, 3):
            A = np.vstack([A, [0, 0, 1.0]])
        lx, ly = np.asarray(x, float) - fx0, np.asarray(y, float) - fy0
        w_ = A[2, 0] * lx + A[2, 1] * ly + A[2, 2]
        return ((A[0, 0] * lx + A[0, 1] * ly + A[0, 2]) / w_ + self.ox,
                (A[1, 0] * lx + A[1, 1] * ly + A[1, 2]) / w_ + self.oy)


@dataclass
class _Obj:
    """One falling object on its way through the pipeline."""
    n: int                         # 1-based, as shown to the user; 1 is the primary
    tr: object                     # TrackResult with this object as .main, in its pass's px
    times: np.ndarray              # that pass's frame times
    geo: Geom
    nums: np.ndarray               # that pass's frame numbers in the file
    preset: str | None = None      # what you said it is
    gray: list | None = None       # that pass's grey frames (primary only, for the clocks)
    zoom: dict | None = None
    view: dict | None = None       # a colour close-up, to name it
    shape: dict | None = None      # what its outline says (shape.classify)
    identity: dict | None = None   # what it is, and who said so
    clip: tuple | None = None      # CLIP's scores, when CLIP ran
    notes: list = field(default_factory=list)
    res: dict | None = None        # the measurement
    p1: dict | None = None         # first-pass arrays and flights
    p2: dict | None = None         # the zoom pass, if any
    first_hop: bool = False        # the measured stretch starts at a bounce, not a release

    @property
    def rigid(self) -> bool:
        idn = self.identity or {}
        if idn.get("shape") in ("round", "rigid"):
            return idn["shape"] == "rigid"
        return bool(self.shape and self.shape.get("shape") == "rigid")

    @property
    def label(self) -> str | None:
        return (self.identity or {}).get("label")


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


def _down2d(t, x, y, d_ref):
    """The picture's down for one object's fall: straight down unless its
    motion clearly says otherwise (a camera rolled more than 10 degrees).
    More than 45 degrees off only when, read upright, it doesn't fall at all."""
    g, n_ok, spread = cam_.gravity_2d(t, x, y, d_ref)
    if n_ok < 3 or spread > 20:
        return DOWN
    ang = math.degrees(math.acos(float(np.clip(g @ DOWN, -1, 1))))
    if ang <= 10:
        return DOWN
    if ang > 45 and _speeds_up_downward(t, x, y, d_ref):
        return DOWN
    return g


def _speeds_up_downward(t, x, y, d_ref) -> bool:
    """Does it clearly speed up toward the picture's bottom in any flight?"""
    for idx in split_flights(t, x, y, DOWN, diam_px=d_ref):
        if len(idx) < 8:
            continue
        fit = robust_fit_2d(t[idx], x[idx], y[idx])
        a, s_ = fit.accel_along(DOWN)
        if a > 4 * s_ and a > 0.3 * float(np.hypot(*fit.accel_vec())):
            return True
    return False


def _minor(x, y, d, f_px, cx, cy):
    """Widths across the line to the picture's centre (see camera.minor_width)."""
    ax, ay = cam_.sensor_axes(x, y, d)
    return cam_.minor_width(x, y, d, f_px, cx, cy, ax, ay)


def _sensor_rows(geo, f, x, y, rs_axis):
    """Where each point sat on the sensor along its readout direction (rolling
    shutter), in the frame it was filmed in."""
    if geo.T is None:
        return np.asarray(y if rs_axis == "y" else x, float).copy()
    out = np.empty(len(f))
    for j, (fi, xi, yi) in enumerate(zip(f, x, y)):
        a, b = geo.to_frame(int(fi), float(xi), float(yi))
        out[j] = b if rs_axis == "y" else a
    return out


def _camera_geometry(objs, info, *, hfov=None, mode="auto", notes=None, pitch=None,
                     roll=None, bg=None, bg_geo=None) -> dict:
    """Which way gravity points as the camera sees it, from every round
    object's fall in 3D (galileo/camera.py), and the focal length used.

    pitch: gravity leaning toward (-) or away from (+) the camera, degrees.
    roll: which way down points in the picture, degrees from straight down.
    The picture is levelled only for a clear pitch, or a roll over 10
    degrees that the falls agree on.
    """
    fp = cam_.focal_prior(info, hfov)
    out = {"f_frac": fp["f_frac"], "f_rel": fp["rel"], "f_source": fp["source"],
           "n": [0.0, 1.0, 0.0], "pitch": 0.0, "pitch_sigma": None, "roll": 0.0,
           "roll_sigma": None, "pitched": False, "rolled": False, "mode": mode,
           "objects": [], "why": None}
    if mode == "upright":
        out["why"] = "assumed level and upright (as asked)"
        return out
    if pitch is not None or roll is not None:
        # You said how the camera was held: gravity's direction follows.
        b = math.radians(float(pitch or 0.0))
        r_ = math.radians(float(roll or 0.0))
        n_use = np.array([math.cos(b) * math.sin(r_), math.cos(b) * math.cos(r_), math.sin(b)])
        out.update(n=[float(v) for v in n_use], pitch=float(pitch or 0.0), pitch_sigma=1.0,
                   roll=float(roll or 0.0), roll_sigma=1.0, pitched=abs(float(pitch or 0)) > 0.5,
                   rolled=abs(float(roll or 0)) > 0.5, mode="given")
        return out
    per = []
    for o in objs:
        geo = o.geo
        f_px = fp["f_frac"] * max(geo.W, geo.H)
        cx, cy = geo.W / 2.0, geo.H / 2.0
        f, t, x, y, d, w0, d_ref = _arrays(o.tr, o.times, geo, rigid=o.rigid)
        g2 = (o.p1 or {}).get("g2", DOWN)
        fl = split_flights(t, x, y, g2, diam_px=d_ref)
        if not fl:
            continue
        entry = {"n": o.n, "g2": [float(g2[0]), float(g2[1])], "flights": len(fl),
                 "points": int(sum(len(i) for i in fl)), "rigid": o.rigid}
        if not o.rigid:
            dm = _minor(x, y, d, f_px, cx, cy)
            hop = [o.first_hop] + [float(t[b[0]] - t[a[-1]]) <= 0.12 for a, b in zip(fl, fl[1:])]
            fits = []
            for j, idx in enumerate(fl):
                if not hop[j]:
                    idx = idx[t[idx] - t[idx[0]] >= 0.06]       # clear of the hand
                if len(idx) < 12 or t[idx[-1]] - t[idx[0]] < 0.15 or \
                        np.isfinite(dm[idx]).sum() < 12:
                    continue
                fits.append(cam_.fit_flight_3d(t[idx], x[idx], y[idx], dm[idx], f_px, cx, cy,
                                               d_ref, w0[idx]))
            G, cov, used, red = cam_.combine(fits)
            if G is not None and np.linalg.norm(G) > 0:
                nvec, beta, sb, roll, sr, _ = cam_.direction(G, cov)
                entry.update(G=G, cov=cov, nvec=nvec, pitch=beta, pitch_sigma=sb, roll=roll,
                             roll_sigma=sr, used=len(used), scatter=red,
                             per_flight=[None if ft is None else
                                         [round(v_, 1) for v_ in cam_.direction(ft["G"], ft["cov"])[1:5]]
                                         for ft in fits])
        per.append(entry)
    out["objects"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items()
                       if k not in ("G", "cov")} for e in per]
    f_frac = fp["f_frac"]
    # ---- what the falls say: K = tan(pitch) / f (independent of f), and the roll
    three_d = [e for e in per if "nvec" in e]
    K_t = sK_t = None
    if three_d:
        # Every object sees the same camera: combine their directions.
        w = np.array([1.0 / max(e["pitch_sigma"] ** 2 + e["roll_sigma"] ** 2, 1e-6)
                      for e in three_d])
        nv = np.sum([wi * e["nvec"] for wi, e in zip(w, three_d)], axis=0)
        nv = nv / np.linalg.norm(nv)
        sb_t = float(1.0 / np.sqrt(np.sum([1.0 / max(e["pitch_sigma"], 1e-3) ** 2
                                           for e in three_d])))
        sr_t = float(1.0 / np.sqrt(np.sum([1.0 / max(e["roll_sigma"], 1e-3) ** 2
                                           for e in three_d])))
        b_t = math.asin(float(np.clip(nv[2], -1, 1)))
        # The roll from the picture's own acceleration (galileo/camera.py's
        # gravity_2d): the 3D one leans on the widths, which can be off.
        g2 = np.asarray(per[0]["g2"], float)
        roll_t = math.degrees(math.atan2(g2[0], g2[1]))
        sr_t = max(sr_t, 2.0)
        n_fl = max(e.get("used", 1) for e in three_d)
        K_t = math.tan(b_t) / f_frac
        sK_t = math.radians(sb_t) / max(math.cos(b_t) ** 2, 1e-6) / f_frac
        out.update(K_ball=K_t, K_ball_sigma=sK_t)
    elif per:
        # Only outlines: the roll from the picture; the pitch from the room, if anywhere.
        g2 = np.asarray(per[0]["g2"], float)
        roll_t, sr_t = math.degrees(math.atan2(g2[0], g2[1])), 8.0
        n_fl = per[0]["flights"]
    else:
        return out
    # ---- what the room's straight lines say (galileo/scene.py)
    vline = hz = None
    if bg is not None:
        down = np.array([math.sin(math.radians(roll_t)), math.cos(math.radians(roll_t))])
        # Lines across the floor also meet on the down axis, at the horizon
        # (K about -1 / (f^2 K) for a tipped camera): within |K| <= 1.2 the
        # two can't be mixed up for an ordinary lens. The ball's own reading,
        # when clear, narrows the search further.
        if K_t is not None and abs(K_t) > 2 * sK_t:
            windows = [(K_t - 3 * sK_t - 0.15, K_t + 3 * sK_t + 0.15)]
        else:
            windows = [(-0.15, 1.2), (-1.2, 0.15)]
        fs = centre = None
        if bg_geo is not None:
            fs = (bg_geo.W, bg_geo.H)
            centre = (bg_geo.W / 2.0 - bg_geo.ox, bg_geo.H / 2.0 - bg_geo.oy)
        sc = {}
        for window in windows:
            try:
                got = scene.read(bg, down, K_window=window, frame_size=fs, centre=centre)
            except Exception as e:  # never lose the measurement over the scenery
                got = {"error": str(e)}
            v_ = got.get("vertical")
            if v_ and (not sc.get("vertical") or v_["lines"] > sc["vertical"]["lines"]):
                sc = got
            elif not sc:
                sc = got
        vline, hz = sc.get("vertical"), sc.get("horizon")
        out["scene"] = {"segments": sc.get("segments"), "vertical": vline, "horizon": hz}
    K, sK, src = K_t, sK_t, "the ball's changing size"
    if vline:
        Ks, sKs = float(vline["K"]), max(float(vline["K_sigma"]), 0.01)
        if K_t is None:
            K, sK, src = Ks, sKs, "the room's uprights"
        elif abs(Ks - K_t) <= 3.0 * math.hypot(sKs, sK_t):
            wa, wb = 1.0 / sKs**2, 1.0 / sK_t**2
            K, sK = (wa * Ks + wb * K_t) / (wa + wb), 1.0 / math.sqrt(wa + wb)
            src = "the room's uprights and the ball's changing size"
        else:
            K, sK, src = Ks, sKs, "the room's uprights"
            if abs(K_t) > 3 * sK_t:
                out["disagree"] = True
    # ---- the focal length from the horizon, when the camera is clearly tipped
    if hz and K is not None and abs(K) > 4 * sK and hz["h"] * K < 0 and hz["lines"] >= 8:
        ff = math.sqrt(-hz["h"] / K)
        rel = 0.5 * math.hypot(hz["h_sigma"] / abs(hz["h"]), sK / abs(K))
        if 0.25 <= ff <= 4.0 and rel < min(out["f_rel"], 0.1):
            out.update(f_frac=ff, f_rel=max(rel, 0.03),
                       f_source="the room's lines (uprights and horizon)")
            f_frac = ff
    if K is not None:
        beta = math.degrees(math.atan(f_frac * K))
        sb = math.degrees(f_frac * sK / (1.0 + (f_frac * K) ** 2))
    else:
        beta, sb = 0.0, None
    # ---- roll: gravity's own direction, checked against the room's uprights
    roll, sr = roll_t, sr_t
    room_roll = False
    if vline:
        dr = (float(vline["roll"]) - roll_t + 90.0) % 180.0 - 90.0
        room_roll = True
        # The uprights fix the camera; the fall should run along them.
        if abs(dr) >= 15.0 or sr is None or vline["roll_sigma"] < 0.5 * sr:
            roll, sr = roll_t + dr, float(vline["roll_sigma"])
        out["roll_room"] = roll_t + dr
        if three_d:
            # How far the fall itself leans from the uprights (checks.vertical_check).
            out["fall_vs_room"] = {"deg": float(-dr),
                                   "sigma": float(math.hypot(sr_t, vline["roll_sigma"]))}
    out.update(pitch=beta, pitch_sigma=sb, roll=roll, roll_sigma=sr, K=K, K_sigma=sK,
               pitch_source=src if K is not None else None)
    room = vline is not None
    min_deg = cam_.PITCH_MIN_DEG if room else cam_.PITCH_MIN_DEG_BALL
    min_z = cam_.PITCH_MIN_Z if room else cam_.PITCH_MIN_Z_BALL
    pitched = (K is not None and sb is not None and abs(beta) >= min_deg
               and abs(K) >= min_z * sK and (room or n_fl >= 2))
    if not pitched and not room and K is not None and sb is not None \
            and abs(beta) >= cam_.PITCH_MIN_DEG_BALL and abs(K) >= 3 * sK:
        out["hint"] = (f"the ball's changing size hints that the camera looks "
                       f"{'down' if beta > 0 else 'up'} about {abs(beta):.0f}°, but not clearly "
                       "enough to correct for it (give --pitch if you know it)")
    if pitched and abs(beta) > cam_.PITCH_MAX_DEG:
        out["why"] = (f"the camera looks almost straight {'down' if beta > 0 else 'up'} the fall "
                      f"({abs(beta):.0f}°), so the picture can't be turned level")
        pitched = False
    if out.get("disagree") and pitched:
        out["why"] = ("the ball's changing size and the room's uprights disagree about the "
                      "camera's angle; went with the room")
    roll_ok = abs(roll) > 10.0
    if roll_ok and abs(roll) > C.MAX_TILT_DEG:
        # Far from the picture's own down: only on solid evidence (bounces that
        # agree, one long clean fall measured to a few degrees, or the room).
        roll_ok = n_fl >= 2 or (sr is not None and sr < 4.0) or room_roll
        if roll_ok and abs(roll) > 45.0 and per and _falls_upright(objs[0]):
            # Lines can't tell a picture turned a quarter from an upright one
            # (uprights and floor lines swap), so a big turn also needs the
            # upright reading to make no sense: nothing speeding up toward
            # the picture's bottom. A generated ball that drifts sideways
            # while it falls is a fall seen upright, and stays one.
            roll_ok = False
        if not roll_ok:
            out["why"] = (f"the fall points {abs(roll):.0f}° from the picture's down, but not "
                          "clearly enough to turn the picture")
            out.pop("fall_vs_room", None)
    rr, bb = math.radians(roll), math.radians(beta)
    if pitched:
        n_use = np.array([math.cos(bb) * math.sin(rr), math.cos(bb) * math.cos(rr), math.sin(bb)])
    elif roll_ok:
        n_use = np.array([math.sin(rr), math.cos(rr), 0.0])
    else:
        n_use = np.array([0.0, 1.0, 0.0])
    out.update(n=[float(v) for v in n_use], pitched=bool(pitched), rolled=bool(roll_ok))
    return out


def _falls_upright(o) -> bool:
    """Does the object clearly speed up toward the picture's bottom in any of
    its flights (the upright reading of the clip)?"""
    f, t, x, y, d, w0, d_ref = _arrays(o.tr, o.times, o.geo, rigid=o.rigid)
    return _speeds_up_downward(t, x, y, d_ref)


def _tilt_helps(objs, camera) -> bool:
    """Do the flights agree better in the level view than as filmed?

    A real tilt makes hops at different places in the picture accelerate
    differently as filmed (perspective), and the level view puts that
    right. A tilt read from widths that are simply off does the opposite.
    """
    o = next((o for o in objs if not o.rigid), None)
    if o is None:
        return False
    f, t, x0, y0, d0, w0, d_ref0 = _arrays(o.tr, o.times, o.geo)
    rr = math.radians(camera["roll"])
    flat = {**camera, "pitched": False,
            "n": [math.sin(rr), math.cos(rr), 0.0] if camera.get("rolled") else [0.0, 1.0, 0.0]}

    def spread(cam):
        lev = _leveler(cam, o.geo)
        x, y, d, d_ref = _level(lev, x0, y0, d0, d_ref0, False, cam, o.geo)
        fl = split_flights(t, x, y, DOWN, diam_px=d_ref)
        fl = [i for i in fl if len(i) >= 12]
        if len(fl) < 2:
            return None
        m = _depth_factors(t, d, fl)[0]
        cx, cy = o.geo.W / 2.0, o.geo.H / 2.0
        xs, ys = cx + (x - cx) * m, cy + (y - cy) * m
        a = []
        for idx in fl:
            fit = robust_fit_2d(t[idx], xs[idx], ys[idx])
            av, sv = fit.accel_along(DOWN)
            a.append((av, sv))
        a = np.array(a)
        if (a[:, 0] <= 0).any():
            return None
        w = 1.0 / np.maximum(a[:, 1], 1e-9) ** 2
        mu = float(np.sum(w * a[:, 0]) / np.sum(w))
        chi = float(np.sum(w * (a[:, 0] - mu) ** 2)) / (len(a) - 1)
        return chi
    s_lev, s_flat = spread(camera), spread(flat)
    if s_lev is None:
        return False
    return s_flat is None or s_lev < s_flat


def _leveler(camera, geo):
    """The camera's levelling map in this pass's pixels."""
    if not camera:
        return cam_.Leveler(1.0, 0.0, 0.0)
    n = np.asarray(camera["n"], float)
    f_px = camera["f_frac"] * max(geo.W, geo.H)
    return cam_.Leveler(f_px, geo.W / 2.0, geo.H / 2.0, nvec=n)


def _arrays(tr, times, geo, notes=None, rigid=False):
    """Track as arrays in (global) working px.

    rigid: follow the outline's thickness-weighted centroid (each pixel counts
    by its depth into the outline, which weighs a lump by its volume: close
    to the centre of mass of a uniform solid of any shape) instead of the
    brightness centre, and use the outline's area as its size.
    """
    main = tr.main
    f = main.f
    t = np.asarray(times, float)[f]
    if rigid:
        x = main.tx + geo.ox
        y = main.ty + geo.oy
        d = 2.0 * np.sqrt(np.clip(main.area, 0, None) / np.pi)   # same-area disc
        d = np.where(d > 0, d, np.nan)
        d_ref = float(np.nanmedian(d)) if np.isfinite(d).any() else 4.0
        return f, t, x, y, d, np.ones(len(f)), d_ref
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


def _speeds(t, x, y, flights):
    """Speed (px per video-second) and direction of travel per frame, from
    each flight's fit; NaN outside flights."""
    n = len(t)
    sp = np.full(n, np.nan)
    ang = np.full(n, np.nan)
    for idx in flights:
        idx = np.asarray(idx)
        if len(idx) < 4:
            continue
        fit = robust_fit_2d(t[idx], x[idx], y[idx])
        vx, vy = fit.vel(t[idx] - fit.t0)
        sp[idx] = np.hypot(vx, vy)
        ang[idx] = np.arctan2(vy, vx)
    return sp, ang


def _shape_geometry(tr, times, t, x, y, fl, dt):
    """What the outline says: round or not (shape.classify), plus its long side."""
    main = tr.main
    stats = shp.outline_stats(main.dets)
    across = main.extent_across(times)
    speed, _ = _speeds(t, x, y, fl)
    cls = shp.classify(stats, across, t, fl, speed, dt)
    idx = np.concatenate(fl)
    L = stats["long"][idx]
    cls["long_px"] = float(np.nanpercentile(L, 90)) if np.isfinite(L).any() else None
    return cls


def _best_frame(t, fl, speed):
    """The frame to name the object from: the slowest (least blurred) one in
    a flight, a moment after the flight starts (clear of the hand)."""
    best, best_v = None, np.inf
    for idx in fl:
        idx = np.asarray(idx)
        late = idx[t[idx] - t[idx[0]] >= 0.06]
        for j in (late if len(late) else idx):
            v = speed[j] if np.isfinite(speed[j]) else 1e9
            if v < best_v:
                best, best_v = int(j), v
    return best if best is not None else int(np.concatenate(fl)[len(np.concatenate(fl)) // 2])


def _view(frames, tr, times, fl, size, keep_full=False):
    """A colour close-up of the object, to name it ("img", centred at "x", "y",
    "size" px across), and with keep_full the whole frame for Gemini and
    Depth Pro ("full", object at "fx", "fy", "fsize" px across)."""
    main = tr.main
    t = np.asarray(times, float)[main.f]
    speed, _ = _speeds(t, main.x, main.y, fl)
    j = _best_frame(t, fl, speed)
    det = main.dets[j]
    img = frames[det.f]
    if img.ndim != 3:
        return None
    half = 1.6 * max(size, 6.0) + 8
    crop = identify._square_crop(img, det.x, det.y, half)
    hh = crop.shape[0] / 2.0
    return {"img": crop, "x": hh, "y": hh, "size": float(max(size, 6.0)),
            "full": img.copy() if keep_full else None, "fx": float(det.x), "fy": float(det.y),
            "fsize": float(max(size, 6.0)), "t": float(t[j])}


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
    # The tracker works in the steadied window's own pixels.
    return [(float(times[fi]), float(xs) - geo.ox, float(ys) - geo.oy)]


def _track(tracker, frames, times, roi_px, seed_pts, sam2_box, max_objects=C.MAX_OBJECTS,
           footprints=None):
    if tracker == "sam2":
        from .sam2_tracker import track_with_sam2
        seed1 = None
        if seed_pts:
            ts, xs, ys = seed_pts[0]
            seed1 = (int(np.argmin(np.abs(np.asarray(times) - ts))), xs, ys)
        tr = track_with_sam2(frames, times, box=sam2_box, seed=seed1)
        tr.seeded = seed1 is not None or sam2_box is not None
        return tr
    return track_falling_object(frames, times, roi=roi_px, seed=seed_pts, max_objects=max_objects,
                                footprints=footprints)


def _preset_list(preset) -> list:
    if preset is None:
        return []
    if isinstance(preset, str):
        preset = [p for p in preset.replace(";", ",").split(",")]
    return [None if (p is None or str(p).strip().lower() in ("", "auto", "none")) else str(p).strip()
            for p in preset]


# ---------------------------------------------------------------------------
def analyze(path: str, *, debug_video: str | None = None, debug_speed: float = 1.0,
            progress: Callable[[str], None] | None = None, **kw) -> dict:
    """Measure one clip.

    Keyword options: preset, size_m, size_rel_unc, label, roi, t_start, t_end,
    manual_ruler, use_gemini, use_depth, tracker, sam2_box, stabilize_mode,
    rs_mode, g_ref, scale_correction, and:

    preset: what the object is (a key of config.OBJECT_PRESETS), or a list
    of them, one per object; "auto" or None lets Galileo work it out.
    seed: (t, x, y), a point on the ball at video time t; x, y as fractions of
    the picture (or original pixels). It picks the primary object. zoom:
    allow the full-resolution second pass. drag_vt: the ball's terminal
    speed in m/s for the air-drag correction (presets carry one).
    identify: "auto" (Gemini if use_gemini, else CLIP if installed), "clip",
    "gemini" or "off". max_objects: how many falling objects to measure.
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


def _run(path: str, *, dbg: dict | None = None, preset=None,
         size_m: float | None = None, size_rel_unc: float = 0.01, label: str | None = None,
         roi=None, t_start: float | None = None, t_end: float | None = None,
         manual_ruler=None, use_gemini: bool = False, use_depth: bool = False,
         tracker: str = "classical", sam2_box=None, stabilize_mode: str = "auto",
         rs_mode: str = "auto", g_ref: float = C.G_EARTH, scale_correction: float = 1.0,
         seed=None, zoom: bool = True, drag_vt: float | None = None, identify_mode: str = "auto",
         max_objects: int = C.MAX_OBJECTS, hfov: float | None = None, camera_mode: str = "auto",
         camera_pitch: float | None = None, camera_roll: float | None = None,
         real_time: bool | None = None,
         progress: Callable[[str], None] | None = None) -> dict:
    t_wall = time.time()
    real_time = C.ASSUME_REAL_TIME if real_time is None else bool(real_time)
    say = progress or (lambda s: None)
    info = probe(path)
    result: dict = {"file": info.to_dict(), "notes": [], "version": 3}
    presets = _preset_list(preset)

    # ---- first pass: the whole picture ------------------------------------------
    fps0 = float(info.fps or 30.0)
    n_est = info.n_frames or int(fps0 * info.duration) or C.MAX_FRAMES
    span_n = n_est
    if t_start is not None or t_end is not None:
        t_a = t_start or 0.0
        t_b = t_end if t_end is not None else (info.duration or n_est / fps0)
        span_n = max(1, int(round((t_b - t_a) * fps0)) + 1)
    step = 1
    if span_n > C.MAX_FRAMES:
        # A long clip: one frame in `step` of all of it (a moving camera makes
        # "where the motion is" a poor guess, and the first bounce stays in),
        # while that leaves MIN_FPS_LONG frames a second. Past that, the
        # stretch with the most motion.
        step = max(1, min(int(np.ceil(span_n / C.MAX_FRAMES)), int(fps0 // C.MIN_FPS_LONG)))
        if t_start is None and t_end is None and span_n > C.MAX_FRAMES * step:
            say("finding the motion")
            t_start, t_end = motion_window(path, info, C.MAX_FRAMES * step)
            if t_start is not None:
                result["notes"].append(f"Long clip: analysed {t_start:.2f}-{t_end:.2f} s, where "
                                       "the motion is.")
        if step > 1:
            result["notes"].append(f"Long clip: used one frame in {step} "
                                   f"({fps0 / step:.0f} a second).")
    n_use = min(int(np.ceil(span_n / step)), C.MAX_FRAMES)
    sc = work_scale(info, n_use, C.WORK_MAX_SIDE, C.FRAME_BUDGET_BYTES)
    say("decoding")
    frames, times, nums = read_frames(path, info, scale=sc, t_start=t_start, t_end=t_end,
                                      max_frames=C.MAX_FRAMES, step=step)
    if len(frames) < 6:
        return _unsure(result, "The clip is too short to measure.", t_wall,
                       stage="decode: too short")
    Hw, Ww = frames[0].shape[:2]
    dt_med = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / (info.fps or 30)
    fps_file = 1.0 / dt_med if dt_med > 0 else (info.fps or 30.0)
    result["work"] = {"scale": sc, "width": Ww, "height": Hw,
                      "frames": len(frames), "fps_measured": fps_file, "step": step}

    say("stabilizing")
    frames_s, T, stab = stabilize(frames, stabilize_mode)
    del frames
    foot = stab.pop("footprints", None)
    result["stabilization"] = {k: v for k, v in stab.items()}
    cw, chh, cox, coy = stab.get("canvas") or (Ww, Hw, 0, 0)
    geo = Geom(W=Ww, H=Hw, scale=sc, T=T, ox=-float(cox), oy=-float(coy), fox=0.0, foy=0.0)
    if cox or coy or (cw, chh) != (Ww, Hw):
        result["notes"].append(
            f"The camera moved a lot ({stab.get('model')}); frames were laid onto a "
            f"{cw}x{chh} canvas around frame {stab.get('reference', 0) + 1}.")
    if stab.get("ignored"):
        result["notes"].append(
            "The background is too plain to see whether the camera moved (the only detail "
            "to follow was the object itself), so the camera was taken as still.")
    roi_px = None
    if roi is not None:
        roi_px = (int(roi[0] * Ww + cox), int(roi[1] * Hw + coy), int(roi[2] * Ww + cox),
                  int(roi[3] * Hw + coy))
    seed_pts = _seed_points(seed, info, times, geo)
    if seed is not None and not seed_pts:
        result["notes"].append("The ball pick is outside the part of the clip that was "
                               "measured, so it was ignored.")
    say("tracking")
    tr = _track(tracker, frames_s, times, roi_px, seed_pts, sam2_box, max(1, int(max_objects)),
                footprints=foot)
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

    objs: list[_Obj] = []
    for k, tro in enumerate((tr.objects or [tr])[:max(1, int(max_objects))]):
        f, t, x, y, d, w0, d_ref = _arrays(tro, times, geo)
        g2 = _down2d(t, x, y, d_ref)
        fl = split_flights(t, x, y, g2, diam_px=d_ref)
        if not fl:
            if k == 0:
                return _unsure(result, "The object moves, but no clean ballistic flight was "
                                       "found.", t_wall, stage="tracking: no flight",
                               track=_track_payload(tro.main, times, geo))
            continue
        o = _Obj(n=len(objs) + 1, tr=tro, times=times, geo=geo, nums=nums)
        o.shape = _shape_geometry(tro, times, t, x, y, fl, dt_med)
        size_v = max(d_ref, o.shape.get("long_px") or 0.0) if o.shape["shape"] == "rigid" \
            else d_ref
        o.view = _view(frames_s, tro, times, fl, size_v, keep_full=use_gemini or use_depth)
        o.p1 = {"f": f, "t": t, "x": x, "y": y, "fl": fl, "d_ref": d_ref, "size": size_v,
                "scale": sc, "tr": tro, "g2": g2}
        objs.append(o)

    # What only the whole picture gives: the light level, for the flicker clock,
    # and the still background, for the room's straight lines.
    bg1 = background(frames_s) if camera_mode != "upright" else None
    gray1 = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in frames_s]
    bright = [float(np.median(g[::4, ::4])) for g in gray1]
    rs_axis = "x" if info.rotation in (90, 270) else "y"
    del frames_s

    # Your names first: they decide which object is the primary one.
    clip_ok, clip_why = (identify.clip_available() if identify_mode in ("auto", "clip")
                         else (False, ""))
    if clip_ok and (len(objs) > 1 or not presets):
        clip_ok = _clip_scores(objs, result["notes"])
    _assign_names(objs, presets, result["notes"], size_m=size_m, size_rel_unc=size_rel_unc,
                  label=label)

    # ---- second pass: zoom in on small objects ----------------------------------
    if zoom:
        for o in objs:
            p1 = o.p1
            plan = _zoom_plan(info, geo, times, p1["t"], p1["x"], p1["y"], p1["f"], p1["fl"],
                              p1["d_ref"], extent=p1["size"])
            if not plan:
                continue
            plan["step"] = step                  # the same frames as the first pass
            say("zooming in on the ball" if len(objs) == 1 else f"zooming in on object {o.n}")
            z = _zoom_pass(path, info, plan, geo, times, o.tr, tracker, want_gray=o.n == 1,
                           down=p1["g2"])
            if z is None:
                o.notes.append(f"Tried to zoom in on the {p1['d_ref']:.0f} px object but lost "
                               "it, so it was measured in the first pass.")
                continue
            gray_z, times_z, geo_z, tr_z, nums_z, view_z = z
            f2, t2, x2, y2, d2, w02, d_ref2 = _arrays(tr_z, times_z, geo_z)
            fl2 = split_flights(t2, x2, y2, p1["g2"], diam_px=d_ref2)
            # The window ends just after the last flight it was planned for; a hop
            # that starts after that is cut short by the window, not the clip.
            if plan["t1"] < float(times[-1]) - 1e-6:
                fl2 = [q for q in fl2 if t2[q[0]] <= plan["t_last"] + 1.5 * dt_med]
            # Likewise at the start, when the window opens partway through the bounces.
            if plan["first_flight"] > 0:
                fl2 = [q for q in fl2 if t2[q[-1]] >= plan["t_first"] - 1.5 * dt_med]
            if not fl2:
                o.notes.append("The zoomed-in pass found no clean flight, so the first pass "
                               "was used.")
                continue
            o.first_hop = plan["first_flight"] > 0
            o.zoom = {"scale": geo_z.scale, "frames": len(times_z),
                      "crop_px": [int(round(v / geo_z.scale)) for v in plan["crop"]],
                      "t0": float(times_z[0]), "t1": float(times_z[-1]),
                      "ball_px_before": p1["d_ref"], "ball_px": d_ref2}
            o.p2 = {"geo": geo_z, "times": times_z, "nums": nums_z, "tr": tr_z,
                    "crop": plan["crop"]}
            o.tr, o.times, o.geo, o.nums, o.gray = tr_z, times_z, geo_z, nums_z, gray_z
            sh2 = _shape_geometry(tr_z, times_z, t2, x2, y2, fl2, dt_med)
            if sh2.get("frames", 0) >= 3:
                o.shape = sh2
            if view_z is not None:
                old = o.view or {}
                o.view = {**view_z, **{k_: old.get(k_) for k_ in ("full", "fx", "fy", "fsize")}}
                o.clip = None          # look again, closer
            if o.n == 1 and dbg is not None:
                dbg["pass2"] = {"geo": geo_z, "times": times_z, "nums": nums_z, "tr": tr_z,
                                "crop": plan["crop"]}
    if objs[0].gray is None:
        objs[0].gray = gray1
    del gray1

    # ---- what is each object? -----------------------------------------------------
    if any(o.identity is None for o in objs):
        say("naming the objects" if len(objs) > 1 else "naming the object")
    if clip_ok:
        clip_ok = _clip_scores([o for o in objs if o.identity is None and o.clip is None],
                               result["notes"])
    _identify_rest(objs, use_gemini=use_gemini, mode=identify_mode, clip_ok=clip_ok,
                   clip_why=clip_why, notes=result["notes"])
    objs = _credible(objs, result["notes"])

    # ---- the camera: which way is down, as it sees it ------------------------------
    say("working out the camera angle")
    camera = _camera_geometry(objs, info, hfov=hfov, mode=camera_mode, notes=result["notes"],
                              pitch=camera_pitch, roll=camera_roll, bg=bg1, bg_geo=geo)
    del bg1
    if camera.get("pitched") and camera.get("pitch_source") == "the ball's changing size":
        # Read from the ball alone: keep the tilt only if the falls agree
        # better with it than without it.
        if not _tilt_helps(objs, camera):
            flat = dict(camera)
            rr = math.radians(camera["roll"])
            flat.update(pitched=False, n=[math.sin(rr), math.cos(rr), 0.0] if camera["rolled"]
                        else [0.0, 1.0, 0.0],
                        hint=f"the ball's changing size hints that the camera looks "
                             f"{'down' if camera['pitch'] > 0 else 'up'} about "
                             f"{abs(camera['pitch']):.0f}°, but the falls don't agree better "
                             "with that, so it was left out")
            camera = flat
    result["camera"] = _camera_payload(camera)
    if camera.get("pitched"):
        b = float(camera["pitch"])
        result["notes"].append(
            f"The camera looks {'down' if b > 0 else 'up'} at the fall by about {abs(b):.0f}° "
            f"(± {max(float(camera['pitch_sigma'] or 0), 0.5):.0f}°), from "
            f"{camera.get('pitch_source') or 'the fall'}. Measured in a level view of the same "
            f"scene; focal length: {camera['f_source']}.")
    elif camera.get("rolled"):
        result["notes"].append(f"The camera is rolled about {abs(camera['roll']):.0f}°; down was "
                               "taken from the fall itself.")
    if camera.get("why"):
        result["notes"].append(f"Camera: {camera['why']}.")

    # ---- measure each one -----------------------------------------------------------
    shared = {"info": info, "fps_file": fps_file, "dt_med": dt_med, "rs_axis": rs_axis,
              "g_ref": g_ref, "rs_mode": rs_mode, "scale_correction": scale_correction,
              "bright": bright, "camera": camera, "real_time": real_time,
              "fps_native": fps_file * step}
    clock = None
    keep_nums = None
    for o in objs:
        if len(objs) > 1:
            say(f"measuring object {o.n}")
        prim = o.n == 1
        o.res = _measure(o, shared, clock=clock, keep_nums=keep_nums, say=say,
                         manual_ruler=manual_ruler if prim else None,
                         use_depth=use_depth, drag_vt=drag_vt if prim else None,
                         want_dbg=dbg is not None)
        if prim:
            clock = o.res["clock_raw"]
            keep = clock.get("keep_frames")
            if keep:
                keep_nums = set(int(v) for v in np.asarray(o.nums)[np.asarray(keep, int)])

    # Other objects Galileo isn't sure about, and that don't fall, are noise.
    kept = [objs[0]]
    for o in objs[1:]:
        st = str(o.res["verdict"].get("stage") or "")
        if not _sure_of(o.identity) and (st.startswith("tracking") or (o.res["fit"]["a_px"] <= 0)):
            continue
        kept.append(o)
    if len(kept) < len(objs):
        n_drop = len(objs) - len(kept)
        result["notes"].append(f"Ignored {n_drop} more moving thing{'s' if n_drop > 1 else ''} "
                               "that didn't fall like an object.")
        objs = kept
        for n, o in enumerate(objs, 1):
            o.n = n

    # ---- one verdict for the clip --------------------------------------------------------
    p = objs[0]
    pr = p.res
    if len(objs) > 1:
        eq = checks.equal_falls_check_g([_g_entry(o) for o in objs])
        pr["tests"] = [eq if t_["name"] == "Equal falls" else t_ for t_ in pr["tests"]]
        for o in objs[1:]:
            o.res["tests"] = [eq if t_["name"] == "Equal falls" else t_ for t_ in o.res["tests"]]
    overall = combine_objects([_summary_for_verdict(o) for o in objs],
                              cross=pr["tests"])
    if dbg is not None:
        dbg["pass1"]["tr"] = p.p1["tr"]          # the primary may have changed with your names
        dbg["final"] = pr.get("dbg")
        dbg["objects"] = [{"n": o.n, "label": o.label, "rigid": o.rigid, "final": o.res.get("dbg"),
                           "verdict": o.res["verdict"], "p2": o.p2} for o in objs]
    result["notes"].extend(p.notes)
    result.update({
        "verdict": overall,
        "tests": pr["tests"],
        "clock": _clean(pr["clock_raw"]),
        "scale": pr["scale"],
        "drag": pr["drag"],
        "gemini_object": pr.get("gemini_object"),
        "fit": pr["fit"],
        "track": pr["track"],
        "plot": pr["plot"],
        "ghost": pr["ghost"],
        "objects": [_object_payload(o) for o in objs],
    })
    if p.zoom:
        result["zoom"] = p.zoom
    if use_gemini and overall["verdict"] != "UNSURE":
        try:
            from . import gemini
            text = gemini.explain(result)
            if text:
                overall["explanation"] = text
        except Exception as e:
            result["notes"].append(f"Gemini explanation unavailable: {e}")
    result["elapsed_s"] = time.time() - t_wall
    say("done")
    return result


# ---------------------------------------------------------------------------
def _clip_scores(objs, notes) -> bool:
    """CLIP's scores (and picture embedding) for each object's close-up.
    False if CLIP can't run."""
    for o in objs:
        if o.view is None:
            continue
        try:
            o.clip = identify.clip_probs(o.view["img"], o.view["x"], o.view["y"], o.view["size"])
        except Exception as e:  # weights missing offline, out of memory...
            notes.append(f"Object recognition (CLIP) unavailable: {e}")
            for o2 in objs:
                o2.clip = None
            return False
    return True


def _assign_names(objs, presets, notes, *, size_m=None, size_rel_unc=0.01, label=None):
    """Your names (--object), anything at all with an optional size
    ("rotisserie chicken=22cm"), matched to the objects they fit best: by
    CLIP when it ran (it reads any words), else by size (bigger objects get
    the bigger known sizes), else in Galileo's order. The object you named
    first becomes the primary, unless you clicked one."""
    named = [p for p in presets if p]
    idns = [identify.from_name(p) for p in named]
    for idn in idns:
        if idn.get("key") is None and not idn.get("size_m"):
            notes.append(f"No size known for '{idn['label']}': give one like "
                         f"\"{idn['label']}=22cm\" for a ruler (or turn on Gemini).")
    clip_all = len(objs) > 1 and all(o.clip for o in objs)
    seeded = bool(getattr(objs[0].tr, "seeded", False))
    if not idns:
        where = []
    elif clip_all:
        try:
            logp = identify.clip_name_logp([o.clip[3] for o in objs], [i["name"] for i in idns])
        except Exception:
            logp = None
        if logp is None:
            where = list(range(min(len(idns), len(objs)))) + [None] * max(0, len(idns) - len(objs))
        elif len(idns) == 1:
            # One name: it's for the fall Galileo tracked best, unless CLIP is
            # sure another object is clearly the one you named.
            p = np.exp(logp[:, 0])
            j = int(np.argmax(p))
            where = [j if (j != 0 and not seeded and p[0] < 0.05 and p[j] >= 0.3
                           and p[j] > 5 * p[0]) else 0]
        else:
            where = identify.match_names([i["name"] for i in idns], logp)
    elif len(idns) > 1 and all(i.get("size_m") for i in idns):
        # No CLIP: bigger objects get the bigger names (assuming they sit at
        # similar distances from the camera).
        by_size = sorted(range(len(objs)),
                         key=lambda j: -(objs[j].p1["size"] / objs[j].p1["scale"]))
        by_name = sorted(range(len(idns)), key=lambda q: -idns[q]["size_m"])
        where = [None] * len(idns)
        for q, j in zip(by_name, by_size):
            where[q] = j
    else:
        where = list(range(min(len(idns), len(objs)))) + [None] * max(0, len(idns) - len(objs))
    for idn, j in zip(idns, where):
        if j is None:
            notes.append(f"Couldn't find the {idn['label']} you named among the falling objects.")
            continue
        objs[j].identity = idn
    # The best-tracked object you named is the primary, unless you clicked one.
    if idns and objs[0].identity is None and not seeded:
        j0 = next((j for j, o in enumerate(objs) if o.identity is not None), 0)
        objs.insert(0, objs.pop(j0))
        for n, o in enumerate(objs, 1):
            o.n = n
    if size_m and (objs[0].identity is None or not objs[0].identity.get("size_m")):
        prev = objs[0].identity or {}
        objs[0].identity = {"key": prev.get("key"), "label": prev.get("label") or label or "object",
                            "size_m": float(size_m), "rel_unc": float(size_rel_unc),
                            "vt": prev.get("vt"), "shape": prev.get("shape"), "source": "you",
                            "confidence": 1.0, "detail": ""}


def _identify_rest(objs, *, use_gemini, mode, clip_ok, clip_why, notes):
    """Name the objects you didn't: Gemini, then CLIP, then look-alikes."""
    mode = (mode or "auto").lower()
    for o in objs:
        mine = o.identity
        if mine is not None and mine.get("size_m"):
            continue
        v = o.view or {}
        if use_gemini and mode in ("auto", "gemini") and v.get("full") is not None:
            try:
                g = identify.gemini_identify(v["full"], (v["fx"], v["fy"], v["fsize"]), o.shape,
                                             hint=(mine or {}).get("label"))
            except Exception as e:  # network, quota, missing key
                notes.append(f"Gemini object ID unavailable: {e}")
                g = None
            if g and mine:
                # Your name, Gemini's estimate of its size.
                if g.get("size_m"):
                    o.identity = {**mine, "size_m": g["size_m"], "rel_unc": g.get("rel_unc"),
                                  "vt": mine.get("vt") or g.get("vt"),
                                  "shape": mine.get("shape") or g.get("shape"),
                                  "detail": f"size estimated by Gemini; {g.get('detail', '')}"}
                continue
            if g:
                o.identity = g
            if o.identity and o.identity.get("size_m"):
                continue
        if mine is None and o.clip is not None:
            o.identity = identify.clip_identify(None, 0, 0, 0, o.shape, probs=o.clip)
    # Look-alikes: a second ball the same size and shape as a named one.
    infos = []
    for o in objs:
        size = (o.shape or {}).get("long_px") if o.rigid else o.p1["d_ref"]
        scale1 = o.p1.get("scale") or o.geo.scale
        infos.append({"identity": o.identity, "shape": "rigid" if o.rigid else "round",
                      "size_orig_px": (size / scale1) if size else None})
    for j, o in enumerate(objs):
        if o.identity and (o.identity.get("size_m") or o.identity.get("rejected")
                           or o.identity.get("source") == "you"):
            continue
        same = identify.same_as(infos, j)
        if same:
            o.identity = same
            infos[j]["identity"] = same
    if not clip_ok and mode in ("auto", "clip") and clip_why and any(
            not (o.identity or {}).get("size_m") for o in objs):
        notes.append(f"To name objects automatically, {clip_why}.")


def _credible(objs, notes):
    """Keep the primary, and other objects that are really objects: named
    ones, or compact things with a good long fall. Drops the wall edges,
    arms and shadows a moving camera or a person can turn into tracks."""
    keep = [objs[0]]
    dropped = 0
    for o in objs[1:]:
        idn = o.identity or {}
        if idn.get("rejected"):
            dropped += 1
            continue
        if not _sure_of(idn):
            # Unnamed, or only a hesitant guess: it has to look like an object.
            sh = o.shape or {}
            fl = o.p1["fl"]
            longest = max(len(i) for i in fl)
            size = max(o.p1["size"], 1.0)
            g2 = o.p1.get("g2", DOWN)
            u = (o.p1["x"] * g2[0] + o.p1["y"] * g2[1])[np.concatenate(fl)]
            if (sh.get("aspect") or 1.0) > 2.5 or longest < 15 or np.ptp(u) < 3 * size:
                dropped += 1
                continue
        keep.append(o)
    for n, o in enumerate(keep, 1):
        o.n = n
    if dropped:
        notes.append(f"Ignored {dropped} other moving thing{'s' if dropped > 1 else ''} that "
                     "didn't look like a falling object (an edge, an arm, a shadow).")
    return keep


def _sure_of(idn) -> bool:
    """Is Galileo confident what this object is (you said, or a sure guess)?"""
    if not idn or not idn.get("label") or idn.get("rejected"):
        return False
    if idn.get("source") == "you" or str(idn.get("source", "")).startswith("same as"):
        return True
    return float(idn.get("confidence") or 0) >= 0.5


def _g_entry(o):
    v = o.res["verdict"]
    return {"n": o.n, "label": o.label, "g": v.get("g"), "sigma": v.get("sigma"),
            "a_px": o.res["fit"]["a_px"], "a_sigma": o.res["fit"]["a_sigma"],
            "ok": v.get("stage") not in (None,) and not str(v.get("stage", "")).startswith(
                "tracking")}


def _summary_for_verdict(o):
    idn = o.identity or {}
    return {"n": o.n, "label": o.label, "verdict": o.res["verdict"],
            "source": idn.get("source"), "named": bool(idn.get("label")),
            "strong": idn.get("source") == "you", "n_points": o.res["fit"].get("n_points", 0)}


def _object_payload(o):
    r = o.res
    v = r["verdict"]
    idn = dict(o.identity or {})
    sh = dict(o.shape or {})
    return {
        "n": o.n, "label": o.label, "primary": o.n == 1,
        "shape": "rigid" if o.rigid else "round",
        "outline": {k: sh.get(k) for k in ("shape", "sure", "why", "circularity", "aspect",
                                           "width_scatter", "long_px")},
        "identity": {k: idn.get(k) for k in ("key", "label", "size_m", "rel_unc", "vt", "source",
                                             "confidence", "detail")} if idn else None,
        "verdict": v, "g": v.get("g"), "sigma": v.get("sigma"),
        "tests": r["tests"], "scale": r["scale"], "drag": r["drag"], "fit": r["fit"],
        "track": r["track"], "spin": r.get("spin"), "zoom": o.zoom, "notes": list(o.notes),
        "view_t": (o.view or {}).get("t"),
    }


def _camera_payload(camera):
    keep = ("f_frac", "f_rel", "f_source", "pitch", "pitch_sigma", "roll", "roll_sigma",
            "pitched", "rolled", "mode", "n", "why", "objects", "K", "K_sigma", "pitch_source",
            "K_ball", "K_ball_sigma", "scene", "disagree", "hint", "roll_room",
            "fall_vs_room")
    return {k: camera.get(k) for k in keep}


# ---------------------------------------------------------------------------
def _measure(o: _Obj, sh: dict, *, clock=None, keep_nums=None, say=None, manual_ruler=None,
             use_depth=False, drag_vt=None, want_dbg=False) -> dict:
    """Fit, clocks, rulers, air drag, checks and verdict for one object.

    The primary object (clock=None) reads the clocks; the others share its k,
    since a clip has one speed.
    """
    say = say or (lambda s_: None)
    notes = o.notes
    info, fps_file, dt_med, rs_axis = sh["info"], sh["fps_file"], sh["dt_med"], sh["rs_axis"]
    g_ref, rs_mode = sh["g_ref"], sh["rs_mode"]
    camera = sh.get("camera")
    tr, times, geo = o.tr, o.times, o.geo
    idn = o.identity or {}
    rigid = o.rigid
    # x0, y0, d0: as filmed (stabilized). x, y, d: in the level view of the
    # camera (the same unless it looks down, up, or is rolled), where gravity
    # points straight down the picture.
    lev = _leveler(camera, geo)
    f, t, x0, y0, d0, w0, d_ref0 = _arrays(tr, times, geo, notes, rigid=rigid)
    x, y, d, d_ref = _level(lev, x0, y0, d0, d_ref0, rigid, camera, geo)
    fl = split_flights(t, x, y, DOWN, diam_px=d_ref)
    if not fl and rigid:
        # The outline's centre wobbles more than the bright core: fall back.
        rigid = False
        f, t, x0, y0, d0, w0, d_ref0 = _arrays(tr, times, geo, notes)
        x, y, d, d_ref = _level(lev, x0, y0, d0, d_ref0, rigid, camera, geo)
        fl = split_flights(t, x, y, DOWN, diam_px=d_ref)
    rows = _sensor_rows(geo, f, x0, y0, rs_axis)
    k_lev = lev.widths(x0, y0, np.ones(len(x0)))      # level view px per filmed px, per point
    stats = shp.outline_stats(tr.main.dets) if rigid else None
    length_px = None
    if rigid and stats is not None and np.isfinite(stats["long"]).any():
        length_px = float(np.nanpercentile(stats["long"] * k_lev, 90))

    # ---- fit ----------------------------------------------------------------------
    say("fitting")
    gdir = DOWN.copy()
    if rigid:
        m_fac, depth_change = np.ones(len(t)), 0.0   # its size changes as it turns
    else:
        m_fac, depth_change = _depth_factors(t, d, fl)
    cxp, cyp = geo.W / 2.0, geo.H / 2.0
    xs = cxp + (x - cxp) * m_fac
    ys = cyp + (y - cyp) * m_fac
    spin = None
    if rigid and fl:
        speed, _ = _speeds(t, xs, ys, fl)
        sf = shp.spin_fit(t, xs, ys, stats, fl, length_px or d_ref, speed=speed, dt=dt_med)
        if sf is not None:
            xs_c, ys_c, spin = sf
            if spin["applied"]:
                xs, ys = xs_c, ys_c
                notes.append(f"It spins ({spin['spin_rev_s']:.1f} turns a second) with its mass "
                             f"{spin['offset_px']:.0f} px off its outline's centre; followed the "
                             "centre of mass instead.")
    n_axis = geo.W if rs_axis == "x" else geo.H
    fit_kw = dict(fps=fps_file, n_px_axis=n_axis, rs_axis=rs_axis, rs_mode=rs_mode, gdir=gdir,
                  r=rows)
    jf = fit_flights(t, xs, ys, w0, fl, **fit_kw)
    gdir, tilt_note = _aim_gravity(jf)
    if tilt_note and not (camera or {}).get("why"):
        notes.append(tilt_note)
    if gdir is not DOWN and not np.allclose(gdir, DOWN):
        fl = split_flights(t, xs, ys, gdir, diam_px=d_ref) or fl
        fit_kw["gdir"] = gdir
        jf = fit_flights(t, xs, ys, w0, fl, **fit_kw)
    # The first flight starts in the hand: cut the push, and if the ball was
    # thrown, let the bounces measure gravity instead.
    fl, launch = _launch(t, xs, ys, fl, jf, gdir, first_hop=o.first_hop)
    fit_kw["exclude"] = launch["exclude"]
    jf = fit_flights(t, xs, ys, w0, fl, **fit_kw)
    rs_rel = rolling_shutter_effect(jf, t, xs, ys, w0)
    naive = {"a_px": jf.a_px, "a_sigma": jf.a_sigma}

    # ---- clocks -------------------------------------------------------------------
    if clock is None:
        say("reading clocks")
        # The file's own frame rate (a long clip may be read one frame in N).
        ev = clocks.metadata_clock({"fps": sh.get("fps_native", fps_file),
                                    "capture_fps": info.capture_fps})
        mf = jf.flights[jf.main]
        mi = mf.idx
        vx, vy = mf.fit.vel(t[mi] - mf.fit.t0)
        # Expected motion per frame, only where the fit is valid (inside the
        # flight), back in filmed pixels.
        back = m_fac[mi] * np.where(np.isfinite(k_lev[mi]) & (k_lev[mi] > 0), k_lev[mi], 1.0)
        disp = {int(fi): float(np.hypot(a, b) * dt_med / bk)
                for fi, a, b, bk in zip(f[mi], vx, vy, back)}
        if o.gray is not None:
            ev.append(clocks.duplicate_clock(o.gray, f, x0 - geo.ox, y0 - geo.oy, d_ref0, disp))
        u_all = xs * gdir[0] + ys * gdir[1]
        ev.append(clocks.interpolation_clock(f[mi], u_all[mi]))
        ev.append(clocks.flicker_clock(sh["bright"], fps_file))
        clock = clocks.combine([e for e in ev if e], fps_file)
        if sh.get("real_time"):
            clock = clocks.real_time(clock)
            if clock.get("ignored_k"):
                kk, src = clock["ignored_k"]
                notes.append(f"A slow-motion clue ({src}) points to {kk:.1f}x slow motion, but "
                             "clips are taken as playing in real time.")
        keep = clock.get("keep_frames")
        sel = np.isin(f, np.asarray(keep)) if keep else None
    else:
        sel = (np.isin(np.asarray(o.nums)[f], np.asarray(sorted(keep_nums)))
               if keep_nums else None)
    o.gray = None

    refit_note = None
    if sel is not None and clock["confidence"] >= C.CLOCK_MIN_CONFIDENCE and sel.sum() >= 6:
        ii = np.flatnonzero(sel)
        fl2 = split_flights(t[ii], xs[ii], ys[ii], gdir, diam_px=d_ref)
        if fl2:
            fl2 = [ii[s_] for s_ in fl2]
            fit_kw["fps"] = fps_file / clock["k"]
            jf2 = fit_flights(t, xs, ys, w0, fl2, **{**fit_kw, "exclude": ()})
            fl2, launch = _launch(t, xs, ys, fl2, jf2, gdir, first_hop=o.first_hop)
            fit_kw["exclude"] = launch["exclude"]
            jf = fit_flights(t, xs, ys, w0, fl2, **fit_kw)
            fl = fl2
            refit_note = (f"Refit on the {len(ii)} original frames "
                          f"({clock['source']}).")

    # ---- rulers ---------------------------------------------------------------------
    say("measuring scale")
    rulers = []
    obj_label = idn.get("label")
    used = np.concatenate([fl_i for fl_i in fl])
    size_m_ = idn.get("size_m")
    rel_ = idn.get("rel_unc") if idn.get("rel_unc") is not None else 0.1
    vt = drag_vt or idn.get("vt")
    rigid_info = None
    if rigid:
        speed, ang = _speeds(t, xs, ys, fl)
        st_len = stats
        st_len = {**stats, "long": np.asarray(stats["long"], float) * k_lev}
        L_fine = getattr(tr, "L_fine", None)
        if L_fine is not None and np.isfinite(L_fine).sum() >= 8:
            st_len = {**stats, "long": np.asarray(L_fine, float) * k_lev}   # measured at half contrast
        samples, rigid_info = shp.rigid_length(st_len, t, fl, speed, ang, dt_med)
        if rigid_info.get("note"):
            notes.append(f"Length: {rigid_info['note']}.")
        rel_ = max(rel_, C.RIGID_MIN_REL_UNC)
    else:
        d_norm = d * m_fac                 # widths at the reference distance
        samples = [v for v in d_norm[used] if np.isfinite(v)]
        if len(samples) < 5:
            samples = [v for v in d_norm if np.isfinite(v)]
        if tr.rest_d and tr.rest is not None:
            rx_, ry_ = tr.rest.x + geo.ox, tr.rest.y + geo.oy
            rd = _level_width(lev, camera, geo, rx_, ry_, tr.rest_d)
            if np.isfinite(rd):
                samples += [rd] * 3
    if size_m_:
        r = ruler_from_size(samples, size_m_, rel_, obj_label or "object")
        if r:
            src = idn.get("source")
            if rigid and rigid_info:
                r.detail = (f"{obj_label or 'object'} = {size_m_ * 1000:.0f} mm long across "
                            f"{rigid_info['length_px']:.1f} px (its longest side as seen)")
            if src and src != "you":
                r.detail += f"; size from {src}"
            rulers.append(r)
    if manual_ruler:
        (p1, p2, L) = manual_ruler
        g_s = geo.scale
        (ax_, bx_), (ay_, by_), _ = lev.forward(np.array([p1[0], p2[0]]) * g_s,
                                                 np.array([p1[1], p2[1]]) * g_s)
        rulers.append(ruler_manual((ax_, ay_), (bx_, by_), L))
    if use_depth and o.view is not None and o.view.get("full") is not None:
        try:
            from .depth import depth_pro_ruler
            dr = depth_pro_ruler(o.view["full"], (o.view["fx"], o.view["fy"]), o.view["fsize"])
            if dr:
                # Measured on the first pass; re-express per analysed pixel (of
                # the level view).
                kv = float(np.nanmedian(k_lev)) if np.isfinite(k_lev).any() else 1.0
                rulers.append(Ruler(name=dr.name, s=dr.s * o.p1["scale"] / geo.scale / kv,
                                    rel_sigma=dr.rel_sigma, detail=dr.detail))
        except Exception as e:
            notes.append(f"Depth Pro unavailable: {e}")
    scale = fuse(rulers)
    if scale and sh["scale_correction"] != 1.0:
        scale["s"] *= sh["scale_correction"]
        scale["correction"] = sh["scale_correction"]
    if scale and not scale["agree"]:
        notes.append(f"Rulers disagree by {scale['spread'] * 100:.0f}%; error bars widened.")

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
    elif scale and size_m_ and 0.036 <= size_m_ <= 0.044 and not rigid:
        mfl = jf.flights[jf.main]
        fac, _ = drag_factor(mfl.fit, t[mfl.idx] - mfl.fit.t0, gdir, scale["s"] * k_clock,
                             PING_PONG_VT)
        if fac > 1.03:
            notes.append(
                f"If this is a ping-pong ball, choose the ping-pong preset (--object ping-pong): "
                f"air drag slows this fall by about {100 * (1 - 1 / fac):.0f}%, and Galileo "
                "corrects for it when it knows the ball.")

    # ---- scale-free checks --------------------------------------------------------------
    say("testing physics")
    # Mention drag only where it matters (light balls), not for a steel ball's 0.05%.
    drag_on = bool(drag) and max(abs(v_ - 1) for v_ in drag["flights"]) > 0.005
    tests = [checks.parabola_check(t, xs_f, ys_f, w0, jf, drag_corrected=drag_on),
             checks.bounce_check(t, jf, dt_med)]
    tests.append(checks.equal_falls_check((a_raw, s_raw), []))
    if o.n == 1 and camera and camera.get("fall_vs_room") and camera.get("mode") != "given":
        fv = camera["fall_vs_room"]
        tests.append(checks.vertical_check(fv["deg"], fv["sigma"]))
    size_px = max(d_ref, 0.5 * length_px) if (rigid and length_px) else d_ref
    rest_xy = None
    if tr.rest is not None:
        rx_, ry_, _ = lev.forward(np.array([tr.rest.x + geo.ox]), np.array([tr.rest.y + geo.oy]))
        rest_xy = (float(rx_[0]), float(ry_[0]))
    lead = np.arange(0, int(jf.flights[0].idx[0]))
    if len(lead) >= 3:
        lx, ly = xs[lead][-6:], ys[lead][-6:]
        if np.ptp(lx) < 0.3 * size_px and np.ptp(ly) < 0.3 * size_px:
            rest_xy = (float(np.median(lx)), float(np.median(ly)))
    if o.first_hop:
        tests.append({"name": "Starts from rest", "status": "skip", "detail": {},
                      "summary": "Measured from a bounce onward (the zoomed-in stretch), so "
                                 "the release isn't in it."})
    else:
        tests.append(checks.release_check(jf, rest_xy, (length_px or d_ref) if rigid else d_ref,
                                          thrown=launch["thrown"]))
    if launch["thrown"]:
        v0 = abs(launch["v0_px"]) * scale["s"] * k_clock if scale else None
        n_after = len(jf.flights) - 1
        if v0 is not None and v0 >= 2.5:
            first = (f"Thrown, not dropped: the ball was already moving at {v0:.1f} m/s when "
                     "first seen")
        else:
            first = "The ball was already falling fast when it was first tracked"
        if launch["exclude"]:
            notes.append(f"{first}, so that first flight is left out and the "
                         f"{n_after} flight(s) after the first bounce measure gravity.")
        else:
            notes.append(f"{first}. That's fine for measuring g; it just doesn't "
                         "start from rest.")
    if launch["trim_s"] > 0:
        notes.append(f"Left out the first {1000 * launch['trim_s']:.0f} ms of the first "
                     "flight, where the hand was still pushing the ball.")

    mf = jf.flights[jf.main]
    # Tracking quality is judged on jitter around a smooth path (third
    # differences), not on the parabola fit, so a smooth but non-physical
    # path still gets a physics verdict.
    jitter = float(np.hypot(_noise_px(x0[mf.idx]), _noise_px(y0[mf.idx])))
    kv = float(np.nanmedian(k_lev[mf.idx])) if np.isfinite(k_lev[mf.idx]).any() else 1.0
    size_img = size_px / max(kv, 1e-6)
    tracking_ok = jitter < 0.25 * size_img + 0.5
    if not tracking_ok:
        notes.append(f"Tracking is jittery ({jitter:.1f} px on a {size_img:.0f} px object).")
    if d_ref0 < 8 and not rigid:
        where = " even after zooming in" if o.zoom else ""
        notes.append(f"The ball is only {d_ref0:.0f} px across{where}; move the camera "
                     "closer or use a bigger ball for a tighter answer.")

    # Does the tracked thing fall at all? A dropped ball ends its flight moving
    # down, having come down a few of its own widths. A hand pulling away doesn't.
    # A big ball close to the camera has no room to come down two of its own
    # widths inside the picture, so for it a fifth of the picture is enough.
    # Any of its flights will do: a bouncing ball's later hops are lower than
    # its first drop.
    need = min(2.0 * d_ref, 0.2 * (abs(gdir[0]) * geo.W + abs(gdir[1]) * geo.H))

    def _drop(fr_):
        pu_m = fr_.fit.cx * gdir[0] + fr_.fit.cy * gdir[1]
        tau_m = t[fr_.idx] - fr_.fit.t0
        u_m = pu_m[0] + pu_m[1] * tau_m + pu_m[2] * tau_m**2
        return float(u_m[-1] - np.min(u_m)), float(pu_m[1] + 2 * pu_m[2] * tau_m[-1])
    travel = _drop(mf)[0]
    falls = any(dr_ >= need and ve_ > 0 for dr_, ve_ in map(_drop, jf.flights))
    if not falls and _sure_of(idn):
        # A ball tossed up is in free flight too; "never falls" is there to
        # catch a hand pulling away, and this is known not to be a hand.
        a_m, s_m_ = mf.fit.accel_along(gdir)
        span = float(np.hypot(np.ptp(xs[mf.idx]), np.ptp(ys[mf.idx])))
        falls = a_m > 5 * s_m_ and span >= need

    light = identify.is_light(idn)
    if light:
        # Drag bends a feather's fall: a changing acceleration is what a real one does.
        for t_ in tests:
            if t_["name"] == "Constant acceleration" and t_["status"] == "fail":
                t_["status"] = "warn"
                t_["summary"] += f" Expected for a {obj_label}: air drag rules its fall."
    # How far the camera model alone could move the answer (focal length, pitch).
    geom_rel, geom_parts = _geometry_error(t, x0, y0, d0, d_ref0, [fr_.idx for fr_ in jf.flights],
                                           w0, camera, geo, rigid)
    v = decide(a_px=a_raw, a_sigma=jf.a_sigma / drag["factor"] if drag else s_raw, scale=scale,
               clock=clock, tests=tests, n_points=len(mf.idx), duration=mf.duration,
               g_ref=g_ref, tracking_ok=tracking_ok, rs_rel=rs_rel, drag=drag, falls=falls,
               light=obj_label if light else None, geom_rel=geom_rel,
               real_time=sh.get("real_time", C.ASSUME_REAL_TIME))
    if v["stage"] == "ruler: missing" and o.n > 1:
        v["headline"] = "Can't tell: no ruler (its size isn't known)."

    # ---- payload ----------------------------------------------------------------------
    k_used = v.get("k", 1.0)
    vt_plot = vt if drag else None
    out = {"verdict": v, "tests": tests, "clock_raw": clock, "scale": scale, "drag": drag,
           "spin": spin, "gemini_object": idn if idn.get("source") == "Gemini" else None}
    if want_dbg:
        out["dbg"] = {"geo": geo, "times": times, "tr": tr, "f": f, "t": t, "x": x0, "y": y0,
                      "d": d0, "xs": xs, "ys": ys, "m_fac": m_fac, "jf": jf_obs, "lev": lev,
                      "jf_free": jf, "drag_shift": D, "gdir": gdir,
                      "scale": scale, "k": k_used, "drag": drag, "g_ref": g_ref,
                      "d_ref": d_ref, "flights": [fr_.idx for fr_ in jf.flights],
                      "excluded": tuple(jf.excluded), "launch": launch, "rigid": rigid,
                      "ghost": _ghost_path(jf_obs, times, gdir, scale, k_used, g_ref, geo.H,
                                           t, xs, ys, vt_plot)}
    out.update({
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
            "shape": "rigid" if rigid else "round", "length_px": length_px,
            "n_points": int(len(mf.idx)), "leveled": not lev.identity,
            "camera_error_rel": geom_rel, "camera_error_parts": geom_parts,
        },
        "track": _track_payload(tr.main, times, geo, rigid=rigid),
        "plot": _plot_payload(jf_obs, t, xs, ys, gdir, scale, k_used, g_ref, d_ref, vt_plot,
                              drag["flights"][0] if drag else 1.0),
        "ghost": _ghost_payload(jf_obs, times, gdir, scale, k_used, g_ref, geo, t, xs, ys,
                                vt_plot, lev=lev, m_fac=m_fac),
    })
    return out


def _level(lev, x0, y0, d0, d_ref0, rigid, camera, geo):
    """Positions and widths in the camera's level view (galileo/camera.py).

    Round objects' widths are first taken across the line to the picture's
    centre (off-axis stretch), which is what the level view's ruler needs.
    """
    if camera and not rigid:
        f_px = camera["f_frac"] * max(geo.W, geo.H)
        d = _minor(x0, y0, d0, f_px, geo.W / 2.0, geo.H / 2.0)
    else:
        d = np.asarray(d0, float)
    if lev.identity:
        x, y = np.asarray(x0, float), np.asarray(y0, float)
    else:
        d = lev.widths(x0, y0, d)
        x, y, z = lev.forward(x0, y0)
        x = np.where(z > 1e-3, x, np.nan)
        y = np.where(z > 1e-3, y, np.nan)
    if np.isfinite(d).any():
        d_ref = float(np.nanmedian(d))
    else:
        k = lev.widths(x0, y0, np.ones(len(x0)))
        d_ref = d_ref0 * (float(np.nanmedian(k)) if np.isfinite(k).any() else 1.0)
    return x, y, d, d_ref


def _level_width(lev, camera, geo, x, y, d):
    """One still ball's width (measured as the average of both axes) in the level view."""
    xa, ya = np.array([float(x)]), np.array([float(y)])
    dd = np.array([float(d)])
    if camera:
        f_px = camera["f_frac"] * max(geo.W, geo.H)
        dd = cam_.minor_width(xa, ya, dd, f_px, geo.W / 2.0, geo.H / 2.0,
                              np.array([np.nan]), np.array([np.nan]))
    return float(lev.widths(xa, ya, dd)[0])


def _geometry_error(t, x0, y0, d0, d_ref0, fl, w0, camera, geo, rigid):
    """How much the camera model alone could move g (relative), and why.

    g is measured again without the extras (no rolling shutter, no drag), in
    units of the object's own width, with the focal length at the ends of its
    error bar, and with the pitch at the ends of its own: the spread is the
    camera's share of the error. A wrong focal length stretches the depth
    axis, so it also turns the pitch (gravity's depth part grows with f).
    """
    if not camera or not fl:
        return 0.0, {}
    f_frac, f_rel = float(camera["f_frac"]), float(camera["f_rel"])
    n = np.asarray(camera["n"], float)
    pitched = bool(camera.get("pitched"))
    fl = [np.asarray(i) for i in fl if len(i) >= 6]
    if not fl:
        return 0.0, {}

    def ratio(ff, nn):
        cam2 = {**camera, "f_frac": ff, "n": list(nn)}
        lev = _leveler(cam2, geo)
        x, y, d, d_ref = _level(lev, x0, y0, d0, d_ref0, rigid, cam2, geo)
        if not (np.isfinite(x[np.concatenate(fl)]).all()):
            return None
        m = np.ones(len(t)) if rigid else _depth_factors(t, d, fl)[0]
        cx, cy = geo.W / 2.0, geo.H / 2.0
        xs, ys = cx + (x - cx) * m, cy + (y - cy) * m
        jf = fit_flights(t, xs, ys, w0, fl, fps=1.0, n_px_axis=1.0, rs_mode="off", gdir=DOWN)
        used = np.concatenate([jf.flights[j].idx for j in (jf.used or range(len(fl)))])
        dn = (d * m)[used]
        dn = dn[np.isfinite(dn)]
        if len(dn) < 3 or jf.a_px <= 0:
            return None
        return jf.a_px / float(np.median(dn))

    r0 = ratio(f_frac, n)
    if not r0:
        return 0.0, {}
    parts = {}

    def n_for(ff):
        if not pitched:
            return n
        v = np.array([n[0], n[1], n[2] * ff / f_frac])
        return v / np.linalg.norm(v)
    lo, hi = f_frac / (1 + f_rel), f_frac * (1 + f_rel)
    rp, rm = ratio(hi, n_for(hi)), ratio(lo, n_for(lo))
    if rp and rm:
        parts["focal length"] = abs(rp - rm) / 2 / r0
    if pitched and camera.get("pitch_sigma"):
        b = math.asin(float(np.clip(n[2], -1, 1)))
        h = np.array([n[0], n[1]]) / max(float(np.hypot(n[0], n[1])), 1e-9)
        sb = math.radians(float(camera["pitch_sigma"]))
        rr = []
        for bb in (b - sb, b + sb):
            rr.append(ratio(f_frac, np.array([math.cos(bb) * h[0], math.cos(bb) * h[1],
                                              math.sin(bb)])))
        if all(rr):
            parts["camera pitch"] = abs(rr[1] - rr[0]) / 2 / r0
    tot = float(np.sqrt(sum(v * v for v in parts.values())))
    return tot, {k: float(v) for k, v in parts.items()}


# ---------------------------------------------------------------------------
def _launch(t, xs, ys, fl, jf, gdir, first_hop=False):
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
    hop = [bool(first_hop)] + [float(t[b[0]] - t[a[-1]]) <= 0.12 for a, b in zip(fl, fl[1:])]
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
    if len(f0) >= 5 and not first_hop:
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


def _zoom_plan(info, geo, times, t, x, y, f, fl, d_ref, extent=None):
    """Where and when to look again, at what resolution. None if it wouldn't help.

    extent: the object's longest side when it isn't round (sets the margin,
    and whether it is small enough to need zooming)."""
    size = max(d_ref, 0.6 * extent) if extent and extent > d_ref else d_ref
    if size >= C.ZOOM_MIN_PX or info.width <= 0:
        return None
    r_c = geo.W / info.width                 # first-pass px per original px
    d_full = size / r_c
    z = min(1.0, C.ZOOM_TARGET_PX / max(d_full, 1e-6))
    if z < 1.25 * r_c:
        return None                          # the first pass was already near full size
    times = np.asarray(times, float)
    lead = 0.3

    # A camera that pans far is laid onto a canvas: the patch to decode (where
    # the object shows in the frames) and the window it is steadied into
    # (where it sits in the reference view) are then two different boxes.
    canvas = geo.T is not None and (geo.ox != 0 or geo.oy != 0 or geo.fox not in (None, geo.ox))

    def window(i_a, i_b):
        t0 = max(float(t[i_a]) - lead, float(times[0]))
        t1 = min(float(t[i_b]) + 0.1, float(times[-1]))
        sel = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
        fx, fy, sx, sy = [], [], [], []
        for fi, xi, yi in zip(f[sel], x[sel], y[sel]):
            a, b = geo.to_frame(int(fi), xi, yi)
            fx.append(float(a))
            fy.append(float(b))
            sx.append(float(xi))
            sy.append(float(yi))
        m = max(2.5 * d_ref, 0.75 * (extent or 0.0)) + 6.0
        if not canvas:
            fx, fy = fx + sx, fy + sy         # where it shows, and where it is stabilized
        box = (max(min(fx) - m, 0.0) / r_c, max(min(fy) - m, 0.0) / r_c,
               min(max(fx) + m, geo.W) / r_c, min(max(fy) + m, geo.H) / r_c)
        dst = box if not canvas else ((min(sx) - m) / r_c, (min(sy) - m) / r_c,
                                      (max(sx) + m) / r_c, (max(sy) + m) / r_c)
        n = int(np.sum((times >= t0 - 1e-9) & (times <= t1 + 1e-9)))
        return t0, t1, box, n, dst

    def cost(zz, box, n, dst):
        a = (box[2] - box[0]) * (box[3] - box[1]) + (dst[2] - dst[0]) * (dst[3] - dst[1])
        return 0.5 * a * zz * zz * 3 * n * (2 if canvas else 1)

    # Every run of consecutive flights is a candidate window. A window cut in
    # the middle of a flight leaves a stub that fits badly, so whole flights
    # only. Among those that fit the memory budget: more frames help, a
    # bigger picture of the object helps more (its edges and centre are
    # found to about a pixel, whatever its size).
    best = None
    for i in range(len(fl)):
        for j in range(i, len(fl)):
            t0, t1, box, n, dst = window(fl[i][0], fl[j][-1])
            if n > C.MAX_FRAMES:
                break
            zz = z
            c = cost(zz, box, n, dst)
            if c > C.ZOOM_BUDGET_BYTES:
                zz = z * float(np.sqrt(C.ZOOM_BUDGET_BYTES / c))
            if zz < 1.25 * r_c:
                continue
            pts = sum(len(fl[q]) for q in range(i, j + 1))
            key = zz * pts ** 0.75
            if best is None or key > best[0]:
                best = (key, t0, t1, box, zz, float(t[fl[j][-1]]), i, float(t[fl[i][0]]), dst)
    if best is None:
        return None
    _, t0, t1, box, zz, t_last, i_first, t_first, dst = best
    Wz, Hz = max(2, round(info.width * zz)), max(2, round(info.height * zz))
    crop = (int(np.floor(box[0] * zz)), int(np.floor(box[1] * zz)),
            min(int(np.ceil(box[2] * zz)), Wz), min(int(np.ceil(box[3] * zz)), Hz))
    win = (int(np.floor(dst[0] * zz)), int(np.floor(dst[1] * zz)),
           int(np.ceil(dst[2] * zz)), int(np.ceil(dst[3] * zz))) if canvas else crop
    if crop[2] - crop[0] >= 16 and crop[3] - crop[1] >= 16:
        return {"z": zz, "t0": t0, "t1": t1, "crop": crop, "W": Wz, "H": Hz, "t_last": t_last,
                "first_flight": i_first, "t_first": t_first, "window": win}
    return None


def _zoom_pass(path, info, plan, geo, times, tr, tracker, want_gray=True, down=DOWN):
    """Decode the object's patch at high resolution and track it there again.

    Returns (grey frames or None, times, geometry, track result, frame
    numbers, colour close-up) or None.
    """
    z = plan["z"]
    x0, y0, x1, y1 = plan["crop"]
    frames, tz, nz = read_frames(path, info, scale=z, t_start=plan["t0"], t_end=plan["t1"],
                                 max_frames=C.MAX_FRAMES, crop=plan["crop"],
                                 step=plan.get("step", 1))
    if len(frames) < 8:
        return None
    ratio = plan["W"] / geo.W
    times = np.asarray(times, float)
    Tz = None
    wx0, wy0, wx1, wy1 = plan.get("window") or plan["crop"]
    if geo.T is not None:
        # Reuse the first pass's camera motion, re-expressed for this crop and
        # the window it is steadied into (in the first pass's canvas pixels,
        # scaled up).
        idx = np.clip(np.searchsorted(times, tz), 1, len(times) - 1)
        idx = np.where(np.abs(times[idx - 1] - tz) < np.abs(times[idx] - tz), idx - 1, idx)
        Tz = rescale_transforms([geo.T[i] for i in idx], ratio, (x0, y0),
                                dst_offset=(wx0 - geo.ox * ratio, wy0 - geo.oy * ratio))
        frames = warp(frames, Tz, size=(wx1 - wx0, wy1 - wy0))
    else:
        wx0, wy0 = x0, y0
    # Follow the same object as the first pass.
    seeds = [(float(times[dt.f]), (dt.x + geo.ox) * ratio - wx0, (dt.y + geo.oy) * ratio - wy0)
             for dt in tr.main.dets if plan["t0"] <= times[dt.f] <= plan["t1"]]
    if tracker == "sam2":
        from .sam2_tracker import track_with_sam2
        ts, sx, sy = seeds[len(seeds) // 4]
        fi = int(np.argmin(np.abs(tz - ts)))
        trz = track_with_sam2(frames, tz, seed=(fi, sx, sy))
        trz.seeded = True
    else:
        trz = track_falling_object(frames, tz, seed=seeds, max_objects=1)
    if trz.main is None or not trz.seeded:
        return None
    d_z = float(np.nanmedian(trz.main.d))
    d_c = float(np.nanmedian(tr.main.d)) * ratio
    if not (0.6 < d_z / max(d_c, 1e-6) < 1.7):
        return None
    geo_z = Geom(W=plan["W"], H=plan["H"], scale=z, ox=float(wx0), oy=float(wy0), T=Tz,
                 fox=float(x0), foy=float(y0))
    view = None
    tt = np.asarray(tz, float)[trz.main.f]
    fl = split_flights(tt, trz.main.x, trz.main.y, down, diam_px=max(d_z, 2.0))
    if fl:
        size = d_z
        st = shp.outline_stats(trz.main.dets)
        L = st["long"]
        if np.isfinite(L).any() and np.nanmedian(L) > 1.3 * d_z:
            size = float(np.nanpercentile(L, 90))
        view = _view(frames, trz, tz, fl, size)
    gray = ([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in frames]
            if want_gray else None)
    del frames
    return gray, tz, geo_z, trz, nz, view


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


def _track_payload(main, times, geo, rigid=False):
    pts = []
    for dt in main.dets:
        x0, y0 = ((dt.tx, dt.ty) if rigid and np.isfinite(dt.tx) else (dt.x, dt.y))
        px, py = geo.to_frame(dt.f, x0 + geo.ox, y0 + geo.oy)
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


def _ghost_payload(jf, times, gdir, scale, k, g_ref, geo, t=None, xs=None, ys=None, vt=None,
                   lev=None, m_fac=None):
    """The same, as fractions of the picture for the web overlay."""
    gp = _ghost_path(jf, times, gdir, scale, k, g_ref, geo.H, t, xs, ys, vt)
    if gp is None:
        return None
    times = np.asarray(times, float)
    gt, gxs, gys = gp
    if m_fac is not None and t is not None and len(gt):
        # Undo the distance scaling, then the levelling: back to the picture.
        m = np.interp(gt, np.asarray(t, float), np.asarray(m_fac, float))
        cx, cy = geo.W / 2.0, geo.H / 2.0
        gxs, gys = cx + (gxs - cx) / m, cy + (gys - cy) / m
        if lev is not None and not lev.identity:
            gxs, gys = lev.backward(gxs, gys)
    out = []
    for tt, gx, gy in zip(gt, gxs, gys):
        fi = int(np.argmin(np.abs(times - tt)))
        px, py = geo.to_frame(fi, gx, gy)
        out.append({"t": float(tt), "x": float(px / geo.W), "y": float(py / geo.H)})
    return {"points": out}
