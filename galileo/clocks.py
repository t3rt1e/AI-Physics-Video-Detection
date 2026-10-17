"""Layer 3: clocks. How many real seconds pass per second of video (k)?

A phone's 8x slow motion makes a real drop read 9.81/64 = 0.15 m/s^2, the
same "floaty" look as AI video. Physics alone can't separate the two, so
Galileo reads other clocks in the file:

  metadata      capture-rate tags (Android com.android.capture.fps) and
                high-frame-rate timestamps
  duplicates    frames repeated k times (slowed without interpolation)
  interpolation motion that is piecewise straight with kinks every k frames
                (slowed with optical-flow or blended frames)
  flicker       mains lighting (120 Hz in the US) aliased into the frame rate;
                weak on its own, reported as a hint

Each clock returns an evidence dict; combine() picks the answer.
"""
from __future__ import annotations

import numpy as np

from . import config as C

STANDARD_CAPTURE_FPS = (24, 25, 30, 48, 50, 60, 90, 100, 120, 240, 480, 960)


def _ev(name, k, confidence, detail, **extra):
    d = {"name": name, "k": float(k), "confidence": float(confidence), "detail": detail}
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
def metadata_clock(meta: dict) -> list[dict]:
    """Capture-rate tags and high-frame-rate timestamps."""
    out = []
    fps = float(meta.get("fps") or 0)
    cap = meta.get("capture_fps")
    if cap and fps and cap > 1.4 * fps:
        k = cap / fps
        out.append(_ev("file metadata", k, 0.95,
                       f"File says it was captured at {cap:g} fps and plays at {fps:g} fps "
                       f"({k:.1f}x slow motion).", capture_fps=float(cap)))
    elif fps >= 90:
        out.append(_ev("frame timestamps", 1.0, 0.9,
                       f"High-speed capture: timestamps run at {fps:.0f} fps of real time. "
                       f"Played back at 30 fps it would look {fps / 30:.0f}x slow.",
                       capture_fps=fps, slowmo_playback=fps / 30.0))
    elif cap and fps and abs(cap / fps - 1) < 0.1:
        out.append(_ev("file metadata", 1.0, 0.85,
                       f"File says it was captured and plays at {fps:g} fps (real time).",
                       capture_fps=float(cap)))
    return out


# ---------------------------------------------------------------------------
def duplicate_clock(gray, frames, xs, ys, diam, disp_per_frame) -> dict | None:
    """Frames repeated k times leave a step pattern in the image differences.

    gray: sequence of working-resolution gray frames indexed by frame number.
    frames/xs/ys: the tracked object's frame numbers and positions (working px).
    disp_per_frame: expected displacement per frame from the naive fit (px).
    """
    frames = np.asarray(frames)
    if len(frames) < 8:
        return None
    pos = {int(f): (float(x), float(y)) for f, x, y in zip(frames, xs, ys)}
    E, used = [], []
    r = int(max(diam, 4) * 1.2) + 2
    H, W = gray[0].shape[:2]
    for f in range(int(frames.min()) + 1, int(frames.max()) + 1):
        if f not in pos or f - 1 not in pos:
            continue
        if disp_per_frame.get(f, 0.0) < 0.75:
            continue
        (x0, y0), (x1, y1) = pos[f - 1], pos[f]
        xa, xb = int(max(min(x0, x1) - r, 0)), int(min(max(x0, x1) + r, W))
        ya, yb = int(max(min(y0, y1) - r, 0)), int(min(max(y0, y1) + r, H))
        if xb - xa < 3 or yb - ya < 3:
            continue
        a = gray[f - 1][ya:yb, xa:xb].astype(np.float32)
        b = gray[f][ya:yb, xa:xb].astype(np.float32)
        E.append(float(np.mean(np.abs(b - a))))
        used.append(f)
    if len(E) < 8:
        return None
    E = np.asarray(E)
    ref = np.percentile(E, 80)
    if ref <= 0.5:
        return None
    dup = E < 0.2 * ref
    if dup.mean() < 0.25:
        return None
    used = np.asarray(used)
    changes = used[~dup]
    if len(changes) < 4:
        return None
    sp = np.diff(changes)
    k = int(np.median(sp))
    consistency = float(np.mean(sp == k))
    if k < 2 or consistency < 0.6:
        return None
    # Keep one frame per run: the frame where the picture changes.
    return _ev("repeated frames", k, 0.6 + 0.35 * consistency,
               f"Each frame is repeated {k} times ({consistency:.0%} of runs), the mark of "
               f"footage slowed {k}x without new frames.",
               keep_frames=[int(c) for c in changes], dup_fraction=float(dup.mean()))


# ---------------------------------------------------------------------------
def interpolation_clock(frames, u) -> dict | None:
    """Slowed with interpolation: curvature bunches up every k frames.

    Between original frames, generated in-betweens move in straight lines,
    so the second difference of position is ~0 except at the original
    frames, where it carries k times the normal curvature.
    frames, u: consecutive-ish frame numbers and position along gravity (px).
    """
    frames = np.asarray(frames)
    u = np.asarray(u, float)
    # Longest run of consecutive frames.
    runs, start = [], 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] != frames[i - 1] + 1:
            runs.append((start, i))
            start = i
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if b - a < 14:
        return None
    fr = frames[a:b]
    uu = u[a:b]
    s = uu[2:] - 2 * uu[1:-1] + uu[:-2]
    centers = fr[1:-1]
    mean_s = float(np.mean(s))
    if mean_s <= 0:
        return None
    best = None
    for q in range(2, 13):
        if len(s) < 3 * q:
            break
        ph = centers % q
        means = np.array([s[ph == p].mean() if np.any(ph == p) else 0.0 for p in range(q)])
        p_star = int(np.argmax(means))
        resid = s - means[ph]
        sig = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        n_star = int(np.sum(ph == p_star))
        n_oth = len(s) - n_star
        other = float(s[ph != p_star].mean())
        z = (means[p_star] - other) / (sig * np.sqrt(1.0 / n_star + 1.0 / max(n_oth, 1)))
        frac = means[p_star] / (q * mean_s)
        if frac > 0.6 and z > 6 and abs(other) < 0.35 * mean_s:
            best = (q, p_star, float(frac), float(z))
            break
    if best is None:
        return None
    q, p_star, frac, z = best
    keep = [int(f) for f in frames if f % q == p_star]
    return _ev("interpolated frames", q, min(0.55 + 0.4 * frac, 0.9),
               f"Motion bends only every {q} frames; the frames between are straight-line "
               f"in-betweens. Footage was slowed {q}x by interpolation.",
               keep_frames=keep, period=q, phase=p_star, z=z)


# ---------------------------------------------------------------------------
def flicker_clock(brightness, fps_file: float) -> dict | None:
    """Mains flicker (100/120 Hz) aliased into the frame rate.

    Strong frame-to-frame alternation means a ~240 fps capture under US
    lighting. Ambiguous on its own (48 fps aliases the same way), so this is
    a hint with low confidence.
    """
    b = np.asarray(brightness, float)
    if len(b) < 32 or not fps_file:
        return None
    k_ = 5
    smooth = np.convolve(b, np.ones(k_) / k_, mode="same")
    x = (b - smooth)[k_:-k_]
    if np.std(x) < 0.05:
        return None
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x))
    band = f > 0.05
    if not band.any():
        return None
    i = int(np.argmax(np.where(band, X, 0)))
    snr = float(X[i] / (np.median(X[band]) + 1e-12))
    if snr < 15:
        return None
    f_obs = float(f[i])
    cands = []
    for F in STANDARD_CAPTURE_FPS:
        for fm in (100.0, 120.0):
            a = (fm / F) % 1.0
            a = min(a, 1 - a)
            if abs(a - f_obs) < 0.02:
                cands.append((F, fm))
    hi = [c for c in cands if c[0] >= 1.5 * fps_file and c[0] in (120, 240, 480, 960)]
    if not hi:
        return None
    F, fm = sorted(hi, key=lambda c: (c[0] != 240, c[0]))[0]
    k = F / fps_file
    return _ev("light flicker", k, 0.45,
               f"Lights flicker at {f_obs:.2f} cycles per frame, which {fm:.0f} Hz mains "
               f"produces at {F} fps capture: a hint of {k:.0f}x slow motion.",
               capture_fps=float(F), snr=snr)


# ---------------------------------------------------------------------------
def combine(evidence: list[dict], fps_file: float) -> dict:
    """Pick k. Confident clocks set it; a weak hint is kept aside for the verdict."""
    ev = [e for e in evidence if e]
    strong = [e for e in ev if e["confidence"] >= C.CLOCK_MIN_CONFIDENCE]
    weak = [e for e in ev if e["confidence"] < C.CLOCK_MIN_CONFIDENCE and e["k"] > 1.4]
    out = {"k": 1.0, "source": None, "confidence": 0.0, "evidence": ev,
           "weak_k": None, "weak_source": None, "keep_frames": None,
           "fps_file": fps_file, "slowmo_playback": None}
    if strong:
        best = max(strong, key=lambda e: e["confidence"])
        out.update(k=round(best["k"], 3), source=best["name"], confidence=best["confidence"],
                   keep_frames=best.get("keep_frames"))
        if best.get("slowmo_playback"):
            out["slowmo_playback"] = best["slowmo_playback"]
        ks = [e["k"] for e in strong]
        if max(ks) / min(ks) > 1.3:
            out["conflict"] = True
            out["confidence"] = min(out["confidence"], 0.5)
    if weak:
        w = max(weak, key=lambda e: e["confidence"])
        out["weak_k"], out["weak_source"] = w["k"], w["name"]
    return out


def real_time(clock: dict) -> dict:
    """Take the clip as playing at the speed it was filmed: k = 1, no slow
    motion and no speed-up. Frames the clocks found to be repeats or blends
    are still left out of the fit; a slow-down they found is only reported
    ("ignored_k")."""
    out = {**clock, "k": 1.0, "weak_k": None, "weak_source": None, "conflict": False,
           "real_time": True, "ignored_k": None}
    if clock.get("confidence", 0) >= C.CLOCK_MIN_CONFIDENCE and abs(clock.get("k", 1.0) - 1) > 0.05:
        out["ignored_k"] = (float(clock["k"]), clock["source"])
        out["source"] = "taken as real time"
    elif clock.get("weak_k"):
        out["ignored_k"] = (float(clock["weak_k"]), clock["weak_source"])
    return out
