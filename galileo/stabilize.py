"""Remove camera shake, pans and zooms by tracking the static background.

Generated clips often drift or push in slowly, and handheld phone clips
shake. A moving camera adds its own acceleration to the object's, so frames
are warped back onto the first frame before tracking. Static-camera clips
are detected and left alone.

Each frame is matched to a keyframe, not to the frame before it. Chaining
frame-to-frame estimates lets anything that moves slowly (a hand pulling
away, a person shifting) leak in a fraction of a pixel per frame, and at
120 to 240 fps that adds up to several pixels of fake camera motion. Against
a keyframe, a moving hand is soon far from where it started and gets
rejected as an outlier. A new keyframe is taken only when too few
background points still match (a real pan).
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f


def _features(img):
    return cv2.goodFeaturesToTrack(img, maxCorners=300, qualityLevel=0.01, minDistance=8,
                                   blockSize=7)


def _match(key, pts, cur, guess):
    """Similarity transform mapping cur onto key, or None. guess maps cur onto key too."""
    if pts is None or len(pts) < 12:
        return None, 0
    G = np.vstack([guess, [0, 0, 1]])
    Ginv = np.linalg.inv(G)
    init = (pts.reshape(-1, 2) @ Ginv[:2, :2].T + Ginv[:2, 2]).astype(np.float32).reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(key, cur, pts, init, winSize=(21, 21), maxLevel=3,
                                          flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    ok = st.ravel() == 1
    if ok.sum() < 10:
        return None, 0
    A, inl = cv2.estimateAffinePartial2D(nxt[ok], pts[ok], method=cv2.RANSAC,
                                         ransacReprojThreshold=1.5)
    if A is None or inl is None or inl.sum() < 8:
        return None, 0
    return A, int(inl.sum())


def estimate_motion(frames) -> tuple[list[np.ndarray], dict]:
    """Per-frame 2x3 similarity transforms mapping frame i onto frame 0."""
    n = len(frames)
    key = _gray(frames[0])
    key_pts = _features(key)
    key_T = np.eye(3)                      # keyframe -> frame 0
    n_key = len(key_pts) if key_pts is not None else 0
    to_ref = [np.eye(3)]
    last = np.eye(3)[:2]                   # current frame -> keyframe, last estimate
    keyframes = 1
    for i in range(1, n):
        cur = _gray(frames[i])
        A, n_in = _match(key, key_pts, cur, last)
        if A is None or n_in < max(12, 0.3 * n_key):
            # Too few background points left in common: start a new keyframe
            # from the previous frame, whose placement is already known.
            key = _gray(frames[i - 1])
            key_pts = _features(key)
            n_key = len(key_pts) if key_pts is not None else 0
            key_T = to_ref[i - 1]
            keyframes += 1
            A, n_in = _match(key, key_pts, cur, np.eye(3)[:2])
            if A is None:
                A = np.eye(3)[:2]
        last = A
        M = np.vstack([A, [0, 0, 1]])
        to_ref.append(key_T @ M)
    # How much did the camera move?
    h, w = frames[0].shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    disp = [float(np.max(np.linalg.norm((T @ corners)[:2] - corners[:2], axis=0)))
            for T in to_ref]
    steps = [np.linalg.inv(a) @ b for a, b in zip(to_ref, to_ref[1:])]
    step_disp = [float(np.max(np.linalg.norm((S @ corners)[:2] - corners[:2], axis=0)))
                 for S in steps] or [0.0]
    info = {"max_drift_px": max(disp), "median_step_px": float(np.median(step_disp)),
            "keyframes": keyframes}
    return [T[:2] for T in to_ref], info


def stabilize(frames, mode="auto", drift_px=4.0, step_px=0.15):
    """Return (frames, transforms, info). Frames are warped only if the camera moved.

    A locked-off camera still shows a little estimation noise, and warping by
    that noise hurts more than it helps, so a tripod shot (tiny typical step,
    small total drift) is left alone.
    """
    if mode == "off" or len(frames) < 3:
        return frames, None, {"applied": False}
    T, info = estimate_motion(frames)
    moved = info["median_step_px"] > step_px or info["max_drift_px"] > drift_px
    if mode == "auto" and not moved:
        info["applied"] = False
        return frames, None, info
    return warp(frames, T), T, {**info, "applied": True}


def warp(frames, T):
    h, w = frames[0].shape[:2]
    return [frames[0] if i == 0 else
            cv2.warpAffine(f, np.asarray(A, np.float64), (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            for i, (f, A) in enumerate(zip(frames, T))]


def rescale_transforms(T, ratio: float, offset=(0.0, 0.0)):
    """Transforms found on downscaled full frames, re-expressed for a crop.

    ratio: crop pixels per original working pixel. offset: the crop's
    top-left corner in the new pixels. The rotation and zoom parts carry
    over; the shift scales with the pixels and absorbs the crop offset.
    """
    o = np.asarray(offset, float)
    out = []
    for A in T:
        A = np.asarray(A, float)
        L = A[:, :2]
        t = L @ o + ratio * A[:, 2] - o
        out.append(np.hstack([L, t[:, None]]))
    return out
