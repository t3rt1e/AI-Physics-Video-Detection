"""Remove camera shake, pans and zooms by tracking the static background.

Generated clips often drift or push in slowly. A moving camera adds its
own acceleration to the object's, so frames are warped back onto the first
frame before tracking. Static-camera clips are detected and left alone.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_motion(frames) -> tuple[list[np.ndarray], dict]:
    """Per-frame 2x3 similarity transforms mapping frame i onto frame 0."""
    n = len(frames)
    to_ref = [np.eye(3)]
    steps = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) if frames[0].ndim == 3 else frames[0]
    for i in range(1, n):
        cur = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) if frames[i].ndim == 3 else frames[i]
        M = np.eye(3)
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=8,
                                      blockSize=7)
        if pts is not None and len(pts) >= 12:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21),
                                                  maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 10:
                A, inl = cv2.estimateAffinePartial2D(nxt[ok], pts[ok], method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.5)
                if A is not None and inl is not None and inl.sum() >= 8:
                    M[:2] = A  # maps current -> previous
        steps.append(M)
        to_ref.append(to_ref[-1] @ M)
        prev = cur
    # How much did the camera move?
    h, w = frames[0].shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    disp = [float(np.max(np.linalg.norm((T @ corners)[:2] - corners[:2], axis=0)))
            for T in to_ref]
    step_disp = [float(np.max(np.linalg.norm((S @ corners)[:2] - corners[:2], axis=0)))
                 for S in steps] or [0.0]
    info = {"max_drift_px": max(disp), "median_step_px": float(np.median(step_disp))}
    return [T[:2] for T in to_ref], info


def stabilize(frames, mode="auto", drift_px=4.0, step_px=0.15):
    """Return (frames, transforms, info). Frames are warped only if the camera moved.

    Chaining frame-to-frame estimates random-walks by a pixel or two even on
    a locked-off camera, and warping by that noise hurts more than it helps,
    so a tripod shot (tiny typical step, small total drift) is left alone.
    """
    if mode == "off" or len(frames) < 3:
        return frames, None, {"applied": False}
    T, info = estimate_motion(frames)
    moved = info["median_step_px"] > step_px or info["max_drift_px"] > drift_px
    if mode == "auto" and not moved:
        info["applied"] = False
        return frames, None, info
    h, w = frames[0].shape[:2]
    out = [frames[0]] + [cv2.warpAffine(f, A, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                         for f, A in zip(frames[1:], T[1:])]
    info["applied"] = True
    return out, T, info
