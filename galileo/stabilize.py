"""Remove camera motion: shake, pans, tilts, rolls, zooms, walking closer.

A moving camera adds its own motion to the object's, so frames are laid
back onto one reference view before tracking. Static-camera clips are
detected and left alone.

Each frame is matched to a keyframe, not to the frame before it. Chaining
frame-to-frame estimates lets anything that moves slowly (a hand pulling
away, a person shifting) leak in a fraction of a pixel per frame, and at
120 to 240 fps that adds up to several pixels of fake camera motion. Against
a keyframe, a moving hand is soon far from where it started and gets
rejected as an outlier. A new keyframe is taken only when too few
background points still match (a real pan).

Small moves (a hand's shake, a slow drift, a zoom) are undone with a
similarity: shift, turn and scale. Bigger ones, such as a pan or tilt to
follow the ball, also change the perspective, so frames are then matched with
a full homography, which is exact for a camera turning about its own centre
(and for a flat background whatever the camera does). The reference view is
then the middle of the motion, on a canvas big enough to hold every frame;
what a frame doesn't cover is filled in from the others, and its outline
(footprint) is kept so nothing cut off by it is mistaken for an object.
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np


def _gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f


def _features(img):
    return cv2.goodFeaturesToTrack(img, maxCorners=300, qualityLevel=0.01, minDistance=8,
                                   blockSize=7)


def _as3(A):
    A = np.asarray(A, np.float64)
    return A if A.shape == (3, 3) else np.vstack([A, [0.0, 0.0, 1.0]])


def _match(key, pts, cur, guess):
    """Background points of `key` found in `cur`, and the similarity mapping
    cur onto key. guess (3x3) maps cur onto key too. Returns (A 3x3 or None,
    inliers, (cur points, key points))."""
    if pts is None or len(pts) < 12:
        return None, 0, None
    Ginv = np.linalg.inv(_as3(guess))
    p = pts.reshape(-1, 2).astype(np.float64)
    q = np.hstack([p, np.ones((len(p), 1))]) @ Ginv.T
    init = (q[:, :2] / q[:, 2:3]).astype(np.float32).reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(key, cur, pts, init, winSize=(21, 21), maxLevel=3,
                                          flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    ok = st.ravel() == 1
    if ok.sum() < 10:
        return None, 0, None
    A, inl = cv2.estimateAffinePartial2D(nxt[ok], pts[ok], method=cv2.RANSAC,
                                         ransacReprojThreshold=1.5)
    if A is None or inl is None or inl.sum() < 8:
        return None, 0, None
    # Keep the matches a looser test accepts, for a homography later.
    src, dst = nxt[ok].reshape(-1, 2), pts[ok].reshape(-1, 2)
    pred = np.hstack([src, np.ones((len(src), 1))]) @ _as3(A).T
    loose = np.hypot(*(pred[:, :2] - dst).T) < 12.0
    return _as3(A), int(inl.sum()), (src[loose].astype(np.float32), dst[loose].astype(np.float32))


def _homography(corr, A):
    """Homography from one frame's matches (falls back to the similarity A)."""
    if corr is None or len(corr[0]) < 16:
        return A, False
    Hm, inl = cv2.findHomography(corr[0], corr[1], cv2.RANSAC, 1.5)
    if Hm is None or inl is None or inl.sum() < 16:
        return A, False
    src, dst = corr[0][inl.ravel() > 0], corr[1][inl.ravel() > 0]
    # Spread over the picture, or the perspective part is a guess.
    span = np.ptp(src, axis=0)
    if span.min() < 40:
        return A, False
    return Hm / Hm[2, 2], True


def estimate_motion(frames, model: str = "auto", big_frac: float = 0.06):
    """Per-frame 3x3 transforms mapping frame i onto frame 0, and info.

    model: "similarity", "homography", or "auto" (homography only when the
    camera moves far enough for perspective to matter, big_frac of the
    picture's size).
    """
    n = len(frames)
    key = _gray(frames[0])
    key_pts = _features(key)
    key_i = 0
    n_key = len(key_pts) if key_pts is not None else 0
    first = (key, key_pts, n_key)          # the first keyframe, to come back to
    steps = [np.eye(3)]                    # frame i -> its keyframe (similarity)
    corrs = [None]
    keys = [0]
    last = np.eye(3)                       # current frame -> keyframe, last estimate
    keyframes = 1
    sim_so_far = [np.eye(3)]
    for i in range(1, n):
        cur = _gray(frames[i])
        A, n_in, corr = _match(key, key_pts, cur, last)
        if key_i != 0 and first[1] is not None and i % 4 == 0:
            # Back to the first keyframe when it is in view again (someone
            # walked in front of a still camera, then out): no drift builds up.
            guess0 = sim_so_far[-1]
            A0, n0, c0 = _match(first[0], first[1], cur, guess0)
            if A0 is not None and n0 >= max(12, 0.5 * first[2]) and n0 >= n_in:
                key, key_pts, n_key, key_i = first[0], first[1], first[2], 0
                A, n_in, corr = A0, n0, c0
        if A is None or n_in < max(12, 0.3 * n_key):
            # Too few background points left in common: start a new keyframe
            # from the previous frame, whose placement is already known.
            key = _gray(frames[i - 1])
            key_pts = _features(key)
            n_key = len(key_pts) if key_pts is not None else 0
            key_i = i - 1
            keyframes += 1
            A, n_in, corr = _match(key, key_pts, cur, np.eye(3))
            if A is None:
                A, corr = np.eye(3), None
        last = A
        steps.append(A)
        corrs.append(corr)
        keys.append(key_i)
        sim_so_far.append(sim_so_far[key_i] @ A)

    def chain(mats):
        out = [np.eye(3)]
        for i in range(1, n):
            out.append(out[keys[i]] @ mats[i])
        return out
    sim = chain(steps)
    h, w = frames[0].shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T

    def disp(T):
        q = T @ corners
        return float(np.max(np.linalg.norm(q[:2] / q[2] - corners[:2], axis=0)))
    # A real pan moves the view for a good part of the clip; a glitch (someone
    # walking right past the lens) only for a moment.
    drift = float(np.percentile([disp(T) for T in sim], 80))
    use_h = model == "homography" or (model == "auto" and drift > big_frac * max(w, h))
    good_h = 0
    if use_h:
        hsteps = [np.eye(3)]
        for i in range(1, n):
            Hm, ok = _homography(corrs[i], steps[i])
            hsteps.append(Hm)
            good_h += int(ok)
        if good_h < 0.6 * (n - 1):
            use_h = False                  # too few frames give a trustworthy homography
        else:
            to_ref = chain(hsteps)
    if not use_h:
        to_ref = sim
    step_disp = [disp(np.linalg.inv(a) @ b) for a, b in zip(to_ref, to_ref[1:])] or [0.0]
    info = {"max_drift_px": max(disp(T) for T in to_ref), "drift80_px": drift,
            "median_step_px": float(np.median(step_disp)), "keyframes": keyframes,
            "model": "homography" if use_h else "similarity"}
    return to_ref, info


def _reference(T, w, h):
    """The frame in the middle of the camera's motion: the one whose view
    centre is the median of all of them (a brief glitch can't pull it)."""
    c = np.array([w / 2.0, h / 2.0, 1.0])
    pts = np.array([(A @ c)[:2] / (A @ c)[2] for A in T])
    med = np.median(pts, axis=0)
    return int(np.argmin(np.hypot(*(pts - med).T)))


def _canvas(T, w, h, max_area: float):
    """Canvas size and the reference frame's offset in it, holding every frame
    (each side grows at most so the area stays under max_area frames)."""
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    pts = []
    for A in T:
        q = A @ corners
        pts.append((q[:2] / q[2]).T)
    pts = np.vstack(pts)
    x0, y0 = float(min(pts[:, 0].min(), 0.0)), float(min(pts[:, 1].min(), 0.0))
    x1, y1 = float(max(pts[:, 0].max(), w)), float(max(pts[:, 1].max(), h))
    grow = np.sqrt(max_area)
    x0 = max(x0, -0.5 * (grow - 1) * w)
    x1 = min(x1, w + 0.5 * (grow - 1) * w)
    y0 = max(y0, -0.5 * (grow - 1) * h)
    y1 = min(y1, h + 0.5 * (grow - 1) * h)
    ox, oy = int(np.ceil(-x0)), int(np.ceil(-y0))
    W, H = int(np.ceil(x1)) + ox, int(np.ceil(y1)) + oy
    return W + W % 2, H + H % 2, ox, oy


def stabilize(frames, mode="auto", drift_px=4.0, step_px=0.15, *, big_frac: float = 0.06,
              max_canvas: float = 2.5):
    """Return (frames, transforms, info). Frames are warped only if the camera moved.

    A locked-off camera still shows a little estimation noise, and warping by
    that noise hurts more than it helps, so a tripod shot (tiny typical step,
    small total drift) is left alone. transforms map each frame's pixels to
    the canvas (3x3). info["canvas"] = (W, H, ox, oy): the canvas size and
    where the reference view's corner sits in it; info["footprints"]: each
    frame's outline on the canvas (only when the canvas is bigger than a
    frame).
    """
    if mode == "off" or len(frames) < 3:
        return frames, None, {"applied": False}
    model = mode if mode in ("similarity", "homography") else "auto"
    T, info = estimate_motion(frames, model=model, big_frac=big_frac)
    moved = info["median_step_px"] > step_px or info["max_drift_px"] > drift_px
    if mode == "auto" and not moved:
        info["applied"] = False
        return frames, None, info
    h, w = frames[0].shape[:2]
    big = info["model"] == "homography" or info["drift80_px"] > big_frac * max(w, h)
    if not big:
        # Small moves: onto the first frame, same size (the way it always was).
        out = warp(frames, T)
        info.update(applied=True, reference=0, canvas=(w, h, 0, 0), footprints=None)
        return out, T, info
    r = _reference(T, w, h)
    Rinv = np.linalg.inv(T[r])
    T = [Rinv @ A for A in T]
    Wc, Hc, ox, oy = _canvas(T, w, h, max_canvas)
    S = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1.0]])
    T = [S @ A for A in T]
    out, foot = _warp_fill(frames, T, (Wc, Hc))
    info.update(applied=True, reference=int(r), canvas=(Wc, Hc, ox, oy), footprints=foot)
    return out, T, info


def warp(frames, T, size=None):
    """Frames onto the reference view (edges smeared outward, for small moves)."""
    h, w = frames[0].shape[:2]
    W, H = size or (w, h)
    out = []
    for f, A in zip(frames, T):
        A = _as3(A)
        if np.allclose(A, np.eye(3)) and (W, H) == (w, h):
            out.append(f)
        elif np.allclose(A[2], [0, 0, 1]):
            out.append(cv2.warpAffine(f, A[:2], (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE))
        else:
            out.append(cv2.warpPerspective(f, A, (W, H), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REPLICATE))
    return out


def _warp_fill(frames, T, size):
    """Frames onto a bigger canvas. Where a frame has no picture, the median
    of the frames that do fills in, so the tracker sees only background
    there. Returns (frames, footprints)."""
    W, H = size
    h, w = frames[0].shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    foot = []
    for A in T:
        q = A @ corners
        foot.append((q[:2] / q[2]).T.astype(np.float32))
    warped = [cv2.warpPerspective(f, A, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
              for f, A in zip(frames, T)]
    ones = np.full((h, w), 255, np.uint8)

    def valid(A):
        m = cv2.warpPerspective(ones, A, (W, H), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.erode(m, np.ones((5, 5), np.uint8)) > 0
    # Median of the frames that cover each pixel (a sample of frames, in bands).
    pick = np.unique(np.linspace(0, len(frames) - 1, min(31, len(frames))).astype(int))
    masks = {i: valid(T[i]) for i in pick}
    ch = 1 if warped[0].ndim == 2 else warped[0].shape[2]
    bg = np.zeros((H, W, ch), np.uint8)
    band = 48
    for y0 in range(0, H, band):
        y1 = min(H, y0 + band)
        stack = np.stack([warped[i][y0:y1].reshape(y1 - y0, W, ch).astype(np.float32)
                          for i in pick])
        m = np.stack([masks[i][y0:y1] for i in pick])[..., None]
        stack = np.where(m, stack, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # pixels no frame covers
            med = np.nanmedian(stack, axis=0)
        bg[y0:y1] = np.nan_to_num(med, nan=0.0).astype(np.uint8)
    bg = bg if ch > 1 else bg[:, :, 0]
    out = []
    for f, A in zip(warped, T):
        m = valid(A)
        g = f.copy()
        g[~m] = bg[~m]
        out.append(g)
    return out, foot


def rescale_transforms(T, ratio: float, offset=(0.0, 0.0), dst_offset=None):
    """Transforms found on the downscaled whole frames, re-expressed for a
    crop of the frames decoded at another scale.

    ratio: new pixels per old working pixel. offset: the crop's top-left
    corner, in new pixels of the frame. dst_offset: where the output window
    starts, in new pixels of the canvas (default: the same as offset, the
    way small moves are handled).
    """
    o = np.asarray(offset, float)
    d = np.asarray(o if dst_offset is None else dst_offset, float)
    Sc = np.diag([ratio, ratio, 1.0])
    Si = np.diag([1.0 / ratio, 1.0 / ratio, 1.0])
    Ts = np.array([[1, 0, o[0]], [0, 1, o[1]], [0, 0, 1.0]])
    Td = np.array([[1, 0, -d[0]], [0, 1, -d[1]], [0, 0, 1.0]])
    return [Td @ Sc @ _as3(A) @ Si @ Ts for A in T]
