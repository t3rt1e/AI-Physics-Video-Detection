"""SAM 2 tracker for cluttered clips (people, busy backgrounds, moving cameras).

Same output as the classical tracker, so the rest of the pipeline doesn't
care which one ran. Needs the official SAM 2 package and torch:

    pip install "git+https://github.com/facebookresearch/sam2.git"

Weights download from Hugging Face on first use (facebook/sam2.1-hiera-small,
about 180 MB). Wants a GPU; on CPU expect seconds per frame.

Seeding: give a box (x0, y0, x1, y1) in working pixels on the first frame
where the object is clearly visible, or let the classical tracker's best
guess seed it.
"""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from .track import Det, Track, TrackResult, _width_along, track_falling_object

MODEL_ID = os.environ.get("GALILEO_SAM2_MODEL", "facebook/sam2.1-hiera-small")
_state: dict = {}


def _predictor():
    if "p" in _state:
        return _state["p"]
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _state["p"] = SAM2VideoPredictor.from_pretrained(MODEL_ID, device=device)
    _state["device"] = device
    return _state["p"]


def _mask_det(mask: np.ndarray, fi: int, prev: Det | None) -> Det | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 6:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    X, Y = xs.astype(float), ys.astype(float)
    ones = np.ones_like(X)
    zeros = np.zeros_like(X)   # a mask has no noise floor to model
    if prev is not None and np.hypot(cx - prev.x, cy - prev.y) > 0.5 * prev.d:
        vx, vy = cx - prev.x, cy - prev.y
        n = np.hypot(vx, vy)
        width = _width_along(ones, zeros, X, Y, cx, cy, (-vy / n, vx / n))
    else:
        width = float(np.nanmean([_width_along(ones, zeros, X, Y, cx, cy, (1.0, 0.0)),
                                  _width_along(ones, zeros, X, Y, cx, cy, (0.0, 1.0))]))
    return Det(f=fi, x=cx, y=cy, d=width, area=float(len(xs)), elong=1.0, contrast=255.0)


def track_with_sam2(frames, times, box=None) -> TrackResult:
    import torch
    seed_f = 0
    if box is None:
        guess = track_falling_object(frames, times)
        if guess.main is None:
            raise RuntimeError("SAM 2 needs a starting box: none given and no moving object found.")
        d0 = guess.main.dets[len(guess.main.dets) // 4]
        r = 1.2 * d0.d
        box = (d0.x - r, d0.y - r, d0.x + r, d0.y + r)
        seed_f = d0.f
    pred = _predictor()
    masks: dict[int, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if _state.get("device") == "cuda" else _nullctx())
        with torch.inference_mode(), ctx:
            state = pred.init_state(video_path=tmp)
            pred.add_new_points_or_box(state, frame_idx=seed_f, obj_id=1,
                                       box=np.asarray(box, np.float32))
            for reverse in (False, True):
                for fi, _ids, logits in pred.propagate_in_video(state, start_frame_idx=seed_f,
                                                                reverse=reverse):
                    masks[int(fi)] = (logits[0] > 0.0).cpu().numpy().squeeze()
    dets, prev = [], None
    for fi in sorted(masks):
        d = _mask_det(masks[fi], fi, prev)
        if d is not None:
            dets.append(d)
            prev = d
    tr = Track(dets=dets, kind="moving", score=1.0)
    return TrackResult(main=tr if len(dets) >= 6 else None, others=[], rest=None, rest_d=None,
                       n_candidates=len(dets), d_fine=np.array([d.d for d in dets]))


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
