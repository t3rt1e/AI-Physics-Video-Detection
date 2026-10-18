"""Second ruler: Apple Depth Pro (metric depth + focal length from one frame).

m per px at the object = depth at the object / focal length in px.

Optional. Needs `pip install torch transformers` (see requirements-ml.txt)
and downloads apple/DepthPro-hf (about 2 GB) on first use. Runs on CPU in
tens of seconds per frame; on the 5070 Ti it is well under a second.
Treat it as a ~25% ruler: good enough to back up or question the object-size
ruler, never a replacement for it on tower clips.
"""
from __future__ import annotations

import numpy as np

from .scale import Ruler

MODEL_ID = "apple/DepthPro-hf"
_state: dict = {}


def _load():
    if "model" in _state:
        return _state["model"], _state["proc"], _state["device"]
    import torch
    from transformers import AutoImageProcessor, DepthProForDepthEstimation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = DepthProForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    _state.update(model=model, proc=proc, device=device)
    return model, proc, device


def depth_pro_ruler(frame_bgr: np.ndarray, xy, d_px: float) -> Ruler | None:
    import torch
    model, proc, device = _load()
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    inputs = proc(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    post = proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
    depth = post["predicted_depth"].float().cpu().numpy()
    focal = post.get("focal_length")
    if focal is None:
        return None
    focal = float(focal)
    x, y = xy
    yy, xx = np.mgrid[0:h, 0:w]
    inner = (xx - x) ** 2 + (yy - y) ** 2 <= (0.3 * d_px) ** 2
    if inner.sum() < 3:
        inner = (xx - x) ** 2 + (yy - y) ** 2 <= max(0.5 * d_px, 2.0) ** 2
    z = float(np.median(depth[inner]))
    if not np.isfinite(z) or z <= 0:
        return None
    return Ruler(name="Depth Pro", s=z / focal, rel_sigma=0.25,
                 detail=f"object {z:.2f} m away, focal length {focal:.0f} px")
