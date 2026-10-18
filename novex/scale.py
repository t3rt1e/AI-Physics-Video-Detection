"""Layer 2: rulers. Turn pixels into meters at the object's own distance.

Each ruler gives s (meters per pixel at the falling object) with a relative
uncertainty. Several rulers are fused in log space; if they disagree the
error bars widen instead of silently picking one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Systematic error of measuring a blob's width (edge blur, shading), px and relative.
EDGE_PX = 0.4
METHOD_REL = 0.015


@dataclass
class Ruler:
    name: str
    s: float            # m per px
    rel_sigma: float
    detail: str

    def to_dict(self):
        return asdict(self)


def robust_median(vals) -> tuple[float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    med = float(np.median(v))
    mad = float(1.4826 * np.median(np.abs(v - med)))
    sem = 1.2533 * mad / np.sqrt(len(v)) if len(v) > 1 else 0.1 * med
    return med, float(sem)


def ruler_from_size(diam_px, size_m: float, rel_unc: float, label: str) -> Ruler | None:
    """The object is its own ruler: known size / measured pixel width."""
    d, sem = robust_median(diam_px)
    if not np.isfinite(d) or d < 4:
        return None
    rel = float(np.sqrt((sem / d) ** 2 + (EDGE_PX / d) ** 2 + METHOD_REL**2 + rel_unc**2))
    return Ruler(name=f"size of {label}", s=size_m / d, rel_sigma=rel,
                 detail=f"{label} = {size_m * 1000:.1f} mm across {d:.1f} px")


def ruler_manual(p1, p2, length_m: float, rel_unc: float = 0.01) -> Ruler:
    """Two clicked points a known distance apart, at the object's distance."""
    px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    rel = float(np.hypot(rel_unc, 1.0 / max(px, 1.0)))
    return Ruler(name="manual ruler", s=length_m / px, rel_sigma=rel,
                 detail=f"{length_m:.3f} m over {px:.1f} px")


def fuse(rulers: list[Ruler]) -> dict | None:
    rulers = [r for r in rulers if r and np.isfinite(r.s) and r.s > 0]
    if not rulers:
        return None
    ls = np.log([r.s for r in rulers])
    sg = np.array([max(r.rel_sigma, 1e-4) for r in rulers])
    w = 1.0 / sg**2
    mean = float(np.sum(w * ls) / np.sum(w))
    sig = float(1.0 / np.sqrt(np.sum(w)))
    chi2 = float(np.sum(((ls - mean) / sg) ** 2))
    dof = len(rulers) - 1
    agree = True
    if dof > 0 and chi2 / dof > 4.0:
        agree = False
        sig *= float(np.sqrt(chi2 / dof))
    spread = float(np.exp(ls.max() - ls.min()) - 1.0) if len(rulers) > 1 else 0.0
    return {"s": float(np.exp(mean)), "rel_sigma": sig, "agree": agree, "chi2": chi2,
            "spread": spread, "rulers": [r.to_dict() for r in rulers]}
