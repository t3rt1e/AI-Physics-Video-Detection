"""Small, site-only adapter to Galileo's public analysis API.

No physics, classifications or private debug structures are copied here.
New result fields are passed through unchanged to the site's All data view.
"""
from pathlib import Path

import numpy as np


def analyse(job_dir: Path, src: Path, opts: dict, say):
    from ..analyze import analyze

    kw = {"use_gemini": bool(opts.get("gemini"))}
    obj = opts.get("object") or "auto"
    if opts.get("seed"):
        seed = opts["seed"]
        kw["seed"] = tuple(float(seed[k]) for k in ("t", "x", "y"))
    if obj == "custom":
        kw.update(size_m=float(opts["size_mm"]) / 1000, size_rel_unc=0.01, label="object")
    elif obj == "gemini":
        kw.update(use_gemini=True, preset=None)
    else:
        kw["preset"] = None if obj == "auto" else obj
    if opts.get("hfov"):
        kw["hfov"] = float(opts["hfov"])
    return analyze(str(src), debug_video=str(job_dir / "debug.mp4"),
                   debug_speed=1.0, progress=say, **kw)


def debug_timing(path: Path, preview: dict) -> dict:
    """Read output timestamps, without depending on the core's private dbg dict.

    A one-to-one whole-clip timeline is safe to synchronize. Cropped or
    downsampled debug renders keep their own frame controls instead of guessing
    the source offset. A render failure must never discard a valid verdict.
    """
    import av

    out = {"debug_times": None, "debug_frame_times": [], "debug_sync": False}
    if not path.is_file():
        return out
    try:
        with av.open(str(path)) as container:
            times = [float(f.pts * f.time_base) for f in container.decode(video=0)
                     if f.pts is not None]
        if not times or not all(np.isfinite(times)) or any(
                b <= a for a, b in zip(times, times[1:])):
            return out
        out["debug_frame_times"] = times
        source = preview.get("times", [])
        if (not preview.get("truncated") and len(source) == len(times)
                and np.allclose(source, times, atol=0.001, rtol=0)):
            out.update(debug_times=times, debug_sync=True)
    except Exception:
        # The browser/download can still expose the core's original artifact.
        pass
    return out
