"""Galileo detector site: upload a clip, get a verdict, scrub it frame by frame.

Run with:  python -m galileo.site            (then open http://localhost:8080)

Each upload becomes a job in <data>/jobs/<id>/:
    input.<ext>     the file as uploaded
    preview.mp4     browser-friendly copy, every frame kept, exact timestamps
    debug.mp4       what Galileo saw (the repo's own debug video)
    frames.json     per-frame times for both videos, so the page can step frame by frame
    result.json     every number the analysis recorded
    job.json        status, options, timings
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .. import config as C

STATIC = Path(__file__).parent / "static"
SAMPLES = Path(__file__).parent / "samples"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
MAX_UPLOAD = 1_500_000_000          # bytes
MAX_PREVIEW_FRAMES = 6000


# ---------------------------------------------------------------------------
def clean(o):
    """JSON-safe copy: numpy to python, NaN/inf to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(obj)))
    os.replace(tmp, path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


# ---------------------------------------------------------------------------
def make_preview(src: str, out: Path, info, max_side: int = 960, on_progress=None):
    """Re-encode every frame as H.264 with its true timestamp. Returns frame info.

    Streams one frame at a time, so a 4K 240 fps clip doesn't fill memory.
    Times are seconds from the first frame, which is also how the debug video
    and the tracker's `t` values are measured.
    """
    from ..video import Writer, iter_frames
    s = min(1.0, max_side / max(info.width, info.height, 1))
    times, nums = [], []
    writer = None
    first = 0.0
    truncated = False
    tmp = out.with_suffix(".part.mp4")
    for img, t, num in iter_frames(src, info, scale=s):
        if len(times) >= MAX_PREVIEW_FRAMES:
            truncated = True
            break
        if writer is None:
            h, w = img.shape[:2]
            writer = Writer(str(tmp), float(info.fps or 30.0), (w, h), crf=21)
        t = float(t)
        if not times:
            first = t
        elif t <= times[-1]:
            t = times[-1] + 1e-4
        times.append(t)
        writer.write(img, t - first)
        nums.append(int(num))
        if on_progress and len(times) % 100 == 0:
            on_progress(len(times))
    if writer is None:
        raise RuntimeError("No frames could be decoded from this file")
    writer.close()
    os.replace(tmp, out)
    t0 = times[0]
    return {"times": [round(t - t0, 6) for t in times], "nums": nums,
            "width": writer.w, "height": writer.h, "truncated": truncated}


def to_webm(src: Path, out: Path) -> None:
    """VP8 copy of an MP4, same frames and timestamps, for browsers without H.264."""
    import av
    from fractions import Fraction
    tmp = out.with_suffix(".part.webm")
    with av.open(str(src)) as inp, av.open(str(tmp), "w", format="webm") as o:
        vs = inp.streams.video[0]
        st = o.add_stream("libvpx", rate=Fraction(vs.average_rate or 30).limit_denominator(1001))
        st.width, st.height, st.pix_fmt = vs.codec_context.width, vs.codec_context.height, "yuv420p"
        st.bit_rate = 3_000_000
        st.options = {"deadline": "realtime", "cpu-used": "8", "lag-in-frames": "0"}
        st.time_base = Fraction(1, 1000)
        last = -1
        for fr in inp.decode(vs):
            pts = int(round(float(fr.pts * vs.time_base) * 1000))
            if pts <= last:
                pts = last + 1
            last = pts
            nf = av.VideoFrame.from_ndarray(fr.to_ndarray(format="bgr24"), format="bgr24")
            nf.pts, nf.time_base = pts, st.time_base
            for pk in st.encode(nf):
                o.mux(pk)
        for pk in st.encode():
            o.mux(pk)
    os.replace(tmp, out)


from .pipeline import analyse, debug_timing


# ---------------------------------------------------------------------------
class Jobs:
    def __init__(self, root: Path):
        self.root = root
        self.dir = root / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.live: dict[str, dict] = {}
        # Anything left "running" by a previous process can't finish.
        for p in self.dir.glob("*/job.json"):
            j = _read_json(p, {})
            if j.get("status") in ("queued", "running"):
                j.update(status="error", error="The server restarted while this was running.")
                _write_json(p, j)

    def path(self, jid: str) -> Path:
        if not jid or not all(c.isalnum() or c in "-_" for c in jid):
            raise HTTPException(400, "Bad job id")
        p = self.dir / jid
        if not p.is_dir():
            raise HTTPException(404, "No such analysis")
        return p

    def get(self, jid: str) -> dict:
        with self.lock:
            if jid in self.live:
                return dict(self.live[jid])
        return _read_json(self.path(jid) / "job.json", {}) or {}

    def update(self, jid: str, **kw):
        with self.lock:
            j = self.live.setdefault(jid, _read_json(self.dir / jid / "job.json", {}) or {})
            j.update(kw)
            snap = dict(j)
        _write_json(self.dir / jid / "job.json", snap)

    def create(self, src_path: Path, filename: str, opts: dict) -> str:
        jid = uuid.uuid4().hex[:10]
        d = self.dir / jid
        d.mkdir()
        ext = src_path.suffix.lower()
        shutil.move(str(src_path), str(d / f"input{ext}"))
        job = {"id": jid, "filename": filename, "status": "queued", "stage": "waiting",
               "created": time.time(), "options": opts, "input": f"input{ext}"}
        with self.lock:
            self.live[jid] = job
        _write_json(d / "job.json", job)
        self.pool.submit(self.run, jid)
        return jid

    def run(self, jid: str):
        d = self.dir / jid
        job = self.get(jid)
        src = d / job["input"]
        t_wall = time.time()
        try:
            from ..video import probe
            self.update(jid, status="running", stage="Reading the video")
            info = probe(str(src))
            self.update(jid, file={"width": info.width, "height": info.height,
                                   "fps": info.fps, "duration": info.duration,
                                   "n_frames": info.n_frames})
            self.update(jid, stage="Preparing frames for playback")
            fr = make_preview(str(src), d / "preview.mp4", info,
                              on_progress=lambda n: self.update(
                                  jid, stage=f"Preparing frames for playback ({n} frames)"))
            _write_json(d / "frames.json", fr)
            self.update(jid, preview=True)

            def say(msg):
                self.update(jid, stage=str(msg).strip().capitalize())

            self.update(jid, stage="Measuring the fall")
            result = analyse(d, src, job["options"], say)
            result = clean(result)
            _write_json(d / "result.json", result)

            fr.update(debug_timing(d / "debug.mp4", fr))
            _write_json(d / "frames.json", fr)
            v = result.get("verdict") or {}
            self.update(jid, status="done", stage="Done", finished=time.time(),
                        elapsed=time.time() - t_wall, verdict=v.get("verdict"),
                        g=v.get("g"), sigma=v.get("sigma"), confidence=v.get("confidence"),
                        headline=v.get("headline"), debug=(d / "debug.mp4").is_file(), debug_sync=fr["debug_sync"])
        except Exception as e:
            traceback.print_exc()
            self.update(jid, status="error", stage="Failed", error=str(e) or type(e).__name__)

    def list(self) -> list[dict]:
        out = []
        for p in self.dir.iterdir():
            if not p.is_dir():
                continue
            try:
                j = self.get(p.name)
            except HTTPException:
                continue
            if j:
                out.append({k: j.get(k) for k in (
                    "id", "filename", "status", "stage", "created", "verdict", "g", "sigma",
                    "confidence", "headline", "error", "file", "elapsed")})
        out.sort(key=lambda j: j.get("created") or 0, reverse=True)
        return out


# ---------------------------------------------------------------------------
def create_app(data_dir: str = "site_data") -> FastAPI:
    root = Path(data_dir)
    jobs = Jobs(root)
    app = FastAPI(title="Galileo detector", version=__version__)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((STATIC / "index.html").read_text(),
                            headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    def status():
        from .. import gemini, identify
        g_ok, g_why = gemini.available()
        c_ok, c_why = identify.clip_available()
        if c_ok:  # the repo only checks torch and transformers; CLIP's image processor needs these too
            for mod, pkg in (("torchvision", "torchvision"), ("PIL", "pillow")):
                try:
                    __import__(mod)
                except Exception:
                    c_ok, c_why = False, f"pip install {pkg} (CLIP needs it)"
                    break
        presets = [{"key": k, "label": v["label"], "shape": v.get("shape", "round"),
                    "size_mm": round(v["size_m"] * 1000, 1)} for k, v in C.OBJECT_PRESETS.items()]
        samples = []
        if SAMPLES.is_dir():
            for p in sorted(SAMPLES.glob("*.mp4")):
                meta = _read_json(p.with_suffix(".label.json"), {}) or {}
                samples.append({"name": p.name, "description": meta.get("description") or p.stem,
                                "ai_generated": bool(meta.get("ai_generated")),
                                "object": meta.get("object")})
        return clean({"version": __version__, "g_ref": C.G_EARTH, "presets": presets,
                      "auto": {"available": c_ok or g_ok, "clip": c_ok, "gemini": g_ok,
                               "detail": c_why if not (c_ok or g_ok) else None},
                      "samples": samples})

    @app.get("/api/jobs")
    def list_jobs():
        return clean(jobs.list())

    @app.post("/api/jobs")
    async def create_job(file: UploadFile = File(...), object: str = Form("auto"),
                         size_mm: str = Form(""), seed: str = Form(""), hfov: str = Form("")):
        name = Path(file.filename or "clip.mp4").name
        ext = os.path.splitext(name)[1].lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(400, f"That doesn't look like a video file ({name}).")
        tmp = root / f"upload_{uuid.uuid4().hex}{ext}"
        size = 0
        with open(tmp, "wb") as fh:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    fh.close()
                    tmp.unlink(missing_ok=True)
                    raise HTTPException(413, "That file is over 1.5 GB.")
                fh.write(chunk)
        try:
            opts = _options(object, size_mm, seed, hfov)
        except HTTPException:
            tmp.unlink(missing_ok=True)
            raise
        return {"id": jobs.create(tmp, name, opts)}

    @app.post("/api/jobs/sample")
    def create_sample(body: dict):
        name = Path(str(body.get("name", ""))).name
        src = SAMPLES / name
        if not src.is_file():
            raise HTTPException(404, "No such sample")
        meta = _read_json(src.with_suffix(".label.json"), {}) or {}
        opts = _options(body.get("object") or meta.get("object") or "steel-38mm",
                        body.get("size_mm") or "", "", "")
        tmp = root / f"upload_{uuid.uuid4().hex}{src.suffix}"
        shutil.copyfile(src, tmp)
        return {"id": jobs.create(tmp, name, opts)}

    @app.post("/api/jobs/{jid}/rerun")
    def rerun(jid: str, body: dict):
        """Measure the same video again with an object (or size) the user chose."""
        d = jobs.path(jid)
        job = jobs.get(jid)
        src = d / job.get("input", "")
        if not src.is_file():
            raise HTTPException(404, "The original video is gone")
        opts = _options(body.get("object") or "auto", body.get("size_mm") or "", "", "")
        tmp = root / f"upload_{uuid.uuid4().hex}{src.suffix}"
        shutil.copyfile(src, tmp)
        return {"id": jobs.create(tmp, job.get("filename") or src.name, opts)}

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        d = jobs.path(jid)
        job = jobs.get(jid)
        out = {"job": clean(job)}
        if job.get("status") == "done":
            out["result"] = _read_json(d / "result.json")
        if (d / "frames.json").exists():
            fr = _read_json(d / "frames.json")
            out["frames"] = fr
        return JSONResponse(out)

    @app.delete("/api/jobs/{jid}")
    def delete_job(jid: str):
        d = jobs.path(jid)
        if jobs.get(jid).get("status") in ("queued", "running"):
            raise HTTPException(409, "Still running")
        shutil.rmtree(d, ignore_errors=True)
        with jobs.lock:
            jobs.live.pop(jid, None)
        return {"ok": True}

    @app.get("/media/{jid}/{which}.mp4")
    def media(jid: str, which: str):
        if which not in ("preview", "debug"):
            raise HTTPException(404)
        p = jobs.path(jid) / f"{which}.mp4"
        if not p.exists():
            raise HTTPException(404, "Not ready yet")
        return FileResponse(p, media_type="video/mp4",
                            headers={"Cache-Control": "private, max-age=3600"})

    webm_lock = threading.Lock()

    @app.get("/media/{jid}/{which}.webm")
    def media_webm(jid: str, which: str):
        """Fallback for browsers that can't play H.264: same video as VP8, made on first use."""
        if which not in ("preview", "debug"):
            raise HTTPException(404)
        d = jobs.path(jid)
        src, out = d / f"{which}.mp4", d / f"{which}.webm"
        if not src.exists():
            raise HTTPException(404, "Not ready yet")
        with webm_lock:
            if not out.exists():
                try:
                    to_webm(src, out)
                except Exception as e:
                    raise HTTPException(500, f"Couldn't convert the video: {e}")
        return FileResponse(out, media_type="video/webm")

    @app.get("/download/{jid}/result.json")
    def dl_result(jid: str):
        p = jobs.path(jid) / "result.json"
        if not p.exists():
            raise HTTPException(404, "Not ready yet")
        return FileResponse(p, media_type="application/json",
                            filename=f"galileo_{jid}_result.json")

    @app.get("/download/{jid}/track.csv")
    def dl_track(jid: str):
        r = _read_json(jobs.path(jid) / "result.json")
        if not r:
            raise HTTPException(404, "Not ready yet")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["t_s", "x_frac", "y_frac", "radius_frac"])
        for p in (r.get("track") or {}).get("points") or []:
            w.writerow([p["t"], p["x"], p["y"], p["r"]])
        return Response(buf.getvalue(), media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="galileo_{jid}_track.csv"'})

    return app


def _options(obj: str, size_mm, seed, hfov) -> dict:
    obj = (obj or "auto").strip()
    if obj not in ("auto", "custom", "gemini") and obj not in C.OBJECT_PRESETS:
        raise HTTPException(400, f"Unknown object: {obj}")
    opts: dict = {"object": obj}
    if obj == "custom":
        try:
            v = float(size_mm)
            assert math.isfinite(v) and 1 <= v <= 5000
        except Exception:
            raise HTTPException(400, "Give the object's size in millimetres (1 to 5000).")
        opts["size_mm"] = v
    if seed:
        try:
            t, x, y = (float(v) for v in str(seed).split(","))
            assert all(math.isfinite(v) for v in (t, x, y))
        except Exception:
            raise HTTPException(400, "seed is 'time,x,y'")
        opts["seed"] = {"t": t, "x": x, "y": y}
    if hfov:
        try:
            opts["hfov"] = float(hfov)
            if not math.isfinite(opts["hfov"]) or not 10 <= opts["hfov"] < 180:
                raise ValueError
        except (ValueError, TypeError):
            raise HTTPException(400, "Field of view must be from 10 to less than 180 degrees.")
    return opts
