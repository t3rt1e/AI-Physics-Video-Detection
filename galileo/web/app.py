"""Galileo web app: side-by-side clip analysis and the live tower readout.

Run with:  python -m galileo serve [--fake-tower | --tower COM5 --camera 0]
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .. import config as C
from ..analyze import analyze

STATIC = Path(__file__).parent / "static"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
GROUPS = ("clips", "drops", "uploads")


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
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def _file_key(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha1(f"{path.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
    return h.hexdigest()[:16]


def _label(path: Path) -> dict:
    meta = {}
    for side in (path.with_suffix(".label.json"), path.with_suffix(".json")):
        if side.exists():
            try:
                meta.update(json.loads(side.read_text()))
            except Exception:
                pass
    return meta


def create_app(data_dir: str = "data", *, tower_port: str | None = None,
               fake_tower: bool = False, camera: int | None = None, cam_size=(1280, 720),
               cam_fps: int = 60, preset: str = "steel-38mm", use_gemini: bool = False,
               calibration: str | None = None) -> FastAPI:
    data = Path(data_dir)
    for g in GROUPS + ("cache",):
        (data / g).mkdir(parents=True, exist_ok=True)
    state: dict = {"clients": set(), "loop": None, "live": None, "tower": None,
                   "camera": None, "last_drop": None, "tower_log": []}
    sem = threading.Semaphore(2)
    scale_corr = 1.0
    cal_info = None
    if calibration:
        cal_info = json.loads(Path(calibration).read_text())
        scale_corr = float(cal_info.get("scale_correction", 1.0))

    # ---- live tower ------------------------------------------------------
    def push(msg: dict):
        loop = state["loop"]
        if loop is None:
            return
        if msg.get("type") == "drop":
            state["last_drop"] = msg
        if msg.get("type") == "tower":
            state["tower_log"] = (state["tower_log"] + [msg])[-50:]
        asyncio.run_coroutine_threadsafe(_broadcast(clean(msg)), loop)

    async def _broadcast(msg):
        dead = []
        for ws in list(state["clients"]):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state["clients"].discard(ws)

    def start_live():
        if not (tower_port or fake_tower):
            return
        from ..live import Camera, LiveSession, SyntheticCamera
        from ..tower import FakeTower, TowerLink
        link = FakeTower() if fake_tower else TowerLink(tower_port)
        if fake_tower:  # keep numbering across restarts
            link.next_id = 1 + len(list((data / "drops").glob("drop_*_webcam.mp4")))
        if camera is not None:
            cam = Camera(camera, *cam_size, fps=cam_fps).start()
        elif fake_tower:
            cam = SyntheticCamera(link)
        else:
            raise SystemExit("--tower needs --camera N (the webcam that films the drop)")
        live = LiveSession(link, cam, str(data), analyze_kwargs={"preset": preset},
                           on_update=push, scale_correction=scale_corr)
        link.start()
        state.update(live=live, tower=link, camera=cam)

    @asynccontextmanager
    async def lifespan(app):
        state["loop"] = asyncio.get_running_loop()
        start_live()
        yield
        if state["tower"]:
            state["tower"].close()
        if state["camera"]:
            state["camera"].close()

    app = FastAPI(title="Galileo", version=__version__, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # ---- pages & status -------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/status")
    def status():
        from .. import gemini
        ok, why = gemini.available()
        t = state["tower"]
        return clean({
            "version": __version__,
            "gemini": {"available": ok, "detail": why, "default_on": use_gemini and ok},
            "tower": None if t is None else {
                "kind": "simulated" if fake_tower else "serial",
                "port": tower_port, "connected": bool(getattr(t, "connected", False))},
            "camera": None if state["camera"] is None else {
                "synthetic": camera is None, "index": camera},
            "presets": {k: v["label"] for k, v in C.OBJECT_PRESETS.items()},
            "default_object": preset,
            "calibration": cal_info,
            "last_drop": _strip(state["last_drop"]),
            "g_ref": C.G_EARTH,
        })

    def _strip(msg):
        if not msg:
            return None
        return {k: v for k, v in msg.items() if k != "result"}

    # ---- clips ------------------------------------------------------------
    def _resolve(clip_id: str) -> Path:
        group, _, name = clip_id.partition("/")
        if group not in GROUPS or not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(400, "Unknown clip id")
        p = data / group / name
        if not p.exists():
            raise HTTPException(404, f"No clip named {clip_id}")
        return p

    @app.get("/api/clips")
    def clips():
        out = []
        for g in GROUPS:
            files = sorted((data / g).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in files:
                if p.suffix.lower() not in VIDEO_EXT:
                    continue
                meta = _label(p)
                out.append({"id": f"{g}/{p.name}", "group": g, "name": p.name,
                            "description": meta.get("description"),
                            "synthetic": bool(meta.get("synthetic")),
                            "ai_generated": bool(meta.get("ai_generated")),
                            "size_mb": round(p.stat().st_size / 1e6, 1)})
        return out

    @app.get("/media/{group}/{name}")
    def media(group: str, name: str):
        p = _resolve(f"{group}/{name}")
        mt = {".mov": "video/quicktime", ".webm": "video/webm"}.get(p.suffix.lower(), "video/mp4")
        return FileResponse(p, media_type=mt)

    preview_lock = threading.Lock()

    @app.get("/preview/{group}/{name}")
    async def preview(group: str, name: str):
        """Same clip as WebM, for browsers that can't decode the original."""
        from ..video import make_preview
        p = _resolve(f"{group}/{name}")
        out = data / "cache" / f"preview_{_file_key(p)}.webm"
        if not out.exists():
            def build():
                with preview_lock:
                    if not out.exists():
                        make_preview(str(p), str(out))
            try:
                await run_in_threadpool(build)
            except Exception as e:
                raise HTTPException(500, f"Couldn't make a preview: {e}")
        return FileResponse(out, media_type="video/webm")

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        name = Path(file.filename or "clip.mp4").name
        stem, ext = os.path.splitext(name)
        if ext.lower() not in VIDEO_EXT:
            raise HTTPException(400, f"Not a video file: {name}")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60] or "clip"
        dest = data / "uploads" / f"{safe}{ext.lower()}"
        i = 1
        while dest.exists():
            dest = data / "uploads" / f"{safe}_{i}{ext.lower()}"
            i += 1
        with open(dest, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        return {"id": f"uploads/{dest.name}"}

    @app.post("/api/analyze")
    async def api_analyze(body: dict):
        p = _resolve(str(body.get("clip", "")))
        obj = body.get("object") or preset
        size_mm = body.get("size_mm")
        gem = bool(body.get("gemini"))
        roi = body.get("roi")
        opts = {"object": obj, "size_mm": size_mm, "gemini": gem, "roi": roi,
                "corr": scale_corr if p.parent.name == "drops" else 1.0, "v": __version__}
        key = _file_key(p) + "_" + hashlib.sha1(json.dumps(opts, sort_keys=True)
                                                .encode()).hexdigest()[:10]
        cache = data / "cache" / f"{key}.json"
        if cache.exists() and not body.get("force"):
            return JSONResponse(json.loads(cache.read_text()))
        kw = {"use_gemini": gem, "roi": roi, "scale_correction": opts["corr"]}
        if obj == "custom" and size_mm:
            kw.update(size_m=float(size_mm) / 1000.0, size_rel_unc=0.01, label="object")
        elif obj == "gemini":
            kw["use_gemini"] = True
        else:
            kw["preset"] = obj

        def run():
            with sem:
                return analyze(str(p), **kw)

        try:
            r = await run_in_threadpool(run)
        except Exception as e:  # report, don't crash the demo
            raise HTTPException(500, f"Analysis failed: {e}")
        r = clean(r)
        r["clip"] = {"id": body["clip"], "url": f"/media/{body['clip']}", **_label(p)}
        cache.write_text(json.dumps(r))
        return JSONResponse(r)

    # ---- tower ------------------------------------------------------------
    @app.post("/api/tower/drop")
    def tower_drop():
        t = state["tower"]
        if t is None or not hasattr(t, "drop"):
            raise HTTPException(400, "The simulated tower isn't running (start with --fake-tower).")
        did = t.drop()
        if did is None:
            raise HTTPException(409, "A drop is already in progress.")
        return {"drop_id": did}

    @app.get("/api/tower/last")
    def tower_last():
        d = state["last_drop"]
        if not d:
            return {"drop": None}
        return clean({"drop": d})

    @app.get("/api/camera.jpg")
    def camera_jpg():
        cam = state["camera"]
        jpg = cam.snapshot_jpeg() if cam else None
        if not jpg:
            raise HTTPException(404, "No camera frame yet.")
        return Response(jpg, media_type="image/jpeg")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        state["clients"].add(sock)
        try:
            while True:
                await sock.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            state["clients"].discard(sock)

    return app
