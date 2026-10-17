"""The live loop: button press -> recording -> analysis -> VIDEOG back to the tower.

    ARM     start recording the webcam
    REL     note when the magnet let go
    GATE    collect (tower truth)
    TOWERG  collect
    REL + tail seconds: stop, analyse the window around the release, send
            VIDEOG,<id>,<g>,<sigma>,<verdict>, log a row in drops.csv

Works with the real tower (TowerLink) or FakeTower, and with a real webcam
(Camera) or SyntheticCamera for rehearsing without hardware.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from .analyze import analyze
from .tower import Drop, format_show, format_videog
from .video import Writer


class Camera:
    """USB webcam on a background thread, frames stamped with the host clock."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 60,
                 mjpg: bool = True, backend: int | None = None):
        self.index, self.req = index, (width, height, fps)
        self.mjpg = mjpg
        self.backend = backend
        self.cap = None
        self.size = (width, height)
        self.fps = float(fps)
        self.latest = None
        self._stop = threading.Event()
        self._rec_q: queue.Queue | None = None
        self._rec_thread = None
        self._rec_ts: list[float] = []
        self._rec_path = None
        self.rec_t0 = None

    def start(self):
        api = self.backend
        if api is None:
            api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open camera {self.index}")
        if self.mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        w, h, fps = self.req
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or fps)
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                time.sleep(0.01)
                continue
            self.latest = frame
            q = self._rec_q
            if q is not None:
                try:
                    q.put_nowait((frame, t))
                except queue.Full:
                    pass  # shows up as a gap in timestamps, never as a timing error

    def snapshot_jpeg(self) -> bytes | None:
        if self.latest is None:
            return None
        ok, buf = cv2.imencode(".jpg", self.latest, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else None

    def start_recording(self, path: str, drop: Drop | None = None):
        self._rec_path = path
        self._rec_ts = []
        self.rec_t0 = time.perf_counter()
        q: queue.Queue = queue.Queue(maxsize=900)
        self._rec_q = q

        def writer():
            wr = None
            while True:
                item = q.get()
                if item is None:
                    break
                frame, t = item
                if wr is None:
                    h, w = frame.shape[:2]
                    wr = Writer(path, self.fps, (w, h), crf=16)
                    self.rec_t0 = t
                wr.write(frame, t - self.rec_t0)
                self._rec_ts.append(t)
            if wr:
                wr.close()

        self._rec_thread = threading.Thread(target=writer, daemon=True)
        self._rec_thread.start()

    def stop_recording(self, drop: Drop | None = None) -> tuple[str, dict]:
        q = self._rec_q
        self._rec_q = None
        if q is not None:
            q.put(None)
        if self._rec_thread:
            self._rec_thread.join(timeout=20)
        meta = {"timestamps": self._rec_ts, "host_clock": True, "camera_index": self.index,
                "size": self.size, "fps_reported": self.fps, "rec_t0": self.rec_t0,
                "description": f"Tower drop {drop.drop_id}" if drop else "Tower drop"}
        _write_sidecar(self._rec_path, meta)
        return self._rec_path, meta

    def close(self):
        self._stop.set()
        if self.cap:
            self.cap.release()


class SyntheticCamera:
    """Rehearsal stand-in for the webcam: renders a drop matching FakeTower.

    Output is synthetic and marked so in its sidecar. Use it to practise the
    flow, never as demo evidence.
    """

    def __init__(self, tower, fps: float = 60.0, size=(1280, 720), ball_mm: float = 38.0,
                 px_per_m: float = 460.0):
        self.tower = tower
        self.fps = fps
        self.size = size
        self.ball_mm = ball_mm
        self.px_per_m = px_per_m
        self.rec_t0 = None
        self._path = None
        self.latest = None

    def start(self):
        return self

    def snapshot_jpeg(self):
        return None

    def start_recording(self, path, drop=None):
        self._path = path
        self.rec_t0 = time.perf_counter()

    def stop_recording(self, drop: Drop | None = None):
        from .synth import Motion, Scene, make_clip
        pre = 0.4
        rel = (drop.rel_host if drop and drop.rel_host else time.perf_counter()) - self.rec_t0
        self.rec_t0 = self.rec_t0 + rel - pre   # clip starts 0.4 s before release
        w, h = self.size
        sc = Scene(width=w, height=h, px_per_m=self.px_per_m, ball_d_m=self.ball_mm / 1000,
                   y0=60.0, seed=int(time.time()) % 1000)
        mo = Motion(g=self.tower.g, t_release=pre + self.tower.lag)
        make_clip(self._path, scene=sc, motion=mo, capture_fps=self.fps, duration=pre + 1.0,
                  rolling_shutter=0.5)
        meta = {"synthetic": True, "rec_t0": self.rec_t0,
                "description": f"Tower drop {drop.drop_id}, simulated" if drop else "Simulated drop"}
        _write_sidecar(self._path, meta)
        return self._path, meta

    def close(self):
        pass


def _write_sidecar(path: str, meta: dict):
    with open(os.path.splitext(path)[0] + ".json", "w") as fh:
        json.dump(meta, fh)


# ---------------------------------------------------------------------------
CSV_FIELDS = ["drop_id", "time", "gate_h_mm", "gate_t_us", "tower_g", "tower_sigma",
              "laptop_tower_g", "video_g", "video_sigma", "verdict", "k", "scale_source",
              "rolling_shutter_rel", "clip", "synthetic"]


class LiveSession:
    """Glue between the tower, the camera and the analysis."""

    def __init__(self, link, camera, data_dir: str, *, analyze_kwargs: dict | None = None,
                 on_update=None, tail_s: float = 1.4, pre_s: float = 0.4,
                 scale_correction: float = 1.0):
        self.link = link
        self.camera = camera
        self.dir = os.path.join(data_dir, "drops")
        os.makedirs(self.dir, exist_ok=True)
        self.csv_path = os.path.join(data_dir, "drops.csv")
        self.kw = dict(analyze_kwargs or {"preset": "steel-38mm"})
        # The tower drops one ball: measure just that, and keep the live loop quick.
        self.kw.setdefault("max_objects", 1)
        self.on_update = on_update or (lambda msg: None)
        self.tail_s, self.pre_s = tail_s, pre_s
        self.scale_correction = scale_correction
        self.drops: dict[int, Drop] = {}
        self.history: list[dict] = []
        self._lock = threading.Lock()
        link.on_event = self._on_event

    # ---- tower events -----------------------------------------------------
    def _on_event(self, ev):
        self.on_update({"type": "tower", "kind": ev.kind, "drop_id": ev.drop_id,
                        "fields": ev.fields, "raw": ev.raw})
        if ev.drop_id is None:
            return
        with self._lock:
            d = self.drops.setdefault(ev.drop_id, Drop(ev.drop_id))
        d.apply(ev)
        if ev.kind == "ARM":
            path = os.path.join(self.dir, f"drop_{ev.drop_id:03d}_webcam.mp4")
            if os.path.exists(path):  # tower rebooted and reused an id: never overwrite
                stamp = datetime.now().strftime("%H%M%S")
                path = os.path.join(self.dir, f"drop_{ev.drop_id:03d}_{stamp}_webcam.mp4")
            self.camera.start_recording(path, d)
            self.link.send(format_show("RECORDING"))
        elif ev.kind == "REL":
            threading.Thread(target=self._finish_later, args=(d,), daemon=True).start()

    def _finish_later(self, d: Drop):
        time.sleep(self.tail_s)
        try:
            self._finish(d)
        except Exception as e:  # keep the loop alive on stage
            self.on_update({"type": "error", "drop_id": d.drop_id, "error": str(e)})
            self.link.send(format_videog(d.drop_id, 0, 0, "UNSURE"))

    def _finish(self, d: Drop):
        path, meta = self.camera.stop_recording(d)
        rel_v = (d.rel_host - (meta.get("rec_t0") or d.rel_host)) if d.rel_host else None
        t0 = max(rel_v - self.pre_s, 0.0) if rel_v is not None else None
        t1 = rel_v + self.tail_s if rel_v is not None else None
        self.on_update({"type": "analyzing", "drop_id": d.drop_id})
        r = analyze(path, t_start=t0, t_end=t1, scale_correction=self.scale_correction,
                    **self.kw)
        v = r["verdict"]
        self.link.send(format_videog(d.drop_id, v.get("g"), v.get("sigma"), v["verdict"]))
        with open(os.path.splitext(path)[0] + ".result.json", "w") as fh:
            json.dump(r, fh, default=_jsonable)
        lt = d.laptop_fit()
        row = {
            "drop_id": d.drop_id, "time": datetime.now().isoformat(timespec="seconds"),
            "gate_h_mm": ";".join(f"{g[1]:.1f}" for g in sorted(d.gates)),
            "gate_t_us": ";".join(str(g[2]) for g in sorted(d.gates)),
            "tower_g": d.tower_g, "tower_sigma": d.tower_sigma,
            "laptop_tower_g": lt["g"] if lt else None,
            "video_g": v.get("g"), "video_sigma": v.get("sigma"), "verdict": v["verdict"],
            "k": v.get("k"),
            "scale_source": (r.get("scale") or {}).get("rulers", [{}])[0].get("name")
            if r.get("scale") else None,
            "rolling_shutter_rel": r.get("fit", {}).get("rolling_shutter", {}).get("effect_rel"),
            "clip": "drops/" + os.path.basename(path),
            "synthetic": bool(meta.get("synthetic")),
        }
        _append_csv(self.csv_path, row)
        summary = {"type": "drop", "drop_id": d.drop_id, "tower_g": d.tower_g,
                   "tower_sigma": d.tower_sigma, "video_g": v.get("g"),
                   "video_sigma": v.get("sigma"), "verdict": v["verdict"],
                   "headline": v.get("headline"), "clip": row["clip"],
                   "synthetic": row["synthetic"], "result": r}
        self.history.append({k: val for k, val in summary.items() if k != "result"})
        self.on_update(summary)


def _append_csv(path: str, row: dict):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CSV_FIELDS})


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def calibrate(csv_path: str) -> dict:
    """Compare video g with tower g over logged drops: the measured scale error."""
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            try:
                tg, vg = float(r["tower_g"]), float(r["video_g"])
            except (TypeError, ValueError):
                continue
            if r.get("synthetic") == "True" or r.get("verdict") not in ("REAL", "SLOWMO"):
                continue
            rows.append((int(r["drop_id"]), tg, vg))
    if not rows:
        return {"n": 0}
    ratio = np.array([vg / tg for _, tg, vg in rows])
    med = float(np.median(ratio))
    return {
        "n": len(rows),
        "median_ratio": med,
        "mean_error_pct": float((ratio.mean() - 1) * 100),
        "spread_pct": float(ratio.std(ddof=1) * 100) if len(rows) > 1 else float("nan"),
        "worst_pct": float(np.max(np.abs(ratio - 1)) * 100),
        "scale_correction": 1.0 / med,
        "drops": [{"drop_id": i, "tower_g": tg, "video_g": vg} for i, tg, vg in rows],
    }
