"""Video in and out, with exact timestamps and capture-rate metadata.

Uses PyAV (bundled FFmpeg) so every frame keeps its presentation timestamp
and the container's tags are readable. Falls back to OpenCV if PyAV is
missing (timestamps then come from the nominal frame rate).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import cv2
import numpy as np

try:
    import av  # type: ignore
except Exception:  # pragma: no cover
    av = None

_CAPTURE_KEYS = ("com.android.capture.fps", "com.apple.quicktime.capture-fps",
                 "capture_fps", "com.apple.quicktime.camera.capture-fps")


@dataclass
class ClipInfo:
    path: str
    width: int                 # display orientation
    height: int
    fps: float                 # nominal playback rate of the file
    n_frames: int
    duration: float
    rotation: int = 0          # counterclockwise display rotation applied to frames
    capture_fps: float | None = None
    codec: str = ""
    tags: dict = field(default_factory=dict)
    timestamps_from: str = "pts"

    def to_dict(self):
        d = asdict(self)
        d["tags"] = {k: v for k, v in self.tags.items() if len(str(v)) < 200}
        return d


def _rotate(img: np.ndarray, rot: int) -> np.ndarray:
    rot %= 360
    if rot == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


def _capture_fps(tags: dict) -> float | None:
    for k, v in tags.items():
        kl = k.lower()
        if kl in _CAPTURE_KEYS or ("capture" in kl and ("fps" in kl or "rate" in kl)):
            try:
                f = float(str(v).split()[0])
                if 1 < f < 10000:
                    return f
            except ValueError:
                pass
    return None


def _sidecar(path: str) -> dict | None:
    p = os.path.splitext(path)[0] + ".json"
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def probe(path: str) -> ClipInfo:
    if av is None:
        return _probe_cv(path)
    with av.open(path) as c:
        vs = c.streams.video[0]
        tags = dict(c.metadata or {})
        tags.update({f"stream:{k}": v for k, v in (vs.metadata or {}).items()})
        rate = vs.average_rate or vs.guessed_rate or vs.base_rate
        fps = float(rate) if rate else 0.0
        n = vs.frames or 0
        dur = float(vs.duration * vs.time_base) if vs.duration else (
            float(c.duration) / 1e6 if c.duration else 0.0)
        rot = 0
        w, h = vs.codec_context.width, vs.codec_context.height
        try:
            fr = next(c.decode(vs))
            rot = int(round(getattr(fr, "rotation", 0) or 0)) % 360
            w, h = fr.width, fr.height
        except Exception:
            pass
        if rot in (90, 270):
            w, h = h, w
        if not n and fps and dur:
            n = int(round(fps * dur))
        info = ClipInfo(path=path, width=w, height=h, fps=fps, n_frames=n, duration=dur,
                        rotation=rot, capture_fps=_capture_fps(tags),
                        codec=vs.codec_context.name, tags=tags)
    side = _sidecar(path)
    if side and side.get("capture_fps") and not info.capture_fps:
        info.capture_fps = float(side["capture_fps"])
    return info


def _probe_cv(path: str) -> ClipInfo:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return ClipInfo(path=path, width=w, height=h, fps=float(fps), n_frames=n,
                    duration=n / fps if fps else 0.0, timestamps_from="nominal fps")


def work_scale(info: ClipInfo, n_frames: int, max_side: int, budget: int) -> float:
    s = min(1.0, max_side / max(info.width, info.height, 1))
    per = info.width * info.height * 3
    if n_frames * per * s * s > budget:
        s = min(s, float(np.sqrt(budget / (n_frames * per))))
    return max(s, 0.25)


def read_frames(path: str, info: ClipInfo | None = None, *, scale: float = 1.0,
                t_start: float | None = None, t_end: float | None = None,
                max_frames: int | None = None, gray: bool = False):
    """Decode frames (display orientation) at `scale`. Returns (frames, times, numbers).

    times are seconds from the first frame of the file, from presentation
    timestamps. numbers are frame indices within the file.
    """
    info = info or probe(path)
    frames, times, nums = [], [], []
    if av is None:
        return _read_cv(path, info, scale, t_start, t_end, max_frames, gray)
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        t_first = None
        for i, fr in enumerate(c.decode(vs)):
            if fr.pts is not None:
                t = float(fr.pts * vs.time_base)
            else:
                t = i / (info.fps or 30.0)
            if t_first is None:
                t_first = t
            t -= t_first
            if t_start is not None and t < t_start - 1e-9:
                continue
            if t_end is not None and t > t_end + 1e-9:
                break
            rot = info.rotation
            sw, sh = fr.width, fr.height
            tw, th = max(2, int(round(sw * scale))), max(2, int(round(sh * scale)))
            fmt = "gray" if gray else "bgr24"
            if (tw, th) != (sw, sh):
                try:
                    img = fr.reformat(width=tw, height=th, format=fmt,
                                      interpolation="AREA").to_ndarray()
                except (TypeError, ValueError, KeyError):  # older PyAV
                    img = fr.reformat(width=tw, height=th, format=fmt).to_ndarray()
            else:
                img = fr.to_ndarray(format=fmt)
            frames.append(_rotate(img, rot))
            times.append(t)
            nums.append(i)
            if max_frames and len(frames) >= max_frames:
                break
    times = np.asarray(times)
    side = _sidecar(path)
    if side and side.get("timestamps") and len(nums) > 0:
        ts = np.asarray(side["timestamps"], float)
        if len(ts) > max(nums):
            reg = regularize_timestamps(ts) if side.get("host_clock") else ts - ts[0]
            times = reg[np.asarray(nums)]
    return frames, times, np.asarray(nums)


def regularize_timestamps(ts) -> np.ndarray:
    """Clean host-clock frame times: remove USB jitter, keep dropped frames as gaps.

    Each frame is assigned an integer slot n at the camera's true period
    (found by least squares), so a dropped frame leaves a two-period gap
    instead of a timing error.
    """
    ts = np.asarray(ts, float)
    if len(ts) < 3:
        return ts - ts[0]
    dt = float(np.median(np.diff(ts)))
    if dt <= 0:
        return ts - ts[0]
    t = ts - ts[0]
    per = t[-1] / max(round(t[-1] / dt), 1)
    c0 = 0.0
    for _ in range(4):
        n = np.round((t - c0) / per).astype(int)
        for i in range(1, len(n)):
            if n[i] <= n[i - 1]:
                n[i] = n[i - 1] + 1
        A = np.vstack([np.ones(len(n)), n]).T
        (c0, per), *_ = np.linalg.lstsq(A, t, rcond=None)
    return per * (n - n[0]).astype(float)


def _read_cv(path, info, scale, t_start, t_end, max_frames, gray):
    cap = cv2.VideoCapture(path)
    frames, times, nums = [], [], []
    i = 0
    fps = info.fps or 30.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if t_start is not None and t < t_start:
            continue
        if t_end is not None and t > t_end:
            break
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img)
        times.append(t)
        nums.append(i - 1)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, np.asarray(times), np.asarray(nums)


def motion_window(path: str, info: ClipInfo, max_frames: int, pad_s: float = 0.4):
    """Find the stretch of a long clip with the most motion (cheap low-res pass)."""
    s = min(1.0, 128 / max(info.width, info.height, 1))
    fr, ts, _ = read_frames(path, info, scale=s, gray=True)
    if len(fr) <= max_frames:
        return None, None
    e = np.array([0.0] + [float(np.mean(cv2.absdiff(a, b))) for a, b in zip(fr, fr[1:])])
    base = np.median(e)
    act = np.clip(e - base, 0, None)
    L = max_frames
    cs = np.concatenate([[0], np.cumsum(act)])
    j = int(np.argmax(cs[L:] - cs[:-L]))
    seg = act[j:j + L]
    on = np.flatnonzero(seg > 0.25 * seg.max()) if seg.max() > 0 else np.arange(len(seg))
    a = j + int(on[0])
    b = j + int(on[-1])
    t0 = max(ts[a] - pad_s, 0.0)
    t1 = ts[b] + pad_s
    # Keep within the frame budget.
    dt = np.median(np.diff(ts)) if len(ts) > 1 else 1 / 30
    if (t1 - t0) / dt > max_frames:
        mid = 0.5 * (ts[a] + ts[b])
        t0, t1 = mid - 0.5 * max_frames * dt, mid + 0.5 * max_frames * dt
    return float(t0), float(t1)


# ---------------------------------------------------------------------------
class Writer:
    """MP4 writer with exact per-frame timestamps (H.264 when available)."""

    def __init__(self, path: str, fps: float, size: tuple[int, int], tags: dict | None = None,
                 crf: int = 18):
        self.path = path
        self.w, self.h = size[0] - size[0] % 2, size[1] - size[1] % 2
        self.fps = fps
        self.n = 0
        self._cv = None
        if av is not None:
            opts = {"movflags": "use_metadata_tags+faststart"}
            self.c = av.open(path, "w", options=opts)
            for k, v in (tags or {}).items():
                self.c.metadata[k] = str(v)
            codec = "libx264" if "libx264" in av.codecs_available else (
                "h264" if "h264" in av.codecs_available else "mpeg4")
            rate = Fraction(fps).limit_denominator(1001)
            self.s = self.c.add_stream(codec, rate=rate)
            self.s.width, self.s.height, self.s.pix_fmt = self.w, self.h, "yuv420p"
            if codec == "libx264":
                self.s.options = {"crf": str(crf), "preset": "veryfast"}
            self.s.time_base = Fraction(1, 90000)
        else:  # pragma: no cover
            self._cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.w, self.h))

    def write(self, img: np.ndarray, t: float | None = None):
        img = img[: self.h, : self.w]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self._cv is not None:
            self._cv.write(img)
            self.n += 1
            return
        t = self.n / self.fps if t is None else t
        f = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="bgr24")
        f.pts = int(round(t * 90000))
        f.time_base = self.s.time_base
        for p in self.s.encode(f):
            self.c.mux(p)
        self.n += 1

    def close(self):
        if self._cv is not None:
            self._cv.release()
            return
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()


def make_preview(path: str, out_path: str, max_side: int = 960) -> str:
    """WebM (VP8) copy for browsers that can't play the original codec.

    Keeps every frame and its timestamp, so the overlay stays in sync with
    the measurement. Display rotation is applied.
    """
    if av is None:
        raise RuntimeError("PyAV is needed to make previews")
    info = probe(path)
    s = min(1.0, max_side / max(info.width, info.height, 1))
    frames, times, _ = read_frames(path, info, scale=s)
    if not frames:
        raise RuntimeError("No frames decoded")
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    tmp = out_path + ".part"
    with av.open(tmp, "w", format="webm") as c:
        st = c.add_stream("libvpx", rate=Fraction(info.fps or 30).limit_denominator(1001))
        st.width, st.height, st.pix_fmt = w2, h2, "yuv420p"
        st.bit_rate = 3_000_000
        st.options = {"deadline": "realtime", "cpu-used": "8", "lag-in-frames": "0"}
        st.time_base = Fraction(1, 1000)
        last = -1
        for img, t in zip(frames, times):
            pts = int(round(t * 1000))
            if pts <= last:
                pts = last + 1
            last = pts
            fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(img[:h2, :w2]), format="bgr24")
            fr.pts = pts
            fr.time_base = st.time_base
            for p in st.encode(fr):
                c.mux(p)
        for p in st.encode():
            c.mux(p)
    os.replace(tmp, out_path)
    return out_path
