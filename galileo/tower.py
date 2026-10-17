"""Galileo Tower link: the hardware-software contract from the game plan.

USB serial, 115200 baud, one message per line, comma-separated fields.

TOWER -> LAPTOP
  ARM,<drop_id>                           countdown started: start recording
  REL,<drop_id>,<t_us>                    magnet released at t_us on the tower clock
  GATE,<drop_id>,<k>,<h_mm>,<t_us>        ball broke gate k, h_mm below the release point
  TOWERG,<drop_id>,<g>,<sigma>            tower's fitted g and its error, in m/s^2
  BOUNCE,<drop_id>,<n>,<t_us>             piezo felt impact n

LAPTOP -> TOWER
  VIDEOG,<drop_id>,<g>,<sigma>,<verdict>  verdict is REAL, AI, SLOWMO or UNSURE
  SHOW,<text>                             free text for the LED matrix

Anything else from the tower (debug prints) is passed through as kind LOG.
FakeTower speaks the same protocol with simulated drops, so the laptop side
can be built and rehearsed before the tower exists.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import config as C

BAUD = 115200
KINDS = {"ARM": 1, "REL": 2, "GATE": 4, "TOWERG": 3, "BOUNCE": 3}


@dataclass
class TowerEvent:
    kind: str
    drop_id: int | None
    fields: dict
    raw: str
    t_host: float = field(default_factory=time.time)
    t_mono: float = field(default_factory=time.perf_counter)


def parse_line(line: str) -> TowerEvent | None:
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    kind = parts[0].upper()
    if kind not in KINDS or len(parts) - 1 < KINDS[kind]:
        return TowerEvent("LOG", None, {"text": line}, line)
    try:
        did = int(parts[1])
        if kind == "ARM":
            f = {}
        elif kind == "REL":
            f = {"t_us": int(parts[2])}
        elif kind == "GATE":
            f = {"k": int(parts[2]), "h_mm": float(parts[3]), "t_us": int(parts[4])}
        elif kind == "TOWERG":
            f = {"g": float(parts[2]), "sigma": float(parts[3])}
        else:  # BOUNCE
            f = {"n": int(parts[2]), "t_us": int(parts[3])}
    except ValueError:
        return TowerEvent("LOG", None, {"text": line}, line)
    return TowerEvent(kind, did, f, line)


def format_videog(drop_id: int, g: float | None, sigma: float | None, verdict: str) -> str:
    g = 0.0 if g is None else g
    sigma = 0.0 if sigma is None else sigma
    return f"VIDEOG,{int(drop_id)},{g:.2f},{sigma:.2f},{verdict}"


def format_show(text: str) -> str:
    return "SHOW," + text.replace(",", " ").replace("\n", " ")[:64]


def fit_gates(h_mm, t_us) -> dict | None:
    """Tower g from gate crossings: h = c0 + c1*t + (g/2)*t^2.

    The free c0 and c1 absorb the magnet's release lag and any offset in the
    measured heights, so only the gate spacing and timing matter.
    """
    h = np.asarray(h_mm, float) / 1000.0
    t = np.asarray(t_us, float) / 1e6
    if len(h) < 3:
        return None
    tc = t - t.mean()
    A = np.vstack([np.ones_like(tc), tc, tc**2]).T
    coef, *_ = np.linalg.lstsq(A, h, rcond=None)
    g = 2 * coef[2]
    sigma = float("nan")
    dof = len(h) - 3
    if dof > 0:
        r = h - A @ coef
        s2 = float(r @ r) / dof
        cov = np.linalg.pinv(A.T @ A) * s2
        sigma = 2 * float(np.sqrt(max(cov[2, 2], 0)))
    return {"g": float(g), "sigma": sigma, "n": int(len(h))}


@dataclass
class Drop:
    """Everything the tower said about one drop."""
    drop_id: int
    arm_host: float | None = None
    rel_host: float | None = None       # perf_counter when REL arrived
    rel_us: int | None = None
    gates: list = field(default_factory=list)      # (k, h_mm, t_us)
    bounces: list = field(default_factory=list)    # (n, t_us)
    tower_g: float | None = None
    tower_sigma: float | None = None

    def laptop_fit(self):
        if len(self.gates) < 3:
            return None
        g = sorted(self.gates)
        return fit_gates([x[1] for x in g], [x[2] for x in g])

    def apply(self, ev: TowerEvent):
        if ev.kind == "ARM":
            self.arm_host = ev.t_mono
        elif ev.kind == "REL":
            self.rel_host = ev.t_mono
            self.rel_us = ev.fields["t_us"]
        elif ev.kind == "GATE":
            self.gates.append((ev.fields["k"], ev.fields["h_mm"], ev.fields["t_us"]))
        elif ev.kind == "TOWERG":
            self.tower_g = ev.fields["g"]
            self.tower_sigma = ev.fields["sigma"]
        elif ev.kind == "BOUNCE":
            self.bounces.append((ev.fields["n"], ev.fields["t_us"]))


# ---------------------------------------------------------------------------
class TowerLink:
    """Real serial link. on_event(TowerEvent) is called from a reader thread."""

    def __init__(self, port: str, baud: int = BAUD, on_event=None):
        import serial  # pyserial
        self._serial_mod = serial
        self.port = port
        self.baud = baud
        self.on_event = on_event or (lambda ev: None)
        self.ser = None
        self._stop = threading.Event()
        self._thread = None
        self.sent: list[str] = []

    @property
    def connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def start(self):
        self.ser = self._serial_mod.Serial(self.port, self.baud, timeout=0.1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(256)
            except Exception as e:  # unplugged
                self.on_event(TowerEvent("LOG", None, {"text": f"serial error: {e}"}, ""))
                time.sleep(0.5)
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                ev = parse_line(line.decode("utf-8", errors="replace"))
                if ev:
                    self.on_event(ev)

    def send(self, line: str):
        self.sent.append(line)
        if self.connected:
            self.ser.write((line + "\n").encode())

    def close(self):
        self._stop.set()
        if self.ser:
            self.ser.close()


class FakeTower:
    """Simulated tower that speaks the real protocol, in real time."""

    def __init__(self, on_event=None, g: float = C.G_GAINESVILLE,
                 heights_mm=(300.0, 600.0, 900.0, 1200.0), countdown_s: float = 3.0,
                 release_lag_s: float = 0.004, timing_jitter_us: float = 20.0,
                 restitution: float = 0.55, floor_mm: float = 1450.0):
        self.on_event = on_event or (lambda ev: None)
        self.g = g
        self.heights = heights_mm
        self.countdown = countdown_s
        self.lag = release_lag_s
        self.jitter = timing_jitter_us
        self.e = restitution
        self.floor_mm = floor_mm
        self.sent: list[str] = []
        self.next_id = 1
        self.connected = True
        self._busy = threading.Lock()

    def start(self):
        return self

    def close(self):
        pass

    def send(self, line: str):
        self.sent.append(line)

    def _emit(self, line: str):
        ev = parse_line(line)
        if ev:
            self.on_event(ev)

    def drop(self, block: bool = False):
        """Run one simulated drop (ARM, countdown, REL, gates, TOWERG, bounces)."""
        if not self._busy.acquire(blocking=False):
            return None
        did = self.next_id
        self.next_id += 1
        th = threading.Thread(target=self._run_drop, args=(did,), daemon=True)
        th.start()
        if block:
            th.join()
        return did

    def _run_drop(self, did: int):
        try:
            t0 = time.perf_counter()
            us = lambda: int((time.perf_counter() - t0) * 1e6) + 1_000_000  # noqa: E731
            self._emit(f"ARM,{did}")
            time.sleep(self.countdown)
            t_rel_us = us()
            self._emit(f"REL,{did},{t_rel_us}")
            # Ball leaves the magnet after a short lag, then falls freely.
            t_free = t_rel_us + self.lag * 1e6
            times = []
            for k, h in enumerate(self.heights, 1):
                tt = t_free + np.sqrt(2 * h / 1000.0 / self.g) * 1e6
                tt += random.gauss(0, self.jitter)
                times.append(tt)
            for k, (h, tt) in enumerate(zip(self.heights, times), 1):
                _sleep_until(t0, tt)
                self._emit(f"GATE,{did},{k},{h:.1f},{int(tt)}")
            fit = fit_gates(self.heights, times)
            self._emit(f"TOWERG,{did},{fit['g']:.3f},{fit['sigma']:.3f}")
            # Bounces on the piezo pad.
            t_hit = t_free + np.sqrt(2 * self.floor_mm / 1000.0 / self.g) * 1e6
            v = np.sqrt(2 * self.g * self.floor_mm / 1000.0)
            for n in range(1, 5):
                _sleep_until(t0, t_hit)
                self._emit(f"BOUNCE,{did},{n},{int(t_hit)}")
                v *= self.e
                t_hit += 2 * v / self.g * 1e6
                if v < 0.3:
                    break
        finally:
            self._busy.release()


def _sleep_until(t0: float, t_us: float):
    target = t0 + (t_us - 1_000_000) / 1e6
    dt = target - time.perf_counter()
    if dt > 0:
        time.sleep(dt)


def list_ports() -> list[str]:
    try:
        from serial.tools import list_ports as lp
        return [f"{p.device}  {p.description}" for p in lp.comports()]
    except Exception:
        return []
