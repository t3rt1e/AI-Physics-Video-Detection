"""Galileo test suite. Run with:  python -m pytest -q

Array-level physics tests run in milliseconds; the video tests render small
synthetic clips with known physics (a few seconds each).
"""
import json

import numpy as np
import pytest

from galileo import checks, clocks
from galileo.physics import fit_flights, split_flights
from galileo.tower import FakeTower, Drop, fit_gates, format_videog, parse_line

A_TRUE = 9.81 * 520.0  # px/s^2


def _fall(t, y0=60.0, t_rel=0.0):
    return y0 + 0.5 * A_TRUE * np.clip(t - t_rel, 0, None) ** 2


def test_fit_recovers_g_with_rolling_shutter():
    rng = np.random.default_rng(0)
    fps, H = 60.0, 720
    alpha = 0.8 / (fps * H)
    t = np.arange(6, 38) / fps
    y = []
    for ti in t:
        yy = _fall(ti)
        for _ in range(60):
            yy = _fall(ti + alpha * yy)
        y.append(yy)
    y = np.array(y) + rng.normal(0, 0.3, len(t))
    x = 640 + rng.normal(0, 0.3, len(t))
    fl = split_flights(t, x, y)
    plain = fit_flights(t, x, y, np.ones(len(t)), fl, fps=fps, n_px_axis=H, rs_mode="off")
    rs = fit_flights(t, x, y, np.ones(len(t)), fl, fps=fps, n_px_axis=H)
    assert plain.a_px / A_TRUE > 1.05          # rolling shutter inflates g...
    assert abs(rs.a_px / A_TRUE - 1) < 0.01    # ...and the fit takes it back out
    assert abs(rs.alpha / alpha - 1) < 0.15


def _bounces(e, t, floor=600.0):
    out = []
    for tt in t:
        y, v, tc = 60.0, 0.0, 0.0
        while True:
            disc = v * v + 2 * A_TRUE * (floor - y)
            th = (-v + np.sqrt(disc)) / A_TRUE
            if tc + th >= tt:
                dt = tt - tc
                out.append(y + v * dt + 0.5 * A_TRUE * dt * dt)
                break
            vi = v + A_TRUE * th
            tc, y, v = tc + th, floor, -e * vi
            if abs(v) < 1:
                out.append(floor)
                break
    return np.array(out)


@pytest.mark.parametrize("e,expected", [(0.7, "pass"), (1.15, "fail")])
def test_bounce_law(e, expected):
    rng = np.random.default_rng(1)
    t = np.arange(120) / 60.0
    y = _bounces(e, t) + rng.normal(0, 0.3, len(t))
    x = 400 + rng.normal(0, 0.3, len(t))
    fl = split_flights(t, x, y)
    jf = fit_flights(t, x, y, np.ones(len(t)), fl, fps=60, n_px_axis=720)
    assert checks.bounce_check(t, jf, 1 / 60)["status"] == expected


def test_parabola_check_catches_easing():
    rng = np.random.default_rng(2)
    t = np.arange(40) / 24.0
    u = t / t[-1]
    y = 60 + 500 * (3 * u**2 - 2 * u**3) + rng.normal(0, 0.3, len(t))
    x = 300 + rng.normal(0, 0.3, len(t))
    fl = split_flights(t, x, y)
    jf = fit_flights(t, x, y, np.ones(len(t)), fl, fps=24, n_px_axis=720)
    assert checks.parabola_check(t, x, y, np.ones(len(t)), jf)["status"] == "fail"


def test_interpolation_clock():
    rng = np.random.default_rng(3)
    k = 4
    y_orig = _fall(np.arange(20) / 30.0)
    frames = np.arange((len(y_orig) - 1) * k + 1)
    y = np.interp(frames / k, np.arange(len(y_orig)), y_orig) + rng.normal(0, 0.2, len(frames))
    ev = clocks.interpolation_clock(frames, y)
    assert ev is not None and ev["k"] == 4
    real = clocks.interpolation_clock(np.arange(60), _fall(np.arange(60) / 240.0)
                                      + rng.normal(0, 0.2, 60))
    assert real is None


def test_metadata_clock():
    ev = clocks.metadata_clock({"fps": 30.0, "capture_fps": 240.0})
    assert ev and ev[0]["k"] == pytest.approx(8.0)
    ev = clocks.metadata_clock({"fps": 240.0, "capture_fps": None})
    assert ev and ev[0]["k"] == 1.0 and ev[0]["slowmo_playback"] == pytest.approx(8.0)


def test_tower_protocol():
    ev = parse_line("GATE,17,3,900.0,1534210")
    assert ev.kind == "GATE" and ev.drop_id == 17 and ev.fields == {"k": 3, "h_mm": 900.0,
                                                                     "t_us": 1534210}
    assert parse_line("hello from the pico").kind == "LOG"
    assert parse_line("") is None
    assert format_videog(17, 9.7134, 0.3, "REAL") == "VIDEOG,17,9.71,0.30,REAL"
    g = 9.793
    h = np.array([300.0, 600.0, 900.0, 1200.0])
    t = 1_000_000 + 4000 + np.sqrt(2 * h / 1000 / g) * 1e6   # 4 ms magnet lag
    fit = fit_gates(h, t)
    assert fit["g"] == pytest.approx(g, rel=1e-6)


def test_fake_tower_runs_a_drop():
    events = []
    ft = FakeTower(on_event=events.append, countdown_s=0.05)
    did = ft.drop(block=True)
    kinds = [e.kind for e in events]
    assert kinds[:2] == ["ARM", "REL"] and kinds.count("GATE") == 4 and "TOWERG" in kinds
    d = Drop(did)
    for e in events:
        d.apply(e)
    assert d.tower_g == pytest.approx(ft.g, rel=0.01)
    assert d.laptop_fit()["g"] == pytest.approx(ft.g, rel=0.01)


def test_split_flights_high_fps_noise_is_not_a_bounce():
    # 240 fps: early in the fall the ball moves well under a pixel per frame,
    # so frame-to-frame noise must not read as bounces.
    rng = np.random.default_rng(4)
    t = np.arange(0, 0.5, 1 / 240)
    y = _fall(t) + rng.normal(0, 0.4, len(t))
    x = 300 + rng.normal(0, 0.4, len(t))
    fl = split_flights(t, x, y, diam_px=12.0)
    assert len(fl) == 1 and len(fl[0]) > 0.8 * len(t)


def _drag_fall(t_end, fps, vt, g=9.81, px_per_m=400.0, vx=0.0):
    dt = 1e-4
    n = int(t_end / dt) + 1
    x = y = 0.0
    u, v = vx, 0.0
    k = g / vt**2
    X, Y = [], []
    for _ in range(n):
        X.append(x)
        Y.append(y)
        sp = np.hypot(u, v)
        u += -k * sp * u * dt
        v += (g - k * sp * v) * dt
        x += u * dt
        y += v * dt
    tf = np.arange(0, t_end, 1 / fps)
    grid = np.arange(n) * dt
    return tf, 500 + np.interp(tf, grid, X) * px_per_m, 100 + np.interp(tf, grid, Y) * px_per_m


def test_drag_correction_recovers_g_for_a_ping_pong_ball():
    from galileo.physics import drag_factor
    t, x, y = _drag_fall(0.45, 120, vt=8.4, vx=0.3)
    fl = split_flights(t, x, y)
    jf = fit_flights(t, x, y, np.ones(len(t)), fl, fps=120, n_px_axis=1080, rs_mode="off")
    g_raw = jf.a_px / 400.0
    f = jf.flights[jf.main]
    fac, q = drag_factor(f.fit, t[f.idx] - f.fit.t0, (0, 1), 1 / 400.0, 8.4)
    assert g_raw < 9.81 * 0.96                     # drag makes it read low...
    assert abs(g_raw * fac / 9.81 - 1) < 0.015     # ...and the correction brings it back


def test_hand_push_is_cut_from_a_throw():
    from galileo.physics import robust_fit_2d, trim_release
    rng = np.random.default_rng(5)
    fps, a_push, push, g = 240.0, 60.0 * 520, 0.06, 9.81 * 520     # px/s^2
    t = np.arange(0, 0.45, 1 / fps)
    v1, y1 = a_push * push, 0.5 * a_push * push**2                 # at release
    tf = np.clip(t - push, 0, None)
    y = np.where(t < push, 0.5 * a_push * t**2, y1 + v1 * tf + 0.5 * g * tf**2)
    y = 50 + y + rng.normal(0, 0.3, len(t))
    x = 300 + rng.normal(0, 0.3, len(t))
    idx = np.arange(len(t))
    whole = 2 * np.polyfit(t, y, 2)[0]
    kept, k = trim_release(t, x, y, idx)
    cut = robust_fit_2d(t[kept], x[kept], y[kept]).accel_along((0, 1))[0]
    assert abs(whole / g - 1) > 0.1                 # the push spoils a plain fit of it all
    assert t[kept[0]] >= push and abs(cut / g - 1) < 0.02
    # A clean drop is left alone.
    yd = 50 + 0.5 * g * t**2 + rng.normal(0, 0.3, len(t))
    assert trim_release(t, x, yd, idx)[1] == 0


def test_verdict_guards():
    from galileo.verdict import decide
    scale = {"s": 1 / 400.0, "rel_sigma": 0.02}
    base = dict(scale=scale, clock=None, tests=[], n_points=40, duration=0.4)
    v = decide(a_px=3900.0, a_sigma=40.0, falls=False, **base)
    assert v["verdict"] == "UNSURE" and v["stage"] == "tracking: not a fall"
    v = decide(a_px=5.0, a_sigma=40.0, **base)
    assert v["verdict"] == "AI" and v["stage"] == "physics: no acceleration"
    v = decide(a_px=-900.0, a_sigma=40.0, **base)
    assert v["verdict"] == "UNSURE" and v["stage"] == "tracking: slows while falling"
    v = decide(a_px=3900.0, a_sigma=900.0, **base)          # +-23%: too wide to call
    assert v["verdict"] == "UNSURE" and v["stage"] == "measurement: too uncertain"
    v = decide(a_px=3500.0, a_sigma=30.0, drag={"factor": 1.12, "vt": 8.4}, **base)
    assert v["verdict"] == "REAL" and v["g"] == pytest.approx(3500 / 400 * 1.12)


def test_pick_seeded_follows_the_click():
    from galileo.track import Det, Track, pick_seeded
    times = np.arange(30) / 60.0

    def tr(x0, vy):
        return Track(dets=[Det(f=i, x=x0, y=50 + vy * i, d=10, area=80, elong=1, contrast=60)
                           for i in range(30)], kind="moving")
    a, b = tr(100, 4.0), tr(300, 5.0)
    assert pick_seeded([a, b], times, [(times[10], 301.0, 100.0)]) is b
    assert pick_seeded([a, b], times, [(times[10], 99.0, 92.0)]) is a
    assert pick_seeded([a, b], times, [(times[10], 600.0, 300.0)]) is None


def test_cli_seed_survives_powershell_commas():
    from galileo.cli import _seed
    assert _seed(["1.25,0.48,0.33"]) == (1.25, 0.48, 0.33)
    assert _seed(["1.25", "0.48", "0.33"]) == (1.25, 0.48, 0.33)


# ---- video end to end ---------------------------------------------------------
from galileo.analyze import analyze  # noqa: E402
from galileo.synth import Motion, Scene, make_clip  # noqa: E402


def _scene():
    return Scene(width=960, height=540, px_per_m=330.0, y0=50.0)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    out = {}
    out["real"] = make_clip(str(d / "real.mp4"), scene=_scene(), motion=Motion(g=9.81),
                            capture_fps=60, duration=0.85, rolling_shutter=0.5)["path"]
    out["ai"] = make_clip(str(d / "ai.mp4"), scene=_scene(), motion=Motion(g=2.3, t_release=0.3),
                          capture_fps=24, duration=1.4)["path"]
    out["slow"] = make_clip(str(d / "slow.mp4"), scene=_scene(), motion=Motion(g=9.81),
                            capture_fps=30, duration=0.85, slowmo={"k": 4, "method": "dup"})["path"]
    return out


@pytest.fixture(scope="module")
def hand_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("hand")
    sc = Scene(width=640, height=360, px_per_m=220.0, ball_d_m=0.040, ball_bgr=(238, 240, 242),
               bg_level=105, shading=0.3, y0=60.0, floor_y=330.0, seed=7)
    m = Motion(g=9.81, restitution=0.85, vt=8.4, kick=0.2, hand=True, t_release=0.3)
    return make_clip(str(d / "hand.mp4"), scene=sc, motion=m, capture_fps=60, duration=1.4)


def test_tracker_follows_the_ball_not_the_hand(hand_clip):
    from galileo.stabilize import stabilize
    from galileo.track import track_falling_object
    from galileo.video import read_frames
    frames, times, _ = read_frames(hand_clip["path"])
    _, T, info = stabilize(frames)
    assert T is None and info["max_drift_px"] < 1.0      # the moving hand isn't camera motion
    tr = track_falling_object(frames, times)
    ball_px = 0.040 * 220.0
    assert abs(np.nanmedian(tr.main.d) / ball_px - 1) < 0.35
    assert np.ptp(tr.main.y) > 150                       # it fell and bounced


def test_debug_video_draws_every_frame(hand_clip, tmp_path):
    from galileo.video import probe, read_frames
    out = str(tmp_path / "hand_debug.mp4")
    r = analyze(hand_clip["path"], preset="ping-pong", debug_video=out)
    assert r.get("debug_video") == out and r["verdict"]["verdict"] == "REAL"
    src, dbg = probe(hand_clip["path"]), probe(out)
    assert dbg.width == src.width and dbg.height > src.height      # picture plus the charts
    frames, _, _ = read_frames(out)
    assert len(frames) == r["work"]["frames"]


def test_panelled_ball_is_measured_whole(tmp_path):
    # A volleyball's panels differ from the wall by different amounts; the
    # width must come from the whole ball, not one panel.
    sc = Scene(width=640, height=480, px_per_m=200.0, ball_d_m=0.21, bg_level=150, noise=2.0,
               panels=((180, 90, 30), (40, 210, 235), (150, 150, 150), (180, 90, 30),
                       (40, 210, 235), (240, 240, 240)), shading=0.25, y0=70.0, seed=11)
    clip = make_clip(str(tmp_path / "volley.mp4"), scene=sc, motion=Motion(g=9.81, t_release=0.2),
                     capture_fps=60, duration=0.75)["path"]
    r = analyze(clip, preset="volleyball")
    assert r["fit"]["diam_px"] == pytest.approx(0.21 * 200.0, rel=0.06)
    assert r["verdict"]["verdict"] == "REAL"


def test_thrown_ball_is_measured_on_its_bounces(tmp_path):
    # Thrown down hard: the hand's push and the fast first flight must not
    # read as fake gravity; the bounce after it measures g.
    sc = Scene(width=640, height=360, px_per_m=200.0, ball_d_m=0.067, ball_bgr=(60, 220, 210),
               y0=40.0, floor_y=330.0, seed=21)
    m = Motion(g=9.81, restitution=0.75, vy0=4.0, push=0.06, t_release=0.15, vx=0.3, vt=21.0)
    clip = make_clip(str(tmp_path / "throw.mp4"), scene=sc, motion=m, capture_fps=120,
                     duration=1.2, rolling_shutter=0.4)["path"]
    r = analyze(clip, preset="tennis")
    v = r["verdict"]
    assert v["verdict"] == "REAL"
    assert v["g"] == pytest.approx(9.81, rel=0.06)
    assert r["fit"]["launch"]["thrown"] and r["fit"]["excluded_flights"] == [0]


def test_small_fast_ball_gets_zoomed_and_drag_corrected(tmp_path, monkeypatch):
    from galileo import config as C
    sc = Scene(width=1280, height=720, px_per_m=350.0, ball_d_m=0.040, ball_bgr=(238, 240, 242),
               bg_level=105, shading=0.3, y0=120.0, floor_y=None, seed=8)
    m = Motion(g=9.81, vt=8.4, vx=0.15, hand=True, t_release=0.25)
    clip = make_clip(str(tmp_path / "pp.mp4"), scene=sc, motion=m, capture_fps=120,
                     duration=0.85, rolling_shutter=0.5)["path"]
    # Force a coarse first pass (as a long 1080p 120 fps clip gets): ball ~7 px.
    monkeypatch.setattr(C, "WORK_MAX_SIDE", 640)
    r = analyze(clip, preset="ping-pong")
    v = r["verdict"]
    assert r.get("zoom") and r["zoom"]["ball_px"] > 1.6 * r["zoom"]["ball_px_before"]
    assert r["drag"]["factor"] > 1.03
    assert v["verdict"] == "REAL"
    assert v["g"] == pytest.approx(9.81, rel=0.05)


def test_real_clip_reads_earth_gravity(clips):
    r = analyze(clips["real"], preset="steel-38mm")
    v = r["verdict"]
    assert v["verdict"] == "REAL"
    assert v["g"] == pytest.approx(9.81, rel=0.06)


def test_ai_clip_is_flagged(clips):
    v = analyze(clips["ai"], preset="steel-38mm")["verdict"]
    assert v["verdict"] == "AI"
    assert v["g"] < 4


def test_slow_motion_is_not_called_fake(clips):
    r = analyze(clips["slow"], preset="steel-38mm")
    v = r["verdict"]
    assert v["verdict"] == "SLOWMO" and v["k"] == pytest.approx(4)
    assert v["g"] == pytest.approx(9.81, rel=0.1)


def test_web_api(clips, tmp_path):
    from fastapi.testclient import TestClient
    from galileo.web.app import create_app
    data = tmp_path / "data"
    (data / "clips").mkdir(parents=True)
    (data / "clips" / "real.mp4").write_bytes(open(clips["real"], "rb").read())
    with TestClient(create_app(str(data))) as c:
        st = c.get("/api/status").json()
        assert "steel-38mm" in st["presets"]
        lst = c.get("/api/clips").json()
        assert lst[0]["id"] == "clips/real.mp4"
        r = c.post("/api/analyze", json={"clip": "clips/real.mp4", "object": "steel-38mm"})
        assert r.status_code == 200
        body = r.json()
        assert body["verdict"]["verdict"] == "REAL"
        json.dumps(body)  # strictly JSON-safe
        assert c.get("/media/clips/real.mp4", headers={"Range": "bytes=0-99"}).status_code == 206
        assert c.post("/api/analyze", json={"clip": "../etc/passwd"}).status_code == 400
        # A click on the ball mid-fall: same answer, and bad picks are refused.
        pts = body["track"]["points"]
        p = pts[len(pts) // 2]
        r2 = c.post("/api/analyze", json={"clip": "clips/real.mp4", "object": "steel-38mm",
                                          "seed": {"t": p["t"], "x": p["x"], "y": p["y"]}})
        assert r2.status_code == 200 and r2.json()["verdict"]["verdict"] == "REAL"
        assert c.post("/api/analyze", json={"clip": "clips/real.mp4",
                                            "seed": {"t": 0.5}}).status_code == 400
