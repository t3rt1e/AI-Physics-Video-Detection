"""Novex test suite. Run with:  python -m pytest -q

Array-level physics tests run in milliseconds; the video tests render small
synthetic clips with known physics (a few seconds each).
"""
import json

import numpy as np
import pytest

from novex import checks, clocks
from novex.physics import fit_flights, split_flights
from novex.tower import FakeTower, Drop, fit_gates, format_videog, parse_line

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
    from novex.physics import drag_factor
    t, x, y = _drag_fall(0.45, 120, vt=8.4, vx=0.3)
    fl = split_flights(t, x, y)
    jf = fit_flights(t, x, y, np.ones(len(t)), fl, fps=120, n_px_axis=1080, rs_mode="off")
    g_raw = jf.a_px / 400.0
    f = jf.flights[jf.main]
    fac, q = drag_factor(f.fit, t[f.idx] - f.fit.t0, (0, 1), 1 / 400.0, 8.4)
    assert g_raw < 9.81 * 0.96                     # drag makes it read low...
    assert abs(g_raw * fac / 9.81 - 1) < 0.015     # ...and the correction brings it back


def test_hand_push_is_cut_from_a_throw():
    from novex.physics import robust_fit_2d, trim_release
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
    from novex.verdict import decide
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
    assert v["real_range"] == pytest.approx([9.80 * 0.85, 9.80 * 1.15])
    # Outside Earth's interval is AI on either side (clips play in real time)...
    v = decide(a_px=4800.0, a_sigma=30.0, **base)               # 12 m/s^2
    assert v["verdict"] == "AI" and v["stage"] == "physics: g too high"
    assert decide(a_px=2400.0, a_sigma=30.0, **base)["verdict"] == "AI"
    # ...unless sped-up footage is allowed (--slowmo): then too fast can't be called.
    assert decide(a_px=4800.0, a_sigma=30.0, real_time=False, **base)["verdict"] == "UNSURE"


def test_pick_seeded_follows_the_click():
    from novex.track import Det, Track, pick_seeded
    times = np.arange(30) / 60.0

    def tr(x0, vy):
        return Track(dets=[Det(f=i, x=x0, y=50 + vy * i, d=10, area=80, elong=1, contrast=60)
                           for i in range(30)], kind="moving")
    a, b = tr(100, 4.0), tr(300, 5.0)
    assert pick_seeded([a, b], times, [(times[10], 301.0, 100.0)]) is b
    assert pick_seeded([a, b], times, [(times[10], 99.0, 92.0)]) is a
    assert pick_seeded([a, b], times, [(times[10], 600.0, 300.0)]) is None


def test_cli_seed_survives_powershell_commas():
    from novex.cli import _seed
    assert _seed(["1.25,0.48,0.33"]) == (1.25, 0.48, 0.33)
    assert _seed(["1.25", "0.48", "0.33"]) == (1.25, 0.48, 0.33)


# ---- video end to end ---------------------------------------------------------
from novex.analyze import analyze  # noqa: E402
from novex.synth import Motion, Scene, make_clip  # noqa: E402


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
    from novex.stabilize import stabilize
    from novex.track import track_falling_object
    from novex.video import read_frames
    frames, times, _ = read_frames(hand_clip["path"])
    _, T, info = stabilize(frames)
    assert T is None and info["max_drift_px"] < 1.0      # the moving hand isn't camera motion
    tr = track_falling_object(frames, times)
    ball_px = 0.040 * 220.0
    assert abs(np.nanmedian(tr.main.d) / ball_px - 1) < 0.35
    assert np.ptp(tr.main.y) > 150                       # it fell and bounced


def test_debug_video_draws_every_frame(hand_clip, tmp_path):
    from novex.video import probe, read_frames
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
    from novex import config as C
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
    # Only with the clocks on (--slowmo); by default clips are taken as real time.
    r = analyze(clips["slow"], preset="steel-38mm", real_time=False)
    v = r["verdict"]
    assert v["verdict"] == "SLOWMO" and v["k"] == pytest.approx(4)
    assert v["g"] == pytest.approx(9.81, rel=0.1)


def test_clips_are_taken_as_real_time_by_default(clips):
    r = analyze(clips["slow"], preset="steel-38mm")
    v = r["verdict"]
    assert v["verdict"] == "AI" and v["k"] == 1.0
    assert v["g"] == pytest.approx(9.81 / 16, rel=0.15)
    assert any("slow motion" in n for n in r["notes"])        # what the clocks found is noted


def test_web_api(clips, tmp_path):
    from fastapi.testclient import TestClient
    from novex.web.app import create_app
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


# ---- several objects, and objects that aren't round ----------------------------------
def test_shape_classifier_tells_a_ball_from_a_box():
    import cv2
    from novex import shape as shp
    from novex.track import Det
    t = np.arange(40) / 60.0
    fl = [np.arange(40)]
    speed = np.full(40, 100.0)

    def dets(poly_at):
        out = []
        for i in range(40):
            h = cv2.convexHull(poly_at(i).astype(np.float32)).reshape(-1, 2)
            out.append(Det(f=i, x=0, y=0, d=10, area=1, elong=1, contrast=60, hull=h))
        return out

    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    disc = dets(lambda i: np.stack([20 * np.cos(ang), 20 * np.sin(ang)], 1))
    st = shp.outline_stats(disc)
    c = shp.classify(st, np.full(40, 40.0), t, fl, speed, 1 / 60)
    assert c["shape"] == "round" and c["sure"]

    def box(i):
        a = 0.2 * i
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return np.array([[-40, -15], [40, -15], [40, 15], [-40, 15]]) @ R.T
    bx = dets(box)
    st = shp.outline_stats(bx)
    across = np.array([np.ptp(b.hull[:, 0]) + 1 for b in bx])
    c = shp.classify(st, across, t, fl, speed, 1 / 60)
    assert c["shape"] == "rigid" and c["sure"]
    assert np.nanmedian(st["long"]) == pytest.approx(81, abs=1.5)


def test_length_is_measured_at_half_contrast():
    import cv2
    from novex.track import measure_length
    bg = np.full((200, 300, 3), 40, np.uint8)
    img = bg.copy()
    rect = cv2.boxPoints(((150.3, 100.2), (120.0, 24.0), 25.0))
    big = np.zeros((800, 1200), np.uint8)
    cv2.fillPoly(big, [np.round((rect - 0.5) * 4 * 16).astype(np.int32)], 255, cv2.LINE_8, 4)
    cov = cv2.resize(big.astype(np.float32) / 255, (300, 200), interpolation=cv2.INTER_AREA)
    img = (bg * (1 - cov[..., None]) + np.array([200, 180, 60]) * cov[..., None]).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    hull = cv2.convexHull((rect * 1.03).astype(np.float32) - 0.03 * np.array([150.3, 100.2],
                                                                                np.float32))
    assert measure_length(img, bg, hull.reshape(-1, 2)) == pytest.approx(120.0, abs=0.6)


def test_pieces_of_one_object_are_not_two_objects():
    from novex.track import Det, Track, select_objects
    times = np.arange(40) / 60.0

    def tr(x0, dx=0.0, d=20.0):
        dets = [Det(f=i, x=x0 + dx * i, y=50 + 0.5 * 2700 * times[i] ** 2, d=d, area=300, elong=1,
                    contrast=60) for i in range(40)]
        t_ = Track(dets=dets, kind="moving", score=100.0)
        t_.falls = True
        return t_
    ball = tr(200)
    piece = tr(207, d=12)         # a panel of the same ball, moving with it
    other = tr(500, d=30)         # a second ball, dropped beside it
    assert select_objects([ball, piece, other], times, ball) == [ball, other]


def test_combined_verdict_from_several_objects():
    from novex.verdict import combine_objects
    real = {"verdict": "REAL", "confidence": 0.9, "headline": "Falls at 9.8 ± 0.3 m/s²: Earth's "
            "gravity.", "reasons": [], "g": 9.8, "sigma": 0.3, "stage": "ok"}
    fake = {"verdict": "AI", "confidence": 0.9, "headline": "Falls at 30% of Earth's gravity.",
            "reasons": [], "g": 3.0, "sigma": 0.2, "stage": "physics: g too low"}
    unsure = {"verdict": "UNSURE", "confidence": 0.0, "headline": "Can't tell: no ruler.",
              "reasons": [], "g": None, "sigma": None, "stage": "ruler: missing"}
    o = lambda n, v, **kw: {"n": n, "label": kw.get("label"), "verdict": v,  # noqa: E731
                            "named": bool(kw.get("label")), "strong": kw.get("strong", False)}
    # A guessed second object that disagrees only lowers the confidence.
    v = combine_objects([o(1, real, label="ball", strong=True), o(2, fake, label="box")])
    assert v["verdict"] == "REAL" and v["confidence"] < 0.9
    # One you named yourself that disagrees can't be explained by a wrong guess.
    v = combine_objects([o(1, real, label="ball", strong=True),
                         o(2, fake, label="basketball", strong=True)])
    assert v["verdict"] == "AI" and v["from_object"] == 2
    # When the first can't be measured, a known second one answers.
    v = combine_objects([o(1, unsure), o(2, real, label="tennis ball")])
    assert v["verdict"] == "REAL" and v["headline"].startswith("Object 2")


@pytest.fixture(scope="module")
def two_balls(tmp_path_factory):
    from novex.synth import Body
    d = tmp_path_factory.mktemp("two")
    sc = Scene(width=960, height=540, px_per_m=300.0, ball_d_m=0.067, ball_bgr=(60, 215, 200),
               x0=300.0, y0=50.0, seed=3,
               bodies=(Body(kind="ball", size_m=0.239, bgr=(40, 110, 230), x0=680.0, y0=80.0,
                            motion=Motion(g=9.81, t_release=0.2)),))
    # (Neither may sit still for half the clip: the backdrop is the median frame.)
    return make_clip(str(d / "two.mp4"), scene=sc, motion=Motion(g=9.81, t_release=0.12),
                     capture_fps=60, duration=1.1)["path"]


@pytest.mark.parametrize("names", [["tennis", "basketball"], ["basketball", "tennis"]])
def test_two_objects_are_measured_separately(two_balls, names):
    r = analyze(two_balls, preset=names, identify_mode="off")
    objs = {o["label"]: o for o in r["objects"]}
    assert set(objs) == {"tennis ball", "basketball"}      # names matched by size, any order
    for o in objs.values():
        assert o["verdict"]["verdict"] == "REAL"
        assert o["g"] == pytest.approx(9.81, rel=0.06)
    assert r["verdict"]["verdict"] == "REAL"
    assert any(t["name"] == "Equal falls" and t["status"] == "pass" for t in r["tests"])


def test_tumbling_phone_is_measured_by_its_length(tmp_path):
    from novex.synth import Body
    sc = Scene(width=640, height=480, px_per_m=500.0, main_ball=False, bg_level=70, seed=5,
               bodies=(Body(kind="box", size_m=0.152, width_m=0.072, bgr=(25, 25, 30), x0=320.0,
                            y0=70.0, motion=Motion(g=9.81, t_release=0.12), omega=10.0,
                            angle0=0.4),))
    clip = make_clip(str(tmp_path / "phone.mp4"), scene=sc, motion=Motion(t_release=0.12),
                     capture_fps=60, duration=0.6)["path"]
    r = analyze(clip, preset="phone")
    v = r["verdict"]
    assert r["objects"][0]["shape"] == "rigid"
    assert v["verdict"] == "REAL" and v["g"] == pytest.approx(9.81, rel=0.05)


def test_off_centre_mass_is_found_from_the_spin(tmp_path):
    # A hammer-like bar: its mass sits a quarter of its length from the
    # outline's centre, so the outline's centre circles the real parabola.
    from novex.synth import Body
    sc = Scene(width=640, height=480, px_per_m=500.0, main_ball=False, bg_level=40, seed=6,
               bodies=(Body(kind="box", size_m=0.20, width_m=0.04, bgr=(60, 180, 240),
                            head_bgr=(160, 160, 170), x0=320.0, y0=70.0,
                            motion=Motion(g=9.81, t_release=0.1), omega=9.0, cm_offset=0.25),))
    clip = make_clip(str(tmp_path / "hammer.mp4"), scene=sc, motion=Motion(t_release=0.1),
                     capture_fps=120, duration=0.55)["path"]
    r = analyze(clip, size_m=0.20, identify_mode="off")
    o = r["objects"][0]
    assert o["shape"] == "rigid" and o["spin"]["applied"]
    assert o["spin"]["offset_px"] == pytest.approx(0.25 * 0.20 * 500.0, rel=0.15)
    assert r["verdict"]["verdict"] == "REAL"
    assert r["verdict"]["g"] == pytest.approx(9.81, rel=0.04)


def test_names_go_to_the_objects_they_fit():
    from novex import identify
    # log P(name | object): object 0 looks like a basketball, object 1 like a tennis ball.
    logp = np.log([[0.1, 0.8], [0.7, 0.2]])
    assert identify.match_names(["tennis", "basketball"], logp) == [1, 0]
    assert identify.resolve("Tennis Ball") == "tennis" and identify.resolve("smartphone") == "phone"
    # Anything can be named, with a size in any unit; known names bring their size.
    assert identify.parse_name("rotisserie chicken=22cm") == ("rotisserie chicken", 0.22)
    assert identify.parse_name("lamp = 8.5 in")[1] == pytest.approx(0.2159)
    chicken = identify.from_name("roast chicken")
    assert chicken["label"] == "rotisserie chicken" and chicken["shape"] == "rigid"
    odd = identify.from_name("grandma's vase=31cm")
    assert odd["label"] == "grandma's vase" and odd["size_m"] == pytest.approx(0.31)
    assert identify.from_name("mystery thing")["size_m"] is None
    assert identify.is_light(identify.from_name("feather"))
    assert not identify.is_light(identify.from_name("brick"))
    infos = [{"identity": identify.from_preset("tennis"), "shape": "round", "size_orig_px": 40.0},
             {"identity": None, "shape": "round", "size_orig_px": 42.0}]
    same = identify.same_as(infos, 1)
    assert same["label"] == "tennis ball" and same["source"] == "same as object 1"


def test_thickness_centroid_weighs_lumps_by_volume():
    # A big lump and a small one: a solid's centre of mass weighs each by its
    # volume (r^3), the outline's area by r^2. Weighting pixels by their depth
    # into the outline gets the volume right.
    import cv2
    from novex.track import detect
    bg = np.full((200, 260, 3), 200, np.uint8)
    img = bg.copy()
    cv2.circle(img, (100, 100), 30, (40, 110, 190), -1, cv2.LINE_AA)
    cv2.circle(img, (140, 100), 12, (40, 110, 190), -1, cv2.LINE_AA)
    d = max(detect(img, bg), key=lambda q: q.area)
    x_vol = (100 * 30 ** 3 + 140 * 12 ** 3) / (30 ** 3 + 12 ** 3)
    assert abs(d.tx - x_vol) < 1.2
    assert abs(d.tx - x_vol) < 0.5 * abs(d.ax - x_vol)


def test_tumbling_lumpy_object_falls_at_g(tmp_path):
    # A "rotisserie chicken": six lumps, tumbling end over end about a tilted
    # axis. Named with its size, it reads Earth's gravity.
    from novex.synth import Body
    sph = ((0.0, 0.0, 0.0, 0.065), (0.055, -0.005, 0.0, 0.05), (-0.07, 0.035, 0.03, 0.026),
           (-0.07, 0.035, -0.03, 0.026), (0.02, -0.045, 0.05, 0.02), (0.02, -0.045, -0.05, 0.02))
    cols = ((40, 110, 190), (50, 125, 205), (30, 85, 160), (30, 85, 160), (45, 105, 180),
            (45, 105, 180))
    sc = Scene(width=640, height=480, px_per_m=400.0, main_ball=False, bg_level=200, seed=9,
               bodies=(Body(kind="spheres", spheres=sph, sphere_bgr=cols, x0=320.0, y0=70.0,
                            motion=Motion(g=9.81, t_release=0.1), omega=2 * np.pi * 2.0,
                            axis=(0.75, 0.25, 0.6), angle0=0.5),))
    clip = make_clip(str(tmp_path / "chicken.mp4"), scene=sc, motion=Motion(t_release=0.1),
                     capture_fps=90, duration=0.6)["path"]
    r = analyze(clip, preset="rotisserie chicken=211mm", identify_mode="off")
    o = r["objects"][0]
    assert o["label"] == "rotisserie chicken" and o["shape"] == "rigid"
    assert r["verdict"]["verdict"] == "REAL"
    assert r["verdict"]["g"] == pytest.approx(9.81, rel=0.06)


def test_drag_ruled_object_is_not_called_fake(tmp_path):
    # A balloon reaches its top speed almost at once and drifts down: that is
    # what a real balloon does, so it can't show gravity, and it isn't AI.
    sc = Scene(width=640, height=480, px_per_m=300.0, ball_d_m=0.28, ball_bgr=(60, 60, 220),
               y0=60.0, seed=12)
    clip = make_clip(str(tmp_path / "balloon.mp4"), scene=sc,
                     motion=Motion(g=9.81, vt=1.2, t_release=0.2), capture_fps=30,
                     duration=1.3)["path"]
    v = analyze(clip, preset="balloon", identify_mode="off")["verdict"]
    assert v["verdict"] == "UNSURE" and v["stage"] == "physics: air drag rules"
    v = analyze(clip, preset="basketball", identify_mode="off")["verdict"]
    assert v["verdict"] == "AI"            # a basketball can't fall like that


def test_cli_takes_any_object_name():
    from novex.cli import _objects
    assert _objects(["rotisserie chicken=22cm", "tennis,auto"]) == [
        "rotisserie chicken=22cm", "tennis", None]


# ---- The camera at any angle ------------------------------------------------
# novex/camera.py (the fall in 3D, the level view), novex/scene.py (the
# room's lines), novex/stabilize.py (pans and turns), rendered with
# novex/synth3d.py.

import math  # noqa: E402

from novex import camera as cam_mod  # noqa: E402
from novex import scene as scene_mod  # noqa: E402
from novex.synth3d import (Ball3D, Camera3D, ball_path, project, render_clip,  # noqa: E402
                             render_still)


def test_level_view_turns_gravity_straight_down():
    n = np.array([0.3, 0.8, 0.52])
    n = n / np.linalg.norm(n)
    assert np.allclose(cam_mod.level_rotation(n) @ n, [0.0, 1.0, 0.0], atol=1e-9)
    lev = cam_mod.Leveler(800.0, 480.0, 270.0, nvec=n)
    u, v = np.array([100.0, 480.0, 900.0]), np.array([50.0, 270.0, 500.0])
    a, b, z = lev.forward(u, v)
    assert (z > 0).all()
    u2, v2 = lev.backward(a, b)
    assert np.allclose(u2, u, atol=1e-6) and np.allclose(v2, v, atol=1e-6)
    # Upside down is half a turn about the camera's axis.
    assert np.allclose(cam_mod.level_rotation([0.0, -1.0, 0.0]) @ [0.0, -1.0, 0.0], [0, 1, 0])


def test_fall_in_3d_reads_how_the_camera_is_tipped():
    # From positions and widths alone: a camera looking 35° down at a drop.
    cam = Camera3D(pos=(0, 1.8, 0), pitch=35, hfov=65)
    ball = Ball3D(pos=(0.1, 1.25, 1.6))
    ts, P = ball_path(ball, 1.0)
    rng = np.random.default_rng(0)
    T = np.arange(32, 86) / 120.0
    C = np.array([[np.interp(t, ts, P[:, k]) for k in range(3)] for t in T])
    u, v, z = project(cam, 0.0, C)
    d = cam.f0 * ball.d_m / z
    u, v = u + rng.normal(0, 0.25, len(T)), v + rng.normal(0, 0.25, len(T))
    d = d + rng.normal(0, 0.25, len(T))
    d_ref = float(np.median(d))
    ft = cam_mod.fit_flight_3d(T, u, v, d, cam.f0, cam.width / 2, cam.height / 2, d_ref)
    nvec, beta, sb, roll, sr, _ = cam_mod.direction(ft["G"], ft["cov"])
    assert abs(beta - 35.0) < max(3 * sb, 2.5)
    assert abs(roll) < 2.0
    assert np.linalg.norm(ft["G"]) * ball.d_m / d_ref == pytest.approx(9.81, rel=0.06)


@pytest.mark.parametrize("pitch,roll", [(0, 0), (25, 0), (-20, 0), (15, 30)])
def test_room_lines_read_the_camera_angle(pitch, roll):
    cam = Camera3D(pos=(0, 1.3 if pitch >= 0 else 0.3, 0), pitch=pitch, roll=roll, hfov=65,
                   width=800, height=450)
    img = render_still(cam)
    f_frac = cam.f0 / cam.width
    K_true = math.tan(math.radians(pitch)) / f_frac
    down = np.array([math.sin(math.radians(roll)), math.cos(math.radians(roll))])
    v = scene_mod.read(img, down, K_window=(-1.2, 1.2))["vertical"]
    assert v is not None
    assert v["K"] == pytest.approx(K_true, abs=0.03)
    assert v["roll"] == pytest.approx(roll, abs=1.0)


@pytest.fixture(scope="module")
def camera_clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("camera")
    small = dict(width=640, height=360, hfov=65)
    jobs = {
        # Looking down 30° at a drop that bounces.
        "down": dict(cam=Camera3D(pos=(0, 1.7, 0), pitch=30, **small),
                     ball=Ball3D(pos=(0.1, 1.25, 1.7))),
        # The same, but the fall is generated (a third of Earth's gravity).
        "down_ai": dict(cam=Camera3D(pos=(0, 1.7, 0), pitch=30, **small),
                        ball=Ball3D(pos=(0.1, 1.25, 1.7), g=3.3)),
        # A phone held sideways.
        "sideways": dict(cam=Camera3D(pos=(0, 1.1, 0), roll=90, **small),
                         ball=Ball3D(pos=(0.0, 1.6, 2.4))),
        # Tilting down to follow the ball while it falls.
        "follow": dict(cam=Camera3D(pos=(0, 1.5, 0), pitch=5, pitch_rate=25, **small),
                       ball=Ball3D(pos=(0.1, 1.6, 2.2))),
    }
    out = {}
    for name, kw in jobs.items():
        out[name] = render_clip(str(d / f"{name}.mp4"), fps=120, duration=1.3, **kw)["path"]
    return out


def test_camera_looking_down_measures_full_gravity(camera_clips):
    r = analyze(camera_clips["down"], preset="ball=74mm", identify_mode="off")
    assert r["camera"]["pitched"] and r["camera"]["pitch"] == pytest.approx(30, abs=3)
    assert r["verdict"]["verdict"] == "REAL"
    assert r["verdict"]["g"] == pytest.approx(9.81, rel=0.08)
    # The old assumption (a level camera) reads it low: about cos(30°) of g.
    flat = analyze(camera_clips["down"], preset="ball=74mm", identify_mode="off",
                   camera_mode="upright")
    assert flat["verdict"]["g"] < 0.92 * 9.81


def test_generated_fall_seen_from_above_is_still_caught(camera_clips):
    r = analyze(camera_clips["down_ai"], preset="ball=74mm", identify_mode="off")
    assert r["verdict"]["verdict"] == "AI"
    assert r["verdict"]["g"] == pytest.approx(3.3, rel=0.15)


def test_sideways_camera(camera_clips):
    r = analyze(camera_clips["sideways"], preset="ball=74mm", identify_mode="off")
    assert r["camera"]["rolled"] and abs(abs(r["camera"]["roll"]) - 90) < 3
    assert r["verdict"]["verdict"] == "REAL"
    assert r["verdict"]["g"] == pytest.approx(9.81, rel=0.08)


def test_camera_that_tilts_to_follow_the_ball(camera_clips):
    r = analyze(camera_clips["follow"], preset="ball=74mm", identify_mode="off")
    assert r["stabilization"]["applied"] and r["stabilization"]["model"] == "homography"
    assert r["verdict"]["verdict"] == "REAL"
    assert r["verdict"]["g"] == pytest.approx(9.81, rel=0.08)


# ---- a ball close to the camera, on a plain backdrop (generated clips) ----------
# Video generators like a big pale ball against a seamless white backdrop. The
# ball then fills a third of the picture (it can't fall two of its own widths
# inside it), looks like a shadow by colour (the backdrop, only darker), and is
# the only detail in the picture (so its motion can pass for the camera's).
@pytest.fixture(scope="module")
def plain_clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("plain")
    out = {}
    for name, g in (("real", 9.81), ("ai", 2.0)):
        sc = Scene(width=640, height=360, px_per_m=700.0, ball_d_m=0.22, ball_bgr=(225, 225, 225),
                   bg_level=245, texture=False, shading=0.3, x0=100.0, y0=95.0, floor_y=280.0,
                   seed=5)
        m = Motion(g=g, restitution=0.85, t_release=0.15, vx=0.4 if g > 5 else 0.2)
        out[name] = make_clip(str(d / f"{name}.mp4"), scene=sc, motion=m, capture_fps=60,
                              duration=1.3 if g > 5 else 2.6)["path"]
    sc = Scene(width=640, height=360, px_per_m=600.0, ball_d_m=0.21, bg_level=240, texture=False,
               noise=1.0, panels=((180, 90, 30), (40, 210, 235), (235, 235, 235), (180, 90, 30),
                                  (40, 210, 235), (235, 235, 235)),
               shading=0.25, x0=120.0, y0=90.0, floor_y=290.0, seed=3)
    out["panels"] = make_clip(str(d / "panels.mp4"), scene=sc,
                              motion=Motion(g=9.81, restitution=0.8, t_release=0.15, vx=0.35),
                              capture_fps=60, duration=1.4)["path"]
    return out


def test_ball_on_a_plain_backdrop_is_not_taken_for_camera_motion(plain_clips):
    from novex.stabilize import stabilize
    from novex.video import read_frames
    for name in ("panels", "real"):          # detail only on the ball; no detail at all
        frames, _, _ = read_frames(plain_clips[name])
        _, T, info = stabilize(frames)
        assert T is None and info["unreliable"] and info["ignored"], name
    r = analyze(plain_clips["panels"], preset="volleyball")
    assert r["verdict"]["verdict"] == "REAL"
    assert r["verdict"]["g"] == pytest.approx(9.81, rel=0.06)


def test_big_pale_ball_close_to_the_camera(plain_clips):
    r = analyze(plain_clips["real"], preset="soccer")
    assert r["fit"]["diam_px"] == pytest.approx(0.22 * 700.0, rel=0.05)
    assert r["verdict"]["verdict"] == "REAL"
    assert r["verdict"]["g"] == pytest.approx(9.81, rel=0.06)
    r = analyze(plain_clips["ai"], preset="soccer")
    assert r["verdict"]["verdict"] == "AI"
    assert r["verdict"]["g"] == pytest.approx(2.0, rel=0.1)


# ---- sizing the object: a typed name, Gemini ---------------------------------------
def test_a_kind_of_object_gets_sized_by_gemini(tmp_path, monkeypatch):
    # "eraser" brings a pencil eraser's 6 cm; the one in the clip is a 14 cm
    # whiteboard eraser. With Gemini on, it sizes the one it sees.
    from novex import gemini, identify
    from novex.synth import Body
    assert identify.sized_by_kind(identify.from_name("eraser"))
    assert not identify.sized_by_kind(identify.from_name("eraser=14cm"))
    assert not identify.sized_by_kind(identify.from_name("volleyball"))
    sc = Scene(width=640, height=480, px_per_m=500.0, main_ball=False, bg_level=70, seed=5,
               bodies=(Body(kind="box", size_m=0.14, width_m=0.05, bgr=(25, 25, 30), x0=320.0,
                            y0=70.0, motion=Motion(g=9.81, t_release=0.12), omega=10.0,
                            angle0=0.4),))
    clip = make_clip(str(tmp_path / "eraser.mp4"), scene=sc, motion=Motion(t_release=0.12),
                     capture_fps=60, duration=0.6)["path"]
    seen = {}

    def fake_identify(frame, xyd, hint=None, crops=None):
        seen.update(hint=hint, crops=len(crops or []))
        return {"object": "whiteboard eraser", "is_round": False, "size_m": 0.14,
                "size_rel_uncertainty": 0.05, "confidence": 0.9, "terminal_speed_m_s": 15,
                "reason": "felt block with a dark handle"}
    monkeypatch.setattr(gemini, "available", lambda: (True, "test"))
    monkeypatch.setattr(gemini, "identify_object", fake_identify)
    monkeypatch.setattr(gemini, "explain", lambda r: None)
    r = analyze(clip, preset="eraser", use_gemini=True)
    assert seen["hint"] == "eraser" and seen["crops"] >= 2      # full-resolution close-ups
    assert r["scale"]["rulers"][0]["detail"].startswith("eraser = 140 mm")
    assert any("Gemini sizes this one at 140 mm" in n for n in r["notes"])
    assert r["verdict"]["verdict"] == "REAL"
    assert r["verdict"]["g"] == pytest.approx(9.81, rel=0.06)
    # Without Gemini the typical size stands, and the clip reads far too slow.
    v = analyze(clip, preset="eraser")["verdict"]
    assert v["verdict"] == "AI" and v["g"] < 6


def test_gemini_moves_on_when_a_model_is_busy(monkeypatch):
    pytest.importorskip("google.genai")
    from novex import gemini

    class Busy(Exception):
        pass
    tried, busy = [], set()

    class Models:
        def generate_content(self, model, contents, config):
            tried.append(model)
            if len(busy) < 2:
                busy.add(model)
            if model in busy:                       # the first two models are busy
                raise Busy("503 UNAVAILABLE")
            return type("R", (), {"text": '{"ok": true}'})()
    monkeypatch.setattr(gemini, "_client", lambda: type("C", (), {"models": Models()})())
    monkeypatch.setattr(gemini, "_RESTING", {})
    resp, used = gemini._generate("hi", json_out=True)
    assert gemini._json(resp) == {"ok": True}
    assert len(tried) == 3 and used == tried[-1] and len(set(tried)) == 3
    # The busy ones are left alone for a while: the next request starts past them.
    tried.clear()
    gemini._generate("hi", json_out=True)
    assert tried == [used]


def test_gemini_replies_are_pooled(monkeypatch):
    # Asked twice, Gemini said "pickleball" once and "practice golf ball" once:
    # the pooled answer keeps the golf ball as a look-alike of another size.
    from novex import gemini, identify
    a = gemini._answer({"candidates": [
        {"object": "pickleball", "is_round": True, "size_m": 0.074, "probability": 0.9},
        {"object": "plastic practice golf ball", "is_round": True, "size_m": 0.042,
         "probability": 0.1}]}, "m1")
    b = gemini._answer({"candidates": [
        {"object": "Plastic practice golf ball", "is_round": True, "size_m": 0.042,
         "probability": 0.8},
        {"object": "pickleball", "is_round": True, "size_m": 0.074, "probability": 0.2}]}, "m2")
    g = gemini.pool([a, b])
    assert g["object"] == "pickleball" and g["size_m"] == 0.074
    assert g["confidence"] == pytest.approx(0.55)
    assert g["alternatives"][0]["size_m"] == 0.042
    assert g["alternatives"][0]["probability"] == pytest.approx(0.45)
    monkeypatch.setattr(gemini, "identify_object", lambda *a_, **k_: g)
    idn = identify.gemini_identify(np.zeros((20, 20, 3), np.uint8), (10, 10, 6))
    assert idn["size_m"] == 0.074 and idn["alternatives"][0]["size_m"] == 0.042
    # One sure reply: a long shot isn't a look-alike worth checking.
    sure = gemini.pool([gemini._answer({"candidates": [
        {"object": "volleyball", "is_round": True, "size_m": 0.21, "probability": 0.95},
        {"object": "beach ball", "is_round": True, "size_m": 0.3, "probability": 0.03}]}, "m")])
    monkeypatch.setattr(gemini, "identify_object", lambda *a_, **k_: sure)
    assert identify.gemini_identify(np.zeros((20, 20, 3), np.uint8), (10, 10, 6))[
        "alternatives"] == []


def test_a_look_alike_of_another_size_makes_it_unsure():
    from novex.verdict import decide, size_doubt
    base = dict(clock=None, tests=[], n_points=40, duration=0.4)
    # Read as a 74 mm pickleball it falls at 9.1 m/s^2; a 42 mm practice golf
    # ball that looks the same would fall at 5.2: the call hangs on which it is.
    v = decide(a_px=9.1 / 0.074 * 60, a_sigma=10.0, scale={"s": 0.074 / 60, "rel_sigma": 0.03},
               **base)
    assert v["verdict"] == "REAL"
    golf = [{"label": "practice golf ball", "size_m": 0.042, "p": 0.45}]
    u = size_doubt(v, "pickleball", 0.074, golf)
    assert u["verdict"] == "UNSURE" and u["stage"] == "ruler: which object?"
    assert "practice golf ball" in u["headline"]
    assert u["size_flip"][0]["g"] == pytest.approx(9.1 * 42 / 74, rel=0.01)
    assert any("too slow" in r for r in u["reasons"])
    # Long shots don't, unless together they make a fair chance (20%).
    assert size_doubt(v, "pickleball", 0.074, [{**golf[0], "p": 0.1}]) is v
    two = [{**golf[0], "p": 0.12}, {"label": "foam ball", "size_m": 0.05, "p": 0.1}]
    assert size_doubt(v, "pickleball", 0.074, two)["verdict"] == "UNSURE"
    # A look-alike of about the same size changes nothing...
    assert size_doubt(v, "pickleball", 0.074, [{"label": "wiffle ball", "size_m": 0.073,
                                                "p": 0.4}]) is v
    # ...and nor does any size when the physics itself breaks.
    broke = {**v, "verdict": "AI", "stage": "physics: Bounce law failed"}
    assert size_doubt(broke, "pickleball", 0.074, golf) is broke


def test_unsure_when_gemini_sees_look_alikes_of_other_sizes(clips, monkeypatch):
    from novex import gemini

    def fake(frame, xyd, hint=None, crops=None):
        return {"object": "orange toy ball", "is_round": True, "size_m": 0.038,
                "size_rel_uncertainty": 0.03, "confidence": 0.55, "terminal_speed_m_s": None,
                "reason": "", "model": "fake", "dimensions_m": [], "samples": 2,
                "alternatives": alts}
    monkeypatch.setattr(gemini, "available", lambda: (True, "test"))
    monkeypatch.setattr(gemini, "identify_object", fake)
    monkeypatch.setattr(gemini, "explain", lambda r: None)
    alts = [{"object": "tennis ball", "size_m": 0.067, "probability": 0.4}]
    v = analyze(clips["real"], use_gemini=True)["verdict"]
    assert v["verdict"] == "UNSURE" and v["stage"] == "ruler: which object?"
    assert v["size_flip"][0]["verdict"] == "AI"
    alts = [{"object": "orange marble", "size_m": 0.036, "probability": 0.4}]
    v = analyze(clips["real"], use_gemini=True)["verdict"]
    assert v["verdict"] == "REAL" and v["g"] == pytest.approx(9.81, rel=0.06)


def test_camera_listens_only_to_falls_that_make_sense():
    from novex.analyze import _camera_votes
    prim = {"n": 1, "g2": [0.0, 1.0], "nvec": [0.02, 1.0, 0.015], "pitch": 0.9,
            "pitch_sigma": 2.1, "roll": 1.0, "roll_sigma": 0.4}
    other = {"n": 2, "g2": [0.0, 1.0], "nvec": [0.0, 1.0, 0.05], "pitch": 2.8,
             "pitch_sigma": 3.4, "roll": -0.2, "roll_sigma": 1.8}
    # A ball rolling toward the camera: its "gravity" runs along the view, and
    # sideways in the picture, while it speeds up nowhere in the picture.
    rolling = {"n": 3, "g2": [0.0, 1.0], "nvec": [0.43, -0.06, 0.90], "pitch": 64.2,
               "pitch_sigma": 0.26, "roll": 98.4, "roll_sigma": 1.5}
    tipped = {"n": 4, "g2": [0.0, 1.0], "nvec": [0.0, 0.71, 0.71], "pitch": 45.0,
              "pitch_sigma": 1.0, "roll": 0.0, "roll_sigma": 1.0}
    outline = {"n": 5, "g2": [0.0, 1.0], "flights": 1}            # no 3D reading
    got = _camera_votes([prim, other, rolling, tipped, outline], 1)
    assert [e["n"] for e in got] == [1, 2]
    assert "speeds up" in rolling["left_out"] and "primary" in tipped["left_out"]


# ---- the newer laws: spin, shadows, floor, sideways, bounciness, size ----------
def _sphere_views(axis, deg, seed=1):
    """Two close-ups of a marked ball, the second turned by deg about axis
    (x right, y down, z away from the camera)."""
    import math
    rng = np.random.default_rng(seed)
    c3 = rng.normal(size=(60, 3))
    c3 /= np.linalg.norm(c3, axis=1, keepdims=True)
    vals = rng.uniform(-1, 1, 60)

    def rot(ax, ang):
        k = np.asarray(ax, float) / np.linalg.norm(ax)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K

    def render(R_, R=60, H=200):
        yy, xx = np.mgrid[0:H, 0:H].astype(float)
        u, v = (xx - 100.3) / R, (yy - 99.6) / R
        w = np.sqrt(np.clip(1 - u ** 2 - v ** 2, 0, 1))
        p = np.stack([u, v, -w], axis=-1) @ R_
        tex = (np.exp((p @ c3.T - 1) / 0.03) * vals).sum(axis=-1)
        img = np.full((H, H), 120.0)
        inside = u ** 2 + v ** 2 <= 1
        img[inside] = ((140 + 60 * tex) * (0.7 + 0.3 * w))[inside]
        return np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    R0 = rot((0.3, 0.5, 0.8), 0.7)
    return render(R0), render(rot(axis, math.radians(deg)) @ R0)


def test_spin_reader_reads_the_turn_about_the_line_of_sight():
    from novex import spin
    for axis, deg, want_z in [((0, 0, 1), 6.0, 6.0), ((0, 0, 1), -10.0, -10.0),
                              ((1, 0, 0), 10.0, 0.0), ((1, 1, 1), 6.0, 6.0 / np.sqrt(3))]:
        a, b = _sphere_views(axis, deg)
        w, match, _ = spin.turn3d(spin.close_up(a, 100.3, 99.6, 120),
                                  spin.close_up(b, 100.0, 100.0, 120))
        assert match > 0.9
        assert np.degrees(w[2]) == pytest.approx(want_z, abs=0.6)


def _spin_clip(path, **motion):
    panels = ((40, 60, 220), (230, 230, 230), (200, 120, 30), (40, 200, 230), (230, 230, 230),
              (60, 160, 60))
    sc = Scene(width=640, height=480, px_per_m=400.0, ball_d_m=0.21, panels=panels, shading=0.2,
               bg_level=40, y0=70.0, floor_y=400.0, x0=200.0, seed=3)
    m = Motion(g=9.81, restitution=0.75, vx=0.6, t_release=0.1, **motion)
    return make_clip(str(path), scene=sc, motion=m, capture_fps=120, duration=1.7)["path"]


def _test(r, name):
    return next(t for t in r["tests"] if t["name"] == name)


def test_bounce_keeps_angular_momentum_about_the_contact(tmp_path):
    # Real: the floor's friction slows the ball sideways and spins it up,
    # together, keeping its angular momentum about the contact point.
    r = analyze(_spin_clip(tmp_path / "grip.mp4", grip=0.7), preset="volleyball")
    t = _test(r, "Spin at bounces")
    assert t["status"] == "pass"
    rows = t["detail"]["bounces"]
    assert rows and all(abs(b["gain"]) < 0.03 for b in rows)
    assert any(b["spin_after"] - b["spin_before"] > 1.0 for b in rows)      # it did spin up
    assert _test(r, "Steady spin")["status"] == "pass"
    assert r["verdict"]["verdict"] == "REAL"
    # Generated: the spin jumps at each bounce with no sideways push to cause it.
    # With gravity right and everything else in order, that one failure is
    # flagged but doesn't decide on its own.
    r = analyze(_spin_clip(tmp_path / "kick.mp4", spin_kick=12.0), preset="volleyball")
    assert _test(r, "Spin at bounces")["status"] == "fail"
    assert r["verdict"]["verdict"] == "REAL" and r["verdict"]["confidence"] <= 0.6
    assert r["verdict"]["lone_failure"] == "Spin at bounces"


def _shadow_clip(path, **scene):
    sc = Scene(width=640, height=480, px_per_m=400.0, ball_d_m=0.067, ball_bgr=(40, 200, 230),
               shading=0.3, bg_level=170, y0=70.0, floor_y=400.0, x0=150.0, seed=4,
               shadow=(0.35, 0.03, 0.5), **scene)
    m = Motion(g=9.81, restitution=0.7, vx=0.5, t_release=0.1)
    return make_clip(str(path), scene=sc, motion=m, capture_fps=120, duration=1.6)["path"]


def test_ball_meets_its_shadow_at_every_bounce(tmp_path):
    r = analyze(_shadow_clip(tmp_path / "lit.mp4"), preset="tennis")
    assert _test(r, "Shadow follows")["status"] == "pass"
    assert _test(r, "Shadow follows")["detail"]["slope_x"] == pytest.approx(0.35, abs=0.08)
    assert _test(r, "Ball meets its shadow")["status"] == "pass"
    assert r["verdict"]["verdict"] == "REAL"
    # A shadow painted a ball-width and a half to the side never meets the ball
    # (on its own, with gravity right, a flag rather than a verdict).
    r = analyze(_shadow_clip(tmp_path / "apart.mp4", shadow_gap=1.6), preset="tennis")
    assert _test(r, "Ball meets its shadow")["status"] == "fail"
    assert r["verdict"]["lone_failure"] == "Ball meets its shadow"


def test_floor_sideways_bounciness_and_size_laws():
    from novex import checks as ck
    from novex.physics import fit_flights
    # One flat floor: bottoms (scaled to the reference distance) agree.
    pts = [{"t": 0.5, "bottom": 400.0, "q": 1.0}, {"t": 1.1, "bottom": 403.0, "q": 1.01},
           {"t": 1.6, "bottom": 398.0, "q": 1.0}]
    assert ck.floor_check(pts, 40.0)["status"] == "pass"
    pts[1]["bottom"] = 440.0                                      # a full width lower
    assert ck.floor_check(pts, 40.0)["status"] == "fail"
    assert ck.floor_check(pts[:1], 40.0)["status"] == "skip"
    # No sideways push: two hops, the second pushed sideways at 30% of g.
    t = np.arange(0, 1.2, 1 / 120)
    a = 9.81 * 400

    def hops(side):
        x, y = np.empty_like(t), np.empty_like(t)
        for k, (t0, t1) in enumerate([(0.0, 0.6), (0.6, 1.2)]):
            sel = (t >= t0) & (t < t1)
            tau = t[sel] - (t0 + t1) / 2
            y[sel] = 300 + 0.5 * a * tau ** 2
            x[sel] = 200 + 50 * tau + (0.5 * side * a * tau ** 2 if k == 1 else 0)
        rng = np.random.default_rng(0)
        return x + rng.normal(0, 0.2, len(t)), y + rng.normal(0, 0.2, len(t))
    fl = [np.flatnonzero(t < 0.595), np.flatnonzero(t > 0.605)]
    for side, want in ((0.0, "pass"), (0.3, "fail")):
        x, y = hops(side)
        jf = fit_flights(t, x, y, np.ones(len(t)), fl, fps=120, n_px_axis=480)
        assert ck.sideways_check(jf, t)["status"] == want
    # Steady bounces: one bounce keeping far more of its speed than the rest.
    b = [{"v_in": v, "restitution_t": e, "restitution_t_sigma": 0.01, "restitution": e,
          "restitution_sigma": 0.01} for v, e in ((1500, 0.76), (1150, 0.78), (900, 0.79),
                                                   (700, 0.80))]
    assert ck.steady_bounce_check(b)["status"] == "pass"
    b[2]["restitution_t"] = 1.05
    assert ck.steady_bounce_check(b)["status"] == "fail"
    # Constant size: a ball whose width swells 40% partway through a flight.
    tt = np.linspace(0, 0.6, 72)
    d = 50.0 / (1 + 0.1 * tt) + np.random.default_rng(1).normal(0, 0.2, len(tt))
    assert ck.size_check(tt, d, [np.arange(len(tt))])["status"] == "pass"
    swell = d * (1 + 0.40 * np.exp(-((tt - 0.3) / 0.06) ** 2))
    assert ck.size_check(tt, swell, [np.arange(len(tt))])["status"] == "warn"


def test_rolling_ball_turns_once_per_circumference():
    from novex import checks as ck
    t = np.arange(60) / 60.0
    x = 100 + 120 * t                               # 120 px/s along the floor
    d = np.full(60, 40.0)                           # radius 20 px: rolling spin 6 rad/s
    for spin_rate, want in ((6.0, "pass"), (0.0, "fail")):
        sp = {"i": np.arange(59), "j": np.arange(1, 60), "t": t[:-1] + 1 / 120,
              "w3": np.column_stack([np.zeros(59), np.zeros(59),
                                     spin_rate + np.random.default_rng(2).normal(0, 0.2, 59)]),
              "readable": 1.0, "tried": 59}
        sp["omega"] = sp["w3"][:, 2]
        assert ck.rolling_check(sp, t, x, d, [np.arange(60)])["status"] == want


def test_newer_laws_can_be_made_advisory(monkeypatch):
    from novex import config as Cfg
    from novex.verdict import decide
    fails = [{"name": "One flat floor", "status": "fail", "summary": "Lands on two floors.",
              "detail": {}, "experimental": True},
             {"name": "Spin at bounces", "status": "fail", "summary": "Spin from nowhere.",
              "detail": {}, "experimental": True}]
    base = dict(a_px=3900.0, a_sigma=30.0, scale={"s": 1 / 400.0, "rel_sigma": 0.02},
                clock=None, tests=fails, n_points=40, duration=0.4)
    assert decide(**base)["verdict"] == "AI"            # two laws broken
    monkeypatch.setattr(Cfg, "EXPERIMENTAL_CHECKS_DECIDE", False)
    v = decide(**base)
    assert v["verdict"] == "REAL" and "Lands on two floors." in v["reasons"]


def test_one_failing_law_alone_does_not_make_a_clip_ai():
    from novex.verdict import decide

    def chk(name, strong=False):
        r = {"name": name, "status": "fail", "summary": f"{name} broke.", "detail": {}}
        if strong:
            r["strong"] = True
        return r
    good = dict(a_px=3920.0, a_sigma=30.0, scale={"s": 1 / 400.0, "rel_sigma": 0.02},
                clock=None, n_points=40, duration=0.4)              # 9.8 m/s^2
    v = decide(tests=[chk("Bounce law")], **good)
    assert v["verdict"] == "REAL" and v["confidence"] <= 0.6
    assert v["lone_failure"] == "Bounce law" and "Bounce law broke." in v["reasons"]
    # Two laws failing, or one failing grossly, is still AI...
    assert decide(tests=[chk("Bounce law"), chk("Constant acceleration")], **good)[
        "verdict"] == "AI"
    assert decide(tests=[chk("Bounce law", strong=True)], **good)["verdict"] == "AI"
    # ...and so is a lone failure when gravity is off too.
    slow = dict(good, a_px=2400.0)                                   # 6 m/s^2
    assert decide(tests=[chk("Bounce law")], **slow)["verdict"] == "AI"


def test_bounce_restitution_comes_from_times_not_shaky_speeds():
    # Four bounces at e = 0.8 under one g; the clip ends partway through the
    # last hop, which is also tracked badly. Every bounce's restitution comes
    # from times (the peaks either side, or whole hops), not from the stub's
    # shaky speeds, and lands on 0.8.
    a = 9.81 * 300.0
    rng = np.random.default_rng(3)
    T = np.sqrt(2 * 300.0 / a)
    hops, tb, v_out = [(0.0, T, 0.0)], T, np.sqrt(2 * a * 300.0)
    for k in range(4):
        v_out *= 0.8
        hops.append((tb, tb + 2 * v_out / a, v_out))
        tb += 2 * v_out / a
    ts, ys, idxs = [], [], []
    for k, (t0, t1, vo) in enumerate(hops):
        cut = t1 if k < len(hops) - 1 else t0 + 0.7 * (t1 - t0)      # the clip ends mid-hop
        tt = np.arange(t0 + 1 / 240, cut - 1 / 480, 1 / 240)
        tau = tt - t0
        yy = 0.5 * a * tt ** 2 if k == 0 else 300.0 - vo * tau + 0.5 * a * tau ** 2
        idxs.append(np.arange(len(ts), len(ts) + len(tt)))
        ts.extend(tt)
        ys.extend(yy + rng.normal(0, 1.5 if k == len(hops) - 1 else 0.3, len(tt)))
    t = np.array(ts)
    y = np.array(ys)
    x = 320 + rng.normal(0, 0.3, len(ts))
    jf = fit_flights(t, x, y, np.ones(len(t)), idxs, fps=240, n_px_axis=720)
    r = checks.bounce_check(t, jf, 1 / 240)
    assert r["status"] == "pass"
    rows = r["detail"]["bounces"]
    assert len(rows) == 4 and all(b["method"] in ("peaks", "hops", "half") for b in rows)
    for b in rows:
        assert b["restitution_best"] == pytest.approx(0.8, abs=0.04)


def test_a_hand_toss_above_the_floor_is_not_a_bounce():
    # Drop, bounce off the floor (e = 0.75), then a hand catches it 120 px
    # above the floor and tosses it up faster than it came: a hand can add
    # energy, a floor can't, so only the floor bounce is held to the law.
    a = 9.81 * 300.0
    rng = np.random.default_rng(5)
    segs = []                                   # (t0, t1, y0, v0): y down, v down positive
    T = np.sqrt(2 * 300.0 / a)
    segs.append((0.0, T, 0.0, 0.0))
    v1 = 0.75 * a * T
    t_hand = T + (v1 - np.sqrt(v1 ** 2 - 2 * a * 120.0)) / a   # rising through 120 px up
    t_turn = t_hand + 2 * (v1 - a * (t_hand - T)) / a         # comes back down to the hand
    segs.append((T, t_turn, 300.0, -v1))
    v_hand_in = v1 - a * (t_hand - T)
    v_toss = 1.4 * v_hand_in
    segs.append((t_turn, t_turn + 2 * v_toss / a - 1 / 120, 180.0, -v_toss))
    ts, ys, idxs = [], [], []
    for t0, t1, y0, v0 in segs:
        tt = np.arange(t0 + 1 / 240, t1 - 1 / 480, 1 / 240)
        tau = tt - t0
        y = y0 + v0 * tau + 0.5 * a * tau ** 2
        if t0 == T:                             # the hop from the floor: stop at the hand
            keep = y <= 300.0
            tt, y = tt[keep], y[keep]
        idxs.append(np.arange(len(ts), len(ts) + len(tt)))
        ts.extend(tt)
        ys.extend(y)
    t = np.array(ts)
    y = np.array(ys) + rng.normal(0, 0.3, len(ts))
    x = 320 + rng.normal(0, 0.3, len(ts))
    jf = fit_flights(t, x, y, np.ones(len(t)), idxs, fps=240, n_px_axis=720)
    assert checks.bounce_check(t, jf, 1 / 240)["status"] == "fail"
    r = checks.bounce_check(t, jf, 1 / 240, diam_px=40.0)
    assert r["status"] == "pass"
    assert [b["off_floor"] for b in r["detail"]["bounces"]] == [False, True]


def test_a_landing_without_a_bounce_ends_the_flight():
    # A can thrown in an arc lands and skids toward the camera instead of
    # bouncing: its fall stops dead without ever turning upward. The flight
    # ends at the landing, and the slide, longer and smoother than the flight
    # and with a tiny but "clear" acceleration, is no free flight to measure.
    rng = np.random.default_rng(6)
    fps = 240.0
    t = np.arange(0, 1.5, 1 / fps)
    t_land, v0 = 0.6, -400.0                            # thrown upward, px/s
    y_land = 100 + v0 * t_land + 0.5 * A_TRUE * t_land ** 2
    s = t - t_land
    y = np.where(t < t_land, 100 + v0 * t + 0.5 * A_TRUE * t ** 2,
                 y_land + 200.0 * s + 0.5 * 20.0 * s ** 2)
    x = 700 - 400 * t + rng.normal(0, 0.3, len(t))
    y = y + rng.normal(0, 0.3, len(t))
    fl = split_flights(t, x, y, diam_px=30.0)
    assert t[fl[0][-1]] == pytest.approx(t_land, abs=0.02)
    jf = fit_flights(t, x, y, np.ones(len(t)), fl, fps=fps, n_px_axis=720)
    assert t[jf.flights[jf.main].idx[-1]] < t_land + 0.02
    assert jf.a_px == pytest.approx(A_TRUE, rel=0.02)


def test_a_tube_that_breaks_into_pieces_is_still_one_object():
    # A white tube on a beige wall breaks into pieces where its colour matches
    # the wall: the track of its top piece ends a frame after the track of its
    # bottom piece begins, half a tube away. That is one tube, not two.
    from novex.track import Det, Track, _same_object
    times = np.arange(60) / 240.0

    def piece(frames, y0, dy):
        return Track(dets=[Det(f=i, x=400.0 - 2 * i, y=y0 + dy * i, d=26, area=900, elong=3.4,
                               contrast=40) for i in frames], kind="moving")
    top = piece(range(0, 31), 140.0, 1.0)
    bottom = piece(range(30, 60), 190.0, 1.0)            # 50 px below it at frames 30-31
    assert _same_object(top, bottom, times) and _same_object(bottom, top, times)
    # Two balls whose paths meet for a frame, far apart for their size, are two.
    ball = Track(dets=[Det(f=i, x=200.0, y=bottom.dets[i - 30].y, d=26, area=530, elong=1.0,
                           contrast=40) for i in range(30, 60)], kind="moving")
    ball_top = Track(dets=[Det(f=i, x=260.0, y=100.0 + i, d=26, area=530, elong=1.0, contrast=40)
                           for i in range(0, 31)], kind="moving")
    assert not _same_object(ball_top, ball, times)
