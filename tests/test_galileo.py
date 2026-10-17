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


# ---- several objects, and objects that aren't round ----------------------------------
def test_shape_classifier_tells_a_ball_from_a_box():
    import cv2
    from galileo import shape as shp
    from galileo.track import Det
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
    from galileo.track import measure_length
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
    from galileo.track import Det, Track, select_objects
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
    from galileo.verdict import combine_objects
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
    from galileo.synth import Body
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
    from galileo.synth import Body
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
    from galileo.synth import Body
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
    from galileo import identify
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
    from galileo.track import detect
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
    from galileo.synth import Body
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
    from galileo.cli import _objects
    assert _objects(["rotisserie chicken=22cm", "tennis,auto"]) == [
        "rotisserie chicken=22cm", "tennis", None]


# ---- The camera at any angle ------------------------------------------------
# galileo/camera.py (the fall in 3D, the level view), galileo/scene.py (the
# room's lines), galileo/stabilize.py (pans and turns), rendered with
# galileo/synth3d.py.

import math  # noqa: E402

from galileo import camera as cam_mod  # noqa: E402
from galileo import scene as scene_mod  # noqa: E402
from galileo.synth3d import (Ball3D, Camera3D, ball_path, project, render_clip,  # noqa: E402
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
