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
