"""Contract checks for the site boundary; the physics lives in the core tests.

Run from the repository root: python -m pytest galileo/site/tests -q
"""
import importlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from galileo.site.app import _options
from galileo.site.pipeline import analyse, debug_timing


def test_site_uses_public_analyzer_and_keeps_new_fields(tmp_path, monkeypatch):
    core = importlib.import_module("galileo.analyze")
    calls = []
    result = {"verdict": {"verdict": "UNSURE"}, "future_core_metric": {"value": 17}}

    def public_analyze(path, **kwargs):
        calls.append((path, kwargs))
        kwargs["progress"]("new upstream stage")
        return result

    monkeypatch.setattr(core, "analyze", public_analyze)
    stages = []
    actual = analyse(tmp_path, tmp_path / "input.mp4", {
        "object": "custom", "size_mm": 38, "hfov": 70,
        "seed": {"t": 0.2, "x": 0.4, "y": 0.5},
    }, stages.append)
    assert actual is result
    assert len(calls) == 1
    assert stages == ["new upstream stage"]
    kwargs = calls[0][1]
    assert kwargs["size_m"] == pytest.approx(0.038)
    assert kwargs["hfov"] == 70
    assert kwargs["seed"] == (0.2, 0.4, 0.5)
    assert kwargs["debug_video"] == str(tmp_path / "debug.mp4")
    assert kwargs["debug_speed"] == 1.0


@pytest.mark.parametrize("obj,expected", [("auto", None), ("steel-38mm", "steel-38mm")])
def test_object_selection_reaches_core(tmp_path, monkeypatch, obj, expected):
    core = importlib.import_module("galileo.analyze")
    captured = {}

    def public_analyze(path, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(core, "analyze", public_analyze)
    analyse(tmp_path, tmp_path / "input.mp4", {"object": obj}, lambda _: None)
    assert captured["preset"] == expected


def test_new_core_presets_are_accepted(monkeypatch):
    from galileo import config
    monkeypatch.setitem(config.OBJECT_PRESETS, "future-object", {"label": "New object", "size_m": 0.2})
    assert _options("future-object", "", "", "")["object"] == "future-object"


@pytest.mark.parametrize("field_of_view", ["nan", "inf", "0", "180", "-20"])
def test_bad_camera_values_do_not_reach_core(field_of_view):
    with pytest.raises(HTTPException) as exc:
        _options("auto", "", "", field_of_view)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("source,truncated,synchronized", [
    ([0, 0.04, 0.08], False, True),
    ([0, 0.02, 0.04, 0.06, 0.08], False, False),
    ([0, 0.04, 0.08], True, False),
    ([0, 0.05, 0.1], False, False),
])
def test_debug_sync_is_only_used_for_matching_timelines(tmp_path, monkeypatch, source, truncated, synchronized):
    import av

    class Video:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, **kwargs):
            return [SimpleNamespace(pts=i, time_base=0.04) for i in range(3)]

    debug = tmp_path / "debug.mp4"
    debug.touch()
    monkeypatch.setattr(av, "open", lambda _: Video())
    mapping = debug_timing(debug, {"times": source, "truncated": truncated})
    assert mapping["debug_sync"] is synchronized
    assert mapping["debug_frame_times"] == [0, 0.04, 0.08]
    assert (mapping["debug_times"] is not None) is synchronized


def test_broken_debug_render_does_not_discard_result(tmp_path, monkeypatch):
    import av
    debug = tmp_path / "debug.mp4"
    debug.write_bytes(b"incomplete output")

    def fail(_):
        raise ValueError("video unavailable")

    monkeypatch.setattr(av, "open", fail)
    assert debug_timing(debug, {"times": [0]}) == {
        "debug_times": None, "debug_frame_times": [], "debug_sync": False,
    }
