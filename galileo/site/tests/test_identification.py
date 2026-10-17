"""Site provider selection and real-core fallback, without network/API requests."""
import builtins
import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from fastapi.testclient import TestClient

from galileo import gemini, identify
from galileo.site.app import create_app
from galileo.site.pipeline import analyse


@pytest.fixture(autouse=True)
def isolated_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def sdk(monkeypatch):
    google = ModuleType("google")
    google.genai = ModuleType("google.genai")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", google.genai)


@pytest.fixture
def calls(monkeypatch):
    recorded = []
    core = importlib.import_module("galileo.analyze")
    def public_analyze(path, **kwargs):
        recorded.append(kwargs)
        return {"verdict": {"verdict": "UNSURE"}}
    monkeypatch.setattr(core, "analyze", public_analyze)
    return recorded


@pytest.mark.parametrize("variable", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_saved_environment_key_enables_gemini(tmp_path, monkeypatch, sdk, calls, variable):
    monkeypatch.setenv(variable, "test-only-not-a-real-key")
    analyse(tmp_path, tmp_path / "input.mp4", {"object": "auto"}, lambda _: None)
    assert calls[0]["use_gemini"] is True
    assert calls[0]["preset"] is None
    assert calls[0].get("identify_mode", "auto") == "auto"  # Keep CLIP fallback.
    assert "test-only-not-a-real-key" not in repr(calls)


def test_no_key_keeps_offline_identification(tmp_path, sdk, calls):
    analyse(tmp_path, tmp_path / "input.mp4", {"object": "auto"}, lambda _: None)
    assert calls[0]["use_gemini"] is False


def test_missing_gemini_sdk_does_not_break_offline_analysis(tmp_path, monkeypatch, calls):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-not-a-real-key")
    original_import = builtins.__import__
    def without_sdk(name, *args, **kwargs):
        if name == "google.genai":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_sdk)
    analyse(tmp_path, tmp_path / "input.mp4", {"object": "auto"}, lambda _: None)
    assert calls[0]["use_gemini"] is False


@pytest.mark.parametrize("preference", [False, True])
def test_explicit_preference_is_preserved(tmp_path, monkeypatch, calls, preference):
    def unexpected_check():
        pytest.fail("An explicit provider preference must not be overridden")
    monkeypatch.setattr(gemini, "available", unexpected_check)
    analyse(tmp_path, tmp_path / "input.mp4", {"object": "auto", "gemini": preference}, lambda _: None)
    assert calls[0]["use_gemini"] is preference


def test_legacy_gemini_object_choice_still_works(tmp_path, calls):
    analyse(tmp_path, tmp_path / "input.mp4", {"object": "gemini"}, lambda _: None)
    assert calls[0]["use_gemini"] is True
    assert calls[0]["preset"] is None


def test_availability_is_checked_per_job(tmp_path, monkeypatch, sdk, calls):
    analyse(tmp_path, tmp_path / "one.mp4", {}, lambda _: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-not-a-real-key")
    analyse(tmp_path, tmp_path / "two.mp4", {}, lambda _: None)
    assert [c["use_gemini"] for c in calls] == [False, True]


@pytest.mark.parametrize("g_ok,c_ok,preferred,fallback", [
    (True, True, "gemini", "clip"),
    (False, True, "clip", None),
    (False, False, None, None),
])
def test_status_reports_provider_and_missing_setup(tmp_path, monkeypatch, g_ok, c_ok, preferred, fallback):
    monkeypatch.setattr(gemini, "available", lambda: (g_ok, "test-model" if g_ok else "no GEMINI_API_KEY set"))
    monkeypatch.setattr(identify, "clip_available", lambda: (c_ok, "test-model" if c_ok else "CLIP not installed"))
    monkeypatch.setitem(sys.modules, "torchvision", ModuleType("torchvision"))
    monkeypatch.setitem(sys.modules, "PIL", ModuleType("PIL"))
    with TestClient(create_app(str(tmp_path))) as client:
        response = client.get("/api/status")
    assert response.status_code == 200
    state = response.json()["auto"]
    assert state["available"] is (g_ok or c_ok)
    assert state["preferred"] == preferred
    assert state["fallback"] == fallback
    assert state["gemini_detail"] == (None if g_ok else "no GEMINI_API_KEY set")


@pytest.mark.parametrize("outcome", ["success", "error", "empty"])
def test_site_uses_core_gemini_priority_and_clip_fallback(tmp_path, monkeypatch, outcome):
    """Run the public pipeline on a real clip; only model answers are replaced."""
    order = []
    monkeypatch.setattr(gemini, "available", lambda: (True, "test-model"))
    monkeypatch.setattr(gemini, "explain", lambda _: None)
    monkeypatch.setattr(identify, "clip_available", lambda: (True, "test-model"))
    monkeypatch.setattr(identify, "clip_probs", lambda *a, **kw: (
        ["steel-38mm"], ["preset"], np.array([1.0]), np.array([1.0]),
    ))
    def gemini_answer(*args, **kwargs):
        order.append("Gemini")
        if outcome == "error":
            raise RuntimeError("simulated Gemini outage")
        if outcome == "empty":
            return None
        return identify.from_preset("steel-38mm", source="Gemini", confidence=1.0)
    def clip_answer(*args, **kwargs):
        order.append("CLIP")
        return identify.from_preset("steel-38mm", source="CLIP", confidence=1.0)
    monkeypatch.setattr(identify, "gemini_identify", gemini_answer)
    monkeypatch.setattr(identify, "clip_identify", clip_answer)
    sample = Path(__file__).parents[1] / "samples" / "synthetic_real_60fps.mp4"
    result = analyse(tmp_path, sample, {"object": "auto"}, lambda _: None)
    assert order[0] == "Gemini"
    source = result["objects"][0]["identity"]["source"]
    if outcome == "success":
        assert source == "Gemini"
        assert "CLIP" not in order
    else:
        assert source == "CLIP"
        assert order.index("Gemini") < order.index("CLIP")
    assert result["verdict"]["verdict"] == "REAL"
