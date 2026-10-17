"""Gemini and Veo: object ID for the ruler, plain-English reasons, the AI twin.

Needs `pip install google-genai` and an API key in GEMINI_API_KEY (or
GOOGLE_API_KEY). Every call is optional: Galileo works offline without it.
Answers are cached on disk so a venue Wi-Fi drop can't break a demo clip
that already ran once.

Model names change often. Override with GALILEO_GEMINI_MODEL and
GALILEO_VEO_MODEL. The defaults were current in October 2026; note that
gemini-2.5-flash was scheduled to shut down on 16 October 2026.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import cv2
import numpy as np

MODEL = os.environ.get("GALILEO_GEMINI_MODEL", "gemini-flash-latest")
VEO_MODEL = os.environ.get("GALILEO_VEO_MODEL", "veo-3.1-generate-preview")
CACHE_DIR = os.environ.get("GALILEO_CACHE", os.path.join(os.path.expanduser("~"), ".galileo_cache"))


def available() -> tuple[bool, str]:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return False, "no GEMINI_API_KEY set"
    try:
        import google.genai  # noqa: F401
    except Exception:
        return False, "google-genai not installed"
    return True, MODEL


def _client():
    from google import genai
    return genai.Client()


def _cache(key: str):
    p = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def _store(key: str, val):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, key + ".json"), "w") as fh:
        json.dump(val, fh)


def _jpeg(img: np.ndarray, max_side: int = 1024, q: int = 88) -> bytes:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


OBJECT_PROMPT = """The red box marks an object that falls during this video. The second image
is a close crop of it. Identify the object and give its real-world size so it
can serve as a ruler.

Return JSON only:
{"object": short name, "is_round": true/false,
 "size_m": for a ball or other round object its diameter in meters; for
           anything else its longest dimension in meters,
 "size_rel_uncertainty": spread of that size across real examples, as a fraction
           (a regulation basketball 0.02, a tennis ball 0.03, an apple 0.15,
           a smartphone 0.05, a book 0.15),
 "terminal_speed_m_s": rough terminal speed in still air in m/s
           (a ping-pong ball 8, a basketball 20, a smartphone 20, a golf ball 40),
 "confidence": 0-1 that the identification is right,
 "reason": one short sentence}"""


def identify_object(frame_bgr: np.ndarray, xyd, hint: str | None = None) -> dict | None:
    """Ask Gemini what the falling object is and how big such objects are.
    hint: what the person who filmed it says it is."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    x, y, d = (float(v) for v in xyd)
    h, w = frame_bgr.shape[:2]
    r = max(int(3 * d), 24)
    x0, x1 = int(max(x - r, 0)), int(min(x + r, w))
    y0, y1 = int(max(y - r, 0)), int(min(y + r, h))
    crop = frame_bgr[y0:y1, x0:x1]
    boxed = frame_bgr.copy()
    cv2.rectangle(boxed, (int(x - d), int(y - d)), (int(x + d), int(y + d)), (0, 0, 255), 3)
    a, b = _jpeg(boxed), _jpeg(crop, 512)
    prompt = OBJECT_PROMPT
    if hint:
        prompt += (f"\n\nThe person who filmed it says it is: {hint}. Give the size of that "
                   "object as it looks here.")
    key = "obj_" + hashlib.sha1(a + b + MODEL.encode() + prompt.encode()).hexdigest()[:20]
    hit = _cache(key)
    if hit:
        return hit
    from google.genai import types
    resp = _client().models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=a, mime_type="image/jpeg"),
                  types.Part.from_bytes(data=b, mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           temperature=0.1),
    )
    data = json.loads(resp.text)
    if isinstance(data, list):
        data = data[0]
    out = {"object": str(data.get("object", "object")),
           "is_round": bool(data.get("is_round", True)),
           "size_m": float(data["size_m"]) if data.get("size_m") else None,
           "size_rel_uncertainty": float(data.get("size_rel_uncertainty", 0.1)),
           "confidence": float(data.get("confidence", 0.5)),
           "terminal_speed_m_s": (float(data["terminal_speed_m_s"])
                                  if data.get("terminal_speed_m_s") else None),
           "reason": str(data.get("reason", "")), "model": MODEL}
    if not out["is_round"]:
        out["size_rel_uncertainty"] = max(out["size_rel_uncertainty"], 0.1)
    _store(key, out)
    return out


EXPLAIN_PROMPT = """You are writing the explanation line for Galileo, a tool that checks whether
a video is real by measuring physics. Using ONLY the numbers below, write two
short plain-English sentences a hackathon judge will understand: what Galileo
measured and why that supports the verdict. No hedging beyond the error bars
given, no new numbers, no em dashes.

{facts}"""


def explain(result: dict) -> str | None:
    ok, why = available()
    if not ok:
        return None
    v = result["verdict"]
    facts = {
        "verdict": v["verdict"], "measured_g_m_s2": _r(v.get("g")), "error_m_s2": _r(v.get("sigma")),
        "earth_g": v.get("g_ref"), "slow_motion_factor": _r(v.get("k")),
        "clock_source": (result.get("clock") or {}).get("source"),
        "object": ((result.get("scale") or {}).get("rulers") or [{}])[0].get("detail"),
        "tests": [f'{t["name"]}: {t["status"]}: {t["summary"]}' for t in result.get("tests", [])
                  if t["status"] != "skip"],
        "headline": v.get("headline"),
    }
    key = "exp_" + hashlib.sha1(json.dumps(facts, sort_keys=True).encode()
                                + MODEL.encode()).hexdigest()[:20]
    hit = _cache(key)
    if hit:
        return hit.get("text")
    from google.genai import types
    resp = _client().models.generate_content(
        model=MODEL, contents=EXPLAIN_PROMPT.format(facts=json.dumps(facts, indent=1)),
        config=types.GenerateContentConfig(temperature=0.3),
    )
    text = (resp.text or "").strip().replace("\u2014", ",")
    _store(key, {"text": text})
    return text


def _r(x):
    return None if x is None else round(float(x), 2)


TWIN_PROMPT = ("A fixed camera films this drop tower. The ball held by the electromagnet at "
               "the top is released and falls straight down onto the pad at the bottom. "
               "Realistic, real-time speed, no slow motion, no camera movement, no cuts.")


def generate_twin(image_path: str, out_path: str, prompt: str | None = None,
                  duration_s: int = 8, resolution: str = "720p", aspect: str = "16:9",
                  poll_s: float = 10.0, log=print) -> str:
    """Veo image-to-video: the AI copy of the tower for the side-by-side demo."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    from google.genai import types
    client = _client()
    img = types.Image.from_file(location=image_path)
    log(f"Asking {VEO_MODEL} for a {duration_s}s clip (takes a minute or two)...")
    op = client.models.generate_videos(
        model=VEO_MODEL, prompt=prompt or TWIN_PROMPT, image=img,
        config=types.GenerateVideosConfig(aspect_ratio=aspect, resolution=resolution,
                                          number_of_videos=1, duration_seconds=duration_s),
    )
    t0 = time.time()
    while not op.done:
        time.sleep(poll_s)
        op = client.operations.get(op)
        log(f"  ...{time.time() - t0:.0f}s")
    if getattr(op, "error", None):
        raise RuntimeError(f"Veo error: {op.error}")
    vid = op.response.generated_videos[0]
    try:
        client.files.download(file=vid.video, destination=out_path)
    except TypeError:
        client.files.download(file=vid.video)
        vid.video.save(out_path)
    if not os.path.exists(out_path):
        vid.video.save(out_path)
    with open(os.path.splitext(out_path)[0] + ".json", "w") as fh:
        json.dump({"generated_by": VEO_MODEL, "prompt": prompt or TWIN_PROMPT,
                   "source_image": os.path.basename(image_path), "ai_generated": True}, fh)
    log(f"Saved {out_path}")
    return out_path
