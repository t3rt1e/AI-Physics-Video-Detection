"""Gemini and Veo: object ID for the ruler, plain-English reasons, the AI twin.

Needs `pip install google-genai` and an API key in GEMINI_API_KEY (or
GOOGLE_API_KEY). Every call is optional: Novex works offline without it.
Answers are cached on disk so a venue Wi-Fi drop can't break a demo clip
that already ran once.

Model names change often. Override with NOVEX_GEMINI_MODEL and
NOVEX_VEO_MODEL. The defaults were current in October 2026.

A free key gets about 20 requests a day per model, so each kind of request
walks its own list of models, moving on when one is busy (503), out of quota
(429) or gone (404): naming objects starts with the Flash models, the written
explanation with the Flash-Lite ones, and both end with the other's.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

import cv2
import numpy as np

MODEL = os.environ.get("NOVEX_GEMINI_MODEL", "gemini-flash-latest")
FLASH = ("gemini-flash-latest", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
         "gemini-3-flash-preview")
LITE = ("gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
        "gemma-4-31b-it")
FALLBACK_MODELS = FLASH + LITE          # for looking at pictures: the stronger ones first
TEXT_MODELS = LITE + FLASH              # for writing a sentence: save the Flash quota
VEO_MODEL = os.environ.get("NOVEX_VEO_MODEL", "veo-3.1-generate-preview")
_HOME = os.path.expanduser("~")
CACHE_DIR = os.environ.get("NOVEX_CACHE", os.path.join(_HOME, ".novex_cache"))
_CLIENT = None
_RESTING: dict = {}                     # model -> time it may be tried again


def available() -> tuple[bool, str]:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return False, "no GEMINI_API_KEY set"
    try:
        import google.genai  # noqa: F401
    except Exception:
        return False, "google-genai not installed"
    return True, MODEL


def _client():
    # One client for the whole run: a client made for a single call can be
    # closed (and its connection with it) before the call is sent.
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        _CLIENT = genai.Client()
    return _CLIENT


def _rest_for(err: Exception) -> float:
    """How long to leave a model alone after it failed (seconds)."""
    s = str(err)
    if "429" in s or "RESOURCE_EXHAUSTED" in s:
        if "PerMinute" in s:
            return 60.0                 # too many this minute
        return 1800.0                   # out of the day's quota (a free key's 20)
    if "404" in s or "NOT_FOUND" in s:
        return 86400.0                  # retired
    if "503" in s or "UNAVAILABLE" in s:
        return 120.0                    # busy
    return 0.0


def _generate(contents, *, json_out: bool = False, temperature: float = 0.1,
              models=None, start: int = 0):
    """One request, moving on to the next model when one is busy, over quota or
    gone (and leaving that one alone for a while). models: the list to walk
    (FALLBACK_MODELS by default, after NOVEX_GEMINI_MODEL); start: begin
    that far down it, so repeated questions get different models' opinions.
    Returns (response, model used)."""
    from google.genai import types
    chain = list(dict.fromkeys([MODEL, *FALLBACK_MODELS] if models is None else models))
    chain = chain[start % len(chain):] + chain[:start % len(chain)]
    now = time.time()
    ready = [m for m in chain if _RESTING.get(m, 0) <= now] or chain
    last = None
    for m in ready:
        cfg = {"temperature": temperature,
               "http_options": types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))}
        if json_out and not m.startswith("gemma"):     # Gemma has no JSON mode
            cfg["response_mime_type"] = "application/json"
        try:
            resp = _client().models.generate_content(
                model=m, contents=contents, config=types.GenerateContentConfig(**cfg))
            if json_out:
                _json(resp)                              # a reply that isn't JSON: next model
            return resp, m
        except Exception as e:  # 503 busy, 429 quota, 404 retired: try the next one
            last = e
            rest = _rest_for(e)
            if rest:
                _RESTING[m] = time.time() + rest
    raise RuntimeError(f"no Gemini model answered ({str(last)[:120]})")


def _json(resp):
    text = (resp.text or "").strip()
    try:
        data = json.loads(text)
    except ValueError:
        # Models without a JSON mode wrap it in prose or a code fence.
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if not m:
            raise
        data = json.loads(m.group(0))
    return data[0] if isinstance(data, list) and data else data


def cache_dir() -> str:
    """CACHE_DIR. The first time, a cache kept under the project's old name
    (~/.galileo_cache) is moved here, so answers already paid for are kept."""
    new, old = os.path.join(_HOME, ".novex_cache"), os.path.join(_HOME, ".galileo_cache")
    if CACHE_DIR == new and not os.path.exists(new) and os.path.isdir(old):
        try:
            os.rename(old, new)
        except OSError:
            pass
    return CACHE_DIR


def _cache(key: str):
    p = os.path.join(cache_dir(), key + ".json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def _store(key: str, val):
    d = cache_dir()
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, key + ".json"), "w") as fh:
        json.dump(val, fh)


def _jpeg(img: np.ndarray, max_side: int = 1024, q: int = 88) -> bytes:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


INTRO_VIEWS = """The first image is one frame of a video; the red box marks an object that
falls (or is thrown) in it. The other images are close-ups of that same object at
full resolution, taken at the moments it moved least; it may show a different side
in each as it turns. Look closely at the close-ups: its shape, materials, colours,
any printing or logos."""

INTRO_CROP = """The red box marks an object that falls during this video. The second image
is a close crop of it."""

LOOK = """Identify it as specifically as you can (the product, if it is a common one)
and give its real-world size so it can serve as a ruler.

The scene tells you what kind of place this is and what people there use, but not
how big the object is: things on a wall or farther back sit at a different distance
from the camera, so next to them it looks bigger than it is. Only something at the
same distance, like the hand that drops it, is a fair size comparison.

If it is a ball, look for what tells the kinds apart: holes (pickleball, wiffle
ball, plastic practice golf ball), felt with a curved seam (tennis ball), panels and
logos (volleyball, soccer ball, basketball), dimples (golf ball), a smooth shine
(ping-pong ball, rubber ball)."""

SIZE = ("for a ball or other round object its diameter in meters; for anything else "
        "its longest dimension in meters")
SPREAD = """spread of that size across real examples, as a fraction
   (a regulation basketball 0.02, a tennis ball 0.03, an apple 0.15, a smartphone 0.05,
   a book 0.15)"""
TERMINAL = """rough terminal speed in still air in m/s (a ping-pong ball 8, a basketball 20,
   a smartphone 20, a golf ball 40)"""

NAME_JSON = f"""Return JSON only:
{{"candidates": the 3 things it most likely is, most likely first, with probabilities
   that add up to 1 (spread them when the images can't tell them apart), each
   {{"object": short name, "is_round": true/false, "size_m": {SIZE},
    "probability": 0-1}},
 "dimensions_m": the first one's [longest, middle, shortest] in meters,
 "size_rel_uncertainty": the first one's {SPREAD},
 "terminal_speed_m_s": the first one's {TERMINAL},
 "reason": one short sentence on what in the images gave it away}}"""

SIZE_JSON = f"""Return JSON only:
{{"object": short name, "is_round": true/false, "size_m": {SIZE},
 "dimensions_m": [longest, middle, shortest] in meters,
 "size_rel_uncertainty": {SPREAD},
 "terminal_speed_m_s": {TERMINAL},
 "confidence": 0-1 that the identification is right,
 "reason": one short sentence on what in the images gave it away}}"""

# Asked this many times, by different models when it can: answers that disagree
# mean "not sure". Each one is a request against the day's quota.
NAME_SAMPLES = max(1, int(os.environ.get("NOVEX_GEMINI_SAMPLES", "2")))
NAME_TEMPERATURE = 0.5


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _answer(data: dict, used: str) -> dict:
    """One reply, as a list of candidates (a single named object is one sure one)."""
    cands = []
    for c in data.get("candidates") or []:
        if isinstance(c, dict) and c.get("object") and (_f(c.get("size_m")) or 0) > 0:
            cands.append({"object": str(c["object"]), "is_round": bool(c.get("is_round", True)),
                          "size_m": _f(c["size_m"]), "p": max(_f(c.get("probability"), 0.0), 0.0)})
    if not cands and data.get("object"):
        cands = [{"object": str(data["object"]), "is_round": bool(data.get("is_round", True)),
                  "size_m": _f(data.get("size_m")), "p": _f(data.get("confidence"), 0.5)}]
    tot = sum(c["p"] for c in cands)
    if len(cands) > 1 and tot > 0:
        for c in cands:
            c["p"] /= tot
    return {"candidates": cands[:3], "model": used,
            "size_rel_uncertainty": _f(data.get("size_rel_uncertainty"), 0.1),
            "terminal_speed_m_s": _f(data.get("terminal_speed_m_s")),
            "reason": str(data.get("reason", "")),
            "dimensions_m": [float(v) for v in (data.get("dimensions_m") or [])
                             if isinstance(v, (int, float))]}


def _norm(name: str) -> str:
    return " ".join(str(name).lower().replace("-", " ").split())


def pool(answers: list) -> dict | None:
    """Several replies about one object, pooled: each name's probability is
    its average over the replies, so a name only one reply gave counts half
    (with two replies). The likeliest name is the answer; the others are the
    alternatives, with their sizes."""
    answers = [a for a in answers if a and a["candidates"]]
    if not answers:
        return None
    names: dict = {}
    for a in answers:
        for c in a["candidates"]:
            e = names.setdefault(_norm(c["object"]), {"object": c["object"], "p": 0.0,
                                                      "sizes": [], "round": []})
            e["p"] += c["p"] / len(answers)
            if c["size_m"]:
                e["sizes"].append(c["size_m"])
            e["round"].append(c["is_round"])
    ranked = sorted(names.values(), key=lambda e: -e["p"])
    top = ranked[0]
    first = next((a for a in answers if _norm(a["candidates"][0]["object"]) == _norm(top["object"])),
                 answers[0])
    size = float(np.median(top["sizes"])) if top["sizes"] else None
    out = {"object": top["object"], "is_round": sum(top["round"]) * 2 >= len(top["round"]),
           "size_m": size, "size_rel_uncertainty": first["size_rel_uncertainty"],
           "confidence": round(top["p"], 3), "terminal_speed_m_s": first["terminal_speed_m_s"],
           "reason": first["reason"], "model": first["model"],
           "dimensions_m": first["dimensions_m"], "samples": len(answers),
           "alternatives": [{"object": e["object"], "size_m": float(np.median(e["sizes"])),
                             "probability": round(e["p"], 3)} for e in ranked[1:6] if e["sizes"]]}
    if not out["is_round"]:
        out["size_rel_uncertainty"] = max(out["size_rel_uncertainty"] or 0.1, 0.1)
    return out


def identify_object(frame_bgr: np.ndarray, xyd, hint: str | None = None,
                    crops: list | None = None) -> dict | None:
    """Ask Gemini what the falling object is and how big such objects are.
    hint: what the person who filmed it says it is (then Gemini only sizes it).
    crops: close-ups of it from the full-resolution video (several moments, so
    several sides); without them, a crop of frame_bgr around xyd.

    Without a hint Gemini is asked NAME_SAMPLES times for its three likeliest
    answers, and the replies are pooled (see pool): look-alikes of different
    sizes come back as "alternatives", which the verdict checks against."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    x, y, d = (float(v) for v in xyd)
    h, w = frame_bgr.shape[:2]
    boxed = frame_bgr.copy()
    m = 0.7 * d + 4                      # snug: a box much bigger than it reads as its size
    cv2.rectangle(boxed, (int(x - m), int(y - m)), (int(x + m), int(y + m)), (0, 0, 255),
                  max(2, int(round(max(h, w) / 500))))
    if crops:
        imgs = [_jpeg(boxed)] + [_jpeg(c, 768) for c in crops[:4]]
        intro = INTRO_VIEWS
    else:
        r = max(int(3 * d), 24)
        x0, x1 = int(max(x - r, 0)), int(min(x + r, w))
        y0, y1 = int(max(y - r, 0)), int(min(y + r, h))
        imgs = [_jpeg(boxed), _jpeg(frame_bgr[y0:y1, x0:x1], 512)]
        intro = INTRO_CROP
    if hint:
        prompt = (f"{intro} {LOOK}\n\nThe person who filmed it says it is: {hint}. Give the "
                  f"size of that object as it looks here.\n\n{SIZE_JSON}")
        n, temp = 1, 0.1
    else:
        prompt, n, temp = f"{intro} {LOOK}\n\n{NAME_JSON}", NAME_SAMPLES, NAME_TEMPERATURE
    key = "obj_" + hashlib.sha1(b"".join(imgs) + prompt.encode()).hexdigest()[:20]
    from google.genai import types
    parts = [types.Part.from_bytes(data=a, mime_type="image/jpeg") for a in imgs] + [prompt]
    answers = []
    for i in range(n):
        k = key if i == 0 else f"{key}_{i}"
        hit = _cache(k)
        if hit is None:
            resp, used = _generate(parts, json_out=True, temperature=temp, start=i)
            hit = _answer(_json(resp), used)
            _store(k, hit)
        answers.append(hit)
    out = pool(answers)
    if out and hint:
        out["alternatives"] = []
    return out


EXPLAIN_PROMPT = """You are writing the explanation line for NØVEX, a tool that checks whether
a video is real by measuring physics. Using ONLY the numbers below, write two
short plain-English sentences a hackathon judge will understand: what NØVEX
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
    key = "exp_" + hashlib.sha1((EXPLAIN_PROMPT + json.dumps(facts, sort_keys=True)).encode()).hexdigest()[:20]
    hit = _cache(key)
    if hit:
        return hit.get("text")
    resp, _ = _generate(EXPLAIN_PROMPT.format(facts=json.dumps(facts, indent=1)),
                        temperature=0.3, models=TEXT_MODELS)
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
