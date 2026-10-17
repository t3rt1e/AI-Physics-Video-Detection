"""Verdict engine: combine the three layers into REAL / AI / SLOWMO / UNSURE.

  g = s * a_px * k**2   (times an air-drag factor for light balls)

Order of evidence:
  1. Not enough clean frames, or a jittery track   -> UNSURE
  2. A scale-free test fails                        -> AI (slow motion can't explain it)
  3. The tracked thing never falls                  -> UNSURE (probably a hand, not the ball)
  4. Falls at a steady speed                        -> AI (a dropped object speeds up)
     Slows down as it falls                         -> UNSURE (a hand lowering or catching it)
  5. No ruler                                       -> UNSURE
  6. Error bar too wide to call either way          -> UNSURE
  7. g (after clocks) matches Earth                 -> REAL, or SLOWMO if k > 1
  8. g too low and no clock explains it             -> AI
  9. g too high                                     -> UNSURE (sped up, or object smaller than assumed)

By default (config.ASSUME_REAL_TIME) every clip is taken as playing at the
speed it was filmed, with no clocks: g inside Earth's interval is REAL, and
outside it, too slow or too fast, is AI.
"""
from __future__ import annotations

import numpy as np

from . import config as C

VERDICTS = ("REAL", "AI", "SLOWMO", "UNSURE")


def _fmt_g(g, sig):
    if sig >= 1:
        return f"{g:.1f} ± {sig:.1f} m/s²"
    if abs(g) < 1:
        return f"{g:.2f} ± {sig:.2f} m/s²"
    return f"{g:.1f} ± {sig:.1f} m/s²"


def decide(*, a_px, a_sigma, scale, clock, tests, n_points, duration, g_ref=C.G_EARTH,
           tracking_ok=True, rs_rel=0.0, drag=None, falls=True, light=None,
           geom_rel=0.0, real_time=C.ASSUME_REAL_TIME) -> dict:
    """falls: the tracked object moves down a good distance and is still moving
    down at the end of its flight (False for a hand pulling away).
    light: the object's name when air drag rules its fall (a feather, a
    balloon, a sheet of paper): then falling slowly or at a steady speed is
    what a real one does, not a sign of AI.
    geom_rel: relative error from the camera model (its focal length and
    angle), which matters when the camera looks down or up at the fall.
    real_time: the clip plays at the speed it was filmed (k = 1), so a g
    outside Earth's interval is AI whichever side it falls on. Otherwise
    clocks may explain a low g (slow motion), and a high g can't be called
    (the footage may be sped up)."""
    reasons: list[str] = []
    out = {"verdict": "UNSURE", "confidence": 0.0, "headline": "", "reasons": reasons,
           "g": None, "sigma": None, "g_naive": None, "g_ref": g_ref, "ratio": None,
           "k": 1.0, "tol": None, "stage": None, "camera_rel": float(geom_rel or 0.0)}

    fails = [t for t in tests if t["status"] == "fail"]
    warns = [t for t in tests if t["status"] == "warn"]

    if n_points < C.MIN_FLIGHT_POINTS or duration < C.MIN_FLIGHT_SECONDS or not tracking_ok:
        out["stage"] = "tracking: too few frames or jitter"
        out["headline"] = "Can't tell: the fall wasn't tracked cleanly enough to measure."
        reasons.append(f"Tracked {n_points} frames over {duration:.2f} s of falling; "
                       f"need {C.MIN_FLIGHT_POINTS}+ frames and {C.MIN_FLIGHT_SECONDS:.2f}+ s.")
        return out

    # Clock: confident clocks set k; conflicting or weak ones become hints.
    k = 1.0
    k_rel = 0.0
    if clock and clock.get("confidence", 0) >= C.CLOCK_MIN_CONFIDENCE:
        k = float(clock["k"])
        k_rel = 0.01 if clock.get("source") in ("file metadata", "frame timestamps") else 0.02
    weak_k = clock.get("weak_k") if clock else None
    if clock and clock.get("conflict"):
        weak_k = weak_k or clock.get("k")
    if real_time:
        k, weak_k = 1.0, None
    out["k"] = k

    if fails and scale is None:
        out.update(verdict="AI", confidence=0.75, stage=f"physics: {fails[0]['name']} failed",
                   headline="The motion breaks the laws of physics.")
        reasons.extend(t["summary"] for t in fails)
        reasons.append("No ruler for meters, so g itself wasn't measured.")
        return out

    wrong_thing = ("Galileo may have followed something other than the ball, like a hand or "
                   "a shadow. Point it at the ball (Pick the ball in the web app, or --seed on "
                   "the command line) and measure again.")
    if not fails and not falls:
        out["stage"] = "tracking: not a fall"
        out["headline"] = "Can't tell: the tracked object never falls."
        reasons.append("It moves up or sideways, or hardly moves at all.")
        reasons.append(wrong_thing)
        return out

    drag_rules = (f"Air drag rules a {light}'s fall: a real one drifts down slowly, so its fall "
                  "can't show gravity. Try something heavier and more compact.") if light else None
    if not fails and (a_px <= 0 or a_px < 2.0 * a_sigma):
        if a_px > -2.0 * a_sigma and light:
            out["stage"] = "physics: air drag rules"
            out["headline"] = "Can't tell: it floats down, as a real one would."
            reasons.append(drag_rules)
            return out
        if a_px > -2.0 * a_sigma:
            # Moves down a good distance at a steady speed: a dropped object can't.
            out.update(verdict="AI", confidence=0.6, stage="physics: no acceleration")
            out["headline"] = "It falls at a steady speed instead of speeding up."
            reasons.append("A dropped object speeds up as it falls; this one moves down without "
                           "accelerating, like an animation (or something floating, such as a "
                           "feather or a balloon).")
            reasons.append("If the tracked points aren't on the ball, point Galileo at it (Pick "
                           "the ball in the web app, or --seed on the command line) and measure "
                           "again.")
            return out
        out["stage"] = "tracking: slows while falling"
        out["headline"] = "Can't tell: the tracked object slows down as it falls."
        reasons.append("A falling ball speeds up; something that slows down on the way down is "
                       "usually a hand lowering it or catching it.")
        reasons.append(wrong_thing)
        return out

    if scale is None:
        out["stage"] = "ruler: missing"
        out["headline"] = "Can't tell: no ruler. Tell Galileo what the object is."
        reasons.append("The fall looks like a clean parabola, but without the object's "
                       "real size the acceleration stays in pixels.")
        return out

    s = scale["s"]
    g_naive = s * a_px
    drag_f = float(drag["factor"]) if drag else 1.0
    g = g_naive * k * k * drag_f
    fit_rel = abs(a_sigma / a_px) if a_px else 1.0
    rel = float(np.sqrt(scale["rel_sigma"] ** 2 + fit_rel**2 + (2 * k_rel) ** 2
                        + (0.5 * rs_rel) ** 2 + (0.3 * (drag_f - 1)) ** 2 + 0.01**2
                        + float(geom_rel or 0.0) ** 2))
    sigma = abs(g) * rel
    spread = C.Z_TOL * rel
    tol = max(C.TOL_REL, spread)
    ratio = g / g_ref
    out.update(g=g, sigma=sigma, g_naive=g_naive, ratio=ratio, tol=tol)

    gtxt = _fmt_g(g, sigma)
    pct = ratio * 100
    drag_line = None
    if drag and abs(drag_f - 1) > 0.005:
        drag_line = (f"Corrected for air drag: {100 * (drag_f - 1):+.0f}% (a light ball, top "
                     f"speed about {drag['vt']:.0f} m/s).")

    if fails:
        out.update(verdict="AI", confidence=0.85, stage=f"physics: {fails[0]['name']} failed")
        out["headline"] = f"Physics breaks: {fails[0]['summary'].split('. ')[0].rstrip('.')}."
        reasons.extend(t["summary"] for t in fails)
        reasons.append(f"The best constant-gravity fit gives g = {gtxt} ({pct:.0f}% of Earth's), "
                       "but the path isn't a free fall.")
        reasons.extend(t["summary"] for t in warns)
        return out

    far = False
    if spread > C.MAX_SPREAD:
        # Too uncertain to confirm gravity, but a far-off value can still be called.
        if ratio * (1 + spread) < 1 - C.TOL_REL or ratio * (1 - spread) > 1 + C.TOL_REL:
            far = True   # clearly off even at the near end of the error bar
            tol = C.TOL_REL
            out["tol"] = tol
        else:
            out["stage"] = "measurement: too uncertain"
            out["headline"] = f"Can't tell: g = {gtxt} is too uncertain to call."
            reasons.append(f"The error bar is ±{100 * rel:.0f}%. Usually the ball is too small "
                           "in the frame, or only a few frames of the fall were tracked.")
            return out

    # Earth's interval for this measurement: real inside it.
    lo, hi = g_ref * (1 - tol), g_ref * (1 + tol)
    out["real_range"] = [float(lo), float(hi)]
    range_line = (f"Counts as real between {lo:.1f} and {hi:.1f} m/s² (Earth's {g_ref:.2f} "
                  f"± {100 * tol:.0f}% for this measurement).")

    if abs(ratio - 1) <= tol and not far:
        out["stage"] = "ok"
        if k >= 1.5:
            out.update(verdict="SLOWMO", confidence=float(min(0.97, clock["confidence"])))
            out["headline"] = f"Real footage in {k:.0f}x slow motion. g = {gtxt}."
            reasons.append(f"As played, it falls at {g_naive * drag_f:.2f} m/s², because "
                           f"{k:.0f}x slow motion divides g by {k * k:.0f}.")
            reasons.append(f"{clock['source'].capitalize()} reveal the slow-down; correcting "
                           f"for it gives {gtxt}, matching Earth.")
        else:
            conf = float(np.clip(1.0 - abs(ratio - 1) / (2 * tol), 0.55, 0.97))
            out.update(verdict="REAL", confidence=conf)
            out["headline"] = f"Falls at {gtxt}: Earth's gravity."
            reasons.append(f"Measured g = {gtxt}, within {abs(ratio - 1) * 100:.0f}% of "
                           f"{g_ref:.2f}.")
            reasons.append(range_line)
            if clock and clock.get("slowmo_playback"):
                sp = clock["slowmo_playback"]
                reasons.append(f"Played back at 30 fps this clip would look {sp:.0f}x slow "
                               f"({g / sp**2:.2f} m/s²); its {sp * 30:.0f} fps timestamps give "
                               "real time.")
        if drag_line:
            reasons.append(drag_line)
        reasons.extend(t["summary"] for t in tests if t["status"] == "pass")
        reasons.extend(t["summary"] for t in warns)
        return out

    if ratio < 1 - tol and light:
        out["stage"] = "physics: air drag rules"
        out["headline"] = f"Can't tell: it falls at {gtxt}, slower than gravity, as a real one would."
        reasons.append(drag_rules)
        return out

    if ratio < 1 - tol:
        k_needed = float(np.sqrt(1 / ratio)) if ratio > 0 else float("inf")
        if weak_k and abs(g * weak_k**2 / g_ref - 1) <= tol:
            out.update(verdict="SLOWMO", confidence=0.55, k=weak_k, stage="ok",
                       g=g * weak_k**2, sigma=sigma * weak_k**2)
            out["headline"] = (f"Probably real {weak_k:.0f}x slow motion "
                               f"(only a weak clock agrees).")
            reasons.append(f"As played it falls at {gtxt}. A {weak_k:.0f}x slow-down, hinted "
                           f"by {clock.get('weak_source')}, would make it exactly Earth's g.")
            return out
        # Judged at the top of the error bar when the bar is wide.
        r_hi = ratio * (1 + spread) if far else ratio
        conf = float(np.clip(0.6 + (1 - tol - r_hi) * 1.5, 0.6, 0.98))
        out.update(verdict="AI", confidence=conf, stage="physics: g too low")
        out["headline"] = f"Falls at {pct:.0f}% of Earth's gravity ({gtxt}): too slow to be real."
        reasons.append(f"Measured g = {gtxt}; Earth's is {g_ref:.2f}.")
        reasons.append(range_line)
        if real_time:
            reasons.append("The clip is taken as playing at the speed it was filmed (no slow "
                           "motion).")
        elif k > 1:
            reasons.append(f"Even after undoing the {k:.0f}x slow motion found by "
                           f"{clock['source']}, it's too slow.")
        else:
            reasons.append("No sign of slow motion in the file: no capture-rate tag, no "
                           "repeated frames, no interpolation pattern.")
            reasons.append(f"Only a hidden {k_needed:.1f}x slow-down with every trace stripped "
                           "could make this real footage.")
        if drag_line:
            reasons.append(drag_line)
        reasons.extend(t["summary"] for t in tests if t["status"] == "pass")
        reasons.extend(t["summary"] for t in warns)
        return out

    # Too fast.
    if real_time:
        # Judged at the bottom of the error bar when the bar is wide.
        r_lo = ratio * (1 - spread) if far else ratio
        conf = float(np.clip(0.6 + (r_lo - 1 - tol) * 1.5, 0.6, 0.98))
        out.update(verdict="AI", confidence=conf, stage="physics: g too high")
        out["headline"] = f"Falls at {pct:.0f}% of Earth's gravity ({gtxt}): too fast to be real."
        reasons.append(f"Measured g = {gtxt}; Earth's is {g_ref:.2f}.")
        reasons.append(range_line)
        reasons.append("The clip is taken as playing at the speed it was filmed (not sped up). "
                       f"A real one would have to be {1 / ratio:.2f}x the size assumed for it.")
        if drag_line:
            reasons.append(drag_line)
        reasons.extend(t["summary"] for t in tests if t["status"] == "pass")
        reasons.extend(t["summary"] for t in warns)
        return out
    out.update(verdict="UNSURE", confidence=0.5, stage="physics: g too high")
    out["headline"] = f"Falls faster than gravity allows ({gtxt})."
    reasons.append("Either the footage is sped up, or the object is smaller than assumed "
                   f"(it would need to be {1 / ratio:.2f}x its assumed size).")
    if drag_line:
        reasons.append(drag_line)
    return out


FLIP_MIN_P = 0.2     # look-alikes this likely, together, that would flip the call: ask


def size_doubt(v: dict, label: str | None, size_m: float | None, alternatives,
               p_main: float | None = None) -> dict:
    """The ruler came from a guess at what the object is. When the other things
    it could be (a practice golf ball rather than a pickleball) would together
    have a fair chance (FLIP_MIN_P) of turning REAL into AI or back, the verdict
    hangs on that guess: say so and ask, rather than call it. g scales with the
    size assumed (g = s * a_px), so each one's g is this g times its size over
    the one used.

    alternatives: [{"label", "size_m", "p"}] from the identification; p_main:
    how likely the one used is."""
    g, sig = v.get("g"), v.get("sigma")
    if (v.get("verdict") not in ("REAL", "AI") or g is None or not sig or not size_m
            or not alternatives or v.get("stage") not in
            ("ok", "physics: g too low", "physics: g too high")):
        return v
    g_ref = v.get("g_ref") or C.G_EARTH
    rel = abs(sig / g) if g else 1.0
    spread = C.Z_TOL * rel
    far = spread > C.MAX_SPREAD
    tol = v.get("tol") or C.TOL_REL

    def call(r):
        if far:
            off = r * (1 + spread) < 1 - C.TOL_REL or r * (1 - spread) > 1 + C.TOL_REL
            return "AI" if off else "UNSURE"
        return "REAL" if abs(r - 1) <= tol else "AI"

    def says(gg, verdict):
        what = {"REAL": "Earth's gravity", "UNSURE": "too uncertain to call"}.get(
            verdict, "too slow to be real" if gg < g_ref else "too fast to be real")
        return f"{_fmt_g(gg, abs(gg) * rel)}, {what}"

    def odds(p):
        return f", {100 * p:.0f}% likely" if p is not None else ""

    flips = []
    for a in alternatives:
        g_a = g * float(a["size_m"]) / float(size_m)
        c = call(g_a / g_ref)
        if c != v["verdict"]:
            flips.append((a, g_a, c))
    p_flip = sum(float(a.get("p") or 0.0) for a, _, _ in flips)
    if not flips or p_flip < FLIP_MIN_P:
        return v
    name = label or "object"
    flips.sort(key=lambda q: -float(q[0].get("p") or 0.0))
    a = flips[0][0]
    out = {**v, "verdict": "UNSURE", "confidence": 0.0, "stage": "ruler: which object?",
           "size_flip": [{"label": a_["label"], "size_m": float(a_["size_m"]), "g": float(gg),
                          "p": a_.get("p"), "verdict": cc} for a_, gg, cc in flips]}
    out["headline"] = (f"Can't tell: it depends on what the object is ({name} or "
                       f"{a['label']}).")
    reasons = [f"As a {name} ({1000 * size_m:.0f} mm{odds(p_main)}) it falls at "
               f"{says(g, v['verdict'])}."]
    for a_, gg, cc in flips:
        reasons.append(f"As a {a_['label']} ({1000 * a_['size_m']:.0f} mm"
                       f"{odds(a_.get('p'))}) it would fall at {says(gg, cc)}.")
    reasons.append("The object's size is the ruler, and these look alike in the video. Tell "
                   "Galileo what it is (the Ruler menu in the web app, or --object on the "
                   "command line).")
    out["reasons"] = reasons
    return out


def _obj_name(o, cap=False) -> str:
    s = f"object {o['n']}" + (f" ({o['label']})" if o.get("label") else "")
    return s[0].upper() + s[1:] if cap else s


def _short(v) -> str:
    if v.get("g") is not None and v.get("sigma") is not None:
        return f"{v['verdict']}, g = {_fmt_g(v['g'], v['sigma'])}"
    return f"{v['verdict']}, {v.get('headline', '').rstrip('.')}"


def combine_objects(objs: list[dict], cross: list | None = None) -> dict:
    """One verdict for a clip with several falling objects.

    objs: per object, primary first: {"n", "label", "verdict", "named" (Galileo
    knows what it is), "strong" (you said what it is)}.

    The primary object (the one you named or clicked, else the best-tracked
    fall) decides. The others:
      * stand in when the primary can't be measured (no ruler, too
        uncertain) and one of them can, if Galileo knows what it is;
      * overturn a REAL call only when you named them yourself and they
        clearly disagree: a guessed size, or a ball that grazes a wall, can
        make one object read wrong in a real clip;
      * otherwise add their numbers to the reasons, and lower the confidence
        when they disagree.
    """
    prim = objs[0]
    v1 = prim["verdict"]
    if len(objs) == 1:
        return v1
    dec = ("REAL", "SLOWMO", "AI")

    def real(v):
        return v["verdict"] in ("REAL", "SLOWMO")

    out = {**v1, "reasons": list(v1.get("reasons", [])), "from_object": prim["n"]}
    lines = []
    if v1["verdict"] in dec:
        strong_clash = None
        agree = 0
        for o in objs[1:]:
            v = o["verdict"]
            if v["verdict"] not in dec:
                lines.append(f"{_obj_name(o, True)}: {v.get('headline', '').rstrip('.')}.")
                continue
            if real(v) == real(v1):
                agree += 1
                lines.append(f"{_obj_name(o, True)} agrees: {_short(v)}.")
                continue
            lines.append(f"{_obj_name(o, True)} disagrees: {_short(v)}.")
            if o.get("strong") and strong_clash is None:
                strong_clash = o
            else:
                out["confidence"] = float(out.get("confidence", 0)) * 0.8
        if strong_clash is not None and real(v1):
            o = strong_clash
            v = o["verdict"]
            out = {**v, "reasons": list(v.get("reasons", [])), "from_object": o["n"],
                   "confidence": float(min(v.get("confidence", 0.6), 0.75))}
            out["headline"] = f"{_obj_name(o, True)}: {v['headline']}"
            out["reasons"].insert(0, f"{_obj_name(prim, True)} falls like a real object "
                                  f"({_short(v1)}) but {_obj_name(o)} doesn't, and in one "
                                  "real scene both would.")
        elif strong_clash is not None:
            out["confidence"] = float(out.get("confidence", 0)) * 0.8
        elif agree:
            out["confidence"] = float(min(0.98, out.get("confidence", 0) + 0.03 * agree))
    else:
        cands = [o for o in objs[1:] if o["verdict"]["verdict"] in dec and o.get("named")]

        def rank(o):
            v = o["verdict"]
            scale_free = v["verdict"] == "AI" and str(v.get("stage", "")).endswith("failed")
            return (scale_free, bool(o.get("strong")), float(v.get("confidence", 0)))
        if cands:
            o = max(cands, key=rank)
            v = o["verdict"]
            out = {**v, "reasons": list(v.get("reasons", [])), "from_object": o["n"]}
            out["headline"] = f"{_obj_name(o, True)}: {v['headline']}"
            out["confidence"] = float(v.get("confidence", 0)) * (1.0 if o.get("strong") else 0.9)
            lines.append(f"{_obj_name(prim, True)} couldn't be measured: "
                         f"{v1.get('headline', '').rstrip('.')}.")
            for q in objs[1:]:
                if q is o:
                    continue
                lines.append(f"{_obj_name(q, True)}: {_short(q['verdict'])}.")
        else:
            for q in objs[1:]:
                lines.append(f"{_obj_name(q, True)}: {_short(q['verdict'])}.")
    out["reasons"].extend(lines)
    for t in cross or []:
        if t.get("name") == "Equal falls" and t.get("status") == "warn" \
                and t["summary"] not in out["reasons"]:
            out["reasons"].append(t["summary"])
    return out
