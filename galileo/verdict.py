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
           tracking_ok=True, rs_rel=0.0, drag=None, falls=True) -> dict:
    """falls: the tracked object moves down a good distance and is still moving
    down at the end of its flight (False for a hand pulling away)."""
    reasons: list[str] = []
    out = {"verdict": "UNSURE", "confidence": 0.0, "headline": "", "reasons": reasons,
           "g": None, "sigma": None, "g_naive": None, "g_ref": g_ref, "ratio": None,
           "k": 1.0, "tol": None, "stage": None}

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

    if not fails and (a_px <= 0 or a_px < 2.0 * a_sigma):
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
                        + (0.5 * rs_rel) ** 2 + (0.3 * (drag_f - 1)) ** 2 + 0.01**2))
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
        out["headline"] = f"Physics breaks: {fails[0]['summary'].split('.')[0]}."
        reasons.extend(t["summary"] for t in fails)
        reasons.append(f"The best constant-gravity fit gives g = {gtxt} ({pct:.0f}% of Earth's), "
                       "but the path isn't a free fall.")
        reasons.extend(t["summary"] for t in warns)
        return out

    if spread > C.MAX_SPREAD:
        # Too uncertain to confirm gravity, but a far-off value can still be called.
        if ratio * (1 + spread) < 1 - C.TOL_REL:
            pass  # clearly too slow even at the top of the error bar: fall through to AI
        elif ratio * (1 - spread) > 1 + C.TOL_REL:
            pass  # clearly too fast even at the bottom: fall through
        else:
            out["stage"] = "measurement: too uncertain"
            out["headline"] = f"Can't tell: g = {gtxt} is too uncertain to call."
            reasons.append(f"The error bar is ±{100 * rel:.0f}%. Usually the ball is too small "
                           "in the frame, or only a few frames of the fall were tracked.")
            return out

    if abs(ratio - 1) <= tol:
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

    if ratio < 1 - tol:
        k_needed = float(np.sqrt(1 / ratio))
        if weak_k and abs(g * weak_k**2 / g_ref - 1) <= tol:
            out.update(verdict="SLOWMO", confidence=0.55, k=weak_k, stage="ok",
                       g=g * weak_k**2, sigma=sigma * weak_k**2)
            out["headline"] = (f"Probably real {weak_k:.0f}x slow motion "
                               f"(only a weak clock agrees).")
            reasons.append(f"As played it falls at {gtxt}. A {weak_k:.0f}x slow-down, hinted "
                           f"by {clock.get('weak_source')}, would make it exactly Earth's g.")
            return out
        conf = float(np.clip(0.6 + (1 - tol - ratio) * 1.5, 0.6, 0.98))
        out.update(verdict="AI", confidence=conf, stage="physics: g too low")
        out["headline"] = f"Falls at {pct:.0f}% of Earth's gravity ({gtxt}): too slow to be real."
        reasons.append(f"Measured g = {gtxt}; Earth's is {g_ref:.2f}.")
        if k > 1:
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
    out.update(verdict="UNSURE", confidence=0.5, stage="physics: g too high")
    out["headline"] = f"Falls faster than gravity allows ({gtxt})."
    reasons.append("Either the footage is sped up, or the object is smaller than assumed "
                   f"(it would need to be {1 / ratio:.2f}x its assumed size).")
    if drag_line:
        reasons.append(drag_line)
    return out
