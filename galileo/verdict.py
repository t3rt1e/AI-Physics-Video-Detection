"""Verdict engine: combine the three layers into REAL / AI / SLOWMO / UNSURE.

  g = s * a_px * k**2

Order of evidence:
  1. Not enough clean frames                 -> UNSURE
  2. A scale-free test fails                 -> AI (slow motion can't explain it)
  3. g (after clocks) matches Earth          -> REAL, or SLOWMO if k > 1
  4. g too low and no clock explains it      -> AI
  5. g too high                              -> UNSURE (sped up, or object smaller than assumed)
"""
from __future__ import annotations

import numpy as np

from . import config as C

VERDICTS = ("REAL", "AI", "SLOWMO", "UNSURE")


def _fmt_g(g, sig):
    if sig >= 1:
        return f"{g:.1f} ± {sig:.1f} m/s²"
    if g < 1:
        return f"{g:.2f} ± {sig:.2f} m/s²"
    return f"{g:.1f} ± {sig:.1f} m/s²"


def decide(*, a_px, a_sigma, scale, clock, tests, n_points, duration, g_ref=C.G_EARTH,
           tracking_ok=True, rs_rel=0.0) -> dict:
    reasons: list[str] = []
    out = {"verdict": "UNSURE", "confidence": 0.0, "headline": "", "reasons": reasons,
           "g": None, "sigma": None, "g_naive": None, "g_ref": g_ref, "ratio": None,
           "k": 1.0, "tol": None}

    fails = [t for t in tests if t["status"] == "fail"]
    warns = [t for t in tests if t["status"] == "warn"]

    if n_points < C.MIN_FLIGHT_POINTS or duration < C.MIN_FLIGHT_SECONDS or not tracking_ok:
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

    if scale is None:
        if fails:
            out.update(verdict="AI", confidence=0.75,
                       headline="The motion breaks the laws of physics.")
            reasons.extend(t["summary"] for t in fails)
            reasons.append("No ruler for meters, so g itself wasn't measured.")
        else:
            out["headline"] = "Can't tell: no ruler. Tell Galileo what the object is."
            reasons.append("The fall looks like a clean parabola, but without the object's "
                           "real size the acceleration stays in pixels.")
        return out

    s = scale["s"]
    g_naive = s * a_px
    g = g_naive * k * k
    fit_rel = abs(a_sigma / a_px) if a_px else 1.0
    rel = float(np.sqrt(scale["rel_sigma"] ** 2 + fit_rel**2 + (2 * k_rel) ** 2
                        + (0.5 * rs_rel) ** 2 + 0.01**2))
    sigma = abs(g) * rel
    tol = max(C.TOL_REL, C.Z_TOL * rel)
    ratio = g / g_ref
    out.update(g=g, sigma=sigma, g_naive=g_naive, ratio=ratio, tol=tol)

    gtxt = _fmt_g(g, sigma)
    pct = ratio * 100

    if fails:
        out.update(verdict="AI", confidence=0.85)
        out["headline"] = f"Physics breaks: {fails[0]['summary'].split('.')[0]}."
        reasons.extend(t["summary"] for t in fails)
        reasons.append(f"The best constant-gravity fit gives g = {gtxt} ({pct:.0f}% of Earth's), "
                       "but the path isn't a free fall.")
        reasons.extend(t["summary"] for t in warns)
        return out

    if abs(ratio - 1) <= tol:
        if k >= 1.5:
            out.update(verdict="SLOWMO", confidence=float(min(0.97, clock["confidence"])))
            out["headline"] = f"Real footage in {k:.0f}x slow motion. g = {gtxt}."
            reasons.append(f"As played, it falls at {g_naive:.2f} m/s², because {k:.0f}x slow "
                           f"motion divides g by {k * k:.0f}.")
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
        reasons.extend(t["summary"] for t in tests if t["status"] == "pass")
        reasons.extend(t["summary"] for t in warns)
        return out

    if ratio < 1 - tol:
        k_needed = float(np.sqrt(1 / ratio))
        if weak_k and abs(g * weak_k**2 / g_ref - 1) <= tol:
            out.update(verdict="SLOWMO", confidence=0.55, k=weak_k,
                       g=g * weak_k**2, sigma=sigma * weak_k**2)
            out["headline"] = (f"Probably real {weak_k:.0f}x slow motion "
                               f"(only a weak clock agrees).")
            reasons.append(f"As played it falls at {gtxt}. A {weak_k:.0f}x slow-down, hinted "
                           f"by {clock.get('weak_source')}, would make it exactly Earth's g.")
            return out
        conf = float(np.clip(0.6 + (1 - tol - ratio) * 1.5, 0.6, 0.98))
        out.update(verdict="AI", confidence=conf)
        frac = f"{pct:.0f}%"
        out["headline"] = f"Falls at {frac} of Earth's gravity ({gtxt}): too slow to be real."
        reasons.append(f"Measured g = {gtxt}; Earth's is {g_ref:.2f}.")
        if k > 1:
            reasons.append(f"Even after undoing the {k:.0f}x slow motion found by "
                           f"{clock['source']}, it's too slow.")
        else:
            reasons.append("No sign of slow motion in the file: no capture-rate tag, no "
                           "repeated frames, no interpolation pattern.")
            reasons.append(f"Only a hidden {k_needed:.1f}x slow-down with every trace stripped "
                           "could make this real footage.")
        reasons.extend(t["summary"] for t in tests if t["status"] == "pass")
        reasons.extend(t["summary"] for t in warns)
        return out

    # Too fast.
    out.update(verdict="UNSURE", confidence=0.5)
    out["headline"] = f"Falls faster than gravity allows ({gtxt})."
    reasons.append("Either the footage is sped up, or the object is smaller than assumed "
                   f"(it would need to be {1 / ratio:.2f}x its assumed size).")
    return out
