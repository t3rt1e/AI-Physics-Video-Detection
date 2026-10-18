"""Layer 1: tests that need neither a ruler nor a clock.

Slowing down a parabola gives another parabola, and scaling pixels to
meters doesn't change a ratio, so none of these can be fooled by slow
motion or a wrong object size. Each check returns a plain dict:

  {"name", "status": pass|warn|fail|skip, "summary", "detail"}
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .physics import JointFit, robust_fit_2d


def _result(name, status, summary, **detail):
    return {"name": name, "status": status, "summary": summary, "detail": detail}


def parabola_check(t, x, y, w0, jf: JointFit, drag_corrected: bool = False) -> dict:
    """Constant acceleration within one fall: compare first and second halves.

    Only a clear, significant change fails. When the halves are too noisy to
    compare, the check says so instead of passing. drag_corrected says the
    positions already had air drag's push taken out (light balls), which only
    changes the wording.
    """
    name = "Constant acceleration"
    if not jf.flights:
        return _result(name, "skip", "No clean fall to test.")
    fl = jf.flights[jf.main]
    idx = fl.idx
    n = len(idx)
    if n < 12:
        return _result(name, "skip", f"Only {n} frames in the fall; need 12 to split it in half.")
    h = n // 2
    parts = [idx[: h + 1], idx[h:]]
    r = np.asarray(jf.rcoord if jf.rcoord is not None else (y if jf.rs_axis == "y" else x), float)
    accs = []
    for p in parts:
        f = robust_fit_2d(t[p], x[p], y[p], w0[p], alpha=jf.alpha, r=r[p], rc=fl.fit.rc)
        accs.append(f.accel_along(jf.gdir))
    (a1, s1), (a2, s2) = accs
    detail = dict(a_first_px=a1, a_second_px=a2, sigma_first=s1, sigma_second=s2,
                  drag_corrected=bool(drag_corrected))
    # A half that clearly runs the wrong way (decelerating fall, rising "drop").
    if (a1 + 2.5 * s1 < 0 and a2 - 2.5 * s2 > 0) or (a2 + 2.5 * s2 < 0 and a1 - 2.5 * s1 > 0):
        return _result(name, "fail",
                       "The object stops accelerating downward partway through the fall.",
                       **detail)
    a_ref = max(abs(a1), abs(a2), 1e-9)
    if a1 <= 0 or a2 <= 0 or s1 / a_ref > 0.35 or s2 / a_ref > 0.35:
        return _result(name, "skip", "Too noisy to compare the two halves of the fall.",
                       **detail)
    ratio = a2 / a1
    z = (a2 - a1) / max(np.hypot(s1, s2), 1e-9)
    detail.update(ratio=ratio, z=float(z))
    lo, hi = C.PARABOLA_RATIO_FAIL
    wlo, whi = C.PARABOLA_RATIO_WARN
    pct = (ratio - 1.0) * 100.0
    if (ratio < lo or ratio > hi) and abs(z) > 4:
        return _result(name, "fail",
                       f"Acceleration changes by {pct:+.0f}% between the first and second half "
                       "of the fall. Real gravity is constant.", **detail)
    if (ratio < wlo or ratio > whi) and abs(z) > 3:
        why = ("lens distortion, or a sign of generated motion" if drag_corrected else
               "air drag, lens distortion, or a sign of generated motion")
        return _result(name, "warn",
                       f"Acceleration drifts {pct:+.0f}% during the fall ({why}).", **detail)
    extra = " after taking out air drag" if drag_corrected else ""
    return _result(name, "pass",
                   f"Acceleration holds steady through the fall ({pct:+.0f}% half to half"
                   f"{extra}).", **detail)


def _u_poly(fit, g):
    """Coefficients (increasing powers of tau) of the position along gravity."""
    return fit.cx * g[0] + fit.cy * g[1]


def _v_sigma(fit, g, tau) -> float:
    """Standard error of the speed along gravity at tau, from the fit."""
    J = np.array([0.0, 1.0, 2.0 * tau])
    var = g[0] ** 2 * float(J @ fit.cov_x @ J) + g[1] ** 2 * float(J @ fit.cov_y @ J)
    return float(np.sqrt(max(var, 0.0)))


def bounce_check(t, jf: JointFit, frame_dt: float) -> dict:
    """Bounce law: same gravity every hop, and hops never gain energy.

    Flights left out of the measurement (a thrown first flight) still count
    for energy, but not for "same gravity": their acceleration is the
    throw's, not gravity's.
    """
    name = "Bounce law"
    fl = jf.flights
    if len(fl) < 2:
        return _result(name, "skip", "No bounces to test.")
    g = jf.gdir
    excluded = set(getattr(jf, "excluded", ()) or ())
    pairs = []
    for j, (a, b) in enumerate(zip(fl, fl[1:])):
        gap = t[b.idx[0]] - t[a.idx[-1]]
        if gap > 6 * frame_dt + 1e-9:
            continue  # not neighbours across a single bounce
        pa, pb = _u_poly(a.fit, g), _u_poly(b.fit, g)
        # u_a(t) - u_b(t) = 0 in absolute time.
        ta0, tb0 = a.fit.t0, b.fit.t0
        tm = 0.5 * (t[a.idx[-1]] + t[b.idx[0]])
        # Solve numerically on a fine grid around the gap (robust to edge cases).
        grid = np.linspace(t[a.idx[-1]] - 2 * frame_dt, t[b.idx[0]] + 2 * frame_dt, 401)
        da = np.polyval(pa[::-1], grid - ta0) - np.polyval(pb[::-1], grid - tb0)
        sgn = np.flatnonzero(np.diff(np.sign(da)))
        tb = grid[sgn[np.argmin(np.abs(grid[sgn] - tm))]] if len(sgn) else tm
        va = np.polyval(np.polyder(pa[::-1]), tb - ta0)
        vb = np.polyval(np.polyder(pb[::-1]), tb - tb0)
        if not (va > 0 and vb < 0):
            continue  # not a bounce (coming in down, going out up): a gap or a glitch
        # A bounce comes in at the hop's top speed; a split near the top of a
        # hop (a tracking hiccup) comes in at almost none.
        tt_a = t[a.idx] - ta0
        v_max = float(np.max(np.abs(np.polyval(np.polyder(pa[::-1]), tt_a)))) if len(tt_a) else 0.0
        if va < 0.25 * v_max:
            continue
        e = -vb / va
        # How well the two fits pin the speeds down at the contact: a short
        # low hop, or a camera that moves, measures them loosely.
        e_sigma = abs(e) * float(np.hypot(_v_sigma(a.fit, g, tb - ta0) / va,
                                          _v_sigma(b.fit, g, tb - tb0) / vb))
        # Same g on both hops: only between hops measured well enough to say
        # (a stub of a hop, cut off by the end of the clip, can't).
        a_big = max(abs(a.a_along), abs(b.a_along))
        precise = all(np.isfinite(h.a_sigma) and h.a_sigma < 0.1 * a_big
                      and t[h.idx[-1]] - t[h.idx[0]] >= 0.15 for h in (a, b))
        compare = j not in excluded and (j + 1) not in excluded and a.a_along != 0 and precise
        pairs.append({"j": j, "t_bounce": float(tb), "v_in": float(va), "v_out": float(vb),
                      "restitution": float(e), "restitution_sigma": e_sigma,
                      "a_ratio": float(b.a_along / a.a_along) if compare else float("nan"),
                      "a_ratio_z": float((b.a_along - a.a_along) /
                                         max(np.hypot(a.a_sigma, b.a_sigma), 1e-9))
                      if compare else float("nan"),
                      "n_after": int(len(b.idx))})
    if not pairs:
        return _result(name, "skip", "Bounces found but the hops around them weren't tracked.")
    # Between two bounces a hop lasts 2v/g, so the time of the hop after a
    # bounce over the time of the hop before it is that bounce's restitution,
    # with no scale, distance or camera motion in it: preferred whenever both
    # hops are whole. Each bounce time is good to about half a frame.
    by_j = {p["j"]: p for p in pairs}
    s_t = 0.5 * frame_dt * np.sqrt(2.0)
    for p in pairs:
        prev, nxt = by_j.get(p["j"] - 1), by_j.get(p["j"] + 1)
        if prev is None or nxt is None:
            continue
        t_in, t_out = p["t_bounce"] - prev["t_bounce"], nxt["t_bounce"] - p["t_bounce"]
        if t_in > 0 and t_out > 0:
            e_t = t_out / t_in
            p["restitution_t"] = float(e_t)
            p["restitution_t_sigma"] = float(e_t * np.hypot(s_t / t_in, s_t / t_out))

    def _e(p):
        if "restitution_t" in p:
            return p["restitution_t"], p["restitution_t_sigma"]
        return p["restitution"], p["restitution_sigma"]
    es = [_e(p)[0] for p in pairs if np.isfinite(_e(p)[0])]
    detail = {"bounces": pairs}
    # Gaining energy, by more than the measurement allows.
    gain = [e for e, s_e in map(_e, pairs) if np.isfinite(e)
            and e > C.RESTITUTION_FAIL and e - 1.0 > 3.0 * s_e]
    if gain:
        return _result(name, "fail",
                       f"The ball leaves the floor faster than it arrived (restitution "
                       f"{max(gain):.2f} > 1). Real bounces lose energy; this looks reversed "
                       "or generated.", **detail)
    lo, hi = C.PARABOLA_RATIO_FAIL
    bad = [p for p in pairs if p["n_after"] >= 8 and np.isfinite(p["a_ratio"])
           and (p["a_ratio"] < lo or p["a_ratio"] > hi) and abs(p["a_ratio_z"]) > 4]
    if bad:
        r = bad[0]["a_ratio"]
        return _result(name, "fail",
                       f"Gravity changes between hops (next hop accelerates {r:.2f}x as hard). "
                       "Every hop should share one g.", **detail)
    if es:
        # Quote the bounces measured well; a loose one (a stub of a hop) can
        # read over 1 within its error.
        shown = [e for e, s_e in map(_e, pairs) if np.isfinite(e) and s_e <= 0.05] or es
        return _result(name, "pass",
                       f"{len(pairs)} bounce(s) lose energy normally (restitution "
                       f"{min(shown):.2f}-{max(shown):.2f}) and hops share one g.", **detail)
    return _result(name, "skip", "Couldn't measure the bounce speeds.", **detail)


def equal_falls_check(a_main: tuple[float, float], others: list[tuple[float, float]]) -> dict:
    """Two objects falling side by side must fall together (same distance from camera)."""
    name = "Equal falls"
    if not others:
        return _result(name, "skip", "Only one falling object.")
    a0, s0 = a_main
    worst = None
    for a1, s1 in others:
        if a0 <= 0 or a1 <= 0:
            continue
        r = a1 / a0
        z = (a1 - a0) / max(np.hypot(s0, s1), 1e-9)
        if worst is None or abs(np.log(r)) > abs(np.log(worst[0])):
            worst = (r, z)
    if worst is None:
        return _result(name, "skip", "Second object not usable.")
    r, z = worst
    detail = {"ratio": float(r), "z": float(z)}
    if (r < 0.7 or r > 1.4) and abs(z) > 4:
        return _result(name, "warn",
                       f"Two falling objects accelerate differently ({r:.2f}x). Fine if one is "
                       "much closer to the camera; otherwise a red flag.", **detail)
    return _result(name, "pass", f"Falling objects keep pace ({r:.2f}x).", **detail)


def equal_falls_check_g(objs: list[dict]) -> dict:
    """Everything in one scene falls with one g. Compares each object's g (in
    m/s^2, each through its own ruler and air-drag correction, so distance
    from the camera doesn't matter) with the first one's.

    objs: [{"n", "label", "g", "sigma"}]; objects without a ruler are left out.
    Never fails on its own: a wrong guess at an object's size looks the same.
    """
    name = "Equal falls"
    with_g = [o for o in objs if o.get("g") is not None and o.get("sigma")]
    if len(with_g) < 2:
        return _result(name, "skip", "Fewer than two objects with a known size, so their falls "
                       "can't be compared.")
    ref = with_g[0]
    worst = None
    for o in with_g[1:]:
        r = o["g"] / ref["g"] if ref["g"] else float("inf")
        z = (o["g"] - ref["g"]) / max(np.hypot(o["sigma"], ref["sigma"]), 1e-9)
        if worst is None or abs(np.log(max(r, 1e-9))) > abs(np.log(max(worst[1], 1e-9))):
            worst = (o, r, z)
    o, r, z = worst

    def nm(q):
        return f"object {q['n']}" + (f" ({q['label']})" if q.get("label") else "")
    detail = {"ratio": float(r), "z": float(z), "objects": [q["n"] for q in with_g]}
    if (r < 0.8 or r > 1.25) and abs(z) > 3:
        return _result(name, "warn",
                       f"{nm(o).capitalize()} falls at {o['g']:.1f} m/s^2 but {nm(ref)} at "
                       f"{ref['g']:.1f}. In one scene they should match, unless a size is "
                       "wrong.".replace("m/s^2", "m/s²"), **detail)
    spread = max(abs(q["g"] / ref["g"] - 1) for q in with_g[1:]) * 100
    return _result(name, "pass", f"All {len(with_g)} objects fall with the same g (within "
                   f"{spread:.0f}%).", **detail)


def vertical_check(ang: float, s_ang: float) -> dict:
    """A real fall speeds up straight down: along the room's own uprights.

    ang: degrees between the way the fall accelerates (in the picture, at
    its centre) and the way the room's vertical lines run there; s_ang its
    error. A generated clip can drop things at a slant; a real one can't.
    Small angles are within what a slightly bent line or the fit allows.
    """
    name = "Falls straight down"
    detail = {"angle_deg": float(ang), "sigma_deg": float(s_ang)}
    if abs(ang) > 25 and abs(ang) > 5 * s_ang:
        return _result(name, "fail",
                       f"It speeds up {abs(ang):.0f}° away from the room's vertical. Real gravity "
                       "pulls straight down.", **detail)
    if abs(ang) > 8 and abs(ang) > 3 * s_ang:
        return _result(name, "warn",
                       f"It speeds up {abs(ang):.0f}° away from the room's vertical (a bent lens, "
                       "or a sign of generated motion).", **detail)
    return _result(name, "pass", f"It speeds up along the room's vertical (within "
                   f"{max(abs(ang), s_ang):.0f}°).", **detail)


def release_check(jf: JointFit, rest_xy, diam_px, thrown: bool = False) -> dict:
    """A dropped object starts from rest at the spot where it was held."""
    name = "Starts from rest"
    if thrown:
        return _result(name, "skip", "Already moving fast when first seen (thrown), so it "
                       "doesn't start from rest.")
    if rest_xy is None or not jf.flights:
        return _result(name, "skip", "Didn't see the object held still before the drop.")
    fl = jf.flights[0]
    f = fl.fit
    g = jf.gdir
    pu = _u_poly(f, g)
    if pu[2] <= 0:
        return _result(name, "skip", "Fall direction unclear.")
    tau_v = -pu[1] / (2 * pu[2])
    xv, yv = f.pos(tau_v)
    dist = float(np.hypot(xv - rest_xy[0], yv - rest_xy[1]))
    ratio = dist / max(diam_px, 1e-6)
    detail = {"vertex_px": [float(xv), float(yv)], "rest_px": [float(rest_xy[0]), float(rest_xy[1])],
              "offset_diameters": ratio}
    if ratio > 10:
        return _result(name, "warn",
                       "The motion doesn't trace back to the spot where the object was held.",
                       **detail)
    if ratio > 1.5:
        return _result(name, "warn",
                       f"The fall's starting point sits {ratio:.1f} ball-widths from where the "
                       "object was held.", **detail)
    return _result(name, "pass", "The fall starts from rest where the object was held.", **detail)


# ---------------------------------------------------------------------------
# More laws: spin, shadows, the floor, sideways pushes, bounciness, size.
# All still need neither a ruler nor a clock: they compare the ball with its
# own width, its own speed or its own earlier self. They are marked
# "experimental" until they have seen enough real clips: config.
# EXPERIMENTAL_CHECKS_DECIDE says whether their failures count in the verdict.

def _exp(r: dict) -> dict:
    r["experimental"] = True
    return r


def _turns(w: float) -> str:
    """rad/s as turns a second, with the direction seen in the picture."""
    rev = w / (2 * np.pi)
    way = "clockwise" if rev > 0 else "anticlockwise"
    return f"{abs(rev):.1f} turns a second {way}" if abs(rev) >= 0.05 else "no spin"


def _side_vel(fit, e, tau) -> tuple[float, float]:
    """Speed along direction e at tau from a flight's fit, and its standard error."""
    vx, vy = fit.vel(np.array([tau]))
    J = np.array([0.0, 1.0, 2.0 * tau])
    var = e[0] ** 2 * float(J @ fit.cov_x @ J) + e[1] ** 2 * float(J @ fit.cov_y @ J)
    return float(vx[0] * e[0] + vy[0] * e[1]), float(np.sqrt(max(var, 0.0)))


def clean_flights(jf: JointFit) -> list[int]:
    """Flights that are clean free falls: their acceleration agrees with the
    main flight's (to 25%, as for the measurement) and their fit is about as
    tight as the others'. A hop ended by a catch, or tracked badly near the
    picture's edge, is left out of the per-hop laws."""
    if not jf.flights:
        return []
    m = jf.flights[jf.main]
    rms = np.median([f.fit.rms for f in jf.flights])
    out = []
    for j, f in enumerate(jf.flights):
        if not (m.a_along and f.a_along > 0 and abs(f.a_along / m.a_along - 1.0) <= 0.25):
            continue
        if f.fit.rms > max(3.0 * rms, 1.5):
            continue
        out.append(j)
    return out


def spin_steady_check(sp: dict | None, flights: list) -> dict:
    """Angular momentum: nothing twists a ball in the air, so it keeps its spin
    from one end of a flight to the other (air slows a spin over seconds, not
    a fraction of one). sp: spin.series output; flights: index arrays."""
    from . import spin as spin_
    name = "Steady spin"
    ok, why = spin_.readable(sp, flights)
    if not ok:
        return _exp(_result(name, "skip", why))
    rows, worst = [], None
    for k, idx in enumerate(flights):
        idx = np.asarray(idx, int)
        if len(idx) < 16:
            continue
        h = len(idx) // 2
        a, b = spin_.rate(sp, idx[:h + 1]), spin_.rate(sp, idx[h:])
        whole = spin_.rate(sp, idx)
        if a is None or b is None or whole is None:
            continue
        (w1, s1, n1), (w2, s2, n2) = a, b
        z = (w2 - w1) / max(np.hypot(s1, s2), 1e-9)
        big = max(abs(w1), abs(w2))
        rows.append({"flight": k, "omega": whole[0], "omega_sigma": whole[1], "first": w1,
                     "second": w2, "z": float(z), "pairs": whole[2]})
        change = abs(w2 - w1)
        if worst is None or change > worst[0]:
            worst = (change, z, big, k, w1, w2)
    if not rows:
        return _exp(_result(name, "skip", "Too few clear views of the ball in any one flight to "
                            "read its spin.", readable=float(sp["readable"])))
    detail = {"flights": rows, "readable": float(sp["readable"])}
    change, z, big, k, w1, w2 = worst
    if change > max(0.5 * big, 2 * np.pi * 1.0) and abs(z) > 5:
        return _exp(_result(name, "fail",
                            f"Its spin changes in midair, from {_turns(w1)} to {_turns(w2)}. Nothing "
                            "in the air can twist a ball like that.", **detail))
    if change > max(0.25 * big, 2 * np.pi * 0.5) and abs(z) > 3:
        return _exp(_result(name, "warn",
                            f"Its spin drifts in midair ({_turns(w1)} to {_turns(w2)}).", **detail))
    w = rows[int(np.argmax([r["pairs"] for r in rows]))]["omega"]
    return _exp(_result(name, "pass", f"It keeps its spin in the air ({_turns(w)}).", **detail))


def bounce_spin_check(sp: dict | None, jf: JointFit, bounces: list, r_px: float,
                      k_inertia: float | None) -> dict:
    """Angular momentum about the contact point. At a bounce every push on the
    ball (the floor's support and its friction) acts where it touches, so none
    of them can turn the ball about that point: its angular momentum about the
    contact point, m r (v + k r w) with v its sideways speed and w its spin
    (clockwise, the way it would roll to the right; k = I / (m r^2), 2/3 for a
    hollow ball, 2/5 for a solid one), is kept by a hard ball that grips, and
    a squashy one loses some (squashing bites into its spin and its sideways
    speed together). It can't grow: a spin kicked up out of nothing, or a
    sideways shove with no spin to pay for it, breaks it.

    bounces: bounce_check's pairs (j, t_bounce, v_in); r_px: the ball's radius
    in the fit's pixels. Picture x to the right."""
    from . import spin as spin_
    name = "Spin at bounces"
    if not bounces:
        return _exp(_result(name, "skip", "No bounces to test."))
    ok, why = spin_.readable(sp, [f.idx for f in jf.flights])
    if not ok:
        return _exp(_result(name, "skip", why))
    g = jf.gdir
    e = np.array([g[1], -g[0]])                 # the picture's sideways, with gravity down
    k = k_inertia or 0.5
    clean = set(clean_flights(jf))
    rows = []
    for p in bounces:
        j = p["j"]
        if j not in clean or j + 1 not in clean:
            continue
        a, b = jf.flights[j], jf.flights[j + 1]
        ra, rb = spin_.rate(sp, a.idx), spin_.rate(sp, b.idx)
        if ra is None or rb is None:
            continue
        tb = p["t_bounce"]
        va, sa = _side_vel(a.fit, e, tb - a.fit.t0)
        vb, sb = _side_vel(b.fit, e, tb - b.fit.t0)
        La, Lb = va + k * r_px * ra[0], vb + k * r_px * rb[0]       # px/s
        s_L = float(np.hypot(np.hypot(sa, k * r_px * ra[1]), np.hypot(sb, k * r_px * rb[1])))
        if k_inertia is None:                   # not knowing the build loosens it
            s_L = float(np.hypot(s_L, 0.2 * k * r_px * max(abs(ra[0]), abs(rb[0]))))
        v_in = abs(p.get("v_in") or 0.0)
        gain = abs(Lb) - abs(La)
        rows.append({"j": j, "t": tb, "v_before": float(va), "v_after": float(vb),
                     "spin_before": ra[0], "spin_after": rb[0], "L_before": float(La),
                     "L_after": float(Lb), "L_sigma": s_L, "v_in": v_in,
                     "gain": float(gain / max(v_in, 1e-9))})
    if not rows:
        return _exp(_result(name, "skip", "The spin couldn't be read on both sides of a clean "
                            "bounce."))
    detail = {"bounces": rows, "k": k_inertia}

    def say(r):
        return (f"its spin goes from {_turns(r['spin_before'])} to {_turns(r['spin_after'])} "
                f"and its sideways speed from {r['v_before']:.0f} to {r['v_after']:.0f} px/s")
    bad = [r for r in rows if r["gain"] * r["v_in"] > 4 * r["L_sigma"] and r["gain"] > 0.10]
    odd = [r for r in rows if r["gain"] * r["v_in"] > 3 * r["L_sigma"] and r["gain"] > 0.05]
    if bad:
        r = max(bad, key=lambda q_: q_["gain"])
        return _exp(_result(name, "fail",
                            f"At a bounce {say(r)}: it comes off turning faster about the spot it "
                            "touched. The floor can only take spin and sideways speed away "
                            "(angular momentum about the contact point can't grow).", **detail))
    if odd:
        r = max(odd, key=lambda q_: q_["gain"])
        return _exp(_result(name, "warn", f"At a bounce {say(r)}, a little more turning than "
                            "the floor can give.", **detail))
    lost = [r for r in rows if r["gain"] < -0.03]
    if lost:
        return _exp(_result(name, "pass", f"At {len(rows)} bounce(s) the floor only takes spin "
                            "and sideways speed away, as friction and squash do.", **detail))
    return _exp(_result(name, "pass", f"At {len(rows)} bounce(s) spin and sideways speed carry "
                        "through as they should.", **detail))


def rolling_check(sp: dict | None, t, x, d, segments: list) -> dict:
    """Rolling without slipping: a ball rolling along the floor turns once for
    every circumference it travels (spin x radius = speed). segments: index
    arrays of stretches where it rolls on the floor."""
    name = "Rolling"
    if not segments:
        return _exp(_result(name, "skip", "It doesn't roll along the floor in the clip."))
    from . import spin as spin_
    ok, why = spin_.readable(sp, segments)
    if not ok:
        return _exp(_result(name, "skip", why))
    rows = []
    for idx in segments:
        idx = np.asarray(idx, int)
        w = spin_.rate(sp, idx)
        if w is None:
            continue
        tt, xx = np.asarray(t, float)[idx], np.asarray(x, float)[idx]
        A = np.vstack([np.ones(len(tt)), tt - tt.mean()]).T
        coef, res, *_ = np.linalg.lstsq(A, xx, rcond=None)
        v = float(coef[1])
        s_v = float(np.sqrt(max(np.sum((xx - A @ coef) ** 2) / max(len(tt) - 2, 1), 1e-12)
                            / max(np.sum((tt - tt.mean()) ** 2), 1e-12)))
        r = 0.5 * float(np.nanmedian(np.asarray(d, float)[idx]))
        ratio = w[0] * r / v if abs(v) > 1e-9 else float("inf")
        s_ratio = abs(ratio) * float(np.hypot(w[1] / max(abs(w[0]), 1e-9), s_v / max(abs(v), 1e-9)))
        rows.append({"t0": float(tt[0]), "t1": float(tt[-1]), "v_px_s": v, "spin": w[0],
                     "ratio": float(ratio), "ratio_sigma": float(s_ratio)})
    if not rows:
        return _exp(_result(name, "skip", "Its turning couldn't be read while it rolled."))
    detail = {"segments": rows}
    r = max(rows, key=lambda q: q["t1"] - q["t0"])
    long_ = r["t1"] - r["t0"] >= 0.4
    if long_ and (r["ratio"] < 0.3 or r["ratio"] > 3.0) and \
            abs(r["ratio"] - 1) > 5 * r["ratio_sigma"]:
        return _exp(_result(name, "fail", f"It rolls along the floor turning {r['ratio']:.1f}x as "
                            "fast as rolling needs (spin x radius = speed). A real ball grips "
                            "the floor and rolls.", **detail))
    if (r["ratio"] < 0.6 or r["ratio"] > 1.6) and abs(r["ratio"] - 1) > 3 * r["ratio_sigma"]:
        return _exp(_result(name, "warn", f"It rolls turning {r['ratio']:.1f}x as fast as rolling "
                            "needs (skidding, or a slippery floor).", **detail))
    return _exp(_result(name, "pass", f"It rolls without slipping (turns {r['ratio']:.2f}x as "
                        "fast as its speed needs).", **detail))


def floor_check(bounce_pts: list, d_ref: float, cy: float | None = None) -> dict:
    """The floor is flat and doesn't move: every bounce lands on it.

    Positions here are scaled to the ball's reference distance (each point
    moved toward the picture's centre by how much nearer the ball is than
    usual). Seen by a level camera, a floor point at the ball's distance sits
    below the centre by a fixed multiple of the ball's width there, and both
    shrink with distance, so in these units every contact lands at one
    height. A camera tipped more than Galileo noticed adds a term that grows
    with distance, so bounces at different distances need three of them.

    bounce_pts: per bounce {"t", "bottom" (the ball's bottom at contact, from
    the fitted hops, scaled), "q" (reference width / width there)}; cy: the
    picture's centre row. The scaling leans on the ball's width, good to a
    couple of percent, so a bounce far from the centre row is placed less
    exactly, and the tolerance grows to match."""
    name = "One flat floor"
    pts = [p for p in bounce_pts if np.isfinite(p["bottom"]) and np.isfinite(p["q"])]
    if len(pts) < 2:
        return _exp(_result(name, "skip", "Fewer than two bounces seen."))
    b = np.array([p["bottom"] for p in pts])
    q = np.array([p["q"] for p in pts])
    detail = {"bounces": [{k: p[k] for k in ("t", "bottom", "q")} for p in pts]}
    if np.ptp(q) > 0.05 * np.median(q):
        if len(pts) < 3:
            return _exp(_result(name, "skip", "Two bounces at different distances from the camera: "
                                "a tipped camera and a sloped floor look alike.", **detail))
        A = np.vstack([np.ones(len(q)), q]).T
        res = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        res = b - np.median(b)
    rel = res / max(d_ref, 1e-9)
    # How well each contact is placed, in widths: squash and the fits' reach
    # (a few percent), plus the scaling's 2% of the distance from the centre row.
    s_q = (0.02 * np.abs(b - cy) / max(d_ref, 1e-9)) if cy is not None else np.zeros(len(b))
    worst = float(np.max(np.abs(rel)))
    over_fail = np.abs(rel) - (0.6 + 4.0 * s_q)
    over_warn = np.abs(rel) - (0.3 + 2.5 * s_q)
    detail["off_widths"] = [float(v) for v in rel]
    detail["tolerance_widths"] = [float(v) for v in 0.3 + 2.5 * s_q]
    # Squash at contact and the fits' reach blur the contact by a few percent
    # of the width; the floor isn't flat to the millimetre either.
    if np.max(over_fail) > 0:
        return _exp(_result(name, "fail", f"The bounces land on different floors: one lands "
                            f"{worst:.1f} ball-widths off where the others put it. A real floor "
                            "doesn't move.", **detail))
    if np.max(over_warn) > 0:
        return _exp(_result(name, "warn", f"The bounces land up to {worst:.1f} ball-widths off one "
                            "flat floor (a sloped floor, a squashy ball, or generated motion).",
                            **detail))
    return _exp(_result(name, "pass", f"All {len(pts)} bounces land on one flat floor (within "
                        f"{max(worst, 0.01):.2f} ball-widths).", **detail))


def sideways_check(jf: JointFit, t, light: bool = False) -> dict:
    """Newton's first law sideways: in the air only gravity pulls (down), so
    nothing should speed a ball up or slow it down sideways. A light ball's
    drag (or a strongly spinning one's curve) can, a little."""
    name = "No sideways push"
    g = jf.gdir
    e = np.array([g[1], -g[0]])
    rows = []
    clean = set(clean_flights(jf))
    for k, fr in enumerate(jf.flights):
        if k not in clean or len(fr.idx) < 12 or fr.duration < 0.2 or \
                not (fr.a_along > 5 * fr.a_sigma > 0):
            continue
        a = fr.fit.accel_vec()
        cov = fr.fit.accel_cov()
        a_s = float(a @ e)
        s_s = float(np.sqrt(max(float(e @ cov @ e), 0.0)))
        rows.append({"flight": k, "ratio": a_s / fr.a_along, "z": a_s / max(s_s, 1e-9)})
    if not rows:
        return _exp(_result(name, "skip", "No flight long enough to test."))
    detail = {"flights": rows}
    r = max(rows, key=lambda q: abs(q["ratio"]) if abs(q["z"]) > 3 else 0.0)
    lim_fail, lim_warn = (0.5, 0.2) if light else (0.2, 0.08)
    if abs(r["ratio"]) > lim_fail and abs(r["z"]) > 5 and not light and len(rows) > 1:
        return _exp(_result(name, "fail", f"In one hop something pushes it sideways at "
                            f"{100 * abs(r['ratio']):.0f}% of gravity. In the air only gravity "
                            "pulls, and only down.", **detail))
    if abs(r["ratio"]) > lim_warn and abs(r["z"]) > 3 and len(rows) > 1:
        return _exp(_result(name, "warn", f"In one hop it drifts sideways at "
                            f"{100 * abs(r['ratio']):.0f}% of gravity (wind, spin, or generated "
                            "motion).", **detail))
    if len(rows) == 1:
        return _exp(_result(name, "pass", "One flight: its sideways speed stays steady.", **detail))
    worst = max(abs(q["ratio"]) for q in rows)
    return _exp(_result(name, "pass", f"Every hop keeps its sideways speed (within "
                        f"{100 * worst:.0f}% of gravity).", **detail))


def steady_bounce_check(bounces: list, clean=None) -> dict:
    """One ball on one floor keeps about the same share of its speed at every
    bounce. Softer landings keep a little more (the share drifts up as the
    bounces die down), so bounces are compared with the others allowing for
    how much harder or softer they landed. A bounce far off the rest isn't
    the same ball on the same floor."""
    name = "Steady bounces"

    def _e(p):
        if "restitution_t" in p:
            return p["restitution_t"], p["restitution_t_sigma"]
        return p["restitution"], p["restitution_sigma"]
    if clean is not None:                       # only bounces between clean hops
        bounces = [p for p in bounces if p["j"] in clean and p["j"] + 1 in clean]
    rows = [(abs(p["v_in"]), *_e(p)) for p in bounces
            if np.isfinite(_e(p)[0]) and 0 < _e(p)[1] < 0.1 and abs(p["v_in"]) > 0]
    if len(rows) < 3:
        return _exp(_result(name, "skip", "Fewer than three bounces measured well enough."))
    v = np.array([r[0] for r in rows])
    e = np.array([r[1] for r in rows])
    s = np.maximum(np.array([r[2] for r in rows]), 0.01)
    worst = None
    for k in range(len(e)):
        others = np.delete(np.arange(len(e)), k)
        ref = float(np.median(e[others]))
        v_ref = float(np.median(v[others]))
        tol = 0.12 + 0.15 * abs(np.log(v[k] / v_ref))
        off = abs(e[k] - ref)
        z = off / float(np.hypot(s[k], np.median(s[others])))
        score = off - tol
        if worst is None or score > worst[0]:
            worst = (score, k, ref, off, tol, z)
    score, k, ref, off, tol, z = worst
    detail = {"restitution": [float(x) for x in e], "landing_px_s": [float(x) for x in v],
              "odd_one": int(k), "others": ref}
    if off > tol + 0.08 and z > 5:
        return _exp(_result(name, "fail", f"One bounce keeps {e[k]:.2f} of its speed while the "
                            f"others keep about {ref:.2f}. One ball on one floor bounces alike.",
                            **detail))
    if off > tol and z > 3:
        return _exp(_result(name, "warn", f"One bounce keeps {e[k]:.2f} of its speed while the "
                            f"others keep about {ref:.2f} (a bump, a spin, or generated motion).",
                            **detail))
    return _exp(_result(name, "pass", f"Every bounce keeps a steady share of its speed "
                        f"({e.min():.2f} to {e.max():.2f}).", **detail))


def size_check(t, d, flights) -> dict:
    """A ball doesn't grow or shrink: its width in the picture changes only as
    it moves toward or away from the camera, and that motion is smooth (its
    distance follows a parabola in a flight, like its height). Something in
    front of it, or a change of light, can fake a swell of a fifth, so that
    only warns; a ball that swells by more than half fails."""
    name = "Constant size"
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    worst, rows = 0.0, []
    for idx in flights:
        idx = np.asarray(idx, int)
        idx = idx[np.isfinite(d[idx]) & (d[idx] > 0)]
        if len(idx) < 15:
            continue
        tt, inv = t[idx], 1.0 / d[idx]
        A = np.vander(tt - tt.mean(), 3, increasing=True)
        wt = np.ones(len(tt))
        for _ in range(6):                               # robust: a swell mustn't bend the path
            sw = np.sqrt(wt)
            coef = np.linalg.lstsq(A * sw[:, None], inv * sw, rcond=None)[0]
            r_ = inv / (A @ coef) - 1.0
            sc = max(float(np.median(np.abs(r_))) * 1.4826, 0.003)
            u_ = r_ / (4.0 * sc)
            wt = np.where(np.abs(u_) < 1, (1 - u_ ** 2) ** 2, 0.0) + 1e-6
        rel = inv / (A @ coef) - 1.0                     # how far off a smooth path, as a share
        k = max(3, len(rel) // 10) | 1
        sm = np.convolve(rel, np.ones(k) / k, mode="valid")
        noise = float(np.median(np.abs(np.diff(rel, 2)))) / np.sqrt(6) * 1.4826 / np.sqrt(k)
        big = float(np.max(np.abs(sm))) if len(sm) else 0.0
        rows.append({"points": int(len(idx)), "swell": big, "noise": noise})
        if big > worst and big > 4 * max(noise, 1e-4):
            worst = big
    if not rows:
        return _exp(_result(name, "skip", "Not enough clear widths to follow its size."))
    detail = {"flights": rows}
    if worst > 0.60:
        return _exp(_result(name, "fail", f"Its size swells and shrinks by {100 * worst:.0f}% "
                            "in midair, far beyond what moving toward or away from the camera "
                            "explains. A real ball keeps its size.", **detail))
    if worst > 0.20:
        return _exp(_result(name, "warn", f"Its size swells and shrinks by {100 * worst:.0f}% "
                            "beyond what moving toward or away from the camera explains "
                            "(something in front of it, changing light, or a generated ball).",
                            **detail))
    return _exp(_result(name, "pass", f"It keeps its size (within {100 * max(worst, 0.005):.0f}% "
                        "of a smooth path).", **detail))


def shadow_checks(obs: dict | None, x, y, d, floor_y, bounce_i: list) -> list[dict]:
    """Light travels in straight lines and arrives at once.

    * "Shadow follows": a far light puts the shadow's centre at the ball's
      foot point plus a fixed multiple of its height, so the shadow slides
      along the floor in step with the ball. (A shadow that never moves is
      part of the still background, so it can't be seen this way.)
    * "Ball meets its shadow": at a bounce the ball is on the floor, so it
      touches its own shadow there.

    obs: shadow.observe output; x, y, d, floor_y: the ball (centre, width) and
    the floor line per tracked point, in the same pixels; bounce_i: the tracked
    point closest to each bounce. Never fails on a scene without a clear shadow.
    """
    follow, meet = "Shadow follows", "Ball meets its shadow"
    if obs is None:
        return [_exp(_result(follow, "skip", "The shadow wasn't looked for.")),
                _exp(_result(meet, "skip", "The shadow wasn't looked for."))]
    x, y, d, fy = (np.asarray(v, float) for v in (x, y, d, floor_y))
    known = np.isfinite(fy) & np.isfinite(d) & (d > 0)
    hc = (fy - y) / d                                    # centre height above the floor, widths
    low = known & (hc < 4.0)
    found = obs["found"] & low & (obs["mass"] > 0.02)
    share = float(found.sum()) / max(int(low.sum()), 1)
    if low.sum() < 10 or share < 0.4:
        why = ("No clear shadow on the floor (soft or overhead light, or a dark floor), so "
               "this can't be tested.")
        return [_exp(_result(follow, "skip", why, share=share)),
                _exp(_result(meet, "skip", why, share=share))]
    i = np.flatnonzero(found)
    ox = (obs["sx"][i] - x[i]) / d[i]
    oy = (obs["sy"][i] - fy[i]) / d[i]
    A = np.vstack([np.ones(len(i)), hc[i]]).T
    cx = np.linalg.lstsq(A, ox, rcond=None)[0]
    cy_ = np.linalg.lstsq(A, oy, rcond=None)[0]
    res = np.hypot(ox - A @ cx, oy - A @ cy_)
    rms = float(np.sqrt(np.median(res ** 2)))
    # Does it slide with the ball? Compare with a dark patch that stays put.
    sx, sx_mean = obs["sx"][i], float(np.mean(obs["sx"][i]))
    travel = float(np.ptp(x[i]) / np.median(d[i]))
    still = float(np.sqrt(np.median(((sx - sx_mean) / d[i]) ** 2)))
    moving = float(np.sqrt(np.median((ox - A @ cx) ** 2)))
    detail = {"share": share, "rms": rms, "slope_x": float(cx[1]), "slope_y": float(cy_[1]),
              "at_floor_x": float(cx[0] + 0.5 * cx[1]), "travel": travel,
              "fit_still": still, "fit_follow": moving}
    out = []
    if rms <= 0.35:
        out.append(_exp(_result(follow, "pass", "Its shadow slides along the floor in step with "
                                f"the ball (within {rms:.2f} ball-widths).", **detail)))
    elif share >= 0.7 and rms > 0.6 and still >= 0.5 * moving:
        out.append(_exp(_result(follow, "warn", "A clear shadow, but it doesn't move in step "
                                f"with the ball (off by {rms:.1f} ball-widths). A shadow cast by "
                                "a real light follows the ball exactly.", **detail)))
    else:
        out.append(_exp(_result(follow, "skip", "A dark patch near the floor, but it doesn't "
                                "track the ball cleanly enough to call it its shadow.", **detail)))
    # Meeting at the bounces: only when the shadow was found and follows.
    rows = []
    for b in bounce_i:
        if b is None or not (0 <= b < len(x)):
            continue
        near = [k for k in range(max(0, b - 1), min(len(x), b + 2)) if obs["found"][k]]
        gap = min((float(obs["gap"][k]) for k in near), default=float("nan"))
        rows.append({"i": int(b), "gap": gap})
    seen = [r for r in rows if np.isfinite(r["gap"])]
    detail2 = {"bounces": rows}
    if out[0]["status"] != "pass" or not seen:
        out.append(_exp(_result(meet, "skip", "The shadow isn't clear at the bounces.", **detail2)))
        return out
    apart = [r for r in seen if r["gap"] >= 0.55]       # 0.2 is touching (the rim left out)
    if len(apart) >= 2 and len(apart) == len(seen):
        out.append(_exp(_result(meet, "fail", f"At every bounce the ball lands "
                                f"{min(r['gap'] for r in apart):.1f}+ widths from its own shadow. "
                                "On the floor, a ball touches its shadow.", **detail2)))
    elif apart:
        out.append(_exp(_result(meet, "warn", f"At {len(apart)} of {len(seen)} bounce(s) the ball "
                                "lands away from its shadow.", **detail2)))
    else:
        out.append(_exp(_result(meet, "pass", f"The ball touches its shadow at {len(seen)} "
                                "bounce(s).", **detail2)))
    return out
