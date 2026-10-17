"""Shared constants for Galileo.

Everything here is a plain number or dict so teammates can tune thresholds
without touching the physics code.
"""

# Reference gravity used for verdicts (m/s^2). Earth's surface gravity runs
# from about 9.78 at the equator to 9.83 at the poles; 9.80 sits in the middle.
G_EARTH = 9.80

# Normal gravity at UF (29.65 N, about 50 m elevation), for tower comparisons.
G_GAINESVILLE = 9.793

# Known objects that can act as their own ruler. size_m is the diameter for
# balls. rel_unc is the relative spread of that size across real objects.
OBJECT_PRESETS = {
    "steel-38mm": {"label": "38 mm steel ball", "size_m": 0.038, "rel_unc": 0.002},
    "steel-25mm": {"label": "25 mm steel ball", "size_m": 0.025, "rel_unc": 0.002},
    "golf": {"label": "golf ball", "size_m": 0.0427, "rel_unc": 0.005},
    "billiard": {"label": "billiard ball", "size_m": 0.0572, "rel_unc": 0.005},
    "tennis": {"label": "tennis ball", "size_m": 0.067, "rel_unc": 0.03},
    "baseball": {"label": "baseball", "size_m": 0.0737, "rel_unc": 0.015},
    "softball": {"label": "softball", "size_m": 0.097, "rel_unc": 0.02},
    "volleyball": {"label": "volleyball", "size_m": 0.21, "rel_unc": 0.02},
    "soccer": {"label": "soccer ball", "size_m": 0.22, "rel_unc": 0.02},
    "basketball": {"label": "basketball", "size_m": 0.239, "rel_unc": 0.02},
    # Light balls feel air drag; results will read low late in a long fall.
    "ping-pong": {"label": "ping-pong ball (drag!)", "size_m": 0.040, "rel_unc": 0.01},
    "apple": {"label": "apple", "size_m": 0.075, "rel_unc": 0.15},
    "orange": {"label": "orange", "size_m": 0.075, "rel_unc": 0.15},
}

# Verdict thresholds -------------------------------------------------------

# A clip is consistent with gravity when |g/g_ref - 1| <= max(TOL_REL, Z_TOL*sigma_rel).
# 15% covers single-ruler scale error on wild footage; tower clips do far better.
TOL_REL = 0.15
Z_TOL = 2.5

# Minimum data for any verdict.
MIN_FLIGHT_POINTS = 6
MIN_FLIGHT_SECONDS = 0.12

# Scale-free parabola test: acceleration in the second half of a fall
# divided by the first half. Rolling shutter and lens distortion can push a
# real clip to roughly 0.85-1.2, so only ratios outside this band fail.
PARABOLA_RATIO_FAIL = (0.6, 1.6)
PARABOLA_RATIO_WARN = (0.8, 1.25)

# Bounce test: a ball can't come off the floor faster than it hit it.
RESTITUTION_FAIL = 1.08

# Slow-motion clocks: minimum confidence before a clock changes the answer.
CLOCK_MIN_CONFIDENCE = 0.6

# Tracking -----------------------------------------------------------------

# Working resolution cap (longest side, px) and memory budget for frames.
WORK_MAX_SIDE = 1280
FRAME_BUDGET_BYTES = 600_000_000
MAX_FRAMES = 900
