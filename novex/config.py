"""Shared constants for Novex.

Everything here is a plain number or dict so teammates can tune thresholds
without touching the physics code.
"""

# Reference gravity used for verdicts (m/s^2). Earth's surface gravity runs
# from about 9.78 at the equator to 9.83 at the poles; 9.80 sits in the middle.
G_EARTH = 9.80

# Normal gravity at UF (29.65 N, about 50 m elevation), for tower comparisons.
G_GAINESVILLE = 9.793

# Known objects that can act as their own ruler. size_m is the diameter for
# balls and the longest side for everything else (shape "rigid": its outline
# changes as it tumbles, so it is measured differently, see novex/shape.py).
# rel_unc is the relative spread of that size across real objects. vt is the
# terminal speed in still air (m/s): light objects feel enough drag to read
# low, and Novex corrects for it when vt is known. For tumbling objects vt
# is a rough average over the ways they can face the air.
OBJECT_PRESETS = {
    "steel-38mm": {"label": "38 mm steel ball", "size_m": 0.038, "rel_unc": 0.002, "vt": 150.0},
    "steel-25mm": {"label": "25 mm steel ball", "size_m": 0.025, "rel_unc": 0.002, "vt": 120.0},
    "golf": {"label": "golf ball", "size_m": 0.0427, "rel_unc": 0.005, "vt": 40.0},
    "billiard": {"label": "billiard ball", "size_m": 0.0572, "rel_unc": 0.005, "vt": 60.0},
    "tennis": {"label": "tennis ball", "size_m": 0.067, "rel_unc": 0.03, "vt": 21.0},
    "baseball": {"label": "baseball", "size_m": 0.0737, "rel_unc": 0.015, "vt": 33.0},
    "softball": {"label": "softball", "size_m": 0.097, "rel_unc": 0.02, "vt": 30.0},
    "volleyball": {"label": "volleyball", "size_m": 0.21, "rel_unc": 0.02, "vt": 16.0},
    "soccer": {"label": "soccer ball", "size_m": 0.22, "rel_unc": 0.02, "vt": 23.0},
    "basketball": {"label": "basketball", "size_m": 0.239, "rel_unc": 0.02, "vt": 20.0},
    "ping-pong": {"label": "ping-pong ball", "size_m": 0.040, "rel_unc": 0.01, "vt": 8.4},
    "pickleball": {"label": "pickleball", "size_m": 0.074, "rel_unc": 0.015, "vt": 15.0},
    "apple": {"label": "apple", "size_m": 0.075, "rel_unc": 0.15, "vt": 40.0},
    "orange": {"label": "orange", "size_m": 0.075, "rel_unc": 0.15, "vt": 40.0},
    # Not round: size_m is the longest side.
    "phone": {"label": "smartphone", "size_m": 0.152, "rel_unc": 0.05, "vt": 20.0,
              "shape": "rigid"},
    "tablet": {"label": "tablet", "size_m": 0.247, "rel_unc": 0.06, "vt": 17.0, "shape": "rigid"},
    "book": {"label": "book", "size_m": 0.23, "rel_unc": 0.15, "vt": 18.0, "shape": "rigid"},
    "cards": {"label": "deck of cards", "size_m": 0.089, "rel_unc": 0.02, "vt": 16.0,
              "shape": "rigid"},
    "water-bottle": {"label": "water bottle", "size_m": 0.21, "rel_unc": 0.10, "vt": 27.0,
                     "shape": "rigid"},
    "soda-can": {"label": "soda can", "size_m": 0.122, "rel_unc": 0.03, "vt": 30.0,
                 "shape": "rigid"},
    "remote": {"label": "TV remote", "size_m": 0.18, "rel_unc": 0.15, "vt": 15.0, "shape": "rigid"},
    "pen": {"label": "pen", "size_m": 0.14, "rel_unc": 0.08, "vt": 11.0, "shape": "rigid"},
    "pencil": {"label": "pencil", "size_m": 0.19, "rel_unc": 0.06, "vt": 9.0, "shape": "rigid"},
    "banana": {"label": "banana", "size_m": 0.19, "rel_unc": 0.12, "vt": 19.0, "shape": "rigid"},
    "egg": {"label": "egg", "size_m": 0.057, "rel_unc": 0.07, "vt": 36.0, "shape": "rigid"},
    "shoe": {"label": "shoe", "size_m": 0.28, "rel_unc": 0.12, "vt": 14.0, "shape": "rigid"},
    "football": {"label": "American football", "size_m": 0.28, "rel_unc": 0.03, "vt": 25.0,
                 "shape": "rigid"},
}

# A non-round object's ruler is never trusted to better than this: a tumbling
# object is often seen foreshortened, and blur can stretch its outline.
RIGID_MIN_REL_UNC = 0.06

# Objects measured per clip (the primary one plus others that fall).
MAX_OBJECTS = 4

# Automatic identification. CLIP is a pretrained image-text model that runs
# offline; it only names objects it is fairly sure of.
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_MIN_PROB = 0.35

# Verdict thresholds -------------------------------------------------------

# A clip is consistent with gravity when |g/g_ref - 1| <= max(TOL_REL, Z_TOL*sigma_rel).
# 15% covers single-ruler scale error on wild footage; tower clips do far better.
TOL_REL = 0.15
Z_TOL = 2.5
# If Z_TOL * relative error exceeds this, a "matches gravity" call would be
# meaningless, so the verdict is "too uncertain" unless g is far off anyway.
MAX_SPREAD = 0.35
# Take every clip as playing at the speed it was filmed (no slow motion, no
# speed-up). Then g inside Earth's interval is real, and outside it, too slow
# or too fast, is AI. False (or --slowmo) reads the file's clocks instead and
# allows real slow motion (SLOWMO) and sped-up footage (can't tell).
ASSUME_REAL_TIME = True

# The newer laws (spin, shadows, one flat floor, no sideways push, steady
# bounces, constant size: checks.py, marked "experimental"). True: their
# failures count like the older laws' (one failing alone, with gravity in
# order, is only a warning either way: verdict.decide). False: they are
# shown in the checks table and the reasons, but can't change the verdict.
EXPERIMENTAL_CHECKS_DECIDE = True

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
# A longer clip is read one frame in N, as long as that leaves this many
# frames a second (plenty for a fall); only past that is it cut to the
# stretch with the most motion.
MIN_FPS_LONG = 60.0

# Zoom pass: if the ball is smaller than ZOOM_MIN_PX in the first pass, the fall
# is decoded again, cropped around the ball's path, at a scale that makes the
# ball about ZOOM_TARGET_PX wide (never above the original resolution). Edge
# errors are about a pixel whatever the size, so a bigger ball reads truer.
ZOOM_MIN_PX = 40.0
ZOOM_TARGET_PX = 64.0
ZOOM_BUDGET_BYTES = 700_000_000

# Gravity must point within this many degrees of the frame's "down".
MAX_TILT_DEG = 30.0
