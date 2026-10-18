"""What is the falling object? Its name gives its size (the ruler), its air
drag, and whether it is round.

Tried in order, each only if it can run here:
  1. what you said: --object, one name per object in any order, anything at
     all ("rotisserie chicken"), with a size if you know it
     ("rotisserie chicken=22cm"); the web app's menu
  2. Gemini, online, with --gemini and a key set: it can name anything
  3. CLIP, a pretrained image-text model that runs offline on the CPU
     (pip install torch transformers; the model downloads once, ~600 MB),
     over a few hundred everyday objects (novex/catalog.py)
  4. the same kind as another object in the clip that looks the same size
     and shape (two tennis balls dropped together)
Otherwise the object gets no ruler, and only its scale-free checks count.

Every answer says where it came from, so a guessed size is never passed
off as a known one.
"""
from __future__ import annotations

import difflib
import hashlib
import math
import os
import re

import cv2
import numpy as np

from . import config as C
from .catalog import CATALOG, LIGHT_VT

ALIASES = {
    "smartphone": "phone", "iphone": "phone", "cellphone": "phone", "cell-phone": "phone",
    "mobile-phone": "phone", "ipad": "tablet", "bottle": "water-bottle", "can": "soda-can",
    "tennis-ball": "tennis", "ping-pong-ball": "ping-pong", "table-tennis": "ping-pong",
    "table-tennis-ball": "ping-pong", "soccer-ball": "soccer", "american-football": "football",
    "golf-ball": "golf", "billiard-ball": "billiard", "pool-ball": "billiard",
    "deck-of-cards": "cards", "playing-cards": "cards", "tv-remote": "remote",
    "remote-control": "remote", "sneaker": "shoe", "steel": "steel-38mm",
    "roast-chicken": "rotisserie-chicken", "rotisserie": "rotisserie-chicken",
    "chicken": "rotisserie-chicken", "roasted-chicken": "rotisserie-chicken",
    "turkey": "roast-turkey", "bread": "bread-loaf", "loaf": "bread-loaf", "doughnut": "donut",
    "cupcake": "muffin", "melon": "cantaloupe", "basketball-ball": "basketball",
    "teddy": "teddy-bear", "stuffed-animal": "plush-toy", "plushie": "plush-toy",
    "cube": "rubiks-cube", "rubik's-cube": "rubiks-cube", "rubiks": "rubiks-cube",
    "die": "dice", "lego": "lego-brick", "toilet-paper-roll": "toilet-paper",
    "glass": "drinking-glass", "cup": "paper-cup", "pan": "frying-pan", "pot": "cooking-pot",
    "knife": "table-knife", "mouse": "mouse", "controller": "game-controller",
    "battery": "aa-battery", "helmet": "bike-helmet", "bag-of-chips": "chip-bag",
    "chips": "chip-bag", "shopping-bag": "plastic-bag", "grocery-bag": "plastic-bag",
    "dry-erase-eraser": "whiteboard-eraser", "board-eraser": "whiteboard-eraser",
    "chalkboard-eraser": "whiteboard-eraser", "expo-eraser": "whiteboard-eraser",
    "felt-eraser": "whiteboard-eraser",
}


def entry(key: str | None) -> dict | None:
    """The known object behind a key: an exact preset or a catalogue entry."""
    if not key:
        return None
    if key in C.OBJECT_PRESETS:
        p = C.OBJECT_PRESETS[key]
        return {"label": p["label"], "size_m": p["size_m"], "rel_unc": p["rel_unc"],
                "vt": p.get("vt"), "shape": p.get("shape", "round")}
    return CATALOG.get(key)


def _slug(name: str) -> str:
    return re.sub(r"[\s_]+", "-", str(name).strip().lower())


def resolve(name: str | None, fuzzy: bool = True) -> str | None:
    """A known object's key from a name you typed, or None."""
    if not name:
        return None
    k = _slug(name)
    for cand in (k, k.rstrip("s"), ALIASES.get(k), ALIASES.get(k.rstrip("s"))):
        if cand and (cand in C.OBJECT_PRESETS or cand in CATALOG):
            return cand
    for key, e in CATALOG.items():                     # the label itself ("loaf of bread")
        if _slug(e["label"]) == k:
            return key
    for key, p in C.OBJECT_PRESETS.items():
        if _slug(p["label"]) == k:
            return key
    if not fuzzy:
        return None
    names = {**{key: key for key in list(C.OBJECT_PRESETS) + list(CATALOG)},
             **{_slug(e["label"]): key for key, e in CATALOG.items()},
             **{_slug(p["label"]): key for key, p in C.OBJECT_PRESETS.items()},
             **{a: v for a, v in ALIASES.items()}}
    close = difflib.get_close_matches(k, list(names), n=1, cutoff=0.88)
    return names[close[0]] if close else None


_UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
          '"': 0.0254, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048}


def parse_name(text: str) -> tuple[str, float | None]:
    """"rotisserie chicken=22cm" -> ("rotisserie chicken", 0.22). Sizes in mm
    when no unit is given (like --size-mm); also cm, m, in, ft."""
    text = str(text).strip()
    m = re.match(r"^(.*?)\s*[=:@]\s*([0-9.]+)\s*([a-zA-Z\"]*)\s*$", text)
    if not m:
        return text, None
    name, num, unit = m.group(1).strip(), float(m.group(2)), m.group(3).lower()
    if unit and unit not in _UNITS:
        return text, None
    return name, num * _UNITS.get(unit or "mm", 0.001)


def shape_of(key: str | None) -> str | None:
    e = entry(key)
    return e["shape"] if e else None


def from_preset(key: str, source: str = "you", confidence: float = 1.0, detail: str = "") -> dict:
    e = entry(key)
    return {"key": key, "label": e["label"], "size_m": e["size_m"], "rel_unc": e["rel_unc"],
            "vt": e.get("vt"), "shape": e.get("shape", "round"), "source": source,
            "confidence": float(confidence), "detail": detail}


def from_name(text: str) -> dict:
    """Your name for an object, known or not, with its size if you gave one.

    A known name brings its size, drag and shape (a size you give wins). An
    unknown one is kept as a label: with a size it is a ruler; without one,
    Gemini may still estimate the size, and CLIP uses the name to tell which
    object you meant."""
    name, size = parse_name(text)
    key = resolve(name)
    if key:
        out = from_preset(key)
        if size:
            out.update(size_m=size, rel_unc=0.02, detail=f"size {1000 * size:.0f} mm from you",
                       explicit_size=True)
        out["name"] = name
        return out
    return {"key": None, "label": name, "name": name, "size_m": size,
            "rel_unc": 0.02 if size else None, "vt": None, "shape": None, "source": "you",
            "confidence": 1.0, "detail": "" if size else "no size known for this name",
            "explicit_size": bool(size)}


def sized_by_kind(idn: dict | None) -> bool:
    """A name you typed whose size is only the catalog's typical one for a kind
    of thing that comes in many sizes or shapes (an "eraser": a pencil's, or a
    whiteboard's), not a size you gave. Worth asking Gemini to size the one in
    the clip."""
    idn = idn or {}
    return bool(idn.get("source") == "you" and idn.get("key") and not idn.get("explicit_size")
                and ((idn.get("rel_unc") or 0.0) >= 0.1 or idn.get("shape") == "rigid"))


def is_light(idn: dict | None) -> bool:
    """Air drag rules its fall (a feather, a balloon, a sheet of paper)."""
    vt = (idn or {}).get("vt")
    return bool(vt and vt < LIGHT_VT)


# How CLIP is asked about each object: describe what sets it apart (holes,
# fuzz, dimples), not just its name. Objects whose size can't be told from a
# picture (steel balls look alike at any size) get a name but no ruler.
CLIP_PHRASES = {
    "tennis": ["a tennis ball, fuzzy felt with a curved seam"],
    "pickleball": ["a pickleball, a hard plastic ball full of round holes",
                   "a perforated plastic wiffle ball"],
    "ping-pong": ["a ping-pong ball, a small smooth white plastic ball",
                  "a small smooth orange ping-pong ball"],
    "golf": ["a golf ball covered in dimples"],
    "baseball": ["a white baseball with red stitches"],
    "softball": ["a yellow softball with red stitches"],
    "billiard": ["a glossy billiard ball with a number"],
    "volleyball": ["a volleyball with curved panels"],
    "soccer": ["a soccer ball with black and white patches"],
    "basketball": ["a basketball, orange with black lines"],
    "apple": ["an apple"],
    "orange": ["an orange, the fruit"],
    "phone": ["a smartphone, a flat glass rectangle", "a mobile phone"],
    "tablet": ["a tablet computer"],
    "book": ["a book with a cover and pages"],
    "cards": ["a deck of playing cards"],
    "water-bottle": ["a plastic water bottle"],
    "soda-can": ["an aluminum soda can"],
    "remote": ["a TV remote control with buttons"],
    "pen": ["a pen", "a marker pen"],
    "pencil": ["a wooden pencil"],
    "banana": ["a yellow banana"],
    "egg": ["a chicken egg"],
    "shoe": ["a shoe", "a sneaker"],
    "football": ["an American football, brown leather with white laces"],
}
NAME_ONLY = {   # recognisable, but no standard size
    "steel ball": ["a shiny steel ball bearing"],
    "ball": ["a rubber ball", "a toy ball"],
    "box": ["a cardboard box"],
    "toy": ["a plastic toy"],
    "rock": ["a rock"],
}
NOT_AN_OBJECT = ["an empty background", "a human hand", "a person's arm", "a person",
                 "a shadow on a wall", "a blurry smudge"]
TEMPLATES = ["a photo of {}.", "a blurry photo of {}.", "a close-up photo of {}.",
             "a photo of {} in mid-air."]

_clip: dict = {}


def clip_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False, "install torch and transformers for offline object recognition"
    return True, C.CLIP_MODEL


def _vocabulary():
    """(keys, kinds, phrase lists) for everything CLIP chooses between."""
    keys, kinds, groups = [], [], []
    for k, ph in CLIP_PHRASES.items():
        keys.append(k)
        kinds.append("preset")
        groups.append(ph)
    for k, e in CATALOG.items():
        if k in CLIP_PHRASES:
            continue
        keys.append(k)
        kinds.append("preset")
        groups.append([e["phrase"]])
    for k, ph in NAME_ONLY.items():
        keys.append(k)
        kinds.append("name")
        groups.append(ph)
    for ph in NOT_AN_OBJECT:
        keys.append(ph)
        kinds.append("none")
        groups.append([ph])
    return keys, kinds, groups


def _text_emb(groups) -> np.ndarray:
    """One normalised text embedding per group of phrases (prompt ensemble)."""
    import torch
    m = _clip
    texts, owner = [], []
    for gi, ph in enumerate(groups):
        for p in ph:
            for tpl in TEMPLATES:
                texts.append(tpl.format(p))
                owner.append(gi)
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            tok = m["proc"](text=texts[i:i + 256], return_tensors="pt", padding=True)
            te = m["model"].get_text_features(**tok)
            if not isinstance(te, torch.Tensor):   # newer transformers return an output object
                te = te.pooler_output if getattr(te, "pooler_output", None) is not None else te[0]
            outs.append((te / te.norm(dim=-1, keepdim=True)).numpy())
    te = np.concatenate(outs)
    owner = np.asarray(owner)
    emb = np.stack([te[owner == gi].mean(0) for gi in range(len(groups))])
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def _load_clip():
    if "model" in _clip:
        return _clip
    from transformers import CLIPModel, CLIPProcessor
    name = os.environ.get("NOVEX_CLIP_MODEL", C.CLIP_MODEL)
    # Otherwise transformers asks the Hugging Face Hub in the background, on
    # every run, whether a safetensors copy of the weights exists, and the Hub
    # answers an anonymous request with a "set HF_TOKEN" warning.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    try:
        # Once downloaded, load it from disk without asking the Hub at all
        # (it also works offline).
        model = CLIPModel.from_pretrained(name, local_files_only=True)
        proc = CLIPProcessor.from_pretrained(name, local_files_only=True)
    except Exception:
        model = CLIPModel.from_pretrained(name)          # first run: download (~600 MB)
        proc = CLIPProcessor.from_pretrained(name)
    _clip.update(model=model.eval(), proc=proc, name=name)
    keys, kinds, groups = _vocabulary()
    # The vocabulary's embeddings take a while on a CPU; keep them on disk.
    from .gemini import cache_dir
    tag = hashlib.sha1(repr((name, groups, TEMPLATES)).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir(), f"clip_vocab_{tag}.npy")
    emb = None
    try:
        if os.path.exists(path):
            emb = np.load(path)
    except Exception:
        emb = None
    if emb is None or emb.shape[0] != len(keys):
        emb = _text_emb(groups)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, emb)
        except Exception:
            pass
    _clip.update(emb=emb, keys=keys, kinds=kinds,
                 scale=float(_clip["model"].logit_scale.exp().item()))
    return _clip


def _square_crop(img, x, y, half):
    h, w = img.shape[:2]
    half = int(max(half, 12))
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REPLICATE)
    xi, yi = int(round(x)) + half, int(round(y)) + half
    return pad[yi - half:yi + half, xi - half:xi + half]


def clip_image_emb(img_bgr, x, y, size_px) -> np.ndarray:
    """CLIP's normalised embedding of the object at (x, y), from two crops."""
    import torch
    m = _load_clip()
    crops = [_square_crop(img_bgr, x, y, 1.5 * size_px), _square_crop(img_bgr, x, y, 1.0 * size_px)]
    rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.ndim == 3 else cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)
           for c in crops]
    with torch.no_grad():
        inp = m["proc"](images=rgb, return_tensors="pt")
        ie = m["model"].get_image_features(**inp)
        if not isinstance(ie, torch.Tensor):
            ie = ie.pooler_output if getattr(ie, "pooler_output", None) is not None else ie[0]
        ie = ie / ie.norm(dim=-1, keepdim=True)
    ie = ie.numpy().mean(0)
    return ie / np.linalg.norm(ie)


def clip_probs(img_bgr, x, y, size_px, emb=None):
    """CLIP's probability for every candidate (known objects, name-only
    objects, and not-an-object) for the object at (x, y). Returns (keys,
    kinds, p, image embedding) with kinds "preset", "name" or "none"."""
    m = _load_clip()
    ie = emb if emb is not None else clip_image_emb(img_bgr, x, y, size_px)
    logits = m["scale"] * (m["emb"] @ ie)
    p = np.exp(logits - logits.max())
    return list(m["keys"]), list(m["kinds"]), p / p.sum(), ie


def clip_name_logp(image_embs: list, names: list) -> np.ndarray:
    """log P(name | object) for any names you typed, known or not, against
    each object's picture, with the whole vocabulary as the alternatives.
    Returns an (objects x names) array."""
    m = _load_clip()
    groups = []
    for nm in names:
        k = resolve(nm)
        if k in CLIP_PHRASES:
            groups.append(CLIP_PHRASES[k])
        elif k in CATALOG:
            groups.append([CATALOG[k]["phrase"]])
        else:
            groups.append([("an " if nm[:1].lower() in "aeiou" else "a ") + nm])
    te = _text_emb(groups)
    out = np.zeros((len(image_embs), len(names)))
    for j, ie in enumerate(image_embs):
        lg_names = m["scale"] * (te @ ie)
        lg_all = np.concatenate([m["scale"] * (m["emb"] @ ie), lg_names])
        mx = lg_all.max()
        out[j] = lg_names - (mx + np.log(np.exp(lg_all - mx).sum()))
    return out


def clip_identify(img_bgr, x, y, size_px, shape_hint: dict | None = None,
                  probs=None) -> dict | None:
    """Name the object at (x, y) in img_bgr with CLIP.

    CLIP gives a probability for every candidate. The size used as the ruler
    is averaged over the likely candidates (each at least a tenth as likely
    as the best), and its error bar grows with how much their sizes differ:
    torn between a tennis ball and a pickleball costs 5%, between a
    ping-pong ball and a basketball far more.

    shape_hint: the outline's verdict (shape.classify); a clearly round or
    clearly not-round outline makes the other kind of object less likely.
    Returns an identity dict; "rejected": True when it isn't an object at all
    (a hand, a shadow, an empty patch of wall).
    """
    keys, kinds, p = (probs if probs is not None else clip_probs(img_bgr, x, y, size_px))[:3]
    p = np.array(p, float)
    if shape_hint and shape_hint.get("sure"):
        want = shape_hint.get("shape")
        for i, (k, kind) in enumerate(zip(keys, kinds)):
            if kind == "preset" and shape_of(k) != want:
                p[i] *= 0.25
    p /= p.sum()
    order = np.argsort(-p)

    def nm(k):
        e = entry(k)
        return e["label"] if e else re.sub(r"^an? ", "", k)
    detail = "CLIP: " + ", ".join(f"{nm(keys[i])} {100 * p[i]:.0f}%" for i in order[:3])
    i0 = int(order[0])
    base = {"key": None, "label": None, "size_m": None, "rel_unc": None, "vt": None,
            "shape": None, "source": "CLIP", "confidence": float(p[i0]), "detail": detail}
    pres = [i for i in order if kinds[i] == "preset"]
    top = pres[0] if pres else None
    none_mass = float(sum(p[i] for i in range(len(p)) if kinds[i] == "none"))
    p_top = float(p[top]) if top is not None else 0.0
    if (kinds[i0] == "none" and p[i0] >= 0.2) or none_mass >= max(0.3, p_top):
        worst = max((i for i in range(len(p)) if kinds[i] == "none"), key=lambda i: p[i])
        return {**base, "rejected": True, "label": nm(keys[worst])}
    if kinds[i0] == "none":
        return {**base, "unsure": True}
    if top is None or p[top] < C.CLIP_MIN_PROB * 0.85 or p[top] < p[i0] * 0.5:
        if kinds[i0] == "name" and p[i0] >= 0.3:
            return {**base, "label": keys[i0]}
        return {**base, "unsure": True}
    likely = [i for i in pres if p[i] >= 0.1 * p[top]]
    w = p[likely] / p[likely].sum()
    ls = np.log([entry(keys[i])["size_m"] for i in likely])
    mu = float(np.sum(w * ls))
    spread = float(np.sqrt(np.sum(w * (ls - mu) ** 2)))
    own = float(np.sqrt(np.sum(w * np.array([entry(keys[i])["rel_unc"] ** 2
                                                 for i in likely]))))
    out = from_preset(keys[top], source="CLIP", confidence=float(p[top]), detail=detail)
    if len(likely) > 1:
        out["size_m"] = float(np.exp(mu))
        out["rel_unc"] = float(np.hypot(own, spread))
        out["detail"] += (f"; size averaged over the likely ones: {1000 * np.exp(mu):.0f} mm "
                          f"±{100 * out['rel_unc']:.0f}%")
    return out


def gemini_identify(view_img, xyd, shape_hint: dict | None = None,
                    hint: str | None = None, crops: list | None = None) -> dict | None:
    """Gemini's answer, matched to a known object when it names one. hint: what
    you said it is, so Gemini sizes that rather than guessing the name.
    crops: full-resolution close-ups of it, to look at closely."""
    from . import gemini
    g = gemini.identify_object(view_img, xyd, hint=hint, crops=crops)
    if not g:
        return None
    key = resolve(g.get("object"))
    conf = float(g.get("confidence", 0.5))
    detail = f"Gemini: {g.get('object')} ({100 * conf:.0f}%)"
    if g.get("reason"):
        detail += f", {g['reason']}"
    round_ = bool(g.get("is_round", True))
    size = g.get("size_m")
    alts = _alternatives(g)
    if alts:
        detail += "; could also be " + ", ".join(
            f"{a['label']} ({1000 * a['size_m']:.0f} mm, {100 * a['p']:.0f}%)" for a in alts)
    if key:
        e = entry(key)
        if not size or abs(math.log(float(size) / e["size_m"])) < math.log(1.3):
            return {**from_preset(key, source="Gemini", confidence=conf, detail=detail),
                    "alternatives": alts}
        # The catalog's typical one is another kind (a pencil eraser, not a
        # whiteboard eraser): keep Gemini's size for the one it looked at.
        rel = float(g.get("size_rel_uncertainty", 0.1))
        return {"key": key, "label": str(g.get("object") or e["label"]), "size_m": float(size),
                "rel_unc": rel if round_ else max(rel, C.RIGID_MIN_REL_UNC),
                "vt": g.get("terminal_speed_m_s") or e.get("vt"),
                "shape": "round" if round_ else "rigid", "source": "Gemini",
                "confidence": conf, "alternatives": alts,
                "detail": detail + f"; sized by Gemini, not as a typical {e['label']} "
                                   f"({1000 * e['size_m']:.0f} mm)"}
    if not size:
        return {"key": None, "label": g.get("object"), "size_m": None, "source": "Gemini",
                "confidence": conf, "detail": detail, "shape": "round" if round_ else "rigid"}
    rel = float(g.get("size_rel_uncertainty", 0.1))
    return {"key": None, "label": str(g.get("object", "object")), "size_m": float(size),
            "rel_unc": rel if round_ else max(rel, C.RIGID_MIN_REL_UNC),
            "vt": g.get("terminal_speed_m_s"), "shape": "round" if round_ else "rigid",
            "source": "Gemini", "confidence": conf, "detail": detail, "alternatives": alts}


ALT_MIN_P = 0.05     # pooled probability below this: noise, not a look-alike


def _alternatives(g) -> list:
    """The other things Gemini says it could plausibly be, with sizes (the
    catalog's typical size when it knows the thing and Gemini's is close)."""
    out = []
    for a in g.get("alternatives") or []:
        p, s = float(a.get("probability") or 0), float(a.get("size_m") or 0)
        if p < ALT_MIN_P or s <= 0:
            continue
        k = resolve(a.get("object"))
        if k and abs(math.log(s / entry(k)["size_m"])) < math.log(1.3):
            s = entry(k)["size_m"]
        out.append({"label": str(a["object"]), "size_m": s, "p": p})
    return out


def same_as(objs: list, j: int) -> dict | None:
    """Object j looks like object i (same shape, within 12% in apparent size):
    assume it is the same kind (two of the same ball). objs: dicts with
    "identity", "shape", "size_orig_px"."""
    me = objs[j]
    for i, o in enumerate(objs):
        idn = o.get("identity") or {}
        if i == j or not idn.get("size_m") or idn.get("source", "").startswith("same as"):
            continue
        if o.get("shape") != me.get("shape"):
            continue
        a, b = o.get("size_orig_px"), me.get("size_orig_px")
        if not a or not b or abs(b / a - 1) > 0.12:
            continue
        out = dict(idn)
        out.update(source=f"same as object {i + 1}", confidence=0.6 * idn.get("confidence", 1.0),
                   detail=f"looks like object {i + 1} ({idn['label']}): same shape, "
                          f"{100 * abs(b / a - 1):.0f}% apart in size")
        # The guess adds its own doubt: they may be different objects that look alike.
        out["rel_unc"] = float(np.hypot(idn.get("rel_unc") or 0.0, 0.05))
        return out
    return None


def p_of(probs, key) -> float:
    """CLIP's probability that an object is `key` (0 if CLIP didn't run)."""
    if not probs:
        return 0.0
    ks, p = probs[0], probs[2]
    return float(p[ks.index(key)]) if key in ks else 0.0


def match_names(names: list, logp) -> list:
    """Which object is which, when you named several: the assignment of your
    names to objects that makes CLIP's answers most likely. logp: an
    (objects x names) array from clip_name_logp. Returns, per name, the
    object index it belongs to (None if there were more names than objects)."""
    from itertools import permutations
    logp = np.asarray(logp, float)
    n_obj = logp.shape[0] if logp.ndim == 2 else 0
    if not names or n_obj == 0:
        return [None] * len(names)
    best, best_s = None, -np.inf
    m = min(len(names), n_obj)
    for perm in permutations(range(n_obj), m):
        sc = sum(logp[j, q] for q, j in enumerate(perm))
        if sc > best_s:
            best, best_s = perm, sc
    return list(best) + [None] * (len(names) - m)
