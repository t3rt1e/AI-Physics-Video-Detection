"""Command line: python -m galileo <command>

  analyze CLIP        measure one clip and print the verdict
  batch FOLDER        measure every clip in a folder and write a CSV
  serve               web app (+ live tower loop with --tower or --fake-tower)
  synth               render rehearsal clips with known physics
  twin                make the AI twin of a tower photo with Veo
  calibrate           compare video g with tower g over logged drops
  ports               list serial ports
  monitor             print raw lines from the tower
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

from . import config as C


def _numbers(tokens):
    """'1,2,3' or 1 2 3 (PowerShell splits unquoted commas into separate words)."""
    if not tokens:
        return None
    if isinstance(tokens, str):
        tokens = [tokens]
    parts = [p for tok in tokens for p in str(tok).replace(";", ",").split(",") if p.strip()]
    return [float(p) for p in parts]


def _roi(tokens):
    v = _numbers(tokens)
    if v is None:
        return None
    if len(v) != 4:
        raise SystemExit("--roi is x0,y0,x1,y1 as fractions of width/height, e.g. "
                         '--roi "0.3,0,0.7,1"')
    return v


def _seed(tokens):
    try:
        v = _numbers(tokens)
    except ValueError:
        v = []
    if v is None:
        return None
    if len(v) != 3:
        raise SystemExit("--seed is T,X,Y: the time in seconds when the ball is in the air, "
                         "then its position as fractions of the width and height (0.5,0.3) or "
                         'in pixels (960,320), e.g. --seed "1.25,0.48,0.33"')
    return tuple(v)


def _objects(tokens):
    """--object names, one per object: a known name ("tennis", "rotisserie
    chicken"), anything else with its size ("grandma's vase=31cm"), or auto.
    Commas separate objects too (PowerShell splits them anyway)."""
    if not tokens:
        return None
    from .identify import parse_name, resolve
    names = [p.strip() for tok in tokens for p in str(tok).replace(";", ",").split(",")
             if p.strip()]
    out = []
    for n in names:
        if n.lower() in ("auto", "?"):
            out.append(None)
            continue
        name, size = parse_name(n)
        if not resolve(name) and not size:
            print(f"Note: '{name}' isn't a known object, so it has no size. Add one like "
                  f'"{name}=22cm" for a ruler.', file=sys.stderr)
        out.append(n)
    return out


def _gemini_on(a) -> bool:
    """--gemini / --no-gemini, else on whenever a key is set and the library is in."""
    if a.gemini is not None:
        return bool(a.gemini)
    from .gemini import available
    return available()[0]


def _analyze_kwargs(a):
    return dict(preset=_objects(a.object), size_m=a.size_mm / 1000 if a.size_mm else None,
                size_rel_unc=a.size_unc, use_gemini=_gemini_on(a), use_depth=a.depth,
                tracker="sam2" if a.sam2 else "classical",
                scale_correction=a.scale_correction, zoom=not a.no_zoom, drag_vt=a.drag_vt,
                identify_mode=a.identify, max_objects=a.max_objects, hfov=a.hfov,
                camera_pitch=a.pitch, camera_roll=a.roll,
                camera_mode="upright" if a.upright else "auto",
                real_time=False if a.slowmo else None)


def _camera_line(r) -> str | None:
    """One line on how the camera was held, when it matters."""
    c = r.get("camera") or {}
    parts = []
    if c.get("pitched"):
        b = c.get("pitch") or 0.0
        parts.append(f"looks {'down' if b > 0 else 'up'} {abs(b):.0f}° "
                     f"(± {max(c.get('pitch_sigma') or 0, 0.5):.0f}°, from {c.get('pitch_source')})")
    if c.get("rolled"):
        parts.append(f"rolled {c.get('roll', 0):.0f}°")
    if not parts:
        return None
    fr = c.get("f_frac") or 0
    hfov = 2 * math.degrees(math.atan(0.5 / fr)) if fr else 0
    return (f"Camera: {', '.join(parts)}; field of view {hfov:.0f}° ({c.get('f_source')}). "
            "Measured in a level view.")


def _obj_line(o) -> str:
    v = o["verdict"]
    name = o.get("label") or "unknown object"
    idn = o.get("identity") or {}
    src = idn.get("source")
    how = {"you": "you said", None: "not identified"}.get(src, src)
    if src in ("CLIP", "Gemini") and idn.get("confidence") is not None:
        how += f", {100 * idn['confidence']:.0f}% sure"
    shape = "round" if o.get("shape") == "round" else "not round"
    g = f"g = {v['g']:.2f} +/- {v['sigma']:.2f} m/s^2" if v.get("g") is not None else "no g"
    line = f"Object {o['n']}: {name} ({shape}; {how}): {v['verdict']}, {g}"
    if v["verdict"] == "UNSURE":
        line += f" [{v.get('stage')}]"
    sp = o.get("spin") or {}
    if sp.get("applied"):
        line += f"; centre of mass {sp['offset_px']:.0f} px off its outline's centre"
    return line


def _debug_path(clip, where):
    """Where the debug video goes: a file, a folder, or next to the clip."""
    stem = os.path.splitext(os.path.basename(clip))[0] + "_debug.mp4"
    if where in (None, "", "auto"):
        return os.path.join(os.path.dirname(os.path.abspath(clip)), stem)
    if os.path.isdir(where) or where.endswith(("/", "\\")):
        os.makedirs(where, exist_ok=True)
        return os.path.join(where, stem)
    return where


def cmd_analyze(a):
    from .analyze import analyze
    dbg = _debug_path(a.clip, a.debug_video) if a.debug_video is not None else None
    r = analyze(a.clip, roi=_roi(a.roi), t_start=a.start, t_end=a.end, seed=_seed(a.seed),
                progress=(lambda s: print(f"  {s}...", file=sys.stderr)) if a.verbose else None,
                debug_video=dbg, debug_speed=a.debug_speed, **_analyze_kwargs(a))
    v = r["verdict"]
    print(f"\n{v['verdict']}  ({v.get('confidence', 0):.0%} confidence)")
    print(v["headline"])
    for line in v["reasons"]:
        print("  -", line)
    objs = r.get("objects") or []
    if len(objs) > 1 or (objs and objs[0].get("shape") != "round"):
        print()
        for o in objs:
            print(_obj_line(o))
    if v.get("explanation"):
        print("\n" + v["explanation"])
    if v.get("stage") and v["verdict"] == "UNSURE":
        print(f"\nStopped at: {v['stage']}")
    clk = r.get("clock") or {}
    if clk.get("source"):
        print(f"\nClock: {clk['source']} -> k = {clk['k']:.2f}")
    cam_line = _camera_line(r)
    if cam_line:
        print(cam_line)
    if r.get("scale"):
        for ru in r["scale"]["rulers"]:
            print(f"Ruler: {ru['name']}: {ru['detail']} (±{ru['rel_sigma'] * 100:.1f}%)")
    z = r.get("zoom")
    if z:
        print(f"Zoom: ball {z['ball_px_before']:.0f} px in the first pass, {z['ball_px']:.0f} px "
              f"measured at {z['scale'] * 100:.0f}% resolution")
    tr = (r.get("track") or {}).get("points") or []
    if tr:
        mid = tr[len(tr) // 2]
        print(f"Tracked {len(tr)} points, e.g. t = {mid['t']:.2f} s at x = {mid['x']:.2f}, "
              f"y = {mid['y']:.2f} (wrong object? rerun with --seed T,X,Y on the ball)")
    for n in r.get("notes", []):
        print("Note:", n)
    print(f"({r['elapsed_s']:.1f} s)")
    if r.get("debug_video"):
        print("Debug video:", r["debug_video"])
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(r, fh, indent=1, default=str)
        print("Full result:", a.json)


VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")


def cmd_batch(a):
    """Measure many clips, one row each, so misreads are easy to sort."""
    from .analyze import analyze
    paths = []
    for item in a.inputs:
        if os.path.isdir(item):
            for p in sorted(glob.glob(os.path.join(item, "*"))):
                if p.lower().endswith(VIDEO_EXT):
                    paths.append(p)
        else:
            paths.extend(sorted(glob.glob(item)) or [item])
    if not paths:
        print("No video files found.")
        return
    if a.json_dir:
        os.makedirs(a.json_dir, exist_ok=True)
    cols = ["file", "verdict", "confidence", "g", "sigma", "g_as_played", "k", "stage",
            "object", "ball_px", "zoomed", "frames", "fps", "flights", "drag_pct",
            "camera_pitch", "camera_roll", "other_objects", "headline", "notes", "debug_video"]
    rows = []
    kw = _analyze_kwargs(a)
    for i, p in enumerate(paths, 1):
        name = os.path.basename(p)
        print(f"[{i}/{len(paths)}] {name} ... ", end="", flush=True)
        try:
            r = analyze(p, debug_video=_debug_path(p, a.debug_dir) if a.debug_dir else None,
                        debug_speed=a.debug_speed, **kw)
        except Exception as e:  # keep going: one bad file shouldn't stop the batch
            print(f"error: {e}")
            rows.append({"file": name, "verdict": "ERROR", "headline": str(e)})
            continue
        v = r["verdict"]
        fit = r.get("fit") or {}
        work = r.get("work") or {}
        drag = r.get("drag") or {}
        row = {
            "file": name, "verdict": v["verdict"], "confidence": round(v.get("confidence") or 0, 2),
            "g": None if v.get("g") is None else round(v["g"], 3),
            "sigma": None if v.get("sigma") is None else round(v["sigma"], 3),
            "g_as_played": None if v.get("g_naive") is None else round(v["g_naive"], 3),
            "k": v.get("k"), "stage": v.get("stage"),
            "ball_px": round(fit["diam_px"], 1) if fit.get("diam_px") else None,
            "zoomed": bool(r.get("zoom")), "frames": work.get("frames"),
            "fps": round(work["fps_measured"], 2) if work.get("fps_measured") else None,
            "flights": " ".join(str(fl["n"]) for fl in fit.get("flights", [])),
            "drag_pct": round(100 * (drag["factor"] - 1), 1) if drag else None,
            "headline": v.get("headline"), "notes": " | ".join(r.get("notes", [])),
            "debug_video": r.get("debug_video"),
            "object": (r.get("objects") or [{}])[0].get("label"),
            "camera_pitch": round((r.get("camera") or {}).get("pitch") or 0.0, 1)
            if (r.get("camera") or {}).get("pitched") else 0.0,
            "camera_roll": round((r.get("camera") or {}).get("roll") or 0.0, 1)
            if (r.get("camera") or {}).get("rolled") else 0.0,
            "other_objects": " | ".join(
                f"{o['n']}: {o.get('label') or '?'} {o['verdict']['verdict']}"
                + (f" g={o['g']:.2f}" if o.get("g") is not None else "")
                for o in (r.get("objects") or [])[1:]),
        }
        rows.append(row)
        g = f"g = {row['g']:.2f}" if row["g"] is not None else "no g"
        print(f"{row['verdict']} ({g}, {row['stage']}, {r['elapsed_s']:.1f} s)")
        if a.json_dir:
            out = os.path.join(a.json_dir, os.path.splitext(name)[0] + ".json")
            with open(out, "w") as fh:
                json.dump(r, fh, indent=1, default=str)
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    counts = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print("\n" + ", ".join(f"{k}: {n}" for k, n in sorted(counts.items())))
    if a.expect:
        want = {"real": ("REAL", "SLOWMO"), "ai": ("AI",)}[a.expect]
        good = sum(1 for row in rows if row["verdict"] in want)
        print(f"Expected {a.expect}: {good} of {len(rows)} right. Rows that missed:")
        for row in rows:
            if row["verdict"] not in want:
                print(f"  {row['file']}: {row['verdict']} ({row.get('stage')}) {row.get('headline')}")
    print(f"Wrote {a.csv}")


def cmd_serve(a):
    import uvicorn
    from .web.app import create_app
    live = bool(a.tower or a.fake_tower or a.camera is not None)
    obj = a.object or ("steel-38mm" if live else "auto")   # uploads: work it out
    app = create_app(data_dir=a.data, tower_port=a.tower, fake_tower=a.fake_tower,
                     camera=a.camera, cam_size=(a.width, a.height), cam_fps=a.fps,
                     preset=obj, use_gemini=_gemini_on(a), calibration=a.calibration)
    print(f"\nGalileo on http://localhost:{a.port}  (data in {os.path.abspath(a.data)})\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def cmd_synth(a):
    from .synth import Motion, Scene, ease_in_out, make_clip
    os.makedirs(a.out, exist_ok=True)
    W, H = 1280, 720
    sc = lambda **k: Scene(**{"width": W, "height": H, "px_per_m": 420.0, "y0": 60.0, **k})  # noqa: E731
    jobs = [
        ("synthetic_real_60fps.mp4", "real drop, 60 fps webcam with rolling shutter",
         dict(scene=sc(), motion=Motion(g=9.793), capture_fps=60, duration=1.1, rolling_shutter=0.5)),
        ("synthetic_ai_floaty_24fps.mp4", "AI-style drop at 2.3 m/s^2 (Veo 3's average)",
         dict(scene=sc(), motion=Motion(g=2.3, t_release=0.4), capture_fps=24, duration=2.0)),
        ("synthetic_ai_easing_24fps.mp4", "AI-style drop with animation easing",
         dict(scene=sc(), motion=Motion(accel_fn=ease_in_out(1.4, 1.1), t_release=0.3),
              capture_fps=24, duration=1.8)),
        ("synthetic_slowmo_240fps_native.mp4", "240 fps capture with real timestamps",
         dict(scene=sc(), motion=Motion(g=9.793, t_release=0.05), capture_fps=240, duration=0.75)),
        ("synthetic_slowmo_8x_tagged.mp4", "240 fps capture played at 30 fps, tagged",
         dict(scene=sc(), motion=Motion(g=9.793, t_release=0.05), capture_fps=240, duration=0.75,
              slowmo={"k": 8, "method": "retime"}, tag_capture_fps=True)),
        ("synthetic_slowmo_4x_repeated.mp4", "30 fps slowed 4x by repeating frames",
         dict(scene=sc(), motion=Motion(g=9.793), capture_fps=30, duration=1.0,
              slowmo={"k": 4, "method": "dup"})),
        ("synthetic_slowmo_4x_interpolated.mp4", "30 fps slowed 4x by interpolation",
         dict(scene=sc(), motion=Motion(g=9.793), capture_fps=30, duration=1.0,
              slowmo={"k": 4, "method": "interp"})),
        ("synthetic_bounce_60fps.mp4", "real drop with bounces",
         dict(scene=sc(floor_y=640.0), motion=Motion(g=9.793, restitution=0.65, t_release=0.1),
              capture_fps=60, duration=1.8, rolling_shutter=0.3)),
    ]
    for name, desc, kw in jobs:
        t0 = time.time()
        make_clip(os.path.join(a.out, name), **kw)
        with open(os.path.join(a.out, os.path.splitext(name)[0] + ".label.json"), "w") as fh:
            json.dump({"synthetic": True, "description": desc,
                       "ball": "38 mm steel (object preset steel-38mm)"}, fh)
        print(f"  {name:40s} {desc} ({time.time() - t0:.1f} s)")
    print(f"\nRehearsal clips in {a.out}. Analyse with --object steel-38mm.")


def cmd_twin(a):
    from .gemini import generate_twin
    generate_twin(a.image, a.out, prompt=a.prompt, duration_s=a.seconds)


def cmd_calibrate(a):
    from .live import calibrate
    c = calibrate(a.csv)
    if not c.get("n"):
        print("No usable drops yet (need rows with both tower g and a REAL video g).")
        return
    print(f"{c['n']} drops: video g / tower g median {c['median_ratio']:.4f}; mean error "
          f"{c['mean_error_pct']:+.2f}%; spread {c['spread_pct']:.2f}%; worst {c['worst_pct']:.2f}%")
    print(f"Pitch line: 'Across {c['n']} tower drops our video g matched the tower to "
          f"{c['spread_pct']:.1f}% (1 sigma).'")
    if a.save:
        with open(a.save, "w") as fh:
            json.dump({"scale_correction": c["scale_correction"], "n": c["n"],
                       "spread_pct": c["spread_pct"]}, fh, indent=1)
        print(f"Saved scale correction {c['scale_correction']:.4f} to {a.save}")


def cmd_ports(a):
    from .tower import list_ports
    ports = list_ports()
    print("\n".join(ports) if ports else "No serial ports found.")


def cmd_monitor(a):
    from .tower import TowerLink
    link = TowerLink(a.port, on_event=lambda ev: print(f"{ev.kind:7s} {ev.raw}")).start()
    print(f"Listening on {a.port} at 115200. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        link.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="galileo", description="Measure physics in video.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def measure_opts(s):
        s.add_argument("--object", nargs="+", metavar="NAME",
                       help="what the falling object is, for its size (the ruler) and air drag. "
                            "Any name: a known one (tennis, phone, rotisserie chicken, and a few "
                            "hundred more; see galileo/catalog.py) or anything with its size, "
                            'e.g. "grandma\'s vase=31cm" (mm, cm, m, in, ft). Several objects: '
                            "one name each, e.g. --object tennis basketball (matched to the "
                            "objects by how they look, or by size). Leave out, or say auto, to "
                            "let Galileo work it out")
        s.add_argument("--identify", default="auto", choices=["auto", "clip", "gemini", "off"],
                       help="how to name objects you didn't: auto (Gemini with --gemini, else "
                            "CLIP if installed), or off")
        s.add_argument("--max-objects", type=int, default=C.MAX_OBJECTS,
                       help="how many falling objects to measure (1 = only the main one)")
        s.add_argument("--size-mm", type=float,
                       help="object size in mm (instead of --object): a ball's diameter, or "
                            "the longest side of anything else")
        s.add_argument("--size-unc", type=float, default=0.01, help="relative size uncertainty")
        s.add_argument("--drag-vt", type=float,
                       help="terminal speed in m/s for the air-drag correction "
                            "(ping-pong 8.4; presets set it for you)")
        s.add_argument("--no-zoom", action="store_true",
                       help="skip the full-resolution second pass on small balls")
        s.add_argument("--gemini", action=argparse.BooleanOptionalAction, default=None,
                       help="Gemini names and sizes the object and explains the verdict (on "
                            "by default when GEMINI_API_KEY is set; --no-gemini turns it off)")
        s.add_argument("--depth", action="store_true", help="add Depth Pro as a second ruler")
        s.add_argument("--sam2", action="store_true", help="track with SAM 2 instead")
        s.add_argument("--scale-correction", type=float, default=1.0)
        s.add_argument("--hfov", type=float, metavar="DEG",
                       help="the camera's field of view across the picture's long side, if "
                            "known (a phone's main camera is about 65-75; it sharpens the "
                            "answer when the camera looks down or up at the fall)")
        s.add_argument("--pitch", type=float, metavar="DEG",
                       help="how far the camera looked down (+) or up (-) from level, if known "
                            "(otherwise read from the room's lines and the ball itself)")
        s.add_argument("--roll", type=float, metavar="DEG",
                       help="how far the camera was turned about its own axis, if known")
        s.add_argument("--upright", action="store_true",
                       help="assume a level, upright camera (skip working out its angle)")
        s.add_argument("--slowmo", action="store_true",
                       help="allow slow-motion and sped-up footage: read the file's clocks "
                            "(capture-rate tags, repeated or blended frames) instead of taking "
                            "the clip as real time")

    s = sub.add_parser("analyze", help="measure one clip")
    s.add_argument("clip")
    measure_opts(s)
    s.add_argument("--seed", nargs="+", metavar="T,X,Y",
                   help='point at the ball: time (s) when it is in the air, then x,y as '
                        'fractions or pixels, e.g. --seed "1.25,0.48,0.33"')
    s.add_argument("--roi", nargs="+", metavar="X0,Y0,X1,Y1",
                   help='region as fractions, e.g. --roi "0.3,0,0.7,1"')
    s.add_argument("--start", type=float)
    s.add_argument("--end", type=float)
    s.add_argument("--json", help="write the full result here")
    s.add_argument("--debug-video", nargs="?", const="auto", metavar="PATH",
                   help="also write a video showing what Galileo tracked and fitted, frame by "
                        "frame (default: CLIP_debug.mp4 next to the clip)")
    s.add_argument("--debug-speed", type=float, default=1.0,
                   help="playback speed of the debug video, e.g. 0.25 for quarter speed")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("batch", help="measure every clip in a folder, write a CSV")
    s.add_argument("inputs", nargs="+", help="folders, files, or wildcards like clips/*.mp4")
    measure_opts(s)
    s.add_argument("--csv", default="galileo_batch.csv", help="where to write the table")
    s.add_argument("--json-dir", help="also save each full result as JSON here")
    s.add_argument("--expect", choices=["real", "ai"],
                   help="what these clips are, to count the misses")
    s.add_argument("--debug-dir", help="also write a debug video for every clip into this folder")
    s.add_argument("--debug-speed", type=float, default=1.0,
                   help="playback speed of the debug videos, e.g. 0.25 for quarter speed")
    s.set_defaults(func=cmd_batch)

    s = sub.add_parser("serve", help="web app and live tower loop")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data", default="data")
    s.add_argument("--tower", help="serial port of the tower, e.g. COM5 or /dev/ttyACM0")
    s.add_argument("--fake-tower", action="store_true", help="simulate the tower")
    s.add_argument("--camera", type=int, default=None, help="webcam index for live drops")
    s.add_argument("--width", type=int, default=1280)
    s.add_argument("--height", type=int, default=720)
    s.add_argument("--fps", type=int, default=60)
    s.add_argument("--object", default=None, choices=sorted(C.OBJECT_PRESETS) + ["auto"],
                   help="what the page assumes is falling until you pick something else: "
                        "'auto' lets Galileo work it out (the default; steel-38mm when a "
                        "tower or webcam is attached)")
    s.add_argument("--gemini", action=argparse.BooleanOptionalAction, default=None,
                   help="on by default when GEMINI_API_KEY is set")
    s.add_argument("--calibration", help="JSON from 'calibrate --save' to apply to live drops")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("synth", help="render rehearsal clips")
    s.add_argument("--out", default="data/clips")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("twin", help="AI twin of a tower photo with Veo")
    s.add_argument("--image", required=True)
    s.add_argument("--out", default="data/clips/ai_twin_veo.mp4")
    s.add_argument("--prompt")
    s.add_argument("--seconds", type=int, default=8, choices=[4, 6, 8])
    s.set_defaults(func=cmd_twin)

    s = sub.add_parser("calibrate", help="video g vs tower g over logged drops")
    s.add_argument("--csv", default="data/drops.csv")
    s.add_argument("--save", help="write the scale correction to this JSON")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("ports", help="list serial ports")
    s.set_defaults(func=cmd_ports)

    s = sub.add_parser("monitor", help="print raw tower lines")
    s.add_argument("--port", required=True)
    s.set_defaults(func=cmd_monitor)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
