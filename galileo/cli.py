"""Command line: python -m galileo <command>

  analyze CLIP        measure one clip and print the verdict
  serve               web app (+ live tower loop with --tower or --fake-tower)
  synth               render rehearsal clips with known physics
  twin                make the AI twin of a tower photo with Veo
  calibrate           compare video g with tower g over logged drops
  ports               list serial ports
  monitor             print raw lines from the tower
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import config as C


def _roi(s):
    if not s:
        return None
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("ROI is x0,y0,x1,y1 as fractions of width/height")
    return v


def cmd_analyze(a):
    from .analyze import analyze
    size_m = a.size_mm / 1000 if a.size_mm else None
    r = analyze(a.clip, preset=a.object, size_m=size_m, size_rel_unc=a.size_unc,
                roi=a.roi, t_start=a.start, t_end=a.end, use_gemini=a.gemini,
                use_depth=a.depth, tracker="sam2" if a.sam2 else "classical",
                scale_correction=a.scale_correction,
                progress=(lambda s: print(f"  {s}...", file=sys.stderr)) if a.verbose else None)
    v = r["verdict"]
    print(f"\n{v['verdict']}  ({v.get('confidence', 0):.0%} confidence)")
    print(v["headline"])
    for line in v["reasons"]:
        print("  -", line)
    if v.get("explanation"):
        print("\n" + v["explanation"])
    clk = r.get("clock") or {}
    if clk.get("source"):
        print(f"\nClock: {clk['source']} -> k = {clk['k']:.2f}")
    if r.get("scale"):
        for ru in r["scale"]["rulers"]:
            print(f"Ruler: {ru['name']}: {ru['detail']} (±{ru['rel_sigma'] * 100:.1f}%)")
    for n in r.get("notes", []):
        print("Note:", n)
    print(f"({r['elapsed_s']:.1f} s)")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(r, fh, indent=1, default=str)
        print("Full result:", a.json)


def cmd_serve(a):
    import uvicorn
    from .web.app import create_app
    app = create_app(data_dir=a.data, tower_port=a.tower, fake_tower=a.fake_tower,
                     camera=a.camera, cam_size=(a.width, a.height), cam_fps=a.fps,
                     preset=a.object, use_gemini=a.gemini, calibration=a.calibration)
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

    s = sub.add_parser("analyze", help="measure one clip")
    s.add_argument("clip")
    s.add_argument("--object", default=None, choices=sorted(C.OBJECT_PRESETS),
                   help="known object to use as the ruler")
    s.add_argument("--size-mm", type=float, help="object diameter in mm (instead of --object)")
    s.add_argument("--size-unc", type=float, default=0.01, help="relative size uncertainty")
    s.add_argument("--roi", type=_roi, help="x0,y0,x1,y1 as fractions, e.g. 0.3,0,0.7,1")
    s.add_argument("--start", type=float)
    s.add_argument("--end", type=float)
    s.add_argument("--gemini", action="store_true", help="Gemini names the object and explains")
    s.add_argument("--depth", action="store_true", help="add Depth Pro as a second ruler")
    s.add_argument("--sam2", action="store_true", help="track with SAM 2 instead")
    s.add_argument("--scale-correction", type=float, default=1.0)
    s.add_argument("--json", help="write the full result here")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_analyze)

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
    s.add_argument("--object", default="steel-38mm", choices=sorted(C.OBJECT_PRESETS))
    s.add_argument("--gemini", action="store_true")
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
