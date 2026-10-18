#!/usr/bin/env python3
"""
Apply a grid of perturbations (re-encode, rescale, fps, blur, grain, multi-pass
"re-upload") to BOTH real and fake videos identically, for detector robustness
evaluation. Requires ffmpeg + ffprobe on PATH.

Usage:
    python perturb_grid.py --real "your_folder" --fake "your_folder" --out perturbed_sink --only kitchen_sink

Output layout:
    perturbed/<perturbation>/<real|fake>/<clip>.<mp4|webm>
    perturbed/manifest.csv   (source, label, perturbation, output, status)

Audio is dropped (-an) since the detectors here are video-only.
"""
import argparse
import csv
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

CODECS = {
    "h264": (["-c:v", "libx264", "-pix_fmt", "yuv420p"], "mp4"),
    "h265": (["-c:v", "libx265", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
              "-x265-params", "log-level=error"], "mp4"),
    "vp9": (["-c:v", "libvpx-vp9", "-b:v", "0", "-pix_fmt", "yuv420p"], "webm"),
}


def build_grid():
    """Each perturbation is a list of stages; stages run sequentially.
    Stage keys: codec, crf, scale (down-then-up factor), fps, blur (sigma), noise (strength)."""
    grid = {}

    # Control: plain re-encode near-transparent quality. Always include this so
    # you can separate "any re-encode" effects from specific perturbations.
    grid["control_h264_crf18"] = [dict(codec="h264", crf=18)]

    # Codec x quality sweep
    for codec in CODECS:
        for crf in (23, 28, 35):
            grid[f"codec_{codec}_crf{crf}"] = [dict(codec=codec, crf=crf)]

    # Down-then-up rescale
    for f in (0.75, 0.5, 0.25):
        grid[f"rescale_{f}"] = [dict(codec="h264", crf=23, scale=f)]

    # Frame-rate change
    for fps in (24, 15, 12):
        grid[f"fps_{fps}"] = [dict(codec="h264", crf=23, fps=fps)]

    # Blur
    for s in (0.5, 1.0, 2.0):
        grid[f"blur_{s}"] = [dict(codec="h264", crf=23, blur=s)]

    # Grain / noise (temporal + uniform)
    for n in (5, 10, 20):
        grid[f"noise_{n}"] = [dict(codec="h264", crf=23, noise=n)]

    # Combinations
    grid["combo_light"] = [dict(codec="h264", crf=26, blur=0.5, noise=5)]
    grid["combo_heavy"] = [dict(codec="h264", crf=30, scale=0.5, blur=1.0, noise=10)]

    # Simulated platform re-upload: two lossy generations at different settings
    grid["reupload_2gen"] = [
        dict(codec="h264", crf=28, scale=0.75),
        dict(codec="h264", crf=30, scale=0.5, fps=24),
    ]

    # Everything stacked into one file: three lossy generations across three
    # codecs, with rescaling, fps change, blur and grain (grain re-applied
    # last so it survives the final encode). Use as a worst-case stress test.
    grid["kitchen_sink"] = [
        dict(codec="h264", crf=28, scale=0.75, blur=0.7, noise=10),
        dict(codec="vp9", crf=33, scale=0.5, fps=24),
        dict(codec="h264", crf=30, noise=6),
    ]
    return grid


def probe_size(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(path)])
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]


def run_stage(src, dst, st, w, h):
    vf = []
    if st.get("scale"):
        f = st["scale"]
        vf.append(f"scale=trunc(iw*{f}/2)*2:trunc(ih*{f}/2)*2")
        vf.append(f"scale={w}:{h}")
    if st.get("fps"):
        vf.append(f"fps={st['fps']}")
    if st.get("blur"):
        vf.append(f"gblur=sigma={st['blur']}")
    if st.get("noise"):
        # fixed seed so grain is reproducible across runs
        vf.append(f"noise=alls={st['noise']}:allf=t+u:all_seed=1234")

    codec_args, _ = CODECS[st["codec"]]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += codec_args + ["-crf", str(st["crf"]), str(dst)]
    subprocess.run(cmd, check=True)


def run_job(job):
    src, label, name, stages, out_root = job
    final_ext = CODECS[stages[-1]["codec"]][1]
    dst = out_root / name / label / f"{src.stem}.{final_ext}"
    if dst.exists():
        return (str(src), label, name, str(dst), "skipped")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        w, h = probe_size(src)
        w, h = w - w % 2, h - h % 2
        with tempfile.TemporaryDirectory() as tmp:
            cur = src
            for i, st in enumerate(stages):
                last = i == len(stages) - 1
                ext = CODECS[st["codec"]][1]
                nxt = dst if last else Path(tmp) / f"stage{i}.{ext}"
                run_stage(cur, nxt, st, w, h)
                cur = nxt
        return (str(src), label, name, str(dst), "ok")
    except Exception as e:
        return (str(src), label, name, str(dst), f"error: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, type=Path)
    ap.add_argument("--fake", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="+",
                    help="run only these perturbation names, e.g. --only kitchen_sink")
    args = ap.parse_args()

    grid = build_grid()
    if args.only:
        unknown = [n for n in args.only if n not in grid]
        if unknown:
            raise SystemExit(f"unknown perturbation(s): {unknown}\navailable: {list(grid)}")
        grid = {k: v for k, v in grid.items() if k in args.only}
    jobs = []
    for label, folder in (("real", args.real), ("fake", args.fake)):
        for src in sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS):
            for name, stages in grid.items():
                jobs.append((src, label, name, stages, args.out))

    print(f"{len(jobs)} jobs ({len(grid)} perturbations)")
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_job, jobs))

    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "label", "perturbation", "output", "status"])
        w.writerows(results)

    failed = [r for r in results if r[4].startswith("error")]
    print(f"done: {len(results) - len(failed)} ok/skipped, {len(failed)} failed")


if __name__ == "__main__":
    main()
