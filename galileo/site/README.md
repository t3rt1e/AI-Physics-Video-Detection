# Galileo website

This folder is the website layer. The detector, physics, object presets, and
classification decisions stay in the main `galileo` package.

## Run

From the repository root (the folder containing `requirements.txt`):

```bash
pip install -r requirements.txt
python -m galileo.site
```

Open http://localhost:8080. The existing `--host`, `--port`, and `--data` flags
still work. Existing analyses in `site_data` remain readable.

No new production dependencies were added. Automatic object identification
continues to use the main project's optional CLIP/Gemini setup. If it isn't
available, the upload form lets you choose an object or enter its size.

## What's new

- A new Galileo orbit logo, neutral surfaces, and small orange/blue accents.
- Light and dark themes. The initial theme follows the operating system; an
  explicit choice is saved in the browser. Motion respects the reduced-motion
  accessibility preference, and the site does not require external fonts.
- Compare two uploads, with independent object/size/camera options. Each clip
  becomes a normal analysis job; there is no separate detector for comparisons.
- Linked playback, shared seeking/speed/frame stepping, and independent controls
  when unlinked. Different frame rates retain their original timestamps. Linked
  frame stepping follows the longer clip; the shorter clip holds its final
  frame. Linking uses elapsed video time, not inferred matching events or
  automatic release-time alignment.
- Select any two analyses in History to compare them without analyzing again.
  Comparison URLs contain both job IDs and work after reloading while those jobs
  remain on the same server. Failed uploads can be retried without resubmitting
  the other successful upload.
- Results focus on the verdict, measured gravity, detected object, video, and a
  brief explanation. Technical details are collapsed initially. Opening them
  reveals overlay toggles, debug playback, the gravity gauge, charts, every
  diagnostic tab, the unmodified result JSON, and downloads. Unknown/new result
  fields remain accessible in **All data**.

The display adds "Likely" to REAL/AI labels to describe the detector's evidence;
it does not change the underlying verdict or threshold. Synthetic examples are
explicitly labeled as synthetic.

## Merge into a newer checkout

**Copy only `galileo/site/` from this ZIP into the latest checkout.** Everything
outside this folder is the exact uploaded snapshot; do not replace a newer
main codebase with that snapshot. Restart the server after updating Python
code. Refresh the browser after updating static files.

No files outside `galileo/site/` were edited. In particular, the detector,
`galileo/web`, root README, requirements, and existing tests are unchanged.

## Boundary with the main codebase

`pipeline.py` calls the public `galileo.analyze.analyze(...)` function at job
execution time. It forwards the selected preset or custom size, camera field of
view, seed, debug-video output path, and progress callback. It does not call
`_run`, inspect the private `dbg` dictionary, copy the detector's implementation,
or infer new verdicts. Results pass through without filtering out new fields.

`app.py` continues to read presets and the Earth-gravity reference directly from
`galileo.config`, and uses the main project's `video.probe`, `iter_frames`, and
`Writer` for previews. New compatible core changes are picked up when the server
restarts. Public API or result-schema breaking changes may still require updates
to this small adapter or the corresponding UI renderer; no frontend can promise
compatibility with arbitrary future breaking changes.

The documented public analysis API doesn't expose the private source-to-debug
frame map. The site therefore reads timestamps from the generated debug video
and synchronizes it only when its complete timeline matches the full preview.
For cropped, downsampled, or otherwise different debug timelines, **Technical
details > Debug video** provides independent playback and frame-step controls.
It never guesses the alignment. Old saved jobs with an exact debug mapping keep
using that mapping. The original debug MP4 remains downloadable.

The preview limit remains 6,000 frames. A truncated preview is marked in the
technical overview; it does not change the core's independent analysis limits.

## Files to know

| File | Responsibility |
| --- | --- |
| `static/index.html`, `static/logo.svg` | App shell and brand |
| `static/app.css`, `static/theme.js` | Appearance, responsive layouts, theme |
| `static/app.js` | Single-clip analysis, results, diagnostic tabs, history |
| `static/compare.js` | Shared upload inputs, comparison setup, paired playback |
| `pipeline.py` | Public detector adapter and safe debug timeline mapping |
| `app.py` | Existing job/media API and previews |
| `tests/test_pipeline.py` | Adapter contract and timeline safety checks |

## Verification for this update

Run the added site tests:

```bash
pip install -r requirements-dev.txt
python -m pytest galileo/site/tests -q
```

14 contract checks passed. These cover forwarding to the public analyzer,
retaining future result fields, newly added presets, camera-input validation,
and matching/mismatched/broken debug timelines.

Headless Chromium checks exercised:

- Light/dark rendering and saved theme after reload.
- Desktop and 390-pixel phone layouts without horizontal page overflow.
- Two actual multipart uploads with independent object settings.
- Comparison reload, linked play/pause/seeking/frame stepping, different frame
  rates and lengths, and independent stepping.
- A failed second upload and retrying only that upload.
- Single upload, invalid-file handling, and custom-size validation.
- Selecting results in History, hidden diagnostics, tab keyboard navigation,
  synchronized debug frame stepping, and independent debug timeline fallback.

End-to-end samples used the main pipeline (not stubbed measurements):

| Synthetic clip | Core verdict | Measured gravity |
| --- | --- | --- |
| Earth-gravity drop, 60 fps | REAL | about 9.88 m/s² |
| Low-gravity drop, 24 fps | AI | about 2.34 m/s² |

These checks validate the website and integration. They do not establish
accuracy on real camera footage or generated footage. CLIP and Gemini inference
were not available in this test environment; their core implementation was not
modified.
