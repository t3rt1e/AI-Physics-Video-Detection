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

## Gemini-first object identification

The website now matches the command line: when the core's
`galileo.gemini.available()` finds a key and the optional SDK, each new job
passes `use_gemini=True` to the public analyzer. Previously, an omitted web
option incorrectly became `False`, so automatic uploads used only CLIP.
Single uploads, both clips in a comparison, and remeasurement jobs all use the
same adapter. Explicit per-job `gemini=False`/`True` choices remain supported.

Choose **Identify automatically** in the upload form. The core prefers Gemini's
identification and falls back to CLIP when Gemini fails or returns no usable
identification. Its existing model retries, cache, manual object/size handling,
and physics remain unchanged. The core may still precompute CLIP candidate
scores before requesting Gemini; seeing CLIP load does not mean Gemini was
skipped. The result's object identity records the provider actually used.

The supplied core reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the **server
process's environment**. It does not search your filesystem for key files.
If your key is already saved as a Windows environment variable, close and
reopen the terminal (or IDE) before restarting the website so it inherits the
updated environment. Ensure the SDK is installed in that same Python environment:

```bash
python -m pip install google-genai
python -m galileo.site
```

If you instead saved the raw key in a text file, load that file into the server
environment before launching. For example, in PowerShell (replace the path):

```powershell
$env:GEMINI_API_KEY = (Get-Content -Raw 'C:\path\to\gemini-key.txt').Trim()
python -m galileo.site
```

**More options → Automatic identification status** shows whether Gemini is
available and CLIP can serve as the backup. If Gemini is unavailable, it shows
the core's setup reason even when CLIP is working. This is a configuration check;
a rejected key, network failure, or exhausted quota is handled when the core
makes its request. Credentials are never added to browser responses or job options.

Provider regression tests cover both environment-variable names, missing keys,
missing SDKs, explicit choices, and per-job checks. Three tests run the real
public analysis pipeline on a sample clip with simulated model responses:
Gemini success, Gemini error, and an empty Gemini response. These verify that
Gemini's identity is preferred and CLIP supplies the fallback without making
external API requests. A live request with your local key could not be tested here.

## What's new

- A complete visual redesign: larger editorial typography, framed panels,
  a motion-study illustration, pointer-responsive lighting, a brief text-decode
  hover, a refined logo, and restrained blue/orange accents on neutral surfaces.
  All pages use the same design, including upload, comparison, history, results,
  progress, and technical details.
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
- Result videos show the object highlight and full measured path immediately,
  without opening Technical details. A dashed line shows the complete recorded
  path; a solid line shows the portion reached during playback. Long tracking
  gaps are not joined. The highlight is drawn only on frames with a nearby
  detection. The initial single-clip view seeks to the first detected frame,
  unless the user already started interacting with the preview. A visible
  **Tracking on/off** button toggles the object and path together.
- The horizontal gravity gauge is visible beside the main video (below it on a
  phone), with the measured value, uncertainty, and Earth comparison range.
  The range uses the core result's `g_ref` and `tol`; the site does not set a new
  classification threshold. Missing measurements have an explicit empty state.
- Comparison videos also show tracking by default, including letterboxing-aware
  coordinates and frame-synchronized overlays. Both results include the gravity
  gauge. Clips still synchronize by elapsed time, not by inferred events.
- Technical details remain collapsed initially. Opening them reveals individual
  overlay toggles, the optional Earth projection, debug playback, charts, every
  diagnostic tab, the unmodified result JSON, and downloads. Closing them keeps
  the chosen tracking overlays visible. Unknown/new result fields remain
  accessible in **All data**.

The display adds "Likely" to REAL/AI labels to describe the detector's evidence;
it does not change the underlying verdict or threshold. Synthetic examples are
explicitly labeled as synthetic.

## Design references

The visual direction draws on Hyperplexed's public tutorials and demos:

- [Futuristic Card Effect](https://codepen.io/Hyperplexed/pen/vYzgeYE)
  ([video](https://youtu.be/jMVhxBB3l0w)): instrument frames, fine grids,
  and layered light, adapted for an analysis interface.
- [Hacked Text Effect](https://codepen.io/Hyperplexed/pen/rNrJgrd)
  ([video](https://youtu.be/W5oawMJaXbU)): a short decode interaction on the
  decorative hero word. Measurements and results never scramble or animate
  through invented values.
- [Hyperplexed's creative vault](https://www.hyperplexed.io/): an example
  collection with interactive cards and strong typographic hierarchy.

These are original site-native HTML/CSS/SVG/JavaScript implementations, with no
remote scripts, fonts, image requests, or new production dependencies. Decorative
motion is disabled for reduced-motion preferences; pointer effects are omitted
on touch devices. Video playback remains under the user's control.

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
| `static/visuals.js` | Shared tracking renderer, gravity gauge, decorative interactions |
| `static/compare.js` | Shared upload inputs, comparison setup, paired playback |
| `pipeline.py` | Public detector adapter and safe debug timeline mapping |
| `app.py` | Existing job/media API and previews |
| `tests/test_pipeline.py` | Adapter contract and timeline safety checks |
| `tests/test_identification.py` | Gemini availability, provider priority, and CLIP fallback |
| `tests/test_visuals.cjs` | Overlay coordinates, detection gaps, and gravity-gauge safety |

## Verification for this update

Run the added site tests:

```bash
pip install -r requirements-dev.txt
python -m pytest galileo/site/tests -q
node --test galileo/site/tests/test_visuals.cjs
```

28 Python adapter/provider checks and 8 dependency-free JavaScript renderer checks passed.
The adapter checks cover forwarding to the public analyzer,
retaining future result fields, newly added presets, camera-input validation,
and matching/mismatched/broken debug timelines. The renderer checks cover
letterboxed-video coordinates, the full path before playback, missing detections,
tracking gaps, absent gravity measurements, dynamic Earth ranges, and bounded
scale ticks. Node is only needed for these optional frontend tests.

Headless Chromium checks exercised:

- Light/dark rendering and saved theme after reload.
- Desktop, 390-pixel, and 360-pixel layouts without horizontal page overflow.
- Default highlight/path and the visible gravity gauge with diagnostics closed.
- Tracking toggles, retaining overlays after closing details, first-detection
  seeking, missing-measurement states, and reduced-motion behavior.
- Two actual multipart uploads with independent object settings.
- Comparison reload, linked play/pause/seeking/frame stepping, different frame
  rates and lengths, and independent stepping.
- A failed second upload and retrying only that upload.
- Single upload, invalid-file handling, and custom-size validation.
- Selecting results in History, hidden diagnostics, tab keyboard navigation,
  and independent debug timeline fallback.

End-to-end samples used the main pipeline (not stubbed measurements):

| Synthetic clip | Core verdict | Measured gravity |
| --- | --- | --- |
| Earth-gravity drop, 60 fps | REAL | about 9.88 m/s² |
| Low-gravity drop, 24 fps | AI | about 2.34 m/s² |

These checks validate the website and integration. They do not establish
accuracy on real camera footage or generated footage. CLIP and Gemini inference
were not available in this test environment; their core implementation was not
modified.
