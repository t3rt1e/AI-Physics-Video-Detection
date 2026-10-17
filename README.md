# Galileo

Catch AI-generated video by measuring physics instead of asking another model.
Real things fall at 9.8 m/s² and every clock in a real scene agrees; generated
video usually doesn't.

```
g = s × a_px × k²
    s     meters per pixel, from the falling object's own size
    a_px  acceleration in pixels per second², from a fitted parabola
    k     real seconds per video second, from the clip's clocks (1 unless slowed)
```

## Quick start

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m galileo synth                 # rehearsal clips with known physics (about a minute)
python -m galileo serve --fake-tower    # open http://localhost:8000
```

Press **Drop the ball** for a simulated tower drop, or pick clips on either side and
press **Measure**. One clip on the command line:

```bash
python -m galileo analyze data/clips/synthetic_ai_floaty_24fps.mp4 --object steel-38mm
```

## Measuring your own clips

```bash
python -m galileo analyze my_drop.mp4 --object ping-pong      # or --size-mm 40 for any ball
python -m galileo analyze my_drop.mp4 --object ping-pong --debug-video
python -m galileo analyze my_drop.mp4 --object ping-pong --seed "1.25,0.48,0.33"
python -m galileo batch path/to/real_clips --object ping-pong --expect real --csv real.csv
```

- **See what Galileo saw.** `--debug-video` writes `my_drop_debug.mp4` next to the clip
  (or give it a path). Every frame shows the blobs it considered (grey rings), other
  moving things with their scores, the tracked ball (one colour per flight, grey where a
  frame was left out of the fit), the fitted path, the velocity (white arrow) and
  acceleration (red arrow, beside a gold 9.8 m/s² line), where a real ball would be (gold
  ring), and the zoom box. The readout gives the time, which flight it's in, the ball's
  position as fractions of the frame (ready for `--seed`), its speed, the fitted
  acceleration and a local one from the frames around it. Underneath, height and
  acceleration against time with a cursor. `--debug-speed 0.25` plays it at quarter
  speed; `batch ... --debug-dir debug` makes one per clip.

- **Name the ball.** `--object` is the ruler (its real size). For light balls it also
  turns on the air-drag correction: a ping-pong ball dropped from a meter falls 5 to 20%
  slower than gravity alone would make it, and Galileo puts that back. `--size-mm` gives
  the size but not the drag, so prefer the preset when there is one (or add
  `--drag-vt 8.4`, the ball's top speed in m/s).
- **Point at the ball when it tracks the wrong thing.** `--seed T,X,Y` is a time in
  seconds when the ball is in the air and its position, either as fractions of the
  width and height (`0.48,0.33`) or in pixels (`920,356`). In the web app, press
  **Pick the ball**, pause on a frame with the ball in the air, and click it. In
  PowerShell, keep the quotes around the numbers.
- **Sort a folder at once.** `batch` measures every clip in a folder and writes a CSV
  with the verdict, g, the stage where it stopped, the ball's size in pixels, whether it
  zoomed in, and the notes. `--expect real` (or `ai`) lists the clips it got wrong, and
  `--json-dir` keeps every full result.
- **Small balls get a second look.** High-frame-rate clips are scaled down to fit in
  memory, which can leave a ping-pong ball 5 to 8 px wide. When the ball comes out
  smaller than about 40 px, Galileo decodes just the patch and stretch of time where it
  flew at up to full resolution and measures it again (`--no-zoom` turns this off).

**Reading "Can't tell".** The verdict says where it stopped (`stage` in the JSON, and
in the CSV):

| Stage | What it means | What to try |
| --- | --- | --- |
| `tracking: nothing moving` / `no flight` | No falling object found | Pick the ball; check the ball contrasts with the background |
| `tracking: not a fall` | The tracked thing moves up or sideways: usually the hand | Pick the ball |
| `tracking: slows while falling` | Something lowered or caught, not dropped | Pick the ball |
| `tracking: too few frames or jitter` | Too short or too shaky a track | Film closer, steadier, with more of the fall in view |
| `measurement: too uncertain` | The error bar is too wide to call | Bigger ball in the frame, longer fall, better light |
| `ruler: missing` | No size given | Add `--object` or `--size-mm` |

## Does anything need training?

**No.** Nothing in Galileo is trained, on your laptop or anywhere else.

- **The measurement is physics.** Background subtraction finds the ball, a weighted
  least-squares parabola gives its acceleration, and the clocks are pattern checks on
  timestamps and frames. It is closed-form math on the CPU: about 1 to 6 seconds per
  clip on a laptop, no GPU, no downloads.
- **The model pieces are used as-is.** Gemini (object ID, plain-English reasons) and
  Veo (the AI twin) run in Google's cloud on your API key. Depth Pro and SAM 2 are
  pretrained and only run inference; both are optional.
- **The one data step is calibration, not training.** Twenty logged tower drops
  measure your setup's real error ("our video g matched the tower to ±X%"). You can
  also apply that ratio as a scale correction; see Calibration below.

| Part | Needed for the demo? | Runs on |
| --- | --- | --- |
| Tracking, fits, clocks, verdict, web app | Yes | Laptop CPU |
| Gemini object ID and reasons | Optional (Gemini prize) | Google cloud |
| Veo AI twin | Run once Saturday, then replay the file | Google cloud |
| Depth Pro second ruler | Optional, for uploaded wild clips | 5070 Ti desktop (CPU works, slowly) |
| SAM 2 tracker | Optional, for cluttered clips | 5070 Ti desktop |

## The demo

| Beat | What to run | What judges see |
| --- | --- | --- |
| Live drop | `serve --tower COM5 --camera 0` (or `--fake-tower` to rehearse) | Button, then TOWER g and VIDEO g side by side within seconds |
| AI twin | `twin --image tower.jpg`, then pick it on the right | Same tower, wrong gravity, strobe view with the ball lagging Earth's rings |
| Slow-motion trap | Pick the 240 fps phone clip | "As played: 0.15 m/s²", Galileo reads the 240 fps clock and reports 9.8 |
| In the wild | Upload any clip of something falling, pick the object as the ruler | Verdict with reasons and error bars |

The left panel follows live drops automatically. Each result shows:

- **The number** with its error bar, and the verdict: Real, AI-generated, Real slow motion, or Can't tell.
- **A strobe view**, like a textbook multiple-exposure photo: the clip's ball next to where
  Earth's gravity puts a ball from the same start, one flash per fixed slice of real time.
- **Distance against time**, measured points and fit against the 9.80 curve.
- **Checks slow motion can't fool**: constant acceleration, the bounce law, equal falls,
  and starting from rest.
- **The video overlay**: the tracked ball in blue and its Earth-gravity twin as a gold ring.

### The slow-motion clip

Use the original file from the phone so its clock survives:

- **iPhone:** AirDrop with Options, All Photos Data on, or export the unmodified original.
  The 240 fps timestamps let Galileo read real time directly.
- **Android:** the file keeps a `com.android.capture.fps` tag; Galileo reads it.
- A slow-mo re-rendered to 30 fps with all tags stripped is still caught if it repeats
  frames or interpolates them. If it has neither, physics alone can't tell it from AI,
  and Galileo says so in the reasons.

## Hooking up the tower

The tower and laptop talk over USB serial at 115200 baud, one comma-separated line per
message (the contract from the game plan):

```
TOWER -> LAPTOP  ARM,id   REL,id,t_us   GATE,id,k,h_mm,t_us   TOWERG,id,g,sigma   BOUNCE,id,n,t_us
LAPTOP -> TOWER  VIDEOG,id,g,sigma,verdict   SHOW,text
```

```bash
python -m galileo ports                        # find the Pico, e.g. COM5 or /dev/ttyACM0
python -m galileo monitor --port COM5          # watch raw lines while wiring
python -m galileo serve --tower COM5 --camera 0
```

On ARM the laptop starts recording the webcam; 1.4 s after REL it cuts the window
around the release, measures it, sends `VIDEOG` back, and appends a row to
`data/drops.csv`. Clips land in `data/drops/`.

`firmware/pico_tower/main.py` is a MicroPython reference for the Pico that speaks this
protocol: button countdown, magnet release, interrupt-timed gates, a gate fit for g,
piezo bounces, and the MAX7219 readout. It has not run on hardware yet; set the pins in
its CONFIG block first.

**Camera setup that measures well**

- Webcam fixed and square to the drop line. If the whole drop doesn't fit, turn the
  webcam to portrait: it fits the tower, and the sensor's rolling shutter then scans
  across the fall instead of along it.
- A matte, brightly coloured ball on the black backdrop. Shiny steel tracks badly.
- A dense ball is the cleanest. A ping-pong ball works with `--object ping-pong`, which
  corrects for its air drag.
- Steady light for the live loop. 60 fps or better.

## Calibration

After 15 to 20 real tower drops:

```bash
python -m galileo calibrate                       # prints the spread for the pitch
python -m galileo calibrate --save data/calibration.json
python -m galileo serve --tower COM5 --camera 0 --calibration data/calibration.json
```

The correction is only applied to tower clips, and the page says when it is on.

## Gemini and Veo

```bash
pip install -r requirements-ml.txt        # or just: pip install google-genai
set GEMINI_API_KEY=...                    # PowerShell: $env:GEMINI_API_KEY="..."  macOS/Linux: export
python -m galileo twin --image tower.jpg --out data/clips/ai_twin_veo.mp4
python -m galileo serve --fake-tower --gemini
```

With Gemini on, it names unknown objects so their typical size can be the ruler, and
writes a two-sentence explanation of each verdict. Answers are cached in
`~/.galileo_cache`, so demo clips that ran once keep working if the venue Wi-Fi drops.
Model names change often: override with `GALILEO_GEMINI_MODEL` (default
`gemini-flash-latest`) and `GALILEO_VEO_MODEL` (default `veo-3.1-generate-preview`).
Veo clips are 24 fps and 4, 6 or 8 s long, and stay downloadable for 2 days.

## How it works

1. **Frames** (`video.py`): PyAV decode with exact timestamps, display rotation, and
   capture-rate tags. Long clips are cut to the stretch with the most motion.
2. **Camera motion** (`stabilize.py`): pans, drift, zooms and handheld shake are tracked
   on the background against keyframes and removed. Matching against a keyframe (not the
   previous frame) keeps a slowly moving hand from leaking in as fake camera motion at
   120 to 240 fps. A locked-off camera is left alone.
3. **Tracking** (`track.py`): background subtraction, then every moving thing is cut into
   flights and scored on how much each flight looks like a dropped ball: long, smooth,
   steady in size, and speeding up downward. A hand pulling away scores near zero; a ball
   that bounces five times collects five flights of evidence. A click on the ball (a
   seed) overrides the score. The ball's width is measured across its direction of
   motion, which motion blur can't stretch; for big multi-coloured balls (a volleyball's
   panels) the outline is used, because the panels don't stand out from the wall equally.
4. **Zoom** (`analyze.py`): if the ball came out small, the stretch of time and patch of
   picture where it flew are decoded again at up to full resolution and tracked there.
5. **Fitting** (`physics.py`): the track is cut into flights at bounces (direction changes
   are judged over 1/60 s, so noise at 240 fps isn't a bounce) and each gets a robust
   parabola in x and y, so sideways motion is fine: g is the part along gravity. One
   rolling-shutter term (bounded by a frame's readout time) is fitted when it clearly
   helps; uncorrected, a 60 fps webcam can read 5 to 10% high. Gravity is the picture's
   "down" unless the camera is clearly tilted (up to 30 degrees); it never flips.
6. **Air drag**: for light balls Galileo works out, from the measured speed, how far drag
   pushed the ball off a vacuum path at every frame, takes that out of the positions and
   fits again. The parabola and the rolling-shutter term then see pure gravity instead of
   bending to absorb drag. Rising and falling hops barely need it; a long first drop
   needs the most.
7. **Checks** (`checks.py`): tests that need neither ruler nor clock. Only a clear,
   significant change fails the constant-acceleration check.
8. **Clocks** (`clocks.py`): capture-rate tags, high-frame-rate timestamps, repeated
   frames, interpolation patterns, and lighting flicker (a hint only).
9. **Rulers** (`scale.py`): object size, plus Depth Pro or a manual ruler, fused with
   error bars that widen when rulers disagree.
10. **Verdict** (`verdict.py`): a scale-free failure means AI. A tracked thing that never
   falls, or slows as it falls, gets "Can't tell" (it is probably a hand); one that falls
   at a steady speed is AI. Otherwise g decides, with a ±15% band; if the error bar is
   wider than ±35% the answer is "too uncertain" unless g is far off anyway. Too slow with
   no clock explaining it is AI; matching Earth after a found slow-down is real slow
   motion.

On synthetic clips with known physics (60 fps rolling-shutter webcam, 240 fps native,
8x tagged, 4x repeated, 4x interpolated, bounces, a panning camera, AI-style floaty and
eased falls, and 1080p 120 fps ping-pong drops with a hand, air drag, sideways kicks at
each bounce and handheld shake) every verdict comes out right and g lands within about
1 to 4% of truth.
Real footage is the real test, so film a few phone drops of the steel ball tonight and
check them before the tower is up.

## Project layout

```
galileo/
  analyze.py     the pipeline, video in, verdict out
  physics.py     fits, flights, rolling shutter          (David)
  checks.py      scale-free tests                        (David)
  clocks.py      slow-motion detection                   (David)
  scale.py       rulers and fusion                       (David)
  verdict.py     verdict and reasons                     (David)
  video.py       decode, timestamps, writer, previews    (Michael)
  track.py       ball tracker and width measurement      (Michael)
  stabilize.py   camera motion removal                   (Michael)
  sam2_tracker.py, depth.py, gemini.py  optional models  (Michael)
  tower.py       serial contract, gate fit, fake tower   (Patrick)
  live.py        webcam capture, live loop, drops.csv    (Patrick + Michael)
  synth.py       synthetic clips with known physics
  web/           FastAPI server and the page
firmware/pico_tower/main.py   reference Pico firmware   (Patrick)
tests/                        python -m pytest -q
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Hackathon rules

SwampHacks requires disclosing AI tool use: say on Devpost that the starting codebase
was written with Claude. Only work done during the event counts, so keep the commit
history from the hacking window.
