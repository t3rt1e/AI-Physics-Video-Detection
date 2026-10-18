# Galileo

Goal : Catch AI-generated video by measuring physics instead of asking another model.
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
python -m galileo analyze two_drops.mp4 --object volleyball tennis   # one name per object
python -m galileo analyze phone_drop.mp4 --object phone              # not a ball: fine
python -m galileo analyze dinner.mp4 --object "rotisserie chicken=22cm"   # anything, with its size
python -m galileo analyze mystery.mp4                                # let Galileo name it (Gemini or CLIP)
python -m galileo serve                                              # web page: upload a clip, nothing else to set
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

- **Throws are fine.** A throw adds speed, not acceleration: the ball is in free fall the
  moment it leaves the hand. Galileo cuts the frames where the hand is still pushing (it
  spots them by their acceleration, far from gravity's). If the ball was already moving
  fast when first seen and it bounces, the flights after the first bounce measure g and
  the throw is shown (dashed in the debug video) but not counted: a hard throw is short,
  fast and often at the edge of the picture. A drop from rest uses every flight.
- **Distance changes are allowed.** A bounce can kick the ball toward or away from the
  camera. Each flight gets its own distance from the ball's apparent size (and a steadily
  changing one when its size clearly trends), so hops at different distances agree.

**Any camera angle, still or moving**

Film the drop however you like: looking down at the floor, up from the floor, with the
phone held sideways, tilting or panning to follow the ball, walking closer, zooming.

- **Tipped up or down.** A camera looking 35° down at a drop sees only cos 35° = 82% of
  the fall in the picture; the rest is the ball moving away and shrinking. The old
  assumption (a level camera) would call that 8 m/s² and say "AI". Galileo reads how the
  camera is tipped from the room: door frames, wall corners and shelf uprights are
  vertical, and the faster they lean together in the picture, the steeper the camera
  looks down (or up). It needs three uprights spread across the picture, or two long ones
  far apart. The ball's changing size says the same thing on its own, but only over two
  or more bounces that agree: a width that reads a little small at speed (motion blur)
  looks exactly like a ball speeding away, so one fall alone is never trusted for this.
  The picture is then turned (a homography, exact for a pinhole camera) into the view of
  a level camera at the same spot, where gravity points straight down the picture and the
  usual measurement applies.
- **Rolled.** A phone held sideways, or at a slant: down is taken from the falls when they
  clearly agree (two or more bounces, or one long clean fall), checked against the room's
  uprights, and the picture is turned upright. Within 10° nothing changes.
- **Moving.** Shake, drift and zoom are undone as before. A pan or tilt big enough for
  perspective to matter is matched with a full homography, and the frames are laid onto a
  canvas big enough to hold the whole motion, centred on its middle; each frame's outline
  is kept so a ball cut off by the edge of the picture isn't measured. Someone walking
  right past a still camera no longer reads as camera motion (Galileo goes back to its
  first keyframe as soon as the room is in view again).
- **The lens.** How steeply the camera looks only matters through the focal length: a wrong
  one changes g by about (its error) × sin²(tilt). Galileo reads it from the file when it
  can, from the room when it sees both uprights and the horizon (floor seams running
  away), and otherwise assumes a phone's main camera with ±30%. That makes a 35° view about
  ±10%; `--hfov 70` (the field of view across the long side, if you know it) brings it to
  a couple of percent. Steeper than about 70°, the fall runs along the line of sight and
  the picture can't be levelled.
- **Say it yourself.** `--pitch 30` (looking down 30°, negative for up) and `--roll 90`
  skip the guessing; `--upright` assumes a level, upright camera (the old behaviour).
- **New check: falls straight down.** When the room's uprights are found, the fall must
  speed up along them. A generated clip can drop things at a slant; a real one can't.

**Several objects, and objects that aren't balls**

- **Every falling object is measured.** Up to four (`--max-objects`), each with its own
  zoom pass, ruler and air drag, sharing the clip's clocks. Each gets its own g and
  verdict, and "Equal falls" compares them in m/s² (each through its own ruler, so
  distance from the camera doesn't matter). Edges, arms and shadows that a moving camera
  or a person turn into tracks are left out: a shadow is the backdrop with every colour
  channel dimmed by the same factor, and anything Galileo can't name has to look like a
  compact object with a good long fall.
- **Name anything, in any order.** `--object volleyball tennis` gives one name per object.
  A name can be anything: a few hundred everyday objects are known with their typical
  size and air drag (`galileo/catalog.py`: balls, fruit, food, kitchen and office
  things, toys, tools; "rotisserie chicken" is 22 cm ±20%), and anything else works with
  its size: `"grandma's vase=31cm"` (mm, cm, m, in or ft; a ball's diameter, or the
  longest side of anything else). A size you measure beats a typical one: ±2% instead
  of ±20%. Galileo matches names to objects by how they look (CLIP reads any words, not
  just its list) or, without CLIP, by size (the bigger object gets the bigger name).
  `auto` in the list leaves one to Galileo. The object you name first is the main one
  unless you `--seed` another. In the web app, "Something else" in the ruler menu does
  the same.
- **Or let Galileo name them.** With nothing named, Gemini (on whenever `GEMINI_API_KEY`
  is set; `--no-gemini` turns it off) or CLIP names each object. Gemini is shown the
  frame and full-resolution close-ups of the object at the moments it moved least (the
  sharpest, and from several sides as it turns), not the working picture, where a small
  object is a smudge. It is asked for the three things the object most likely is, with
  their sizes and odds, and asked twice (by two different models when it can); the two
  replies are pooled. That matters because Gemini is often sure and wrong about a small
  ball: the same pickleball came back as a pickleball, a tennis ball or a practice golf
  ball depending on the call, each time "85 to 98% sure". When look-alikes of another
  size together have a fair chance (20%) of flipping the call, the verdict hangs on what
  the object is, so Galileo says so instead of guessing: "Can't tell: it depends on what
  the object is (pickleball or plastic practice golf ball)", with the g each would give,
  and asks you to name it. Look-alikes of about the same size (a pickleball and a wiffle
  ball) don't matter, and nor does any size when the physics itself breaks. CLIP is a pretrained image classifier that reads text, so it picks from the
  whole catalogue with no training; it runs offline on the CPU (`pip install torch
  transformers`; the model downloads once, about 600 MB). It is asked about each
  object's look, not just its name ("a hard plastic ball full of round holes"), and when
  it hesitates between objects of different sizes the ruler's error bar grows to match:
  torn between a tennis ball and a pickleball costs a few percent; between a ping-pong
  ball and a basketball, the answer becomes "too uncertain". Gemini can name anything at
  all and estimate its size, and sizes a name you gave without one. It also sizes the
  one in the clip when your name only brings a typical size for a kind of thing that
  comes in many sizes: "eraser" is a 6 cm pencil eraser in the catalogue, and a 14 cm
  whiteboard eraser read as one falls at 4.3 m/s² instead of 10 (with Gemini on, it is
  sized at 14 cm; offline, say "whiteboard eraser" or give its size). Two objects that look
  alike in size and shape are taken to be the same kind. A guessed size can never
  overturn your own: the main object's verdict stands unless an object you named
  yourself disagrees.
- **Or skip the name.** `--depth` asks Depth Pro (pretrained, about 2 GB once, needs
  `torch`, `torchvision` and `transformers`) how far away each object is, which gives
  meters per pixel whatever the object is. It is a looser ruler (about ±25%), so it is
  best next to a size.
- **Things air drag rules.** A feather, a balloon, a sheet of paper, a beach ball reach
  their top speed almost at once and drift down. That is what the real ones do, so for
  those Galileo says it can't tell instead of calling a slow fall fake.
- **Not round is fine, and nothing about the shape needs to be known.** Galileo tells a
  ball from anything else by its outline on the sharpest frames (roundness, length
  against width) and whether its width flickers as it turns. To find the centre of
  gravity of anything else it uses two things:
  - *The outline, weighted by thickness.* Each pixel of the silhouette counts by its
    distance in from the edge. A solid is thickest where its outline is widest, so this
    weighs each lump by its volume rather than its area: for a lumpy object like a
    chicken it lands much nearer the real centre of mass than the outline's plain centre.
  - *The motion.* In free fall only the centre of mass moves on a parabola; the body
    turns about it. So whatever the outline's centre does in step with the turning is the
    gap between the two, and Galileo takes it out. It reads the turning from the outline's
    long axis (steady for something spinning in the picture's plane, uneven for a
    tumble toward the camera), or, for a lumpy roundish outline with no clear axis, from
    the rhythm of its area and stretch. This finds a hammer's centre of mass near its
    heavy head, which no outline could. A correction is kept only when it clearly fits
    better, is smaller than the object, and can be told apart from the fall itself
    (roughly half a turn or more).

  The ruler is the longest side, measured where the contrast falls to half at each end
  (blur-proof, and right for two-tone objects), read near the top of its range since a
  tumble often shows it foreshortened, with at least ±6% allowed for that.

**Reading "Can't tell".** The verdict says where it stopped (`stage` in the JSON, and
in the CSV):

| Stage | What it means | What to try |
| --- | --- | --- |
| `tracking: nothing moving` / `no flight` | No falling object found | Pick the ball; check the ball contrasts with the background |
| `tracking: not a fall` | The tracked thing moves up or sideways: usually the hand | Pick the ball |
| `tracking: slows while falling` | Something lowered or caught, not dropped | Pick the ball |
| `tracking: too few frames or jitter` | Too short or too shaky a track | Film closer, steadier, with more of the fall in view |
| `measurement: too uncertain` | The error bar is too wide to call | Bigger ball in the frame, longer fall, better light, or give the object's measured size |
| `physics: air drag rules` | A feather, balloon or sheet of paper: drag, not gravity, sets its fall | Drop something heavier and more compact |
| `ruler: missing` | No size given, and Galileo couldn't name it | Add `--object` or `--size-mm` |

## Does anything need training?

**No.** Nothing in Galileo is trained, on your laptop or anywhere else.

- **The measurement is physics.** Background subtraction finds the ball, a weighted
  least-squares parabola gives its acceleration, and the clocks are pattern checks on
  timestamps and frames. It is closed-form math on the CPU: about 1 to 6 seconds per
  clip on a laptop, no GPU, no downloads.
- **The model pieces are used as-is.** Gemini (object ID, plain-English reasons) and
  Veo (the AI twin) run in Google's cloud on your API key. CLIP (naming objects
  offline), Depth Pro and SAM 2 are pretrained and only run inference; all are optional.
- **The one data step is calibration, not training.** Twenty logged tower drops
  measure your setup's real error ("our video g matched the tower to ±X%"). You can
  also apply that ratio as a scale correction; see Calibration below.

| Part | Needed for the demo? | Runs on |
| --- | --- | --- |
| Tracking, fits, clocks, verdict, web app | Yes | Laptop CPU |
| CLIP object names (offline) | Optional, for clips where you don't name the object | Laptop CPU |
| Gemini object ID and reasons | Optional (Gemini prize) | Google cloud |
| Veo AI twin | Run once Saturday, then replay the file | Google cloud |
| Depth Pro second ruler | Optional, for uploaded wild clips | 5070 Ti desktop (CPU works, slowly) |
| SAM 2 tracker | Optional, for cluttered clips | 5070 Ti desktop |

## The demo

| Beat | What to run | What judges see |
| --- | --- | --- |
| Live drop | `serve --tower COM5 --camera 0` (or `--fake-tower` to rehearse) | Button, then TOWER g and VIDEO g side by side within seconds |
| AI twin | `twin --image tower.jpg`, then pick it on the right | Same tower, wrong gravity, strobe view with the ball lagging Earth's rings |
| Slow-motion trap | Pick the 240 fps phone clip (`--slowmo` on the command line) | "As played: 0.15 m/s²", Galileo reads the 240 fps clock and reports 9.8 |
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

By default every clip is taken as playing at the speed it was filmed: no slow motion,
no speed-up. Then g inside Earth's interval (shown with every result, e.g. "counts as real
between 8.3 and 11.3 m/s²") is real, and outside it, too slow or too fast, is AI. To let
Galileo read the file's clocks and allow real slow motion (and sped-up footage), pass
`--slowmo` (or set `ASSUME_REAL_TIME = False` in `config.py`). Then use the original
file from the phone so its clock survives:

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
python -m galileo serve --fake-tower
```

With a key set, Gemini is on by default (`--no-gemini` turns it off). It names unknown
objects so their typical size can be the ruler, sizes the one in the clip when a name
only gives a typical size, and writes a two-sentence explanation of each verdict.

A free key gets about 20 requests a day per model. Naming an object costs two requests
(`GALILEO_GEMINI_SAMPLES=1` makes it one, at the cost of the cross-check) and the
explanation one, so a clip with one object costs three. Naming walks the Flash models
(`gemini-flash-latest`, then 3.7, 3.6, 3.5 and 3 Flash) and then the Flash-Lite ones; the
explanation starts with Flash-Lite, to leave the Flash quota for naming. A model that is
busy (503), out of quota (429) or retired (404) is skipped and left alone for a while.
When none answers, CLIP names the object and the run goes on. Answers are cached in
`~/.galileo_cache`, so a clip that ran once costs nothing again, and demo clips keep
working if the venue Wi-Fi drops: run your demo clips once beforehand.
Model names change often: override with `GALILEO_GEMINI_MODEL` (default
`gemini-flash-latest`) and `GALILEO_VEO_MODEL` (default `veo-3.1-generate-preview`).
Veo clips are 24 fps and 4, 6 or 8 s long, and stay downloadable for 2 days.

## How it works

1. **Frames** (`video.py`): PyAV decode with exact timestamps, display rotation, and
   capture-rate tags. A clip over 900 frames is read one frame in N (a 240 fps clip at
   120 a second), so a moving camera can't hide the first bounce; only when that would
   leave under 60 frames a second is it cut to the stretch with the most motion.
2. **Camera motion** (`stabilize.py`): pans, drift, zooms and handheld shake are tracked
   on the background against keyframes and removed. Matching against a keyframe (not the
   previous frame) keeps a slowly moving hand from leaking in as fake camera motion at
   120 to 240 fps, and coming back to the first keyframe whenever it is in view keeps
   someone walking past the lens from leaving a fake drift. Small moves are a similarity
   (shift, turn, zoom); a pan or tilt that moves the view more than 6% of the picture for
   a good part of the clip is a homography, laid onto a canvas around the middle of the
   motion. A locked-off camera is left alone, and so is a plain backdrop (a seamless
   white studio wall, typical of generated clips) where the only detail to follow is the
   ball itself: its motion would otherwise pass for the camera's.
3. **Tracking** (`track.py`): background subtraction, then every moving thing is cut into
   flights and scored on how much each flight looks like a dropped ball: long, smooth,
   steady in size (or in area, for something long that tumbles), and speeding up
   downward. A hand pulling away scores near zero; a ball that bounces five times collects
   five flights of evidence. A click on the ball (a seed) overrides the score. Every
   track with a clean fall of its own is a falling object, unless it is a shadow, a piece
   of an object already chosen (a volleyball panel), or the rest of its path after a
   dropout; colour keeps two objects apart where their paths cross. A pale ball on a
   bright backdrop has a shadow's colour, but nothing solid moves along with it, so it
   still counts. Travel is counted in the ball's own widths, but at most a sixth of the
   picture each, so a big ball close to the camera isn't judged to barely move. The ball's width is
   measured across its direction of motion, which motion blur can't stretch; for big
   multi-coloured balls (a volleyball's panels) the outline is used, because the panels
   don't stand out from the wall equally.
4. **Zoom** (`analyze.py`): if an object came out small, the stretch of time and patch of
   picture where it flew are decoded again at up to full resolution and tracked there.
   The window holds whole flights only (a hop cut short by the window fits badly),
   chosen for the most frames at the biggest picture of the object.
5. **Shape and name** (`shape.py`, `identify.py`, `catalog.py`): round or not from the
   outline, then what it is (you, Gemini, CLIP over a few hundred everyday objects, or a
   look-alike), which gives its size and air drag. Non-round objects are followed by
   their centre of mass: the thickness-weighted outline, corrected by the motion (see
   above).
5b. **Camera angle** (`camera.py`, `scene.py`): which way gravity points as the camera sees
   it. The room's uprights converge toward the vertical vanishing point; its distance from
   the picture's centre gives K = tan(pitch) / f without knowing f. Each round object's
   flights are also followed in 3D (depth from the widths alone, Z = f·D/d, so a bend in
   the path from lens distortion or drag can't pass for depth; the rest from the
   positions at that depth). A 3D reading counts only if it points, in the picture, where
   the object speeds up in the picture (a ball rolling toward the camera reads as
   "gravity" along the view and would tip the camera 60°), and another object's only if
   it agrees with the main one's. The balls and the room are combined when they agree,
   the room wins when they don't, and the room's horizon, if seen, gives f. Then every position and width is
   mapped into the level view (`Leveler`), and how far the focal length and the tilt could
   move g is added to the error bar.
6. **Fitting** (`physics.py`): the track is cut into flights at bounces (direction changes
   are judged over 1/60 s, so noise at 240 fps isn't a bounce; a bounce is low in its hop
   and arrives fast, and need not land where the last one did: a ball bouncing away
   from the camera lands higher in the picture each time) and each gets a robust
   parabola in x and y, so sideways motion is fine: g is the part along gravity. One
   rolling-shutter term (bounded by a frame's readout time) is fitted when it clearly
   helps; uncorrected, a 60 fps webcam can read 5 to 10% high. This runs in the level
   view, so gravity points down the picture; the rolling-shutter term still uses each
   point's row on the sensor as filmed.
7. **Air drag**: for light balls Galileo works out, from the measured speed, how far drag
   pushed the ball off a vacuum path at every frame, takes that out of the positions and
   fits again. The parabola and the rolling-shutter term then see pure gravity instead of
   bending to absorb drag. Rising and falling hops barely need it; a long first drop
   needs the most.
8. **Checks** (`checks.py`): tests that need neither ruler nor clock. Only a clear,
   significant change fails the constant-acceleration check. The bounce law reads each
   bounce's restitution from times, never trusting a short hop's speeds if it can help
   it: a free hop's speed is g times the time from its peak, so the restitution is
   (time from the bounce up to the next peak) / (time down from the last peak), or, with
   a moving camera (which shifts where a hop seems to peak but not when the ball hits
   the floor), the ratio of whole hops bounce to bounce. The fits' speeds at the contact
   come last: a short hop's fit can read its acceleration 15 to 20% off, which turns a
   real 0.84 bounce into 1.1, so their error bar includes that. Only a gain beyond the
   measurement's error fails; a turnaround well above the floor the others land on (a
   hand catching and tossing it) isn't tested, since a hand can add energy. Hops are
   compared for "same g" only when both are clean (0.2 s and 15 points or more, tracked
   about as tightly as the rest). With the room's uprights in view, the fall must also
   speed up along them ("Falls straight down").
8b. **Newer laws** (`checks.py`, `spin.py`, `shadow.py`), all still free of ruler and
   clock, marked "(new)" in the output until they have seen more clips:
   - **Steady spin**: nothing twists a ball in the air, so it keeps its spin from one end
     of a hop to the other. The spin is read from the ball's markings: for each pair of
     close-ups, the 3D turn that carries one onto the other is fitted on the sphere
     (every spot of the second looked up where it was in the first). The turn about the
     line of sight reads to a few tenths of a radian a second on a volleyball; a ball
     whose readings scatter more than that (faint markings, or tumbling about an axis
     across the picture, like a small pickleball) is skipped.
   - **Spin at bounces**: every push at a bounce acts where the ball touches the floor,
     so none can turn it about that point: its angular momentum about the contact point
     (m r (v + k r ω), sideways speed v, spin ω, k = 2/3 for a hollow ball, 2/5 for a
     solid one) is kept by a hard ball that grips and partly lost by a squashy one (its
     squash bites into spin and sideways speed together), but never grows. A spin kicked
     up out of nothing, or a sideways shove with no spin to pay for it, fails.
   - **Rolling**: a ball rolling along the floor turns once per circumference (ω r = v).
   - **One flat floor**: in positions scaled to the ball's reference distance, every
     contact of a level camera lands at one height (a tipped camera needs three bounces
     to tell from a sloped floor).
   - **No sideways push**: in the air only gravity pulls, and only down, so no hop
     speeds up or slows down sideways (beyond what drag or spin allow).
   - **Steady bounces**: one ball on one floor keeps a similar share of its speed at
     every bounce (a little more as the landings get softer).
   - **Constant size**: the ball's width changes only as it moves toward or away from
     the camera, smoothly. A swell of a fifth only warns (a hand in front of it, or a
     change of light, can fake that); one of more than half fails.
   - **Shadow follows** and **Ball meets its shadow** (`shadow.py`): light travels in
     straight lines and arrives at once. Each frame is divided by the still background;
     the darker patch near where the ball lands is the shadow candidate. A far light puts
     its centre at the ball's foot point plus a fixed multiple of the height, so a real
     shadow slides along in step with the ball, and at a bounce the ball touches it. A
     shadow that never moves is part of the background and can't be seen this way.
   Hops tracked badly (a catch, the picture's edge) are left out of these, and with a
   moving camera the ones that need a fixed floor or sideways speed are skipped.
   `config.EXPERIMENTAL_CHECKS_DECIDE` (default True) says whether their failures count
   toward the verdict (alongside another failure, since one failing alone is only a
   warning; see Verdict); set it to False to show them without letting them count.
9. **Clocks** (`clocks.py`): capture-rate tags, high-frame-rate timestamps, repeated
   frames, interpolation patterns, and lighting flicker (a hint only). Read once, on the
   main object, and shared: a clip has one speed.
10. **Rulers** (`scale.py`): object size, plus Depth Pro or a manual ruler, fused with
   error bars that widen when rulers disagree.
11. **Verdict** (`verdict.py`): a scale-free failure means AI. A tracked thing that never
   falls, or slows as it falls, gets "Can't tell" (it is probably a hand); one that falls
   at a steady speed is AI. Otherwise g decides, with a ±15% band (wider when the error
   bar is): inside it is real, outside it, too slow or too fast, is AI. If the error bar
   is wider than ±35% the answer is "too uncertain" unless g is far off anyway. With
   `--slowmo`, a clock that explains a low g makes it real slow motion, and too fast
   can't be called (the footage may be sped up). With several objects, the main one decides; another stands in when the main
   one can't be measured and Galileo knows what the other one is, and the rest add their
   numbers to the reasons (lowering the confidence when they disagree). One check
   failing on its own, with nothing gross about it, doesn't make a clip AI: real footage
   trips one law now and then through a slip in measuring, while a generated clip tends
   to break several, break one grossly (a bounce gaining 25%+ by its hop times, gravity
   changing 2.5x between hops, the acceleration doubling within a fall), or get g wrong.
   So a lone ordinary failure is weighed as a warning: gravity and the other checks
   decide, the confidence is capped at 60%, and the reasons say so. Other objects
   that don't clearly speed up on the way down (a ball rolling across the floor, one
   carried in a hand) are dropped however sure Galileo is of what they are, unless you
   named them. When the ruler is Gemini's guess and a look-alike of another size would
   flip the call, the answer is "it depends on what the object is" (see naming above).

On synthetic clips with known physics (60 fps rolling-shutter webcam, 240 fps native,
8x tagged, 4x repeated, 4x interpolated, bounces, a panning camera, AI-style floaty and
eased falls, and 1080p 120 fps ping-pong drops with a hand, air drag, sideways kicks at
each bounce and handheld shake, two different balls falling side by side, a tumbling
phone, a spinning bar with its mass off-centre, and a lumpy six-part "chicken" tumbling
end over end about a tilted axis) every verdict comes out right and g lands within
about 1 to 4% of truth. Without the motion correction, the bar's outline centre reads
30% high.
`synth3d.py` renders a ball in a furnished room through a full 3D camera. With it: a
camera looking 35° down (9.97 m/s², 7.5 with the old level-camera assumption), looking
25° up (9.74), held sideways (9.69), tilting 25°/s to follow the ball (10.28), panning
(10.05), walking toward the scene (10.04), zooming in (9.98), handheld (10.08), and a
generated 3.5 m/s² fall filmed from above (3.21, called AI); true g is 9.81.
Real footage is the real test, so film a few phone drops of the steel ball tonight and
check them before the tower is up.

## Project layout

```
galileo/
  analyze.py     the pipeline, video in, verdict out
  physics.py     fits, flights, rolling shutter          (David)
  checks.py      scale-free tests                        (David)
  spin.py        the ball's spin from its markings (3D turn on the sphere)
  shadow.py      the ball's shadow on the floor, frame by frame
  clocks.py      slow-motion detection                   (David)
  scale.py       rulers and fusion                       (David)
  verdict.py     verdict and reasons                     (David)
  video.py       decode, timestamps, writer, previews    (Michael)
  track.py       tracker: objects, widths, lengths       (Michael)
  shape.py       round or not, centre of mass, length
  identify.py    what each object is: you, Gemini, CLIP
  catalog.py     a few hundred everyday objects: size, drag, shape
  stabilize.py   camera motion removal: shake to full pans (Michael)
  camera.py      the camera's tilt and roll from the fall in 3D; the level view
  scene.py       the room's uprights and horizon: tilt, roll, focal length
  sam2_tracker.py, depth.py, gemini.py  optional models  (Michael)
  tower.py       serial contract, gate fit, fake tower   (Patrick)
  live.py        webcam capture, live loop, drops.csv    (Patrick + Michael)
  synth.py       synthetic clips with known physics
  synth3d.py     the same in 3D: a furnished room and a camera at any angle, moving
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
