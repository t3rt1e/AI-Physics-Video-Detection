"use strict";
/* Galileo detector page. Hash routes: #/ (new), #/history, #/job/<id> */

const $ = (s, r = document) => r.querySelector(s);
const view = $("#view");
const VERDICT = {
  REAL: "Real", AI: "AI-generated", SLOWMO: "Real, slow motion", UNSURE: "Can't tell",
};
let STATUS = null;
let current = null;      // the open job page (so it can be torn down)

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).detail || m; } catch (e) { /* keep status text */ }
    throw new Error(m);
  }
  return r.json();
}

/* ---------------------------------------------------------------- router */
window.addEventListener("hashchange", route);
async function route() {
  if (current) { current.destroy(); current = null; }
  const h = location.hash.replace(/^#\/?/, "");
  document.querySelectorAll(".top nav a").forEach((a) => a.classList.remove("on"));
  if (!STATUS) { try { STATUS = await api("/api/status"); } catch (e) { STATUS = { presets: [], samples: [], auto: {} }; } }
  window.scrollTo(0, 0);
  if (h.startsWith("job/")) { current = new JobPage(h.slice(4)); }
  else if (h === "history") { $(".nav-hist").classList.add("on"); await historyPage(); }
  else { $(".nav-new").classList.add("on"); homePage(); }
}

/* ------------------------------------------------------------------ home */
function homePage() {
  const presets = STATUS.presets || [];
  const round = presets.filter((p) => p.shape === "round"), other = presets.filter((p) => p.shape !== "round");
  const opt = (p) => `<option value="${esc(p.key)}">${esc(p.label)} (${p.size_mm} mm)</option>`;
  const autoOk = STATUS.auto && STATUS.auto.available;
  view.innerHTML = `
  <div class="home">
    <section class="lede">
      <h1>Does it fall like it should?</h1>
      <p>Upload a clip of something dropped, thrown or bouncing. Galileo measures its acceleration frame by frame and compares it with real gravity.</p>
      <ul class="how">
        <li><b>Real footage</b> falls at about 9.8 m/s². Slow motion is caught from the file's clock.</li>
        <li><b>Generated footage</b> usually falls too slowly, speeds up unevenly, or tilts the wrong way.</li>
        <li><b>You see the evidence:</b> the tracked ball on every frame, a debug video, and every number it recorded.</li>
      </ul>
      <div class="samples">
        <h2>No clip handy? Try a rehearsal clip</h2>
        <div class="chips" id="samples">${(STATUS.samples || []).map((s) =>
          `<button class="chip" data-name="${esc(s.name)}" data-obj="${esc(s.object || "")}">${esc(s.description)}</button>`).join("") || '<span class="hint">None installed.</span>'}</div>
      </div>
    </section>
    <form class="drop" id="form">
      <label class="zone" id="zone">
        <input class="visually-hidden" type="file" id="file" accept="video/*,.mp4,.mov,.m4v,.webm,.mkv,.avi">
        <strong id="zone-title">Choose a video or drop it here</strong>
        <span id="zone-sub">MP4, MOV, WebM · up to 1.5 GB · a few seconds is plenty</span>
      </label>
      <label class="field">What is falling?
        <select id="obj">
          ${autoOk ? '<option value="auto">Let Galileo identify it</option>' : ""}
          <optgroup label="Balls">${round.map(opt).join("")}</optgroup>
          <optgroup label="Other objects">${other.map(opt).join("")}</optgroup>
          <option value="custom">Something else (I'll enter its size)</option>
        </select>
        <small>${autoOk ? "Galileo works out what it is from how it looks. If it can't, you'll be asked to tell it." : "Its real size turns pixels into metres. Automatic naming isn't available: run pip install torch torchvision transformers pillow in the Python that runs this site, then restart it."}</small>
      </label>
      <label class="field" id="size-wrap" hidden>Size in millimetres
        <input id="size" type="number" min="1" max="5000" step="any" placeholder="e.g. 220">
        <small>The ball's diameter, or the longest side of anything else.</small>
      </label>
      <details class="adv"><summary>Advanced</summary>
        <label class="field">Camera field of view, degrees (optional)
          <input id="hfov" type="number" min="10" max="180" step="any" placeholder="auto">
          <small>Across the long side. Helps when the camera looks steeply up or down.</small>
        </label>
      </details>
      <p class="err" id="err" role="alert" hidden></p>
      <button class="btn" id="go" type="submit" disabled>Check this video</button>
    </form>
  </div>`;
  const file = $("#file"), go = $("#go"), err = $("#err");
  const objSel = $("#obj");
  objSel.value = autoOk ? "auto" : "tennis";
  const sync = () => { $("#size-wrap").hidden = objSel.value !== "custom"; };
  objSel.onchange = sync;
  sync();
  const picked = (f) => {
    if (!f) return;
    $("#zone-title").innerHTML = `<span class="picked">${esc(f.name)}</span>`;
    $("#zone-sub").textContent = `${(f.size / 1e6).toFixed(1)} MB · choose another to replace it`;
    go.disabled = false;
  };
  file.onchange = () => picked(file.files[0]);
  const zone = $("#zone");
  ["dragenter", "dragover"].forEach((e) => zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", (ev) => { if (ev.dataTransfer.files[0]) { file.files = ev.dataTransfer.files; picked(file.files[0]); } });
  const fail = (m) => { err.textContent = m; err.hidden = false; go.disabled = false; go.textContent = "Check this video"; };
  $("#form").onsubmit = async (ev) => {
    ev.preventDefault();
    err.hidden = true;
    if (objSel.value === "custom" && !$("#size").value) { fail("Enter the object's size in millimetres."); return; }
    const fd = new FormData();
    fd.append("file", file.files[0]);
    fd.append("object", objSel.value);
    fd.append("size_mm", $("#size").value);
    fd.append("hfov", $("#hfov").value);
    go.disabled = true; go.textContent = "Uploading…";
    try { const r = await api("/api/jobs", { method: "POST", body: fd }); location.hash = `#/job/${r.id}`; }
    catch (e) { fail(e.message); }
  };
  $("#samples").addEventListener("click", async (ev) => {
    const b = ev.target.closest(".chip"); if (!b) return;
    try {
      const r = await api("/api/jobs/sample", { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: b.dataset.name }) });
      location.hash = `#/job/${r.id}`;
    } catch (e) { fail(e.message); }
  });
}

/* --------------------------------------------------------------- history */
async function historyPage() {
  view.innerHTML = `<h1 style="font-size:30px">Past results</h1><div class="hist" id="hist"><p class="hint">Loading…</p></div>`;
  const list = await api("/api/jobs");
  const box = $("#hist");
  if (!list.length) { box.innerHTML = `<div class="empty">Nothing here yet. <a href="#/">Check a video</a> and it will be kept here.</div>`; return; }
  box.innerHTML = list.map((j) => {
    const st = j.status === "done" ? (j.verdict || "UNSURE") : j.status;
    const lab = j.status === "done" ? VERDICT[st] : ({ running: "Working", queued: "Waiting", error: "Failed" }[j.status] || j.status);
    return `<a class="hrow" href="#/job/${j.id}"><span class="pill ${st}">${esc(lab)}</span>
      <span><span class="nm">${esc(j.filename)}</span><br><span class="sub">${esc(j.error || j.headline || j.stage || "")}</span></span>
      <span class="gg">${j.g != null ? num(j.g, 1) + " m/s²" : ""}</span>
      <span class="sub">${new Date((j.created || 0) * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}</span></a>`;
  }).join("");
}

/* ------------------------------------------------------------- job page */
class JobPage {
  constructor(id) {
    this.id = id; this.dead = false; this.data = null; this.frames = null; this.idx = 0;
    this.mode = "original"; this.playing = false; this.rate = 1; this.loop = true;
    this.show = { ball: true, earth: true, trail: true };
    this.rendered = false; this.videosReady = false;
    this.keys = (e) => this.onKey(e);
    document.addEventListener("keydown", this.keys);
    view.innerHTML = `<p class="hint">Loading…</p>`;
    this.poll();
  }
  destroy() { this.dead = true; clearTimeout(this.timer); document.removeEventListener("keydown", this.keys); cancelAnimationFrame(this.raf); }

  async poll() {
    if (this.dead) return;
    let d;
    try { d = await api(`/api/jobs/${this.id}`); }
    catch (e) { view.innerHTML = `<div class="empty">${esc(e.message)}. <a href="#/">Start a new check</a></div>`; return; }
    if (this.dead) return;
    this.data = d;
    const j = d.job;
    if (j.status === "error") { view.innerHTML = `<div class="card"><h2>This video couldn't be checked</h2><p class="err">${esc(j.error)}</p><p>Try a different file, or a shorter clip with the falling object clearly visible. <a href="#/">Start over</a></p></div>`; return; }
    if (d.frames && !this.frames) { this.frames = d.frames; }
    if (!this.rendered && (j.status === "done" || j.preview)) { this.render(); }
    else if (this.rendered) { this.update(); }
    else { view.innerHTML = this.working(j); }
    if (j.status !== "done") this.timer = setTimeout(() => this.poll(), 1200);
  }
  stepsHtml(j) {
    const st = (j.stage || "").toLowerCase();
    let k = /read/.test(st) ? 0 : /prepar/.test(st) ? 1 : /debug|draw/.test(st) ? 3 : 2;
    if (j.status === "queued") k = 0;
    const L = ["Reading the video", "Preparing frames for playback", "Measuring the fall", "Drawing the debug video"];
    return `<ol class="steps" role="status">${L.map((l, i) => `<li class="${i < k ? "done" : i === k ? "now" : ""}">${l}</li>`).join("")}</ol>`;
  }
  working(j) {
    return `<div class="card workcard"><h2>${esc(j.filename)}</h2>${this.stepsHtml(j)}<p class="hint">This usually takes under a minute.</p></div>`;
  }

  /* ---- layout ---- */
  render() {
    this.rendered = true;
    const j = this.data.job;
    view.innerHTML = `
      <div id="verdict"></div>
      <div class="bench">
        <section class="player" aria-label="Video player">
          <div class="ptabs">
            <div class="seg" role="group" aria-label="What to show">
              <button data-mode="original" aria-pressed="true">Video</button>
              <button data-mode="debug" aria-pressed="false">Debug view</button>
              <button data-mode="both" aria-pressed="false">Side by side</button>
            </div>
            <div class="opts" id="ovopts">
              <label><input type="checkbox" data-ov="ball" checked><span class="sw" style="border-color:#3ea6ff"></span>Tracked ball</label>
              <label><input type="checkbox" data-ov="trail" checked><span class="sw" style="border-color:#8fd0ff;border-style:dotted"></span>Path</label>
              <label><input type="checkbox" data-ov="earth" checked><span class="sw" style="border-color:#ffc400"></span>Where real gravity puts it</label>
            </div>
          </div>
          <div class="stage" id="stage">
            <div class="pane" id="pane-o"><div class="box" id="box-o"><video id="vo" muted playsinline preload="auto"></video><canvas id="ov"></canvas></div><span class="cap">Video</span></div>
            <div class="pane vis0" id="pane-d"><div class="box"><video id="vd" muted playsinline preload="auto"></video></div><span class="cap">What Galileo saw</span><div class="away" id="away" hidden>This frame is outside the part of the clip Galileo analysed.</div></div>
            <div class="working" id="working" hidden><div><div class="spin"></div><span id="working-t">Working…</span></div></div>
          </div>
          <div class="ctl">
            <div class="scrub" id="scrub"><canvas id="strip" height="22"></canvas><input type="range" id="range" min="0" max="0" value="0" aria-label="Frame"><div class="head" id="head"></div></div>
            <div class="btns">
              <button class="ib" id="b-start" title="First frame (Home)" aria-label="First frame">⏮</button>
              <button class="ib" id="b-m10" title="Back 10 frames (Shift+←)" aria-label="Back 10 frames">−10</button>
              <button class="ib" id="b-prev" title="Previous frame (←)" aria-label="Previous frame"><svg viewBox="0 0 24 24"><path d="M15.4 5.4 14 4l-8 8 8 8 1.4-1.4L8.8 12z"/></svg></button>
              <button class="ib play" id="b-play" title="Play / pause (Space)" aria-label="Play"><svg id="ico-play" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg></button>
              <button class="ib" id="b-next" title="Next frame (→)" aria-label="Next frame"><svg viewBox="0 0 24 24"><path d="m8.6 5.4 1.4-1.4 8 8-8 8-1.4-1.4 6.6-6.6z"/></svg></button>
              <button class="ib" id="b-p10" title="Forward 10 frames (Shift+→)" aria-label="Forward 10 frames">+10</button>
              <button class="ib" id="b-end" title="Last frame (End)" aria-label="Last frame">⏭</button>
              <label class="speed">Speed <select id="rate"><option value="0.05">0.05×</option><option value="0.1">0.1×</option><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
              <label class="speed"><input type="checkbox" id="loop" checked> Loop</label>
              <div class="fr"><b id="fr-n">–</b><small id="fr-t"></small></div>
            </div>
            <p class="hint"><kbd>←</kbd> <kbd>→</kbd> step a frame · <kbd>Shift</kbd> jumps 10 · <kbd>Space</kbd> plays · click the chart to jump to a time</p>
          </div>
        </section>
        <aside class="side">
          <div class="card why" id="why"></div>
          <div class="card chartwrap" id="chartcard"><h2>Distance fallen</h2>
            <p class="legend"><span><i style="border-color:var(--blue)"></i>measured</span><span><i style="border-color:var(--ink)"></i>fit</span><span><i style="border-color:var(--gold);border-top-style:dashed"></i>real gravity</span></p>
            <canvas id="chart" width="560" height="340"></canvas></div>
        </aside>
      </div>
      <div class="sections" id="sections"></div>`;
    this.vo = $("#vo"); this.vd = $("#vd");
    this.fallbackOnError(this.vo, "preview"); this.fallbackOnError(this.vd, "debug");
    this.bind();
    this.update();
  }

  /* ---- data that changes as the job finishes ---- */
  update() {
    const j = this.data.job, r = this.data.result;
    if (this.frames && !this.videosReady) this.setupVideos();
    const w = $("#working");
    if (w) { w.hidden = j.status === "done"; $("#working-t").innerHTML = this.stepsHtml(j); }
    if (j.status === "done" && r && !this.done) {
      this.done = true;
      this.frames = this.data.frames;
      this.result = r;
      this.prepare();
      $("#chartcard").hidden = false;
      this.drawVerdict();
      this.drawSections();
      if (j.debug) { this.vd.src = `/media/${this.id}/debug.mp4`; }
      else $("[data-mode=debug]").disabled = $("[data-mode=both]").disabled = true;
      this.drawStrip(); this.go(this.idx, true);
    } else if (!this.done) {
      $("#verdict").innerHTML = "";
      $("#why").innerHTML = `<h2>Measuring…</h2><p class="hint">You can already step through the video while Galileo works.</p>`;
      $("#chartcard").hidden = true;
      $("[data-mode=debug]").disabled = $("[data-mode=both]").disabled = true;
    }
  }

  /* Browsers without H.264 get the same video as WebM (made on first use). */
  fallbackOnError(v, which) {
    v.addEventListener("error", () => {
      if (v.dataset.fb || !v.getAttribute("src") || !v.getAttribute("src").endsWith(".mp4")) return;
      v.dataset.fb = "1";
      const w = $("#working"); if (w && this.done) { $("#working-t").textContent = "Converting the video for this browser…"; w.hidden = false; }
      v.src = `/media/${this.id}/${which}.webm`;
      v.addEventListener("loadeddata", () => { if (w && this.done) w.hidden = true; this.go(this.idx, true); }, { once: true });
    });
  }

  setupVideos() {
    this.videosReady = true;
    const T = this.frames.times;
    this.n = T.length;
    const dts = []; for (let i = 1; i < T.length; i++) dts.push(T[i] - T[i - 1]);
    dts.sort((a, b) => a - b);
    this.dt = dts.length ? dts[dts.length >> 1] : 1 / 30;
    this.vo.src = `/media/${this.id}/preview.mp4`;
    const rg = $("#range"); rg.max = this.n - 1;
    const box = $("#box-o");
    box.style.aspectRatio = `${this.frames.width} / ${this.frames.height}`;
    this.vo.addEventListener("loadeddata", () => this.go(this.idx, true), { once: true });
    this.vo.addEventListener("ended", () => { if (this.loop) this.go(0, true), this.play(); else this.pause(); });
    window.addEventListener("resize", () => this.drawOverlay());
    this.go(0, true);
  }

  prepare() {
    const r = this.result;
    this.track = (r.track && r.track.points) || [];
    this.ghost = (r.ghost && r.ghost.points) || [];
    this.W = this.frames.width; this.H = this.frames.height;
    // per-frame lookups by nearest time
    const T = this.frames.times;
    const near = (pts) => {
      const out = new Array(T.length).fill(null);
      if (!pts.length) return out;
      let k = 0;
      for (let i = 0; i < T.length; i++) {
        while (k + 1 < pts.length && Math.abs(pts[k + 1].t - T[i]) <= Math.abs(pts[k].t - T[i])) k++;
        if (Math.abs(pts[k].t - T[i]) <= this.dt * 0.55) out[i] = k;
      }
      return out;
    };
    // Track times come from the original timeline (seconds from the first frame), same as T.
    this.trackAt = near(this.track);
    this.ghostAt = near(this.ghost);
    const fl = (r.fit && r.fit.flights) || [];
    this.flights = fl;
    const k = (r.verdict && r.verdict.k) || (r.clock && r.clock.k) || 1;
    this.k = k;
    const p = r.plot;
    this.plotOrigin = null;
    if (p && p.points && p.points.length && fl.length) this.plotOrigin = fl[0].t0 - p.points[0][0] * k;
  }

  /* ---- verdict ---- */
  drawVerdict() {
    const r = this.result, v = r.verdict || {}, j = this.data.job;
    const st = v.verdict || "UNSURE";
    const conf = v.confidence ? `${Math.round(v.confidence * 100)}% confident` : "";
    $("#verdict").innerHTML = `<section class="verdict ${st}">
      <div><h1 class="vword">${VERDICT[st] || st}</h1>
        <p class="vhead">${esc(v.headline || "")}</p>
        <p class="vmeta">${[conf, j.filename, j.elapsed ? `measured in ${num(j.elapsed, 0)} s` : ""].filter(Boolean).map(esc).join("  ·  ")}</p></div>
      <div class="gauge">${this.gauge(v)}</div></section>`;
    this.countUp();
    this.drawWhy();
  }
  countUp() {
    const el = document.querySelector(".gauge .cu"); if (!el) return;
    const to = +el.dataset.v;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const t0 = performance.now(), D = 900;
    const f = (t) => { const p = Math.min(1, (t - t0) / D), e = 1 - Math.pow(1 - p, 3); el.textContent = (to * e).toFixed(1); if (p < 1) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  }
  drawWhy() {
    const v = this.result.verdict || {};
    const rs = (v.reasons || []).slice(0, 4);
    $("#why").innerHTML = `<h2>Why</h2><ul class="whylist">${rs.map((x) => `<li>${esc(x)}</li>`).join("") || "<li>No reasons recorded.</li>"}</ul>
      <button class="linkbtn" type="button" data-open="checks">See every check</button>`;
    $("#why").querySelector("[data-open]").onclick = () => this.openDebug("checks");
  }
  gauge(v) {
    if (v.g == null) return `<p class="nog"><b>No acceleration measured.</b> ${esc(v.stage ? "Stopped at: " + v.stage + "." : "")} Open “What it did” below for the reasons, or pick the ball by name and size when you upload.</p>`;
    const MAXG = Math.max(14, Math.ceil((v.g + (v.sigma || 0)) / 2) * 2 + 2), W = 640, L = 12, R = 628;
    const X = (g) => L + (R - L) * Math.min(g, MAXG) / MAXG;
    const ref = v.g_ref || 9.8, tol = v.tol || 0.15;
    const col = { REAL: "var(--real)", AI: "var(--ai)", SLOWMO: "var(--slow)", UNSURE: "var(--unsure)" }[v.verdict] || "var(--ink)";
    let ticks = "";
    for (let g = 0; g <= MAXG; g += 2) ticks += `<line x1="${X(g)}" x2="${X(g)}" y1="62" y2="68" stroke="var(--line2)"/><text x="${X(g)}" y="82" text-anchor="middle">${g}</text>`;
    const lo = Math.max(0, v.g - (v.sigma || 0)), hi = v.g + (v.sigma || 0);
    const note = v.k && Math.abs(v.k - 1) > 0.02 ? `Corrected for ${num(v.k, 1)}× slow motion` : "";
    return `<svg viewBox="0 0 ${W} 112" role="img" aria-label="Measured ${num(v.g, 1)} metres per second squared against Earth's ${ref}">
      <text class="gnum" x="${L}" y="26"><tspan class="cu" data-v="${v.g}">${num(v.g, 1)}</tspan> <tspan font-size="14" font-weight="600" fill="var(--ink2)">m/s²${v.sigma ? ` ± ${num(v.sigma, 1)}` : ""}</tspan></text>
      ${note ? `<text class="gnote" x="${R}" y="24" text-anchor="end">${esc(note)}</text>` : ""}
      <rect x="${L}" y="44" width="${R - L}" height="18" rx="9" fill="var(--paper)"/>
      <rect x="${X(ref * (1 - tol))}" y="44" width="${X(ref * (1 + tol)) - X(ref * (1 - tol))}" height="18" fill="var(--realSoft)" stroke="var(--real)" stroke-opacity=".5"/>
      <line x1="${X(ref)}" x2="${X(ref)}" y1="38" y2="68" stroke="var(--real)" stroke-width="2"/>
      <text x="${X(ref)}" y="100" text-anchor="middle" style="fill:var(--real);font-weight:700">Earth ${ref}</text>
      <g class="mk" style="--dx:${X(0) - X(v.g)}px"><rect x="${X(lo)}" y="49" width="${Math.max(3, X(hi) - X(lo))}" height="8" rx="4" fill="${col}" opacity=".3"/>
      <circle cx="${X(v.g)}" cy="53" r="9" fill="${col}" stroke="#fff" stroke-width="3"/></g>
      ${ticks}</svg>`;
  }

  /* ---- sections ---- */
  drawSections() {
    const r = this.result, notes = r.notes || [];
    const raw = `<section class="card"><h2>Everything it recorded</h2><p class="hint">The full result, exactly as saved.</p><div class="tree" id="tree"></div>
      <div class="dl"><a class="btn ghost small" href="/download/${this.id}/result.json">Download result.json</a><a class="btn ghost small" href="/download/${this.id}/track.csv">Download tracked points (CSV)</a>${this.data.job.debug ? `<a class="btn ghost small" href="/media/${this.id}/debug.mp4" download>Download debug video</a>` : ""}</div></section>`;
    const tabs = [
      ["frame", "This frame", `<div class="card live" id="live"></div>`],
      ["checks", "Checks", `<section class="card"><h2>Checks gravity can't fake</h2>${this.testsTable(r.tests || [])}</section>`],
      ["fall", "The fall", this.fitCard(r)],
      ["scale", "Scale and clock", this.scaleClockCard(r)],
      ["camera", "Camera", this.cameraCard(r)],
      ["object", "Object", this.objectCard(r)],
      ["flights", "Flights", this.flightsCard(r)],
      ["points", "Tracked points", this.pointsCard()],
      ["file", "File", this.fileCard(r)],
      ["notes", "Notes", notes.length ? `<section class="card"><h2>Notes from the analysis</h2>${notes.map((n) => `<p class="note">${esc(n)}</p>`).join("")}</section>` : "", notes.length],
      ["raw", "All data", raw],
    ].filter((t) => t[2]);
    this.tabs = tabs.map((t) => t[0]);
    $("#sections").innerHTML = `${this.remeasureCard(r)}
      <section class="debug" id="debug">
        <button class="dbg-toggle" id="dbg-toggle" type="button" aria-expanded="false" aria-controls="dbg-body">
          <span><b>Debug details</b><small>Checks, fit, camera, tracked points and raw data</small></span>
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="m6 9 6 6 6-6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
        <div id="dbg-body" hidden>
          <div class="tabs" role="tablist" aria-label="Debug details">${tabs.map((t, i) => `<button role="tab" id="tab-${t[0]}" aria-controls="panel-${t[0]}" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}" data-tab="${t[0]}">${esc(t[1])}${t[3] ? `<span class="badge">${t[3]}</span>` : ""}</button>`).join("")}</div>
          ${tabs.map((t, i) => `<div class="panel" role="tabpanel" id="panel-${t[0]}" aria-labelledby="tab-${t[0]}" ${i ? "hidden" : ""} tabindex="0">${t[2]}</div>`).join("")}
        </div>
      </section>`;
    $("#tree").appendChild(this.tree(r, true));
    $("#dbg-toggle").onclick = () => this.openDebug(null, true);
    const list = $(".tabs");
    list.addEventListener("click", (e) => { const b = e.target.closest("[data-tab]"); if (b) this.selectTab(b.dataset.tab); });
    list.addEventListener("keydown", (e) => {
      const i = this.tabs.indexOf(document.activeElement.dataset.tab); if (i < 0) return;
      let j = null;
      if (e.key === "ArrowRight") j = (i + 1) % this.tabs.length; else if (e.key === "ArrowLeft") j = (i - 1 + this.tabs.length) % this.tabs.length;
      else if (e.key === "Home") j = 0; else if (e.key === "End") j = this.tabs.length - 1;
      if (j !== null) { e.preventDefault(); this.selectTab(this.tabs[j], true); }
    });
    $("#sections").addEventListener("click", (e) => {
      const tr = e.target.closest("tr[data-i]"); if (tr) { this.pause(); this.go(+tr.dataset.i); window.scrollTo({ top: $(".bench").offsetTop - 70, behavior: "smooth" }); }
    });
    const rs = $("#rm-obj");
    if (rs) {
      rs.onchange = () => { $("#rm-size-wrap").hidden = rs.value !== "custom"; };
      $("#rm-go").onclick = async () => {
        const err = $("#rm-err"); err.hidden = true;
        if (rs.value === "custom" && !$("#rm-size").value) { err.textContent = "Enter the object's size in millimetres."; err.hidden = false; return; }
        $("#rm-go").disabled = true;
        try {
          const n = await api(`/api/jobs/${this.id}/rerun`, { method: "POST", headers: { "content-type": "application/json" },
            body: JSON.stringify({ object: rs.value, size_mm: $("#rm-size").value }) });
          location.hash = `#/job/${n.id}`;
        } catch (e) { err.textContent = e.message; err.hidden = false; $("#rm-go").disabled = false; }
      };
    }
  }
  selectTab(id, focus) {
    document.querySelectorAll("[data-tab]").forEach((b) => { const on = b.dataset.tab === id; b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; if (on && focus) b.focus(); });
    document.querySelectorAll(".panel").forEach((p) => { p.hidden = p.id !== "panel-" + id; });
    if (id === "frame") this.drawLive();
  }
  openDebug(tab, toggle) {
    const body = $("#dbg-body"), btn = $("#dbg-toggle");
    const open = toggle ? body.hidden : true;
    body.hidden = !open; btn.setAttribute("aria-expanded", open);
    if (open && tab) { this.selectTab(tab); body.scrollIntoView({ behavior: "smooth", block: "start" }); }
  }
  kv(rows, cls = "") {
    const rs = rows.filter((x) => x && x[1] !== undefined && x[1] !== null && x[1] !== "");
    return rs.length ? `<dl class="kv ${cls}">${rs.map(([k, val]) => `<dt>${esc(k)}</dt><dd>${val}</dd>`).join("")}</dl>` : "";
  }
  testsTable(tests) {
    if (!tests.length) return `<p class="hint">No checks ran.</p>`;
    return `<div class="tw"><table><tbody>${tests.map((t) => `<tr><td style="width:1%"><span class="pill ${esc(t.status)}">${esc({ pass: "Passed", fail: "Failed", warn: "Doubtful", skip: "Skipped" }[t.status] || t.status)}</span></td>
      <td><b>${esc(t.name)}</b><br>${esc(t.summary)}${t.detail && Object.keys(t.detail).length ? `<details class="t"><summary>Numbers</summary>${this.kv(Object.entries(t.detail).map(([k, x]) => [this.nice(k), this.fmt(x)]))}</details>` : ""}</td></tr>`).join("")}</tbody></table></div>`;
  }
  nice(k) { return k.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()); }
  fmt(x) {
    if (x === null || x === undefined) return "–";
    if (typeof x === "number") return Math.abs(x) >= 1000 || (Math.abs(x) < 0.01 && x !== 0) ? x.toPrecision(4) : String(+x.toFixed(4));
    if (typeof x === "boolean") return x ? "yes" : "no";
    if (Array.isArray(x)) return x.map((y) => this.fmt(y)).join(", ");
    if (typeof x === "object") return esc(JSON.stringify(x));
    return esc(x);
  }
  fitCard(r) {
    const f = r.fit || {}, v = r.verdict || {}, rs = f.rolling_shutter || {};
    return `<section class="card"><h2>The fall</h2>${this.kv([
      ["Measured g", v.g != null ? `${num(v.g, 2)} ± ${num(v.sigma, 2)} m/s²` : null],
      ["Earth's g", `${num(v.g_ref, 2)} m/s²`],
      ["Ratio to Earth", v.ratio != null ? `${num(v.ratio * 100, 0)}%` : null],
      ["Allowed band", v.tol != null ? `±${num(v.tol * 100, 0)}%` : null],
      ["Stopped at", v.stage],
      ["Acceleration in the picture", f.a_px != null ? `${num(f.a_px, 1)} ± ${num(f.a_sigma, 1)} px/s²` : null],
      ["Without air-drag correction", f.a_px_naive != null && f.a_px_drag_corrected != null ? `${num(f.a_px_naive, 1)} px/s²` : null],
      ["Object width", f.diam_px != null ? `${num(f.diam_px, 1)} px` : null],
      ["Distance fallen", f.fall_diameters != null ? `${num(f.fall_diameters, 1)} widths` : null],
      ["Points in the fit", f.n_points],
      ["Jitter", f.jitter_px != null ? `${num(f.jitter_px, 2)} px` : null],
      ["Distance change", f.depth_change_rel != null ? `${num(f.depth_change_rel * 100, 1)}%` : null],
      ["Rolling shutter", rs.alpha != null ? `${rs.axis || ""} axis, effect ${num((rs.effect_rel || 0) * 100, 2)}%` : null],
      ["Thrown", f.launch ? (f.launch.thrown ? "yes" : "no, dropped") : null],
      ["Air drag", r.drag ? `factor ${num(r.drag.factor, 3)} (top speed ${num(r.drag.vt, 1)} m/s)` : "not applied"],
      ["Camera error added", f.camera_error_rel != null ? `±${num(f.camera_error_rel * 100, 1)}%` : null],
      ["Analysis time", r.elapsed_s != null ? `${num(r.elapsed_s, 1)} s` : null],
    ])}</section>`;
  }
  scaleClockCard(r) {
    const s = r.scale, c = r.clock, st = r.stabilization;
    const rulers = s && s.rulers ? `<h3>Rulers</h3>${s.rulers.map((x) => `<p style="margin:4px 0"><b>${esc(x.name)}</b>: ${esc(x.detail)}</p>`).join("")}` : "";
    const ev = c && c.evidence && c.evidence.length ? `<h3>Clock evidence</h3>${c.evidence.map((x) => `<p style="margin:4px 0"><b>${esc(x.name)}</b> (${num(x.k, 2)}×, ${num((x.confidence || 0) * 100, 0)}%): ${esc(x.detail)}</p>`).join("")}` : "";
    return `<section class="card"><h2>Scale and clock</h2>${this.kv([
      ["Metres per pixel", s ? `${(s.s * 1000).toFixed(3)} mm` : null],
      ["Ruler uncertainty", s ? `±${num(s.rel_sigma * 100, 1)}%` : null],
      ["Rulers agree", s ? this.fmt(s.agree) : null],
      ["Real seconds per video second", c ? `${num(c.k, 2)}×` : null],
      ["Clock source", c ? c.source : null],
      ["Clock confidence", c && c.evidence && c.evidence.length ? `${num(c.confidence * 100, 0)}%` : (c ? "no clock evidence found" : null)],
      ["Capture rate", c && c.fps_file ? `${num(c.fps_file, 1)} fps` : null],
      ["Slow motion on playback", c && c.slowmo_playback ? `${num(c.slowmo_playback, 1)}×` : null],
      ["Camera shake removed", st ? `${num(st.max_drift_px, 1)} px at most` : null],
      ["Zoom pass", r.zoom ? `ball ${num(r.zoom.ball_px_before, 0)} → ${num(r.zoom.ball_px, 0)} px` : null],
    ])}${rulers}${ev}</section>`;
  }
  cameraCard(r) {
    const c = r.camera || {};
    if (!r.camera) return "";
    return `<section class="card"><h2>Camera</h2>${this.kv([
      ["Looks down by", c.pitch != null ? `${num(c.pitch, 1)}° ± ${num(c.pitch_sigma, 1)}` : null],
      ["Rolled by", c.roll != null ? `${num(c.roll, 1)}°` : null],
      ["Levelled in analysis", c.pitched || c.rolled ? "yes" : "no, close enough to level"],
      ["Focal length", c.f_frac != null ? `${num(c.f_frac, 2)} × picture width (±${num((c.f_rel || 0) * 100, 0)}%)` : null],
      ["Focal length from", c.f_source],
      ["Fall vs room's verticals", c.fall_vs_room && c.fall_vs_room.deg != null ? `${num(c.fall_vs_room.deg, 1)}° off (±${num(c.fall_vs_room.sigma, 1)})` : null],
      ["Why", c.why || c.hint],
    ], "left")}</section>`;
  }
  objectCard(r) {
    const o = (r.objects || [])[0];
    if (!o) return "";
    const id = o.identity || {}, ol = o.outline || {};
    const others = (r.objects || []).slice(1);
    return `<section class="card"><h2>The object</h2>${this.kv([
      ["Named as", id.label], ["How it was named", id.source], ["Naming confidence", id.confidence != null ? `${num(id.confidence * 100, 0)}%` : null],
      ["Real size", id.size_m ? `${num(id.size_m * 1000, 1)} mm (±${num((id.rel_unc || 0) * 100, 1)}%)` : null],
      ["Shape", o.shape], ["Outline", ol.why],
      ["Roundness", ol.circularity != null ? num(ol.circularity, 2) : null],
      ["Width flicker", ol.width_scatter != null ? `${num(ol.width_scatter * 100, 1)}%` : null],
    ], "left")}
    ${others.length ? `<h3>Other falling objects</h3><div class="tw"><table><thead><tr><th>#</th><th>Object</th><th class="num">g (m/s²)</th><th>Result</th></tr></thead><tbody>${others.map((x) => `<tr><td>${x.n}</td><td>${esc(x.label || "?")}</td><td class="num">${x.g != null ? num(x.g, 2) : "–"}</td><td><span class="pill ${(x.verdict || {}).verdict || "UNSURE"}">${VERDICT[(x.verdict || {}).verdict] || "Can't tell"}</span></td></tr>`).join("")}</tbody></table></div>` : ""}</section>`;
  }
  flightsCard(r) {
    const fl = (r.fit && r.fit.flights) || [];
    if (!fl.length) return "";
    const ex = new Set((r.fit.excluded_flights) || []);
    return `<section class="card"><h2>Flights</h2><p class="hint">Each flight is one stretch between bounces, fitted on its own. Click a row to jump to it.</p><div class="tw"><table>
      <thead><tr><th>#</th><th class="num">From</th><th class="num">To</th><th class="num">Frames</th><th class="num">Acceleration (px/s²)</th><th class="num">Fit error (px)</th><th>Used</th></tr></thead><tbody>
      ${fl.map((f, i) => `<tr class="click" data-i="${this.frameAt(f.t0)}"><td>${i + 1}</td><td class="num">${num(f.t0, 3)} s</td><td class="num">${num(f.t1, 3)} s</td><td class="num">${f.n}</td><td class="num">${num(f.a_px, 1)} ± ${num(f.a_sigma, 1)}</td><td class="num">${num(f.rms_px, 2)}</td><td>${ex.has(i) ? "left out" : "yes"}</td></tr>`).join("")}</tbody></table></div></section>`;
  }
  remeasureCard(r) {
    const v = r.verdict || {}, o = (r.objects || [])[0] || {}, id = o.identity || {};
    const stuck = v.verdict === "UNSURE" && /ruler|uncertain/.test(v.stage || "");
    const named = id.label ? `Galileo took it for <b>${esc(id.label)}</b> (${esc(id.source === "you" ? "your choice" : id.source || "guess")}${id.confidence != null && id.source !== "you" ? `, ${num(id.confidence * 100, 0)}% sure` : ""}).` : "Galileo couldn't tell what is falling.";
    const presets = (STATUS.presets || []).map((p) => `<option value="${esc(p.key)}">${esc(p.label)} (${p.size_mm} mm)</option>`).join("");
    return `<section class="card" id="remeasure"><details ${stuck ? "open" : ""}><summary style="cursor:pointer;font-weight:700;font-size:17px">${stuck ? "Tell Galileo what is falling" : "Wrong object? Measure again"}</summary>
      <p>${stuck ? "Galileo couldn't settle on a size for the object, so the result isn't reliable. " : ""}${named} Pick what it is, or give its size, and the same video is measured again.</p>
      <div class="row2"><label class="field">What is falling?<select id="rm-obj">${presets}<option value="custom">Something else (I'll enter its size)</option></select></label>
      <label class="field" id="rm-size-wrap" hidden>Size in millimetres<input id="rm-size" type="number" min="1" max="5000" step="any"></label></div>
      <p class="err" id="rm-err" hidden></p><button class="btn" id="rm-go" type="button" style="margin-top:12px">Measure again</button></details></section>`;
  }
  fileCard(r) {
    const f = r.file || {}, w = r.work || {}, tags = f.tags || {};
    return `<section class="card"><h2>The file</h2><div class="grid2"><div>${this.kv([
      ["Name", esc((this.data.job.filename || f.path || "").split(/[\\/]/).pop())], ["Size", `${f.width} × ${f.height} px`], ["Frame rate", `${num(f.fps, 2)} fps`],
      ["Frames", f.n_frames], ["Length", `${num(f.duration, 2)} s`], ["Codec", f.codec],
      ["Capture rate tag", f.capture_fps ? `${num(f.capture_fps, 1)} fps` : "none"], ["Rotation", f.rotation ? `${f.rotation}°` : "none"],
      ["Timestamps from", f.timestamps_from], ["Analysed at", w.width ? `${w.width} × ${w.height} px, ${w.frames} frames` : null],
      ["Measured frame rate", w.fps_measured ? `${num(w.fps_measured, 2)} fps` : null], ["Version", r.version],
    ], "left")}</div><div><h3 style="margin-top:0">Metadata tags</h3>${Object.keys(tags).length ? this.kv(Object.entries(tags).map(([k, x]) => [k, esc(x)]), "left") : '<p class="hint">None. Generated and re-encoded clips often have few or none.</p>'}</div></div></section>`;
  }
  pointsCard() {
    const P = this.track;
    if (!P.length) return "";
    return `<section class="card"><h2>Tracked points</h2><p class="hint">${P.length} positions of the ball. Click a row to jump to that frame.</p><div class="scroll" id="pts"><table>
      <thead><tr><th class="num">Frame</th><th class="num">Time (s)</th><th class="num">x (% of width)</th><th class="num">y (% of height)</th><th class="num">Width (px)</th></tr></thead><tbody>
      ${P.map((p) => { const i = this.frameAt(p.t); return `<tr class="click" data-i="${i}" id="pt-${i}"><td class="num">${i + 1}</td><td class="num">${num(p.t, 4)}</td><td class="num">${num(p.x * 100, 2)}</td><td class="num">${num(p.y * 100, 2)}</td><td class="num">${num(2 * p.r * this.W, 1)}</td></tr>`; }).join("")}</tbody></table></div></section>`;
  }
  tree(o, open) {
    const wrap = document.createElement("div");
    const leaf = (k, v) => { const d = document.createElement("div"); d.className = "lf"; d.innerHTML = `<span class="k">${esc(k)}:</span> ${esc(typeof v === "number" ? +v.toPrecision(7) : v)}`; return d; };
    const build = (key, val, parent, isOpen) => {
      if (val !== null && typeof val === "object") {
        const entries = Array.isArray(val) ? val.map((x, i) => [i, x]) : Object.entries(val);
        const d = document.createElement("details"); if (isOpen) d.open = true;
        const s = document.createElement("summary"); s.textContent = `${key} ${Array.isArray(val) ? `[${val.length}]` : `{${entries.length}}`}`;
        d.appendChild(s);
        let built = false;
        const fill = () => { if (built) return; built = true; entries.slice(0, 400).forEach(([k, x]) => build(k, x, d, false)); if (entries.length > 400) d.appendChild(leaf("…", `${entries.length - 400} more (see result.json)`)); };
        d.addEventListener("toggle", () => { if (d.open) fill(); });
        if (isOpen) fill();
        parent.appendChild(d);
      } else parent.appendChild(leaf(key, val));
    };
    build("result", o, wrap, open);
    return wrap;
  }

  /* ---- player ---- */
  bind() {
    document.querySelectorAll("[data-mode]").forEach((b) => b.onclick = () => this.setMode(b.dataset.mode));
    document.querySelectorAll("[data-ov]").forEach((c) => c.onchange = () => { this.show[c.dataset.ov] = c.checked; this.drawOverlay(); });
    $("#b-start").onclick = () => { this.pause(); this.go(0); };
    $("#b-end").onclick = () => { this.pause(); this.go(this.n - 1); };
    $("#b-prev").onclick = () => this.step(-1);
    $("#b-next").onclick = () => this.step(1);
    $("#b-m10").onclick = () => this.step(-10);
    $("#b-p10").onclick = () => this.step(10);
    $("#b-play").onclick = () => (this.playing ? this.pause() : this.play());
    $("#rate").onchange = (e) => { this.rate = +e.target.value; this.vo.playbackRate = this.rate; this.vd.playbackRate = this.rate; };
    $("#loop").onchange = (e) => { this.loop = e.target.checked; };
    const rg = $("#range");
    rg.oninput = () => { this.pause(); this.go(+rg.value); };
    $("#chart").onclick = (e) => this.chartClick(e);
  }
  onKey(e) {
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== "checkbox") return;
    if (!this.videosReady) return;
    const k = e.key;
    if (k === "ArrowRight" || k === ".") { e.preventDefault(); this.step(e.shiftKey ? 10 : 1); }
    else if (k === "ArrowLeft" || k === ",") { e.preventDefault(); this.step(e.shiftKey ? -10 : -1); }
    else if (k === " ") { e.preventDefault(); this.playing ? this.pause() : this.play(); }
    else if (k === "Home") { e.preventDefault(); this.pause(); this.go(0); }
    else if (k === "End") { e.preventDefault(); this.pause(); this.go(this.n - 1); }
  }
  setMode(m) {
    this.mode = m;
    document.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === m)));
    $("#stage").classList.toggle("both", m === "both");
    $("#pane-o").classList.toggle("vis0", m === "debug");
    $("#pane-d").classList.toggle("vis0", m === "original");
    $("#ovopts").style.visibility = m === "debug" ? "hidden" : "visible";
    this.go(this.idx, true);
  }
  tOrig(i) { const T = this.frames.times; const nx = i + 1 < T.length ? T[i + 1] : T[i] + this.dt; return (T[i] + nx) / 2; }
  tDebug(i) { const d = this.frames.debug_times; return d && d[i] != null ? d[i] + this.dt / 2 : null; }
  frameAt(t) { const T = this.frames.times; let lo = 0, hi = T.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (T[m] <= t + 1e-6) lo = m; else hi = m - 1; } return lo; }
  step(d) { this.pause(); this.go(this.idx + d); }
  go(i, force) {
    if (!this.frames) return;
    i = Math.max(0, Math.min(this.n - 1, i));
    const changed = i !== this.idx;
    this.idx = i;
    if (!this.playing || force) {
      if (this.vo.readyState > 0 || force) { try { this.vo.currentTime = this.tOrig(i); } catch (e) { /* not ready */ } }
      this.syncDebug();
    }
    if (changed || force) this.refresh();
  }
  syncDebug() {
    if (!this.done || !this.data.job.debug || this.mode === "original") return;
    const t = this.tDebug(this.idx);
    $("#away").hidden = t !== null;
    if (t !== null) { try { this.vd.currentTime = t; } catch (e) { /* not ready */ } }
  }
  play() {
    if (!this.videosReady) return;
    this.playing = true;
    $("#b-play").setAttribute("aria-label", "Pause");
    $("#ico-play").innerHTML = '<path d="M6 5h4v14H6zm8 0h4v14h-4z"/>';
    this.vo.playbackRate = this.rate;
    if (this.idx >= this.n - 1) this.go(0, true);
    this.vo.play().catch(() => this.pause());
    this.tick();
  }
  pause() {
    if (!this.playing) return;
    this.playing = false;
    $("#b-play").setAttribute("aria-label", "Play");
    $("#ico-play").innerHTML = '<path d="M8 5v14l11-7z"/>';
    this.vo.pause(); this.vd.pause();
    cancelAnimationFrame(this.raf);
    this.go(this.idx, true);
  }
  tick() {
    if (!this.playing) return;
    const i = this.frameAt(this.vo.currentTime);
    if (i !== this.idx) {
      this.idx = i;
      if (this.mode !== "original" && this.done && this.data.job.debug) {
        const t = this.tDebug(i);
        $("#away").hidden = t !== null;
        if (t === null) this.vd.pause();
        else {
          if (Math.abs(this.vd.currentTime - t) > 0.12 || this.vd.paused) { this.vd.currentTime = t - this.dt / 2; }
          if (this.vd.paused) { this.vd.playbackRate = this.rate; this.vd.play().catch(() => {}); }
        }
      }
      this.refresh();
    }
    this.raf = requestAnimationFrame(() => this.tick());
  }

  /* ---- per-frame drawing ---- */
  refresh() {
    const i = this.idx, T = this.frames.times;
    $("#range").value = i;
    $("#fr-n").textContent = `Frame ${i + 1} of ${this.n}`;
    $("#fr-t").textContent = `${T[i].toFixed(3)} s · source frame ${this.frames.nums[i]}`;
    const head = $("#head"), w = $("#scrub").clientWidth;
    head.style.left = `${(this.n > 1 ? i / (this.n - 1) : 0) * (w - 3)}px`;
    if (!this.done) return;
    this.drawOverlay(); this.drawLive(); this.drawChart();
    const row = document.getElementById("pt-" + i);
    document.querySelectorAll("#pts tr.cur").forEach((r) => r.classList.remove("cur"));
    if (row) { row.classList.add("cur"); if (this.playing === false) { const sc = $("#pts"); if (sc && sc.contains(row)) sc.scrollTop = row.offsetTop - sc.clientHeight / 2; } }
  }
  drawOverlay() {
    const cv = $("#ov"); if (!cv || !this.done) return;
    const b = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(b.width * dpr); cv.height = Math.round(b.height * dpr);
    const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, b.width, b.height);
    const W = b.width, H = b.height, i = this.idx;
    const ti = this.trackAt[i], gi = this.ghostAt[i];
    c.lineWidth = 2;
    if (this.show.trail) {
      c.strokeStyle = "rgba(143,208,255,.75)"; c.setLineDash([3, 4]); c.beginPath();
      let started = false;
      for (let k = 0; k < this.track.length; k++) { const p = this.track[k]; if (p.t > this.frames.times[i] + 1e-6) break; const x = p.x * W, y = p.y * H; if (!started) { c.moveTo(x, y); started = true; } else c.lineTo(x, y); }
      c.stroke(); c.setLineDash([]);
    }
    if (this.show.earth && gi !== null) {
      const g = this.ghost[gi]; const rad = ti !== null ? this.track[ti].r * W : 10;
      c.strokeStyle = "#ffc400"; c.lineWidth = 2.5; c.beginPath(); c.arc(g.x * W, g.y * H, Math.max(rad, 4), 0, 7); c.stroke();
    }
    if (this.show.ball && ti !== null) {
      const p = this.track[ti];
      c.strokeStyle = "#3ea6ff"; c.lineWidth = 2.5; c.beginPath(); c.arc(p.x * W, p.y * H, Math.max(p.r * W, 4), 0, 7); c.stroke();
    }
  }
  drawLive() {
    const i = this.idx, ti = this.trackAt[i], gi = this.ghostAt[i], T = this.frames.times, r = this.result, v = r.verdict || {};
    const p = ti !== null ? this.track[ti] : null, g = gi !== null ? this.ghost[gi] : null;
    const fl = this.flights.findIndex((f) => T[i] >= f.t0 - 1e-6 && T[i] <= f.t1 + 1e-6);
    let gap = null;
    if (p && g) gap = Math.hypot((p.x - g.x) * this.W, (p.y - g.y) * this.H);
    // Speed from neighbouring track points, in metres per second where a ruler exists
    let speed = null;
    const s = r.scale && r.scale.s, fw = r.fit && r.fit.diam_px, wk = r.work;
    if (p && ti > 0 && ti < this.track.length - 1 && s && wk) {
      const a = this.track[ti - 1], b = this.track[ti + 1], dt = b.t - a.t;
      if (dt > 0 && b.t - a.t < 6 * this.dt) {
        // px in the analysis scale -> m: the final width in px (diam_px) matches the ball's 2r in fractions of the width
        const pxw = fw && p.r ? fw / (2 * p.r) : wk.width;
        speed = Math.hypot((b.x - a.x) * pxw, (b.y - a.y) * pxw * (this.H / this.W)) * s * (v.k || 1) / dt;
      }
    }
    $("#live").innerHTML = `<h2>This frame ${p ? '<span class="tag">ball tracked</span>' : '<span class="tag off">no track here</span>'}</h2>${this.kv([
      ["Frame", `${i + 1} of ${this.n}`], ["Time", `${T[i].toFixed(3)} s`],
      ["Ball position", p ? `${num(p.x * 100, 1)}%, ${num(p.y * 100, 1)}%` : null],
      ["Ball width", p ? `${num(2 * p.r * this.W, 1)} px` : null],
      ["Flight", fl >= 0 ? `${fl + 1} of ${this.flights.length}` : (p ? "between flights" : null)],
      ["Speed", speed != null ? `${num(speed, 2)} m/s` : null],
      ["Where real gravity puts it", g ? `${num(g.x * 100, 1)}%, ${num(g.y * 100, 1)}%` : null],
      ["Gap to that spot", gap != null ? `${num(gap, 1)} px` : null],
    ])}`;
  }
  drawStrip() {
    const cv = $("#strip"), dpr = window.devicePixelRatio || 1, w = cv.clientWidth;
    cv.width = w * dpr; cv.height = 22 * dpr;
    const c = cv.getContext("2d"); c.scale(dpr, dpr);
    const n = this.n, X = (i) => (n > 1 ? i / (n - 1) : 0) * w;
    c.fillStyle = "#e7eaee"; c.beginPath(); c.roundRect(0, 0, w, 22, 5); c.fill();
    const d = this.frames.debug_times;
    if (d) { // analysed window
      let a = d.findIndex((x) => x !== null), b = d.length - 1 - [...d].reverse().findIndex((x) => x !== null);
      if (a >= 0) { c.fillStyle = "#cfd8e2"; c.fillRect(X(a), 0, Math.max(2, X(b) - X(a)), 22); }
    }
    const cols = ["#3ea6ff", "#1f5fd1", "#16795a", "#8a6410", "#bf3b27", "#7a4fd1"];
    this.flights.forEach((f, k) => { const a = this.frameAt(f.t0), b = this.frameAt(f.t1); c.fillStyle = cols[k % cols.length]; c.fillRect(X(a), 6, Math.max(2, X(b) - X(a)), 10); });
    c.fillStyle = "rgba(20,32,43,.35)";
    this.trackAt.forEach((t, i) => { if (t !== null) c.fillRect(X(i), 17, 1, 5); });
  }

  /* ---- chart ---- */
  chartGeom() {
    const p = this.result && this.result.plot; if (!p) return null;
    const pts = p.points, all = pts.map((q) => q[0]).concat(p.earth.map((q) => q[0])), ys = pts.map((q) => q[1]).concat(p.earth.map((q) => q[1]), p.fit.map((q) => q[1]));
    const x1 = Math.max(...all), y1 = Math.max(...ys, 0.01), y0 = Math.min(0, ...ys);
    return { p, x0: 0, x1, y0, y1 };
  }
  drawChart() {
    const cv = $("#chart"), g = this.chartGeom();
    if (!g) { $("#chartcard").hidden = true; return; }
    const dpr = window.devicePixelRatio || 1, cssW = cv.clientWidth || 560, cssH = cssW * 0.6;
    cv.width = cssW * dpr; cv.height = cssH * dpr; cv.style.height = cssH + "px";
    const c = cv.getContext("2d"); c.scale(dpr, dpr);
    const m = { l: 46, r: 10, t: 8, b: 34 }, w = cssW - m.l - m.r, h = cssH - m.t - m.b;
    const X = (t) => m.l + (t - g.x0) / (g.x1 - g.x0) * w, Y = (d) => m.t + h - (d - g.y0) / (g.y1 - g.y0) * h;
    this.cg = { m, w, h, g, cssW };
    c.font = "12px " + getComputedStyle(document.body).fontFamily; c.fillStyle = "#51606e"; c.strokeStyle = "#dde2e7"; c.lineWidth = 1;
    const nice = (span, n) => { const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag; };
    const sx = nice(g.x1, 5), sy = nice(g.y1 - g.y0, 4);
    c.textAlign = "center";
    for (let t = 0; t <= g.x1 + 1e-9; t += sx) { c.beginPath(); c.moveTo(X(t), m.t); c.lineTo(X(t), m.t + h); c.stroke(); c.fillText(t.toFixed(sx < 1 ? 2 : 1), X(t), m.t + h + 15); }
    c.textAlign = "right";
    for (let d = Math.ceil(g.y0 / sy) * sy; d <= g.y1 + 1e-9; d += sy) { c.beginPath(); c.moveTo(m.l, Y(d)); c.lineTo(m.l + w, Y(d)); c.stroke(); c.fillText(d.toFixed(sy < 1 ? 2 : 1), m.l - 6, Y(d) + 4); }
    c.textAlign = "center"; c.fillText("time (s of real time)", m.l + w / 2, cssH - 4);
    c.save(); c.translate(11, m.t + h / 2); c.rotate(-Math.PI / 2); c.fillText("fallen (m)", 0, 0); c.restore();
    const line = (arr, col, dash) => { c.strokeStyle = col; c.lineWidth = 2.5; c.setLineDash(dash || []); c.beginPath(); arr.forEach((q, k) => (k ? c.lineTo(X(q[0]), Y(q[1])) : c.moveTo(X(q[0]), Y(q[1])))); c.stroke(); c.setLineDash([]); };
    line(g.p.earth, "#d9a400", [7, 5]);
    line(g.p.fit, "#14202b");
    const cols = ["#1f5fd1", "#16795a", "#8a6410", "#bf3b27", "#7a4fd1"];
    g.p.points.forEach((q) => { c.fillStyle = cols[q[2] % cols.length]; c.beginPath(); c.arc(X(q[0]), Y(q[1]), 3, 0, 7); c.fill(); });
    if (this.plotOrigin !== null) {
      const tc = (this.frames.times[this.idx] - this.plotOrigin) / this.k;
      if (tc >= g.x0 && tc <= g.x1) { c.strokeStyle = "#bf3b27"; c.lineWidth = 1.5; c.beginPath(); c.moveTo(X(tc), m.t); c.lineTo(X(tc), m.t + h); c.stroke(); }
    }
  }
  chartClick(e) {
    if (!this.cg || this.plotOrigin === null) return;
    const b = e.target.getBoundingClientRect(), x = e.clientX - b.left, { m, w, g } = this.cg;
    const t = g.x0 + (x - m.l) / w * (g.x1 - g.x0), tv = this.plotOrigin + t * this.k;
    this.pause(); this.go(this.frameAt(tv));
  }
}

route();
