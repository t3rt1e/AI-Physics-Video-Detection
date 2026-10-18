"use strict";
/* Novex detector page. Hash routes: #/ (new), #/history, #/job/<id> */

const $ = (s, r = document) => r.querySelector(s);
const view = $("#view");
const VERDICT = {
  REAL: "Likely real", AI: "Likely AI-generated", SLOWMO: "Real, slow motion", UNSURE: "Inconclusive",
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

/* Shared, small UI helpers. The backend supplies every preset and verdict. */
const icons = {
  upload: '<path d="M12 16V4m-4 4 4-4 4 4M4 15v5h16v-5"/>',
  arrow: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  compare: '<rect x="3" y="5" width="7" height="14" rx="2"/><rect x="14" y="5" width="7" height="14" rx="2"/>',
  play: '<path d="m9 5 11 7-11 7Z"/>',
  file: '<path d="M14 3H5v18h14V8Zm0 0v5h5M8 12h8M8 16h5"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  orbit: '<circle cx="12" cy="12" r="3"/><ellipse cx="12" cy="12" rx="10" ry="5" transform="rotate(-35 12 12)"/>',
};
function icon(name, cls = '') { return `<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true">${icons[name] || icons.file}</svg>`; }
function toast(message) {
  const box = $("#toast"); box.textContent = message; box.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { box.hidden = true; }, 3600);
}
function cssColor(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function summaryText(v = {}) {
  return { REAL: "The measured motion is consistent with real-world gravity.",
    AI: "The detector found motion that does not match its physics checks.",
    SLOWMO: "The motion fits real-world gravity after accounting for slow motion.",
    UNSURE: "There isn't enough reliable evidence for a clear result." }[v.verdict]
    || "The detector returned a result. Open the details to explore it.";
}
function stateLabel(j) { return j.status === "done" ? VERDICT[j.verdict] || j.verdict || "Inconclusive" : ({running:"Analyzing",queued:"In queue",error:"Couldn't analyze"}[j.status] || j.status); }
function sampleLabel(s) {
  return s.name.includes('floaty') ? 'Low-gravity drop' : s.name.includes('easing') ? 'Uneven motion' : s.name.includes('240') ? 'High frame rate' : 'Earth-gravity drop';
}

/* ---------------------------------------------------------------- router */
let routeVersion = 0;
window.addEventListener("hashchange", route);
async function route() {
  const version = ++routeVersion;
  if (current) { current.destroy(); current = null; }
  const h = location.hash.replace(/^#\/?/, "");
  document.querySelectorAll(".top nav a").forEach((a) => { a.classList.remove("on"); a.removeAttribute("aria-current"); });
  view.innerHTML = '<div class="loading-state" role="status"><div class="skeleton"></div><p>Getting things ready…</p></div>';
  if (!STATUS) {
    try { STATUS = await api("/api/status"); }
    catch (e) {
      if (version !== routeVersion) return;
      view.innerHTML = `<div class="empty"><h1>Can't reach NØVEX</h1><p>Check that the local server is running, then try again.</p><button class="btn" id="retry">Try again</button></div>`;
      $("#retry").onclick = route; return;
    }
  }
  if (version !== routeVersion) return;
  window.scrollTo(0, 0);
  let active = '.nav-new';
  if (/^job\/[a-zA-Z0-9_-]+$/.test(h)) { current = new JobPage(h.slice(4)); }
  else if (h === "history") { active = '.nav-hist'; current = historyPage(); }
  else if (h === "compare" || /^compare\/[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$/.test(h)) {
    active = '.nav-compare'; current = h === 'compare' ? new CompareUpload() : new ComparePage(h.split('/').slice(1));
  }
  else { current = homePage(); }
  $(active).classList.add("on"); $(active).setAttribute("aria-current", "page");
}

/* ------------------------------------------------------------------ home */
function homePage() {
  const controller = new AbortController();
  view.innerHTML = `
    <section class="home" aria-labelledby="home-title">
      <div class="lede">
        <span class="eyebrow"><i></i> A CLOSER LOOK AT REALITY</span>
        <h1 id="home-title">FOLLOW<br>THE <span class="decode-word" aria-label="fall"><span aria-hidden="true" data-decode="FALL">FALL</span></span><span class="hero-period">.</span></h1>
        <p>A video can look convincing.<br>See what its motion has to say.</p>
        <button class="hero-anchor" type="button" id="start-analysis">Start an analysis <span>↘</span></button>
      </div>
      <div class="motion-study" data-light aria-label="Illustration of a falling object's path, not an analysis result">
        <div class="study-label"><span><i></i> MOTION STUDY</span><span>001 / GRAVITY</span></div>
        <div class="study-grid" aria-hidden="true"></div>
        <svg class="study-art" viewBox="0 0 480 300" aria-hidden="true">
          <defs><radialGradient id="sphere" cx="30%" cy="22%" r="80%"><stop stop-color="#ffcf9c"/><stop offset=".38" stop-color="#f59b5b"/><stop offset="1" stop-color="#6e341b"/></radialGradient></defs>
          <path class="study-guide" d="M60 48H420M60 140H420M60 232H420M60 35V250M240 35V250M420 35V250"/>
          <path class="study-trajectory" d="M80 52Q250 52 357 221"/>
          <path class="study-projection" d="M357 221H420M357 221V252"/>
          <g class="study-points"><circle cx="80" cy="52" r="4"/><circle cx="140" cy="59" r="5"/><circle cx="203" cy="84" r="6"/><circle cx="260" cy="123" r="7"/><circle cx="311" cy="171" r="8"/></g>
          <circle class="study-halo" cx="357" cy="221" r="35"/>
          <circle cx="357" cy="221" r="23" fill="url(#sphere)"/>
          <path class="study-brackets" d="M312 209v-28h28m34 0h28v28m0 25v28h-28m-34 0h-28v-28"/>
          <text x="77" y="34" class="study-coordinate">RELEASE</text><text x="102" y="240" class="study-coordinate">THE PATH TELLS A STORY.</text>
        </svg>
        <div class="study-bottom"><span>Small motion.<br><b>Extraordinary evidence.</b></span><span class="study-axis" aria-hidden="true">↓<small>g</small></span></div>
      </div>
    </section>
    <form class="drop home-upload" id="form" novalidate data-light>
      <div class="form-title" id="analysis-start"><span class="step-number">01 /</span><div><h2>Bring a clip. Follow the evidence.</h2><p>Best with a clearly visible object falling or bouncing.</p></div><span class="form-label">NEW ANALYSIS ${icon('upload')}</span></div>
      ${uploadMarkup('single')}
      <div class="upload-action"><span class="upload-note">Your file is processed by this NØVEX server.</span><button class="btn primary" id="go" type="submit" disabled>Analyze video ${icon('arrow')}</button></div>
      <p class="err" id="err" role="alert" hidden></p>
    </form>
    <section class="sample-section" aria-labelledby="sample-heading">
      <div class="section-heading"><div><span class="eyebrow">02 / THE EXAMPLE COLLECTION</span><h2 id="sample-heading">Know the motion. See the result.</h2></div><span class="hint">Synthetic clips. Real physics checks.</span></div>
      <div class="sample-grid" id="samples">${(STATUS.samples || []).map((sample, i) => `<button class="sample-card ${sample.ai_generated ? 'sample-orange' : ''}" type="button" data-light data-name="${esc(sample.name)}"><span class="sample-top"><span>STUDY / 0${i+1}</span><span class="sample-arrow">↗</span></span><svg class="sample-art" viewBox="0 0 240 80" aria-hidden="true"><path class="sample-baseline" d="M10 66H230"/><path d="M20 14Q120 ${sample.ai_generated ? '8 212 35' : '14 212 61'}"/><circle cx="212" cy="${sample.ai_generated ? '35' : '61'}" r="7"/></svg><span class="sample-copy"><b>${esc(sampleLabel(sample))}</b><small>${sample.ai_generated ? 'Simulated physics anomaly' : 'Simulated natural motion'}</small></span></button>`).join('') || '<p class="hint">No sample clips installed.</p>'}</div>
    </section>
    <a class="compare-callout" href="#/compare" data-light><span class="compare-callout-icon">${icon('compare')}</span><span><span class="eyebrow">A SECOND PERSPECTIVE</span><b>Two clips. A clearer picture.</b><small>Compare motion and results side by side.</small></span><span class="callout-arrow">↗</span></a>
    <aside class="proverb-note" aria-labelledby="proverb-title">
      <div class="proverb-script"><span class="proverb-characters" lang="zh-Hans">以假乱真</span><span class="proverb-pinyin" lang="zh-Latn-pinyin">Yǐ jiǎ luàn zhēn</span></div>
      <div class="proverb-copy"><span class="proverb-label">A CHINESE PROVERB</span><h2 id="proverb-title">Substitute a Fake for the Real</h2><p>A convincing imitation can blur what is real.<br>A reminder to look closer in the age of AI video.</p></div>
    </aside>`;
  $('#start-analysis').onclick = () => { $('#analysis-start').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'}); $('#single-file').focus({preventScroll:true}); };
  const upload = new UploadInput('single', () => { $("#go").disabled = !upload.file; });
  const go = $("#go"), err = $("#err");
  const fail = (e) => { if (controller.signal.aborted) return; err.textContent = e.message; err.hidden = false; go.disabled = !upload.file; go.innerHTML = `Analyze video ${icon('arrow')}`; };
  $("#form").onsubmit = async (ev) => {
    ev.preventDefault(); err.hidden = true;
    if (!upload.validate()) return;
    go.disabled = true; go.textContent = 'Uploading your clip…'; upload.setDisabled(true);
    try { const r = await upload.submit(controller.signal); if (!controller.signal.aborted) location.hash = `#/job/${r.id}`; }
    catch (e) { fail(e); }
    finally { if (!controller.signal.aborted) upload.setDisabled(false); }
  };
  $("#samples").onclick = async (ev) => {
    const b = ev.target.closest('[data-name]'); if (!b || b.disabled) return;
    document.querySelectorAll('[data-name]').forEach(x => x.disabled = true);
    try {
      const r = await api('/api/jobs/sample', {method:'POST', signal:controller.signal, headers:{'content-type':'application/json'}, body:JSON.stringify({name:b.dataset.name})});
      if (!controller.signal.aborted) location.hash = `#/job/${r.id}`;
    } catch (e) { fail(e); document.querySelectorAll('[data-name]').forEach(x => x.disabled = false); }
  };
  return { destroy() { controller.abort(); } };
}

/* --------------------------------------------------------------- history */
function historyPage() {
  const controller = new AbortController();
  view.innerHTML = `<div class="page-heading"><div><span class="eyebrow">YOUR WORKSPACE</span><h1>Past analyses</h1><p>Every clip, with its evidence ready to revisit.</p></div><a class="btn ghost" href="#/">${icon('upload')} New analysis</a></div><div class="history-toolbar" id="history-toolbar" hidden><span id="history-count">Choose two results to compare.</span><button class="btn small" id="compare-selected" disabled>Compare selected ${icon('compare')}</button></div><div class="hist" id="hist" role="status"><div class="skeleton"></div></div>`;
  (async () => {
    try {
      const list = await api('/api/jobs', {signal:controller.signal});
      if (controller.signal.aborted) return;
      const box = $("#hist"); box.removeAttribute('role');
      if (!list.length) { box.innerHTML = `<div class="empty">${icon('file')}<h2>Your first result belongs here.</h2><p>Analyze a clip to start building your collection.</p><a class="btn" href="#/">Analyze a video</a></div>`; return; }
      $("#history-toolbar").hidden = false;
      box.innerHTML = list.map(j => `<div class="hrow"><label class="history-select"><input type="checkbox" value="${esc(j.id)}" aria-label="Select ${esc(j.filename)} for comparison"></label><a class="history-link" href="#/job/${esc(j.id)}"><span class="file-glyph">${icon('file')}</span><span class="history-title"><b class="nm">${esc(j.filename)}</b><span class="sub">${new Date((j.created || 0) * 1000).toLocaleString([], {dateStyle:'medium',timeStyle:'short'})}</span></span><span class="pill ${esc(j.status === 'done' ? j.verdict : j.status)}">${esc(stateLabel(j))}</span><span class="sample-arrow">↗</span></a></div>`).join('');
      box.onchange = () => {
        const chosen = [...box.querySelectorAll('input:checked')];
        box.querySelectorAll('input:not(:checked)').forEach(x => x.disabled = chosen.length >= 2);
        $("#compare-selected").disabled = chosen.length !== 2;
        $("#history-count").textContent = chosen.length ? `${chosen.length} of 2 clips selected` : 'Choose two results to compare.';
      };
      $("#compare-selected").onclick = () => { const ids = [...box.querySelectorAll('input:checked')].map(x => x.value); if (ids.length === 2) location.hash = `#/compare/${ids.join('/')}`; };
    } catch(e) { if (!controller.signal.aborted) $("#hist").innerHTML = `<div class="empty"><p>${esc(e.message)}</p><button class="btn" id="history-retry">Try again</button></div>`, $("#history-retry") && ($("#history-retry").onclick = route); }
  })();
  return {destroy() {controller.abort();}};
}

/* ------------------------------------------------------------- job page */
class JobPage {
  constructor(id) {
    this.id = id; this.dead = false; this.data = null; this.frames = null; this.idx = 0;
    this.mode = "original"; this.playing = false; this.rate = 1; this.loop = true;
    this.show = { ball: true, earth: false, trail: true };
    this.interacted = false;
    this.controller = new AbortController(); this.debugOpen = false;
    this.redraw = () => { if (this.done && !this.dead) { this.drawStrip(); this.drawOverlay(); if (this.debugOpen) this.drawChart(); } };
    window.addEventListener("resize", this.redraw); window.addEventListener("themechange", this.redraw);
    this.rendered = false; this.videosReady = false;
    this.keys = (e) => this.onKey(e);
    document.addEventListener("keydown", this.keys);
    view.innerHTML = `<p class="hint">Loading…</p>`;
    this.poll();
  }
  destroy() {
    this.dead = true; this.controller.abort(); clearTimeout(this.timer);
    document.removeEventListener("keydown", this.keys); window.removeEventListener("resize", this.redraw); window.removeEventListener("themechange", this.redraw);
    cancelAnimationFrame(this.raf); cancelAnimationFrame(this.countRaf);
    [this.vo, this.vd, this.independentVideo].filter(Boolean).forEach(v => {v.pause(); v.removeAttribute('src'); v.load();});
  }

  async poll() {
    if (this.dead) return;
    let d;
    try { d = await api(`/api/jobs/${this.id}`, { signal: this.controller.signal }); }
    catch (e) { if (this.dead) return; view.innerHTML = `<div class="empty"><h2>Could not load this analysis</h2><p>${esc(e.message)}</p><button class="btn" id="job-retry">Try again</button></div>`; this.rendered = false; this.videosReady = false; $("#job-retry").onclick = () => route(); return; }
    if (this.dead) return;
    this.data = d;
    const j = d.job;
    if (j.status === "error") { view.innerHTML = `<div class="card workcard"><span class="eyebrow">ANALYSIS INTERRUPTED</span><h2>This clip needs another look.</h2><p>Try a shorter clip with the falling object clearly visible.</p><details class="adv"><summary>Technical details</summary><p class="err">${esc(j.error)}</p></details><p><a class="btn" href="#/">Try another video</a></p></div>`; return; }
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
    return `<div class="card workcard"><h2>${esc(j.filename)}</h2>${this.stepsHtml(j)}<p class="hint">You can leave this page and find the result in History.</p></div>`;
  }

  /* ---- layout ---- */
  render() {
    this.rendered = true;
    const j = this.data.job;
    view.innerHTML = `
      <div class="result-page-heading"><a href="#/history">← All analyses</a><span>${esc(j.filename)}</span><a href="#/compare">Compare clips ↗</a></div>
      <div id="verdict"></div>
      <div class="bench">
        <section class="player" aria-label="Video player">
          <div class="player-heading"><span class="eyebrow"><i></i> THE MOTION</span><button type="button" class="tracking-toggle" id="tracking-toggle" aria-pressed="true" disabled>Tracking on ${icon("orbit")}</button></div>
          <div class="ptabs" id="analysis-tools" hidden>
            <div class="seg" role="group" aria-label="What to show">
              <button data-mode="original" aria-pressed="true">Video</button>
              <button data-mode="debug" aria-pressed="false">Debug view</button>
              <button data-mode="both" aria-pressed="false">Video + debug</button>
            </div>
            <div class="opts" id="ovopts">
              <label><input type="checkbox" data-ov="ball" checked><span class="sw" style="border-color:#3ea6ff"></span>Tracked object</label>
              <label><input type="checkbox" data-ov="trail" checked><span class="sw" style="border-color:#8fd0ff;border-style:dotted"></span>Path</label>
              <label><input type="checkbox" data-ov="earth"><span class="sw" style="border-color:#ffb86b"></span>Where real gravity puts it</label>
            </div>
          </div>
          <div class="stage" id="stage">
            <div class="pane" id="pane-o"><div class="box" id="box-o"><video id="vo" muted playsinline preload="auto"></video><canvas id="ov" aria-label="Detected object and measured path"></canvas></div><span class="cap">ORIGINAL + TRACKING</span></div>
            <div class="pane vis0" id="pane-d"><div class="box"><video id="vd" muted playsinline preload="auto"></video></div><span class="cap">What NØVEX saw</span><div class="away" id="away" hidden>This frame is outside the part of the clip NØVEX analysed.</div></div>
            <div class="working" id="working" hidden><div><div class="spin"></div><span id="working-t">Working…</span></div></div>
          </div>
          <div class="tracking-legend" id="tracking-legend"><span><i class="tracking-ring"></i>Object</span><span><i class="tracking-line"></i>Full measured path</span><small id="track-status">Waiting for tracking…</small></div>
          <div class="ctl">
            <div class="scrub" id="scrub"><canvas id="strip" height="22"></canvas><input type="range" id="range" min="0" max="0" value="0" aria-label="Frame"><div class="head" id="head"></div></div>
            <div class="btns">
              <button class="ib expert-control" id="b-start" hidden title="First frame (Home)" aria-label="First frame">⏮</button>
              <button class="ib expert-control" id="b-m10" hidden title="Back 10 frames (Shift+←)" aria-label="Back 10 frames">−10</button>
              <button class="ib" id="b-prev" title="Previous frame (←)" aria-label="Previous frame"><svg viewBox="0 0 24 24"><path d="M15.4 5.4 14 4l-8 8 8 8 1.4-1.4L8.8 12z"/></svg></button>
              <button class="ib play" id="b-play" title="Play / pause (Space)" aria-label="Play"><svg id="ico-play" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg></button>
              <button class="ib" id="b-next" title="Next frame (→)" aria-label="Next frame"><svg viewBox="0 0 24 24"><path d="m8.6 5.4 1.4-1.4 8 8-8 8-1.4-1.4 6.6-6.6z"/></svg></button>
              <button class="ib expert-control" id="b-p10" hidden title="Forward 10 frames (Shift+→)" aria-label="Forward 10 frames">+10</button>
              <button class="ib expert-control" id="b-end" hidden title="Last frame (End)" aria-label="Last frame">⏭</button>
              <label class="speed">Speed <select id="rate"><option value="0.05">0.05×</option><option value="0.1">0.1×</option><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
              <label class="speed expert-control" hidden><input type="checkbox" id="loop" checked> Loop</label>
              <div class="fr"><b id="fr-n">–</b><small id="fr-t"></small></div>
            </div>
            <p class="hint expert-control" hidden><kbd>←</kbd> <kbd>→</kbd> step a frame · <kbd>Shift</kbd> jumps 10 · <kbd>Space</kbd> plays · click the chart to jump to a time</p>
          </div>
        </section>
        <aside class="side">
          <div class="card gravity-card" id="gravity-card" hidden></div>
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
      const sync = !!this.frames.debug_times;
      $("[data-mode=debug]").disabled = $("[data-mode=both]").disabled = !sync;
      if (j.debug && sync) this.vd.src = `/media/${this.id}/debug.mp4`;
      this.drawStrip();
      const first = this.trackAt.findIndex(index => index !== null);
      // Start at the first actual detection so the highlight is useful on arrival.
      // Never interrupt someone already inspecting or playing the preview.
      this.go(!this.interacted && !this.playing && first >= 0 ? first : this.idx, true);
      $("#tracking-toggle").disabled = !this.track.length;
    } else if (!this.done) {
      $("#verdict").innerHTML = "";
      $("#why").innerHTML = `<h2>Measuring…</h2><p class="hint">You can already step through the video while NØVEX works.</p>`;
      $("#chartcard").hidden = true;
      $("[data-mode=debug]").disabled = $("[data-mode=both]").disabled = true;
    }
  }

  /* Browsers without H.264 get the same video as WebM (made on first use). */
  fallbackOnError(v, which) {
    v.addEventListener("error", () => {
      if (this.dead) return;
      if (v.dataset.fb || !v.getAttribute("src") || !v.getAttribute("src").endsWith(".mp4")) return;
      v.dataset.fb = "1";
      const w = $("#working"); if (w && this.done) { $("#working-t").textContent = "Converting the video for this browser…"; w.hidden = false; }
      v.src = `/media/${this.id}/${which}.webm`;
      v.addEventListener("loadeddata", () => { if (this.dead) return; if (w && this.done) w.hidden = true; this.go(this.idx, true); }, { once: true });
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
    this.vo.addEventListener("loadeddata", () => { if (!this.dead) this.go(this.idx, true); }, { once: true });
    this.vo.addEventListener("ended", () => { if (this.loop) this.go(0, true), this.play(); else this.pause(); });
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
    const v = this.result.verdict || {}, st = v.verdict || "UNSURE";
    const obj = this.result.objects?.[0]?.identity?.label || this.result.objects?.[0]?.label;
    $("#verdict").innerHTML = `<section class="verdict ${esc(st)}"><div><span class="eyebrow"><i></i> ANALYSIS COMPLETE</span><h1 class="vword">${esc(VERDICT[st] || st)}</h1><p class="vhead">${esc(summaryText(v))}</p></div><div class="result-seal" aria-hidden="true">${icon(st === 'REAL' || st === 'SLOWMO' ? 'check' : 'orbit')}</div>${obj ? `<p class="vmeta">OBJECT MEASURED <b>${esc(obj)}</b></p>` : ''}</section>`;
    const ref = v.g_ref ?? STATUS.g_ref ?? 9.8;
    $("#gravity-card").hidden = false;
    $("#gravity-card").innerHTML = `<span class="eyebrow">GRAVITY, IN CONTEXT</span><h2>How does it compare?</h2><div class="gravity-value"><strong>${Number.isFinite(v.g) ? num(v.g, 2) : '—'}</strong><span>m/s²<small>measured gravity</small></span></div><div class="gauge" id="gravity-gauge">${SiteVisuals.gauge(v, ref)}</div><p class="earth-reference">Earth reference <b>${num(ref,2)} m/s²</b></p>`;
    this.drawWhy();
  }
  drawWhy() {
    const v = this.result.verdict || {};
    const explanation = v.verdict === 'UNSURE' ? "The clip may need a clearer view, a longer fall, or a more accurate object size." : v.verdict === 'AI' ? "An unusual fall is a reason to look closer. It doesn't establish how the video was made." : "The measured fall passes the detector's checks. That alone can't establish the video's origin.";
    $("#why").innerHTML = `<span class="eyebrow">READING THE RESULT</span><h2>What this tells us.</h2><p>${explanation}</p><div class="why-divider"></div><p class="hint">The complete evidence is one click away.</p><button class="btn ghost small" type="button" data-open="overview">Technical details ${icon('arrow')}</button>`;
    $("#why [data-open]").onclick = () => this.openDebug('overview');
  }
  /* ---- sections ---- */
  drawSections() {
    const r = this.result, notes = r.notes || [];
    const raw = `<section class="card"><h2>Everything it recorded</h2><p class="hint">The full result, exactly as saved.</p><div class="tree" id="tree"></div>
      <div class="dl"><a class="btn ghost small" href="/download/${this.id}/result.json">Download result.json</a><a class="btn ghost small" href="/download/${this.id}/track.csv">Download tracked points (CSV)</a>${this.data.job.debug ? `<a class="btn ghost small" href="/media/${this.id}/debug.mp4" download>Download debug video</a>` : ""}</div></section>`;
    const tabs = [
      ["overview", "Overview", `<div class="grid2"><section class="card"><h3>From the detector</h3><p>${esc(r.verdict?.headline || 'No headline provided.')}</p><ul class="reasons">${(r.verdict?.reasons || []).map(x => `<li>${esc(x)}</li>`).join('')}</ul><p class="hint">${r.verdict?.confidence != null ? `Detector confidence: ${num(r.verdict.confidence * 100, 0)}% · not a calibrated probability of origin` : ''}</p>${this.frames.truncated ? '<p class="note">The playback preview contains the first 6,000 frames. The detector analyzes the original file using its own limits.</p>' : ''}</section><div id="chart-slot"></div></div>`],
      ["debugvideo", "Debug video", this.data.job.debug && !this.frames.debug_times ? `<p class="hint">This debug render has its own timeline. Use these controls to inspect it without guessing how it aligns with the original.</p><video id="independent-debug" class="independent-debug" controls playsinline muted preload="metadata"></video><div class="btns"><button class="ib" id="debug-back" aria-label="Previous debug frame">←</button><button class="ib" id="debug-next" aria-label="Next debug frame">→</button><span id="debug-frame-count" class="hint"></span></div>` : ''],
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
          <span><b>Technical details</b><small>Debug video, physics checks, charts, and all recorded data</small></span>
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="m6 9 6 6 6-6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
        <div id="dbg-body" hidden>
          <div class="tabs" role="tablist" aria-label="Technical details">${tabs.map((t, i) => `<button role="tab" id="tab-${t[0]}" aria-controls="panel-${t[0]}" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}" data-tab="${t[0]}">${esc(t[1])}${t[3] ? `<span class="badge">${t[3]}</span>` : ""}</button>`).join("")}</div>
          ${tabs.map((t, i) => `<div class="panel" role="tabpanel" id="panel-${t[0]}" aria-labelledby="tab-${t[0]}" ${i ? "hidden" : ""} tabindex="0">${t[2]}</div>`).join("")}
        </div>
      </section>`;
    $("#tree").appendChild(this.tree(r, true));
    $("#chart-slot").appendChild($("#chartcard"));
    const independent = $("#independent-debug");
    if (independent) {
      this.independentVideo = independent;
      independent.src = `/media/${this.id}/debug.mp4`;
      independent.onerror = () => { if (!independent.dataset.fb && !this.dead) {independent.dataset.fb = '1'; independent.src = `/media/${this.id}/debug.webm`;} };
      const times = this.frames.debug_frame_times || [];
      const index = () => { let lo=0,hi=times.length-1; while(lo<hi) {const m=(lo+hi+1)>>1;if(times[m]<=independent.currentTime+1e-6)lo=m;else hi=m-1;}return lo; };
      const step = delta => {independent.pause();const i=Math.max(0,Math.min(times.length-1,index()+delta));independent.currentTime=times[i]+((times[i+1] ?? times[i]+1/30)-times[i])/2;};
      $("#debug-back").disabled = $("#debug-next").disabled = !times.length;
      $("#debug-back").onclick = () => step(-1); $("#debug-next").onclick = () => step(1);
      independent.ontimeupdate = () => { if(!this.dead) $("#debug-frame-count").textContent = times.length ? `Frame ${index()+1} of ${times.length}` : ''; };
    }
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
    if (id === "overview") { this.drawChart(); }
    if (id !== "debugvideo") this.independentVideo?.pause();
  }
  openDebug(tab, toggle) {
    const body = $("#dbg-body"), btn = $("#dbg-toggle");
    const open = toggle ? body.hidden : true;
    this.debugOpen = open;
    $("#analysis-tools").hidden = !open;
    document.querySelectorAll('.expert-control').forEach(x => x.hidden = !open);
    if (!open) { this.setMode('original'); this.independentVideo?.pause(); }
    this.drawOverlay(); this.drawStrip();
    if(open) {this.drawChart();this.drawLive();}
    body.hidden = !open; btn.setAttribute("aria-expanded", open);
    if(open) requestAnimationFrame(() => {if(!this.dead) this.drawChart();});
    if (open && tab) { this.selectTab(tab); body.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }); }
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
      ["Metres per pixel", s?.s != null ? `${(s.s * 1000).toFixed(3)} mm` : null],
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
    const named = id.label ? `NØVEX took it for <b>${esc(id.label)}</b> (${esc(id.source === "you" ? "your choice" : id.source || "guess")}${id.confidence != null && id.source !== "you" ? `, ${num(id.confidence * 100, 0)}% sure` : ""}).` : "NØVEX couldn't tell what is falling.";
    const presets = (STATUS.presets || []).map((p) => `<option value="${esc(p.key)}">${esc(p.label)} (${p.size_mm} mm)</option>`).join("");
    return `<section class="card" id="remeasure"><details ${stuck ? "open" : ""}><summary style="cursor:pointer;font-weight:700;font-size:17px">${stuck ? "Tell NØVEX what is falling" : "Wrong object? Measure again"}</summary>
      <p>${stuck ? "NØVEX couldn't settle on a size for the object, so the result isn't reliable. " : ""}${named} Pick what it is, or give its size, and the same video is measured again.</p>
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
    document.querySelectorAll("[data-ov]").forEach((c) => c.onchange = () => { this.show[c.dataset.ov] = c.checked; this.syncTracking(); });
    $("#tracking-toggle").onclick = () => {
      const on = !(this.show.ball || this.show.trail);this.show.ball=on;this.show.trail=on;
      document.querySelectorAll('[data-ov=ball],[data-ov=trail]').forEach(input => input.checked=on);
      this.syncTracking();
    };
    $(".player").addEventListener('pointerdown', () => {this.interacted=true;}, {signal:this.controller.signal});
    $(".player").addEventListener('keydown', () => {this.interacted=true;}, {signal:this.controller.signal});
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
    if (e.defaultPrevented || /INPUT|SELECT|TEXTAREA|BUTTON|A|SUMMARY/.test(document.activeElement.tagName) || document.activeElement.isContentEditable) return;
    if (!this.videosReady) return;
    const k = e.key;
    if (["ArrowRight","ArrowLeft",".",","," ","Home","End"].includes(k)) this.interacted=true;
    if (k === "ArrowRight" || k === ".") { e.preventDefault(); this.step(e.shiftKey ? 10 : 1); }
    else if (k === "ArrowLeft" || k === ",") { e.preventDefault(); this.step(e.shiftKey ? -10 : -1); }
    else if (k === " ") { e.preventDefault(); this.playing ? this.pause() : this.play(); }
    else if (k === "Home") { e.preventDefault(); this.pause(); this.go(0); }
    else if (k === "End") { e.preventDefault(); this.pause(); this.go(this.n - 1); }
  }
  setMode(m) {
    this.mode = m;
    if (m === "original") this.vd.pause();
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
    $("#fr-t").textContent = `${T[i].toFixed(3)} s`;
    const head = $("#head"), w = $("#scrub").clientWidth;
    head.style.left = `${(this.n > 1 ? i / (this.n - 1) : 0) * (w - 3)}px`;
    if (!this.done) return;
    this.drawOverlay(); if (this.debugOpen) { this.drawLive(); this.drawChart(); }
    const row = document.getElementById("pt-" + i);
    document.querySelectorAll("#pts tr.cur").forEach((r) => r.classList.remove("cur"));
    if (row) { row.classList.add("cur"); if (this.playing === false) { const sc = $("#pts"); if (sc && sc.contains(row)) sc.scrollTop = row.offsetTop - sc.clientHeight / 2; } }
  }
  syncTracking() {
    const on=this.show.ball || this.show.trail;
    $("#tracking-toggle").setAttribute('aria-pressed',on);
    $("#tracking-toggle").innerHTML=`Tracking ${on ? 'on' : 'off'} ${icon('orbit')}`;
    $("#pane-o .cap").textContent=on ? 'ORIGINAL + TRACKING' : 'ORIGINAL';
    this.drawOverlay();
  }
  drawOverlay() {
    const cv = $("#ov"); if (!cv || !this.done) return;
    const gi=this.ghostAt[this.idx];
    const state=SiteVisuals.drawTracking(cv,{points:this.track,time:this.frames.times[this.idx],tolerance:this.dt*.55,width:this.W,height:this.H,
      ball:this.show.ball,trail:this.show.trail,ghost:this.show.earth && gi!==null ? this.ghost[gi] : null});
    $("#tracking-legend").hidden=!(this.show.ball || this.show.trail);
    $("#track-status").textContent=!state?.count ? 'No object track available' : state.tracked ? '' : 'No detection on this frame';
  }
  drawLive() {
    if (!$("#live")) return;
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
    c.fillStyle = cssColor("--line"); c.beginPath(); c.roundRect(0, 0, w, 22, 5); c.fill();
    if (!this.debugOpen) return;
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
    if (!Array.isArray(p.points) || !p.points.length || !Array.isArray(p.earth) || !Array.isArray(p.fit)) return null;
    const pts = p.points, all = pts.map((q) => q[0]).concat(p.earth.map((q) => q[0])), ys = pts.map((q) => q[1]).concat(p.earth.map((q) => q[1]), p.fit.map((q) => q[1]));
    const x1 = Math.max(...all, 0.001), y1 = Math.max(...ys, 0.01), y0 = Math.min(0, ...ys);
    return { p, x0: 0, x1, y0, y1 };
  }
  drawChart() {
    if (!$("#chart") || !this.debugOpen) return;
    const cv = $("#chart"), g = this.chartGeom();
    if (!g) { $("#chartcard").hidden = true; return; }
    const dpr = window.devicePixelRatio || 1, cssW = cv.clientWidth || 560, cssH = cssW * 0.6;
    cv.width = cssW * dpr; cv.height = cssH * dpr; cv.style.height = cssH + "px";
    const c = cv.getContext("2d"); c.scale(dpr, dpr);
    const m = { l: 46, r: 10, t: 8, b: 34 }, w = cssW - m.l - m.r, h = cssH - m.t - m.b;
    const X = (t) => m.l + (t - g.x0) / (g.x1 - g.x0) * w, Y = (d) => m.t + h - (d - g.y0) / (g.y1 - g.y0) * h;
    this.cg = { m, w, h, g, cssW };
    c.font = "12px " + getComputedStyle(document.body).fontFamily; c.fillStyle = cssColor("--ink2"); c.strokeStyle = cssColor("--line"); c.lineWidth = 1;
    const nice = (span, n) => { const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag; };
    const sx = nice(g.x1, 5), sy = nice(g.y1 - g.y0, 4);
    c.textAlign = "center";
    for (let t = 0; t <= g.x1 + 1e-9; t += sx) { c.beginPath(); c.moveTo(X(t), m.t); c.lineTo(X(t), m.t + h); c.stroke(); c.fillText(t.toFixed(sx < 1 ? 2 : 1), X(t), m.t + h + 15); }
    c.textAlign = "right";
    for (let d = Math.ceil(g.y0 / sy) * sy; d <= g.y1 + 1e-9; d += sy) { c.beginPath(); c.moveTo(m.l, Y(d)); c.lineTo(m.l + w, Y(d)); c.stroke(); c.fillText(d.toFixed(sy < 1 ? 2 : 1), m.l - 6, Y(d) + 4); }
    c.textAlign = "center"; c.fillText("time (s of real time)", m.l + w / 2, cssH - 4);
    c.save(); c.translate(11, m.t + h / 2); c.rotate(-Math.PI / 2); c.fillText("fallen (m)", 0, 0); c.restore();
    const line = (arr, col, dash) => { c.strokeStyle = col; c.lineWidth = 2.5; c.setLineDash(dash || []); c.beginPath(); arr.forEach((q, k) => (k ? c.lineTo(X(q[0]), Y(q[1])) : c.moveTo(X(q[0]), Y(q[1])))); c.stroke(); c.setLineDash([]); };
    line(g.p.earth, cssColor("--orange"), [7, 5]);
    line(g.p.fit, cssColor("--ink"));
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

$(".skip-link").onclick = e => { e.preventDefault(); view.focus(); };
route();
