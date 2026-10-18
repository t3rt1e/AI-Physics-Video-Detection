"use strict";

/* Novex front end: two clips side by side, plus the live tower strip.
   No framework, no network beyond this server, so it runs offline at the venue. */

const $ = (sel, el = document) => el.querySelector(sel);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const COLORS = {};
const VERDICT = {
  REAL: { word: "Real", icon: "✓" },
  AI: { word: "AI-generated", icon: "✕" },
  SLOWMO: { word: "Real, slow motion", icon: "◷" },
  UNSURE: { word: "Can't tell", icon: "?" },
};
const STATUS_MARK = { pass: "✓", warn: "!", fail: "✕", skip: "–" };
// Other falling objects in the overlay (the first one uses the clip colour).
const OBJECT_COLORS = ["#c0399b", "#2f8f3a", "#b8860b", "#1f6fb2"];
const STATUS_WORD = { pass: "Pass", warn: "Caution", fail: "Fail", skip: "Not tested" };

let STATUS = null;
let CLIPS = [];
const panels = {};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

function fmtG(g, s) {
  if (g == null) return "–";
  const d = g < 1 ? 2 : 2;
  return s != null ? `${g.toFixed(d)} ± ${s.toFixed(d)}` : g.toFixed(d);
}

function fmtDt(dt) {
  const inv = 1 / dt;
  if (Math.abs(inv - Math.round(inv)) < 1e-6 && inv >= 2) return `1/${Math.round(inv)} s`;
  return `${dt.toFixed(2)} s`;
}

function sizeCanvas(cv, w, h) {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(1, Math.round(w * dpr));
  const H = Math.max(1, Math.round(h * dpr));
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function nearestByT(pts, t) {
  if (!pts || !pts.length) return null;
  let lo = 0, hi = pts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (pts[mid].t < t) lo = mid + 1; else hi = mid;
  }
  const a = pts[Math.max(lo - 1, 0)], b = pts[lo];
  return Math.abs(a.t - t) <= Math.abs(b.t - t) ? a : b;
}

function interpD(pts, t) {
  // pts: [[t, d], ...] sorted by t
  if (!pts.length || t < pts[0][0] || t > pts[pts.length - 1][0]) return null;
  for (let i = 1; i < pts.length; i++) {
    if (pts[i][0] >= t) {
      const [t0, d0] = pts[i - 1], [t1, d1] = pts[i];
      return t1 === t0 ? d1 : d0 + (d1 - d0) * (t - t0) / (t1 - t0);
    }
  }
  return pts[pts.length - 1][1];
}

function niceStep(range, target) {
  const raw = range / Math.max(target, 1);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * p) return m * p;
  return 10 * p;
}

/* ------------------------------------------------------------------------ */
class Panel {
  constructor(root, side) {
    this.side = side;
    root.appendChild($("#panel-tpl").content.cloneNode(true));
    this.root = root;
    const q = (s) => $(s, root);
    Object.assign(this, {
      clipSel: q(".clip-select"), objSel: q(".object-select"), fileIn: q(".upload input"),
      btn: q(".analyze"), note: q(".clip-note"), video: q(".video"), overlay: q(".overlay"),
      busy: q(".busy"), key: q(".overlay-key"), msg: q(".message"), res: q(".result"),
      gNum: q(".g-num"), verdict: q(".verdict"), vIcon: q(".verdict-icon"), vWord: q(".verdict-word"),
      headline: q(".headline"), expl: q(".explanation"), reasons: q(".reasons"),
      plate: q(".plate-canvas"), plateCap: q(".plate-caption"), legend: q(".legend"),
      chart: q(".chart-canvas"), tip: q(".tooltip"), tests: q(".tests tbody"), facts: q(".facts"),
      pointsBody: q(".points tbody"), pickBtn: q(".pick"), pickClear: q(".pick-clear"),
      pickHint: q(".pick-hint"), pickLayer: q(".pick-layer"), wrap: q(".video-wrap"),
      objTable: q(".objects"), objBody: q(".objects tbody"), keyMore: q(".key-more"),
    });
    this.result = null;
    this.clipId = null;
    this.seed = null;      // {t, x, y}: a click on the ball, x and y as fractions of the picture
    this.picking = false;

    this.clipSel.addEventListener("change", () => this.setClip(this.clipSel.value));
    this.objSel.addEventListener("change", () => this.onObject());
    this.btn.addEventListener("click", (e) => this.measure({ force: e.shiftKey }));
    this.fileIn.addEventListener("change", () => this.upload());
    const redraw = () => this.drawOverlay();
    for (const ev of ["timeupdate", "seeked", "loadedmetadata", "pause"]) this.video.addEventListener(ev, redraw);
    this.video.addEventListener("play", () => this.loop());
    this.video.addEventListener("error", () => {
      // Browser can't decode this codec (HEVC, or H.264 in some builds):
      // switch to the server's WebM preview, which keeps the same timestamps.
      if (this.clipId && !this.video.src.includes("/preview/")) {
        this.video.src = `/preview/${this.clipId}`;
        return;
      }
      this.note.hidden = false;
      this.note.textContent = "This browser can't play the clip. The measurement still works; only the overlay needs playback.";
    });
    this.pickBtn.addEventListener("click", () => this.setPicking(!this.picking));
    this.pickClear.addEventListener("click", () => { this.seed = null; this.syncPick(); this.measure(); });
    this.pickLayer.addEventListener("click", (e) => this.onPick(e));
    this.chart.addEventListener("mousemove", (e) => this.hoverChart(e));
    this.chart.addEventListener("mouseleave", () => { this.tip.hidden = true; this.drawChart(); });
    new ResizeObserver(() => this.redrawAll()).observe(root);
  }

  fillClips(selected) {
    const groups = { drops: "Tower drops", clips: "Demo clips", uploads: "Uploads" };
    this.clipSel.innerHTML = "";
    for (const g of ["drops", "clips", "uploads"]) {
      const items = CLIPS.filter((c) => c.group === g);
      if (!items.length) continue;
      const og = document.createElement("optgroup");
      og.label = groups[g];
      for (const c of items) {
        const o = document.createElement("option");
        o.value = c.id;
        let tag = "";
        if (c.synthetic) tag = " (synthetic)";
        else if (c.ai_generated) tag = " (AI-generated)";
        const base = c.description ? c.description.charAt(0).toUpperCase() + c.description.slice(1) : c.name;
        o.textContent = base + tag;
        o.title = c.name;
        og.appendChild(o);
      }
      this.clipSel.appendChild(og);
    }
    if (!CLIPS.length) {
      const o = document.createElement("option");
      o.textContent = "No clips yet: upload one";
      o.value = "";
      this.clipSel.appendChild(o);
    }
    if (selected) this.clipSel.value = selected;
  }

  fillObjects() {
    this.objSel.innerHTML = "";
    const add = (v, t, parent = this.objSel) => {
      const o = document.createElement("option"); o.value = v; o.textContent = t; parent.appendChild(o);
    };
    const auto = STATUS.auto_id || {};
    add("auto", auto.clip ? "Work it out (recognise each object)" : "Work it out (needs CLIP or Gemini)");
    const shapes = STATUS.shapes || {};
    for (const [title, want] of [["Balls (diameter)", "round"], ["Other objects (longest side)", "rigid"]]) {
      const og = document.createElement("optgroup");
      og.label = title;
      for (const [k, label] of Object.entries(STATUS.presets)) {
        if ((shapes[k] || "round") === want) add(k, label, og);
      }
      if (og.children.length) this.objSel.appendChild(og);
    }
    add("custom", "Known size (enter mm)…");
    add("other", "Something else (name it)…");
    if (STATUS.gemini.available) add("gemini", "Ask Gemini what it is");
    this.objSel.value = STATUS.default_object;
    this.sizeMm = null;
    this.otherName = null;
  }

  onObject() {
    if (this.objSel.value === "other") {
      // Anything at all: a name, and its size if you know it ("rotisserie chicken 22cm").
      const v = window.prompt("What is falling? Add its size if you know it (a ball's diameter, or the longest side), e.g. \"rotisserie chicken 22cm\":", this.otherName || "");
      const txt = (v || "").trim();
      if (!txt) { this.objSel.value = STATUS.default_object; return; }
      const m = txt.match(/^(.*?)[\s=:]+([0-9.]+)\s*(mm|cm|m|in|inch|inches|ft)?$/i);
      this.otherName = m ? `${m[1].trim()}=${m[2]}${(m[3] || "mm").toLowerCase()}` : txt;
      const o = [...this.objSel.options].find((x) => x.value === "other");
      if (o) o.textContent = `Something else: ${txt}`;
    }
    if (this.objSel.value === "custom") {
      const v = window.prompt("Size of the falling object in millimetres (a ball's diameter, or the longest side of anything else):", this.sizeMm || "38");
      const mm = parseFloat(v);
      if (!(mm > 0)) { this.objSel.value = STATUS.default_object; return; }
      this.sizeMm = mm;
    }
    if (this.clipId) this.measure();
  }

  /* -- pick the ball: one click tells the tracker which object to follow -- */
  setPicking(on) {
    this.picking = on && !!this.clipId;
    if (this.picking) this.video.pause();
    this.syncPick();
  }

  syncPick() {
    this.pickBtn.setAttribute("aria-pressed", this.picking ? "true" : "false");
    this.pickBtn.textContent = this.picking ? "Cancel pick" : (this.seed ? "Pick again" : "Pick the ball");
    this.pickHint.hidden = !this.picking;
    this.pickLayer.hidden = !this.picking;
    this.pickClear.hidden = !this.seed || this.picking;
    this.drawOverlay();
  }

  pictureRect() {
    // Where the picture sits inside the player (object-fit: contain leaves bars).
    const v = this.video;
    const W = this.wrap.clientWidth, H = this.wrap.clientHeight;
    if (!v.videoWidth || !W || !H) return null;
    const vr = v.videoWidth / v.videoHeight, er = W / H;
    if (vr > er) return { dw: W, dh: W / vr, ox: 0, oy: (H - W / vr) / 2 };
    return { dw: H * vr, dh: H, ox: (W - H * vr) / 2, oy: 0 };
  }

  onPick(e) {
    const g = this.pictureRect();
    if (!g) return;
    const rect = this.wrap.getBoundingClientRect();
    const x = (e.clientX - rect.left - g.ox) / g.dw;
    const y = (e.clientY - rect.top - g.oy) / g.dh;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    this.seed = { t: this.video.currentTime, x, y };
    this.picking = false;
    this.syncPick();
    this.measure();
  }

  setClip(id, { measure = true } = {}) {
    if (!id) return;
    this.clipId = id;
    this.clipSel.value = id;
    this.seed = null;
    this.picking = false;
    this.syncPick();
    const c = CLIPS.find((x) => x.id === id) || {};
    this.video.src = `/media/${id}`;
    this.result = null;
    this.res.hidden = true;
    this.msg.hidden = true;
    this.key.hidden = true;
    const bits = [];
    if (c.description) bits.push(/[.!?]$/.test(c.description) ? c.description : c.description + ".");
    if (c.synthetic) bits.push("Synthetic rehearsal clip, rendered with known physics.");
    if (c.ai_generated) bits.push("Generated by AI.");
    this.note.hidden = !bits.length;
    this.note.textContent = bits.join(" ");
    this.drawOverlay();
    if (measure) this.measure();
  }

  async upload() {
    const f = this.fileIn.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    this.setBusy(true, "Uploading…");
    try {
      const { id } = await api("/api/upload", { method: "POST", body: fd });
      await refreshClips();
      this.setClip(id);
    } catch (e) {
      this.setBusy(false);
      this.showMessage(`Upload failed: ${e.message}`);
    } finally {
      this.fileIn.value = "";
    }
  }

  setBusy(on, text = "Measuring…") {
    this.busy.hidden = !on;
    $(".busy-text", this.busy).textContent = text;
    this.btn.disabled = on;
  }

  showMessage(text) {
    this.msg.hidden = false;
    this.msg.textContent = text;
  }

  async measure({ force = false } = {}) {
    if (!this.clipId) return;
    const id = this.clipId;
    this.setBusy(true);
    this.msg.hidden = true;
    const t0 = performance.now();
    try {
      const body = {
        clip: id, object: this.objSel.value === "other" ? this.otherName : this.objSel.value,
        size_mm: this.sizeMm,
        gemini: STATUS.gemini.default_on || this.objSel.value === "gemini", force,
        seed: this.seed,
      };
      const r = await api("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      if (this.clipId !== id) return; // user moved on
      this.show(r);
    } catch (e) {
      this.showMessage(`Couldn't measure this clip: ${e.message}`);
    } finally {
      if (this.clipId === id) this.setBusy(false);
      this.lastMs = performance.now() - t0;
    }
  }

  showLive(msg) {
    // A finished tower drop: load its clip and result without re-measuring.
    refreshClips().then(() => {
      this.clipId = msg.clip;
      this.seed = null;
      this.picking = false;
      this.syncPick();
      this.fillClips(msg.clip);
      this.video.src = `/media/${msg.clip}`;
      this.note.hidden = false;
      this.note.textContent = `Drop ${msg.drop_id} from the tower` + (msg.synthetic ? " (simulated tower and camera)." : ".");
      this.show(msg.result);
      this.video.play().catch(() => {});
    });
  }

  show(r) {
    this.result = r;
    this.prepTrack();
    const v = r.verdict || {};
    const vv = VERDICT[v.verdict] || VERDICT.UNSURE;
    this.res.hidden = false;
    this.gNum.textContent = v.g != null ? fmtG(v.g, v.sigma) : "–";
    this.verdict.dataset.v = v.verdict || "UNSURE";
    this.vIcon.textContent = vv.icon;
    this.vWord.textContent = v.verdict === "SLOWMO" && v.k ? `Real, ${Math.round(v.k)}x slow motion` : vv.word;
    this.headline.textContent = v.headline || "";
    this.expl.hidden = !v.explanation;
    this.expl.textContent = v.explanation || "";
    this.reasons.innerHTML = "";
    const seen = new Set();
    for (const line of v.reasons || []) {
      if (seen.has(line)) continue;
      seen.add(line);
      const li = document.createElement("li");
      li.textContent = line;
      this.reasons.appendChild(li);
    }
    for (const n of r.notes || []) {
      const li = document.createElement("li");
      li.textContent = n;
      this.reasons.appendChild(li);
    }
    // Lost the ball? Point at the pick button.
    const stage = v.stage || "";
    this.pickBtn.classList.toggle("suggest", stage.startsWith("tracking") && !this.seed);
    // No ruler, or one that hangs on what the object is? Point at the Ruler menu.
    this.objSel.classList.toggle("suggest", stage.startsWith("ruler"));
    this.renderObjects(r);
    this.renderTests(r.tests || []);
    this.renderFacts(r);
    this.renderPoints(r);
    this.key.hidden = !(r.track && r.track.points && r.track.points.length);
    this.redrawAll();
  }

  prepTrack() {
    const r = this.result;
    this.dtTol = 0.05;
    if (r && r.track && r.track.points && r.track.points.length > 2) {
      const ts = r.track.points.map((p) => p.t);
      const d = [];
      for (let i = 1; i < ts.length; i++) d.push(ts[i] - ts[i - 1]);
      d.sort((a, b) => a - b);
      this.dtTol = Math.max(0.6 * d[d.length >> 1], 0.005);
    }
  }

  renderObjects(r) {
    const objs = r.objects || [];
    const show = objs.length > 1 || (objs.length === 1 && objs[0].shape !== "round");
    this.objTable.hidden = !show;
    this.keyMore.hidden = objs.length < 2;
    this.objBody.innerHTML = "";
    if (!show) return;
    for (const o of objs) {
      const v = o.verdict || {};
      const tr = document.createElement("tr");
      const n = document.createElement("td");
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = o.n === 1 ? COLORS.clipPlate : OBJECT_COLORS[(o.n - 2) % OBJECT_COLORS.length];
      n.append(sw, String(o.n));
      const name = document.createElement("td");
      const idn = o.identity || {};
      const src = idn.source;
      let how = !src ? "not recognised" : src === "you" ? "you said" : src;
      if ((src === "CLIP" || src === "Gemini") && idn.confidence != null) how += `, ${Math.round(idn.confidence * 100)}% sure`;
      how += o.shape === "round" ? "; round" : "; not round, followed by its centre of mass";
      if (o.spin && o.spin.applied) how += ` (${o.spin.offset_px.toFixed(0)} px off its outline's centre)`;
      name.textContent = o.label || "Unknown object";
      const sm = document.createElement("span");
      sm.className = "how";
      sm.textContent = how;
      name.appendChild(sm);
      const g = document.createElement("td");
      g.textContent = v.g != null ? fmtG(v.g, v.sigma) : "–";
      const res = document.createElement("td");
      const vv = VERDICT[v.verdict] || VERDICT.UNSURE;
      res.textContent = v.verdict === "UNSURE" ? `${vv.word}: ${(v.headline || "").replace(/^Can't tell: /, "")}` : vv.word;
      tr.append(n, name, g, res);
      this.objBody.appendChild(tr);
    }
  }

  renderTests(tests) {
    this.tests.innerHTML = "";
    for (const t of tests) {
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = t.experimental ? `${t.name} (new)` : t.name;
      const st = document.createElement("td");
      st.innerHTML = `<span class="st st-${t.status}"><b aria-hidden="true">${STATUS_MARK[t.status] || ""}</b>${STATUS_WORD[t.status] || t.status}</span>`;
      const sum = document.createElement("td");
      sum.textContent = t.summary;
      tr.append(name, st, sum);
      this.tests.appendChild(tr);
    }
  }

  renderFacts(r) {
    const f = this.facts;
    f.innerHTML = "";
    const add = (k, v) => {
      if (v == null || v === "") return;
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      f.append(dt, dd);
    };
    const file = r.file || {};
    add("File", `${(file.path || "").split(/[\\/]/).pop()}, ${file.width}×${file.height}`);
    if (r.work) add("Frame rate", `${r.work.fps_measured.toFixed(2)} fps in the file` + (file.capture_fps ? `, captured at ${file.capture_fps} fps` : ""));
    if (r.work) add("Frames analysed", `${r.work.frames} at ${Math.round(r.work.scale * 100)}% size`);
    if (r.zoom) add("Zoomed in", `ball ${r.zoom.ball_px_before.toFixed(0)} px, then ${r.zoom.ball_px.toFixed(0)} px at ${Math.round(r.zoom.scale * 100)}% resolution (${r.zoom.frames} frames)`);
    if (this.seed) add("Ball picked", `at ${this.seed.t.toFixed(2)} s`);
    const sc = r.scale;
    if (sc) for (const ru of sc.rulers) add(`Ruler: ${ru.name}`, `${ru.detail} (±${(ru.rel_sigma * 100).toFixed(1)}%)`);
    if (sc && sc.correction) add("Calibration", `scale × ${sc.correction.toFixed(4)} from tower drops`);
    const clk = r.clock || {};
    add("Clock", clk.source ? `${clk.source}: k = ${clk.k.toFixed(2)}` : "File timestamps, no slow motion found (k = 1)");
    const fit = r.fit;
    if (fit) {
      add("Acceleration", `${fit.a_px.toFixed(0)} ± ${fit.a_sigma.toFixed(0)} px/s²`);
      const rs = fit.rolling_shutter;
      if (rs && rs.alpha) add("Rolling shutter", `corrected, changed g by ${(rs.effect_rel * 100).toFixed(1)}%`);
      add("Ball size", `${fit.diam_px.toFixed(1)} px`);
      add("Flights", fit.flights.map((x) => `${x.n} frames`).join(", "));
    }
    if (r.drag && Math.abs(r.drag.factor - 1) > 0.005) add("Air drag", `corrected ${((r.drag.factor - 1) * 100).toFixed(1)}% (top speed ${r.drag.vt.toFixed(1)} m/s)`);
    if (r.stabilization && r.stabilization.applied) add("Camera motion", `removed (${r.stabilization.model === "homography" ? "pans and turns, with perspective" : "shake and drift"}; moved up to ${r.stabilization.max_drift_px.toFixed(0)} px)`);
    else if (r.stabilization && r.stabilization.ignored) add("Camera motion", "taken as still (plain backdrop: the only detail to follow was the object)");
    const cam = r.camera || {};
    if (cam.pitched || cam.rolled) {
      const bits = [];
      if (cam.pitched) bits.push(`looks ${cam.pitch > 0 ? "down" : "up"} ${Math.abs(cam.pitch).toFixed(0)}° (from ${cam.pitch_source})`);
      if (cam.rolled) bits.push(`rolled ${Math.abs(cam.roll).toFixed(0)}°`);
      const hfov = cam.f_frac ? 2 * Math.atan(0.5 / cam.f_frac) * 180 / Math.PI : null;
      add("Camera angle", bits.join(", ") + `; measured in a level view` + (hfov ? ` (field of view ${hfov.toFixed(0)}°, ${cam.f_source})` : ""));
    }
    if (r.fit && r.fit.camera_error_rel > 0.02) add("Camera uncertainty", `±${(r.fit.camera_error_rel * 100).toFixed(0)}% on g (mostly the lens's field of view; give it for a tighter answer)`);
    if (r.elapsed_s != null) add("Time to measure", `${r.elapsed_s.toFixed(1)} s`);
  }

  renderPoints(r) {
    this.pointsBody.innerHTML = "";
    const pts = (r.plot && r.plot.points) || [];
    for (const [t, d, j] of pts) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${t.toFixed(4)}</td><td>${d.toFixed(4)}</td><td>${j + 1}</td>`;
      this.pointsBody.appendChild(tr);
    }
  }

  redrawAll() {
    this.drawOverlay();
    this.drawPlate();
    this.drawChart();
  }

  loop() {
    if (this._raf) return;
    const step = () => {
      this.drawOverlay();
      if (!this.video.paused && !this.video.ended) this._raf = requestAnimationFrame(step);
      else this._raf = null;
    };
    this._raf = requestAnimationFrame(step);
  }

  /* -- video overlay: tracked ball and its Earth-gravity twin ------------ */
  drawOverlay() {
    const cv = this.overlay, v = this.video;
    const W = cv.clientWidth, H = cv.clientHeight;
    const ctx = sizeCanvas(cv, W, H);
    ctx.clearRect(0, 0, W, H);
    const r = this.result;
    if (!r || !r.track || !v.videoWidth) return;
    const vr = v.videoWidth / v.videoHeight, er = W / H;
    let dw, dh, ox, oy;
    if (vr > er) { dw = W; dh = W / vr; ox = 0; oy = (H - dh) / 2; }
    else { dh = H; dw = H * vr; oy = 0; ox = (W - dw) / 2; }
    const t = v.currentTime;
    if (this.seed && Math.abs(this.seed.t - t) < 0.15) {
      const sx = ox + this.seed.x * dw, sy = oy + this.seed.y * dh;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(sx - 10, sy); ctx.lineTo(sx - 4, sy); ctx.moveTo(sx + 4, sy); ctx.lineTo(sx + 10, sy);
      ctx.moveTo(sx, sy - 10); ctx.lineTo(sx, sy - 4); ctx.moveTo(sx, sy + 4); ctx.lineTo(sx, sy + 10);
      ctx.stroke();
    }
    const pts = r.track.points;
    ctx.fillStyle = COLORS.clipPlate;
    ctx.globalAlpha = 0.55;
    for (const p of pts) {
      if (p.t > t + 1e-4) break;
      ctx.beginPath();
      ctx.arc(ox + p.x * dw, oy + p.y * dh, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    const cur = nearestByT(pts, t);
    const rad = cur ? Math.max(7, cur.r * dw + 4) : 8;
    if (r.ghost && r.ghost.points) {
      const g = nearestByT(r.ghost.points, t);
      if (g && Math.abs(g.t - t) <= this.dtTol && g.y <= 1.02) {
        ctx.strokeStyle = COLORS.earthPlate;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(ox + g.x * dw, oy + g.y * dh, rad, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    if (cur && Math.abs(cur.t - t) <= this.dtTol) {
      ctx.strokeStyle = COLORS.clipPlate;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(ox + cur.x * dw, oy + cur.y * dh, rad, 0, Math.PI * 2);
      ctx.stroke();
      if ((r.objects || []).length > 1) {
        ctx.fillStyle = COLORS.clipPlate;
        ctx.font = `600 13px ${COLORS.sans}`;
        ctx.fillText("1", ox + cur.x * dw + rad + 3, oy + cur.y * dh - rad);
      }
    }
    // Every other falling object: its trail, a ring, and its number.
    const objs = r.objects || [];
    for (const o of objs.slice(1)) {
      const op = (o.track && o.track.points) || [];
      if (!op.length) continue;
      const col = OBJECT_COLORS[(o.n - 2) % OBJECT_COLORS.length];
      ctx.fillStyle = col;
      ctx.globalAlpha = 0.55;
      for (const p of op) {
        if (p.t > t + 1e-4) break;
        ctx.beginPath();
        ctx.arc(ox + p.x * dw, oy + p.y * dh, 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      const c = nearestByT(op, t);
      if (c && Math.abs(c.t - t) <= this.dtTol) {
        const rr = Math.max(7, c.r * dw + 4);
        ctx.strokeStyle = col;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(ox + c.x * dw, oy + c.y * dh, rr, 0, Math.PI * 2);
        ctx.stroke();
        ctx.font = `600 13px ${COLORS.sans}`;
        ctx.fillText(String(o.n), ox + c.x * dw + rr + 3, oy + c.y * dh - rr);
      }
    }
  }

  /* -- strobe plate: the fall as a multiple-exposure photograph ---------- */
  drawPlate() {
    const cv = this.plate;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    const ctx = sizeCanvas(cv, W, H);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    const r = this.result;
    const plot = r && r.plot;
    const ink = COLORS.plateInk;
    if (!plot) {
      ctx.fillStyle = ink;
      ctx.globalAlpha = 0.6;
      ctx.font = `14px ${COLORS.sans}`;
      ctx.textAlign = "center";
      ctx.fillText("No ruler,", W / 2, H / 2 - 9);
      ctx.fillText("no strobe view", W / 2, H / 2 + 11);
      ctx.globalAlpha = 1;
      this.plateCap.textContent = "Pick the object's size to turn pixels into meters.";
      cv.setAttribute("aria-label", "No strobe view without a ruler.");
      return;
    }
    const first = plot.points.filter((p) => p[2] === 0).map((p) => [p[0], p[1]]).sort((a, b) => a[0] - b[0]);
    const tEnd = plot.t_first_end;
    const dMax = Math.max(0.05, ...first.map((p) => p[1]));
    let dt = 1 / 30, best = 1e9;
    for (const c of [1 / 120, 1 / 60, 1 / 30, 1 / 20, 1 / 15, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5, 1]) {
      const n = tEnd / c;
      if (n >= 5 && Math.abs(n - 11) < best) { best = Math.abs(n - 11); dt = c; }
    }
    const top = 30, bottom = 14, left = 44, right = 8;
    const ph = H - top - bottom;
    const ys = ph / (dMax * 1.06);
    const x1 = left + (W - left - right) * 0.3;
    const x2 = left + (W - left - right) * 0.78;
    // Ruler, like the meter stick in a textbook strobe photograph.
    const step = niceStep(dMax, 6);
    ctx.strokeStyle = ink;
    ctx.fillStyle = ink;
    ctx.font = `11px ${COLORS.sans}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    ctx.moveTo(left - 6, top);
    ctx.lineTo(left - 6, top + ph);
    ctx.stroke();
    for (let d = 0; d <= dMax * 1.06 + 1e-9; d += step / 2) {
      const y = top + d * ys;
      const major = Math.abs(d / step - Math.round(d / step)) < 1e-6;
      ctx.beginPath();
      ctx.moveTo(left - 6, y);
      ctx.lineTo(left - (major ? 12 : 9), y);
      ctx.stroke();
      if (major) ctx.fillText(`${+(d).toFixed(2)}`, left - 14, y);
    }
    ctx.textAlign = "right";
    ctx.fillText("m", left - 14, top - 14);
    ctx.globalAlpha = 0.75;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.font = `12px ${COLORS.sans}`;
    ctx.fillText("This clip", x1, 18);
    ctx.fillText("Earth", x2, 18);
    ctx.globalAlpha = 1;
    const rb = Math.max(3.5, Math.min(11, (plot.diam_m / 2) * ys));
    const poly = (c, t) => c[0] + c[1] * t + c[2] * t * t;
    let earthGone = false, flashes = 0;
    for (let k = 0; k * dt <= tEnd + 1e-9; k++) {
      const t = k * dt;
      let d = interpD(first, t);
      let measured = true;
      if (d == null) { d = Math.max(0, poly(plot.fit_coef, t)); measured = false; }
      const deI = interpD(plot.earth, t);
      const de = deI != null ? deI : poly(plot.earth_coef, t);
      const yc = top + d * ys;
      const ye = top + de * ys;
      ctx.globalAlpha = 0.28;
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x1 + rb, yc);
      if (ye <= top + ph) ctx.lineTo(x2 - rb, ye);
      else ctx.lineTo(x1 + rb + (x2 - x1 - 2 * rb) * (top + ph - yc) / Math.max(ye - yc, 1e-6), top + ph);
      ctx.stroke();
      ctx.globalAlpha = measured ? 1 : 0.45;
      ctx.fillStyle = COLORS.clipPlate;
      ctx.beginPath();
      ctx.arc(x1, yc, rb, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (ye <= top + ph) {
        ctx.strokeStyle = COLORS.earthPlate;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x2, ye, rb, 0, Math.PI * 2);
        ctx.stroke();
      } else earthGone = true;
      flashes++;
    }
    if (earthGone) {
      ctx.fillStyle = COLORS.earthPlate;
      ctx.font = `11px ${COLORS.sans}`;
      ctx.textAlign = "center";
      ctx.fillText("off the plate", x2, H - 3);
    }
    const v = r.verdict || {};
    let cap = `A flash every ${fmtDt(dt)} of real time` + (plot.origin === "release" ? ", from the release." : ", from the first tracked frame.");
    if (v.verdict === "AI" && earthGone) cap += " A real ball would have fallen off the bottom by now.";
    else if (v.verdict === "AI") cap += " The rings show where a real ball would be.";
    else if (v.verdict === "REAL" || v.verdict === "SLOWMO") cap += " The ball keeps pace with Earth's gravity.";
    this.plateCap.textContent = cap;
    cv.setAttribute("aria-label", `Strobe view: ${flashes} flashes, ${fmtDt(dt)} apart. ${cap}`);
  }

  /* -- chart: distance fallen against real time --------------------------- */
  chartGeom() {
    const r = this.result, plot = r && r.plot;
    if (!plot) return null;
    const W = this.chart.clientWidth, H = this.chart.clientHeight;
    const pad = { l: 46, r: 92, t: 10, b: 30 };
    const pts = plot.points;
    const tMax = Math.max(plot.t_first_end * 1.05, ...pts.map((p) => p[0]));
    const dMax = Math.max(...pts.map((p) => p[1]), ...plot.fit.map((p) => p[1])) * 1.12 || 1;
    const X = (t) => pad.l + (t / tMax) * (W - pad.l - pad.r);
    const Y = (d) => H - pad.b - (d / dMax) * (H - pad.t - pad.b);
    return { W, H, pad, tMax, dMax, X, Y, plot };
  }

  drawChart(hoverT = null) {
    const cv = this.chart;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    const ctx = sizeCanvas(cv, W, H);
    ctx.clearRect(0, 0, W, H);
    const g = this.chartGeom();
    this.legend.innerHTML = "";
    if (!g) return;
    const { pad, tMax, dMax, X, Y, plot } = g;
    // Grid and axes.
    ctx.font = `12px ${COLORS.sans}`;
    ctx.fillStyle = COLORS.ink2;
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 1;
    const ys = niceStep(dMax, 4);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let d = 0; d <= dMax + 1e-9; d += ys) {
      const y = Math.round(Y(d)) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      ctx.fillText(`${+d.toFixed(2)}`, pad.l - 6, y);
    }
    const xs = niceStep(tMax, 5);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let t = 0; t <= tMax + 1e-9; t += xs) {
      const x = Math.round(X(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
      ctx.fillText(`${+t.toFixed(2)}`, x, H - pad.b + 6);
    }
    ctx.textAlign = "left";
    ctx.fillText("s", W - pad.r + 6, H - pad.b + 6);
    ctx.textBaseline = "alphabetic";
    ctx.fillText("m", 4, pad.t + 10);

    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);
    ctx.clip();
    const line = (arr, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      arr.forEach(([t, d], i) => (i ? ctx.lineTo(X(t), Y(d)) : ctx.moveTo(X(t), Y(d))));
      ctx.stroke();
    };
    line(plot.earth, COLORS.earth);
    line(plot.fit, COLORS.clip);
    for (const [t, d] of plot.points) {
      ctx.fillStyle = COLORS.paper;
      ctx.beginPath(); ctx.arc(X(t), Y(d), 6, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = COLORS.clip;
      ctx.beginPath(); ctx.arc(X(t), Y(d), 4, 0, Math.PI * 2); ctx.fill();
    }
    if (hoverT != null) {
      ctx.strokeStyle = COLORS.ink3;
      ctx.lineWidth = 1;
      const x = Math.round(X(hoverT)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
    }
    ctx.restore();

    // Direct labels at the line ends (text in ink, the line carries the colour).
    ctx.font = `12px ${COLORS.sans}`;
    ctx.fillStyle = COLORS.ink2;
    ctx.textBaseline = "middle";
    const endFit = plot.fit[plot.fit.length - 1];
    const yFit = Math.min(Math.max(Y(endFit[1]), pad.t + 6), H - pad.b - 6);
    let yEarth = null;
    const lastEarth = plot.earth[plot.earth.length - 1];
    if (lastEarth[1] <= dMax) yEarth = Y(lastEarth[1]);
    else yEarth = pad.t + 6;
    let lf = yFit, le = yEarth;
    if (Math.abs(lf - le) < 15) { if (lf < le) { lf -= 8; le += 8; } else { lf += 8; le -= 8; } }
    const drag = plot.drag_vt != null;
    ctx.fillText(`g = ${plot.g_fit.toFixed(2)}`, W - pad.r + 6, lf);
    ctx.fillText(drag ? "Earth + air" : `Earth ${plot.g_ref.toFixed(2)}`, W - pad.r + 6, le);

    const fitText = drag && plot.g_fit_drag_corrected != null
      ? `Best fit, g = ${plot.g_fit.toFixed(2)} m/s² as filmed (${plot.g_fit_drag_corrected.toFixed(2)} after air drag)`
      : `Best fit, g = ${plot.g_fit.toFixed(2)} m/s²`;
    const earthText = drag
      ? `Earth's gravity, ${plot.g_ref.toFixed(2)} m/s², with air drag on a light ball`
      : `Earth's gravity, ${plot.g_ref.toFixed(2)} m/s²`;
    this.legend.innerHTML =
      `<span><i class="dot" style="background:${COLORS.clip}"></i>Tracked ball</span>` +
      `<span><i style="background:${COLORS.clip}"></i>${fitText}</span>` +
      `<span><i style="background:${COLORS.earth}"></i>${earthText}</span>`;
    cv.setAttribute("aria-label", `Distance fallen against time. Fit g ${plot.g_fit.toFixed(2)}, Earth ${plot.g_ref.toFixed(2)} m/s squared. Data in the tracked points table.`);
  }

  hoverChart(e) {
    const g = this.chartGeom();
    if (!g) return;
    const rect = this.chart.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const t = ((x - g.pad.l) / (g.W - g.pad.l - g.pad.r)) * g.tMax;
    if (t < 0 || t > g.tMax) { this.tip.hidden = true; this.drawChart(); return; }
    this.drawChart(t);
    const poly = (c, x) => c[0] + c[1] * x + c[2] * x * x;
    const fit = poly(g.plot.fit_coef, t);
    const eI = interpD(g.plot.earth, t);
    const earth = eI != null ? eI : poly(g.plot.earth_coef, t);
    let near = null;
    for (const p of g.plot.points) if (!near || Math.abs(p[0] - t) < Math.abs(near[0] - t)) near = p;
    const meas = near && Math.abs(g.X(near[0]) - x) < 10 ? `<br>Tracked: ${near[1].toFixed(3)} m` : "";
    this.tip.innerHTML = `t = ${t.toFixed(3)} s<br>Fit: ${fit.toFixed(3)} m${meas}<br>Earth: ${earth.toFixed(3)} m`;
    this.tip.hidden = false;
    const tx = Math.min(x + 12, g.W - this.tip.offsetWidth - 4);
    this.tip.style.left = `${Math.max(tx, 0)}px`;
    this.tip.style.top = `8px`;
  }
}

/* ------------------------------------------------------------------------ */
async function refreshClips() {
  CLIPS = await api("/api/clips");
  for (const p of Object.values(panels)) p.fillClips(p.clipId);
}

function pickDefault(side) {
  const isAI = (c) => c.ai_generated || /(^|[_\-\s])(ai|veo|twin|sora|fake)/i.test(c.name);
  const real = CLIPS.filter((c) => !isAI(c));
  const fake = CLIPS.filter(isAI);
  if (side === "a") return (CLIPS.find((c) => c.group === "drops") || real.find((c) => /real/i.test(c.name)) || real[0] || CLIPS[0] || {}).id;
  return (fake[0] || CLIPS[1] || CLIPS[0] || {}).id;
}

function renderStatus() {
  const ul = $("#status");
  ul.innerHTML = "";
  const item = (on, text) => {
    const li = document.createElement("li");
    li.className = on ? "on" : "off";
    li.innerHTML = `<span class="dot" aria-hidden="true"></span>`;
    li.append(text);
    ul.appendChild(li);
  };
  const t = STATUS.tower;
  if (t) item(t.connected, t.kind === "simulated" ? "Tower: simulated" : `Tower on ${t.port}`);
  else item(false, "No tower connected");
  item(STATUS.gemini.available, STATUS.gemini.available ? "Gemini ready" : "Gemini off");
  if (STATUS.calibration) item(true, `Calibrated on ${STATUS.calibration.n} drops`);
}

function setupTower() {
  if (!STATUS.tower) return;
  $("#tower").hidden = false;
  const sim = $("#sim-drop");
  if (STATUS.tower.kind === "simulated") {
    sim.hidden = false;
    sim.addEventListener("click", async () => {
      sim.disabled = true;
      try { await api("/api/tower/drop", { method: "POST" }); }
      catch (e) { $("#tower-state").textContent = e.message; sim.disabled = false; }
    });
  }
  connectWS();
}

function connectWS() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (e) => onLive(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connectWS, 1500);
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") for (const p of Object.values(panels)) if (p.picking) p.setPicking(false);
});

function onLive(m) {
  const state = $("#tower-state");
  if (m.type === "tower") {
    if (m.kind === "ARM") { state.textContent = `Drop ${m.drop_id}: counting down, camera recording.`; $("#tower-g").textContent = "–"; $("#video-g").textContent = "–"; $("#tower-agree").textContent = ""; }
    else if (m.kind === "REL") state.textContent = `Drop ${m.drop_id}: released.`;
    else if (m.kind === "GATE") state.textContent = `Drop ${m.drop_id}: gate ${m.fields.k} at ${m.fields.h_mm} mm.`;
    else if (m.kind === "TOWERG") $("#tower-g").innerHTML = `${m.fields.g.toFixed(2)} <small>m/s²</small>`;
  } else if (m.type === "analyzing") {
    state.textContent = `Drop ${m.drop_id}: measuring the video…`;
  } else if (m.type === "error") {
    state.textContent = `Drop ${m.drop_id}: video analysis failed (${m.error}).`;
    $("#sim-drop").disabled = false;
  } else if (m.type === "drop") {
    $("#sim-drop").disabled = false;
    const vg = m.video_g, tg = m.tower_g;
    $("#video-g").innerHTML = vg != null ? `${vg.toFixed(2)} <small>± ${(m.video_sigma || 0).toFixed(2)} m/s²</small>` : "–";
    if (tg != null) $("#tower-g").innerHTML = `${tg.toFixed(2)} <small>m/s²</small>`;
    const vv = VERDICT[m.verdict] || VERDICT.UNSURE;
    let agree = `Verdict: ${vv.word}.`;
    if (vg != null && tg) agree = `Video and tower agree within ${(Math.abs(vg / tg - 1) * 100).toFixed(1)}%. Verdict: ${vv.word}.`;
    $("#tower-agree").textContent = agree + (m.synthetic ? " (Simulated drop.)" : "");
    state.textContent = `Drop ${m.drop_id} done.`;
    const li = document.createElement("li");
    li.innerHTML = `<span>Drop ${m.drop_id}</span><span>tower ${tg != null ? tg.toFixed(2) : "–"}</span><span>video ${vg != null ? vg.toFixed(2) : "–"}</span><span>${vv.word}</span>`;
    $("#drops").prepend(li);
    if ($("#follow").checked && m.result) panels.a.showLive(m);
  }
}

async function init() {
  const root = document.documentElement;
  Object.assign(COLORS, {
    clip: cssVar("--clip"), earth: cssVar("--earth"), ink: cssVar("--ink"), ink2: cssVar("--ink-2"),
    ink3: cssVar("--ink-3"), paper: cssVar("--paper"), plateInk: cssVar("--plate-ink"),
    gridLine: "#d6ddd4", clipPlate: "#7fa8f2", earthPlate: "#e2a64c", sans: getComputedStyle(root).getPropertyValue("--sans"),
  });
  STATUS = await api("/api/status");
  renderStatus();
  CLIPS = await api("/api/clips");
  panels.a = new Panel($("#panel-a"), "a");
  panels.b = new Panel($("#panel-b"), "b");
  for (const p of Object.values(panels)) { p.fillObjects(); p.fillClips(); }
  const a = pickDefault("a"), b = pickDefault("b");
  if (a) panels.a.setClip(a);
  if (b && b !== a) panels.b.setClip(b);
  setupTower();
}

init().catch((e) => {
  document.body.insertAdjacentHTML("beforeend", `<p class="message" style="margin:24px">NØVEX's server didn't answer: ${e.message}</p>`);
});
