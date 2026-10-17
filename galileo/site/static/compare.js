"use strict";
/* Shared uploads and two-clip playback. Analysis always uses /api/jobs. */
function uploadMarkup(prefix, label = '') {
  const presets = STATUS.presets || [], auto = !!STATUS.auto?.available;
  const options = presets.map(p => `<option value="${esc(p.key)}">${esc(p.label)} (${p.size_mm} mm)</option>`).join('');
  return `<div class="upload-fields" id="${prefix}-fields">
    <label class="zone" id="${prefix}-zone">
      <input class="visually-hidden" type="file" id="${prefix}-file" aria-label="${label ? esc(label) + ': ' : ''}Choose video" accept="video/*,.mp4,.mov,.m4v,.webm,.mkv,.avi">
      <span class="upload-symbol">${icon('upload')}</span><strong id="${prefix}-title">Drop your video here</strong>
      <span id="${prefix}-sub">or <u>browse files</u></span><small>MP4, MOV, WebM & more · up to 1.5 GB</small>
    </label>
    <label class="field">What is falling?<select id="${prefix}-object">${auto ? '<option value="auto">Identify automatically</option>' : ''}${options}<option value="custom">Other object — enter its size</option></select>
      <small>${auto ? "We'll identify the object, or ask you to help." : 'Choose the object so Galileo can estimate its real size.'}</small></label>
    <label class="field" id="${prefix}-size-wrap" hidden>Size in millimetres<input id="${prefix}-size" type="number" min="1" max="5000" step="any" placeholder="e.g. 38"><small>Diameter of a ball, or the object's longest side.</small></label>
    <details class="adv"><summary>More options</summary><label class="field">Camera field of view (optional)<input id="${prefix}-hfov" type="number" min="10" max="179.99" step="any" placeholder="Automatic"><small>Degrees across the long side of the video.</small></label>
      ${!auto ? `<details class="t"><summary>Automatic identification setup</summary><p class="hint">${esc(STATUS.auto?.detail || 'Automatic identification is not installed on this server.')} See requirements-ml.txt in the project for the optional dependencies.</p></details>` : ''}</details>
    <p class="err" id="${prefix}-error" role="alert" hidden></p>
  </div>`;
}
class UploadInput {
  constructor(prefix, onChange) {
    this.root = document.getElementById(`${prefix}-fields`); this.prefix = prefix; this.file = null; this.onChange = onChange;
    this.input = this.q('file'); this.object = this.q('object');
    this.object.value = STATUS.auto?.available ? 'auto' : (STATUS.presets || []).some(p => p.key === 'tennis') ? 'tennis' : STATUS.presets?.[0]?.key || 'custom';
    const sync = () => { const custom = this.object.value === 'custom'; this.q('size-wrap').hidden = !custom; this.q('size').required = custom; this.q('size').disabled = !custom; this.onChange?.(); };
    this.object.onchange = sync; this.q('size').oninput = () => this.onChange?.(); this.q('hfov').oninput = () => this.onChange?.();
    this.q('size-wrap').hidden = this.object.value !== 'custom'; this.q('size').required = this.object.value === 'custom'; this.q('size').disabled = this.object.value !== 'custom';
    this.input.onchange = () => this.pick(this.input.files[0]);
    const zone = this.q('zone');
    ['dragenter','dragover'].forEach(type => zone.addEventListener(type, e => {e.preventDefault(); if (!this.input.disabled) zone.classList.add('over');}));
    ['dragleave','drop'].forEach(type => zone.addEventListener(type, e => {e.preventDefault(); zone.classList.remove('over');}));
    zone.addEventListener('drop', e => { if (!this.input.disabled) { if (e.dataTransfer.files.length !== 1) this.error('Choose one video for this slot.'); else this.pick(e.dataTransfer.files[0]); }});
  }
  q(key) { return document.getElementById(`${this.prefix}-${key}`); }
  error(message) { this.q('error').textContent = message; this.q('error').hidden = !message; }
  pick(file) {
    if (!file) return;
    if (!/\.(mp4|mov|m4v|webm|mkv|avi)$/i.test(file.name) || file.size <= 0 || file.size > 1500000000) {
      this.error(file.size > 1500000000 ? 'Choose a video smaller than 1.5 GB.' : 'Choose a non-empty MP4, MOV, M4V, WebM, MKV or AVI video.');
      this.input.value = ''; this.file = null; this.q('zone').classList.remove('has-file'); this.q('title').textContent = 'Drop your video here'; this.q('sub').textContent = 'or browse files'; this.onChange?.(); return;
    }
    this.file = file; this.error(''); this.q('zone').classList.add('has-file');
    this.q('title').textContent = file.name; this.q('sub').textContent = `${(file.size / 1e6).toFixed(1)} MB · choose another file`;
    this.onChange?.();
  }
  validate() {
    if (!this.file) { this.error('Choose a video to continue.'); return false; }
    const invalid = this.root.querySelector(':invalid');
    if (invalid) { const details = invalid.closest('details'); if (details) details.open = true; invalid.reportValidity(); return false; }
    return true;
  }
  setDisabled(value) { this.root.querySelectorAll('input,select').forEach(x => x.disabled = value || (x === this.q('size') && this.object.value !== 'custom')); this.q('zone').setAttribute('aria-disabled', value); }
  async submit(signal) {
    const fd = new FormData(); fd.append('file', this.file); fd.append('object', this.object.value);
    fd.append('size_mm', this.q('size').value); fd.append('hfov', this.q('hfov').value);
    return api('/api/jobs', {method:'POST', body:fd, signal});
  }
}

class CompareUpload {
  constructor() {
    this.controller = new AbortController(); this.ids = [null, null]; this.busy = false;
    view.innerHTML = `<div class="page-heading compare-heading"><div><span class="eyebrow">A SECOND PERSPECTIVE</span><h1>Side by side.<br><span>See the difference.</span></h1><p>Two clips, measured independently. Play them together and compare what the physics says.</p></div><span class="compare-emblem" aria-hidden="true">${icon('compare')}</span></div>
      <form id="compare-form" novalidate><div class="compare-upload-grid">${['A','B'].map((label,i) => `<section class="drop compare-upload ${i ? 'clip-orange' : 'clip-blue'}"><div class="clip-title"><span class="clip-letter">${label}</span><div><h2>Clip ${label}</h2><p>${i ? 'A different clip. A fresh perspective.' : 'Choose your first point of reference.'}</p></div></div>${uploadMarkup('upload-' + i, 'Clip ' + label)}<p class="hint" id="upload-status-${i}" role="status"></p></section>`).join('')}</div>
      <div class="compare-submit"><p class="hint">Each clip can contain a different object.<br>For a useful comparison, try similar scenes.</p><button type="submit" class="btn primary" id="compare-go" disabled>Analyze both clips ${icon('arrow')}</button></div><p class="err" id="compare-error" role="alert" hidden></p></form>
      <div class="compare-bottom"><span>Just exploring?</span><button class="linkbtn" id="compare-demo" type="button">Try an example comparison ↗</button><a href="#/history">Use past results ↗</a></div>`;
    this.uploads = [0,1].map(i => new UploadInput('upload-' + i, () => { this.ids[i] = null; document.getElementById('upload-status-' + i).textContent = ''; this.sync(); }));
    $('#compare-form').onsubmit = e => { e.preventDefault(); this.submit(); };
    $('#compare-demo').onclick = () => this.demo();
  }
  destroy() { this.controller.abort(); }
  sync() { $('#compare-go').disabled = this.busy || !this.uploads.every(u => u.file); }
  lock(value) { this.busy = value; this.uploads.forEach(u => u.setDisabled(value)); $('#compare-demo').disabled = value; this.sync(); }
  async submit() {
    if (this.busy || !this.uploads.every(u => u.validate())) return;
    this.lock(true); $('#compare-error').hidden = true; $('#compare-go').textContent = 'Uploading both clips…';
    const results = await Promise.allSettled(this.uploads.map(async (u,i) => {
      if (this.ids[i]) return;
      document.getElementById('upload-status-' + i).textContent = 'Uploading…';
      const result = await u.submit(this.controller.signal); this.ids[i] = result.id;
      if (!this.controller.signal.aborted) document.getElementById('upload-status-' + i).textContent = 'Uploaded · analysis started';
    }));
    if (this.controller.signal.aborted) return;
    if (this.ids.every(Boolean)) { location.hash = `#/compare/${this.ids.join('/')}`; return; }
    results.forEach((r,i) => { if (r.status === 'rejected') document.getElementById('upload-status-' + i).textContent = 'Upload failed · ready to retry'; });
    $('#compare-error').textContent = results.filter(r => r.status === 'rejected').map(r => r.reason.message).join(' ');
    $('#compare-error').hidden = false; $('#compare-go').innerHTML = `Retry missing upload ${icon('arrow')}`; this.lock(false);
  }
  async demo() {
    const samples = STATUS.samples || [], real = samples.find(s => s.name.includes('real_')), ai = samples.find(s => s.name.includes('floaty'));
    if (!real || !ai) { toast('The example clips are not available on this server.'); return; }
    this.lock(true); $('#compare-demo').textContent = 'Preparing examples…';
    const results = await Promise.allSettled([real,ai].map(s => api('/api/jobs/sample', {method:'POST', signal:this.controller.signal, headers:{'content-type':'application/json'}, body:JSON.stringify({name:s.name})})));
    if (this.controller.signal.aborted) return;
    if (results.every(r => r.status === 'fulfilled')) location.hash = `#/compare/${results.map(r => r.value.id).join('/')}`;
    else { this.lock(false); $('#compare-demo').textContent = 'Try an example comparison ↗'; toast('An example could not start. Any successful analysis is in History. Please try again.'); }
  }
}

class ComparePage {
  constructor(ids) {
    this.ids = ids; this.controller = new AbortController(); this.dead = false; this.linked = true; this.playing = false; this.rate = 1; this.time = 0; this.timers = [];
    this.clips = ids.map(id => ({id, data:null, frames:null, loaded:false, failed:false}));
    view.innerHTML = `<div class="page-heading"><div><span class="eyebrow">THE BIGGER PICTURE</span><h1>Compare clips</h1><p>One view. Two independent analyses.</p></div><a class="btn ghost" href="#/compare">${icon('upload')} New comparison</a></div>
      <div class="comparison-grid">${this.clips.map((clip,i) => `<section class="comparison-card ${i ? 'clip-orange' : 'clip-blue'}" id="clip-${i}" aria-label="Clip ${i ? 'B' : 'A'}"><div class="comparison-card-head"><span class="clip-letter">${i ? 'B' : 'A'}</span><span class="clip-name" id="clip-name-${i}">Loading clip…</span></div>
        <div class="comparison-stage"><video id="clip-video-${i}" muted playsinline preload="auto"></video><div class="comparison-placeholder" id="clip-placeholder-${i}"><div class="spin"></div><span>Preparing video…</span></div></div>
        <div class="clip-controls"><button class="ib clip-prev" type="button" aria-label="Previous frame, clip ${i ? 'B' : 'A'}" disabled>←</button><button class="ib clip-play" type="button" aria-label="Play clip ${i ? 'B' : 'A'}" disabled>▶</button><button class="ib clip-next" type="button" aria-label="Next frame, clip ${i ? 'B' : 'A'}" disabled>→</button><input class="clip-range" type="range" min="0" max="0" value="0" aria-label="Frame, clip ${i ? 'B' : 'A'}" disabled><span class="clip-time" id="clip-time-${i}">—</span></div>
        <div class="comparison-result" id="clip-result-${i}" aria-live="polite"><div class="skeleton"></div><p class="hint">Waiting for the analysis…</p></div></section>`).join('')}</div>
      <section class="comparison-transport card" aria-label="Comparison playback"><div class="transport-heading"><label class="link-switch"><input type="checkbox" id="link-playback" checked><span class="switch-track"></span>Linked playback</label><label class="speed">Speed <select id="compare-rate"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><span id="pair-time">0.00 s</span></div>
        <div class="pair-controls"><button class="ib" id="pair-start" aria-label="Restart both clips" disabled>↺</button><button class="ib" id="pair-prev" aria-label="Previous comparison frame" disabled>←</button><button class="btn" id="pair-play" disabled>${icon('play')} Play both</button><button class="ib" id="pair-next" aria-label="Next comparison frame" disabled>→</button><input type="range" id="pair-range" min="0" max="0" step="0.001" value="0" aria-label="Comparison time" disabled></div><p class="hint" id="sync-note">Playback is available when both videos are ready.</p><p class="err" id="pair-error" role="alert" hidden></p></section>
      <div class="comparison-footer"><span>${icon('orbit')} Each result comes directly from Galileo's analysis of that clip.</span><a href="#/history">View all analyses ↗</a></div>`;
    this.clips.forEach((clip,i) => {
      clip.video = document.getElementById('clip-video-' + i); clip.box = document.getElementById('clip-' + i);
      clip.video.addEventListener('loadeddata', () => { if (this.dead) return; clip.loaded = true; clip.failed = false; document.getElementById('clip-placeholder-' + i).hidden = true; this.ready(); });
      clip.video.addEventListener('error', () => {
        if (this.dead || !clip.video.getAttribute('src')) return;
        if (!clip.video.dataset.fallback) { clip.video.dataset.fallback = '1'; clip.video.src = `/media/${clip.id}/preview.webm`; }
        else { clip.failed = true; clip.loaded = false; document.getElementById('clip-placeholder-' + i).innerHTML = '<span>This browser could not play the video. You can still open the analysis below.</span>'; document.getElementById('clip-placeholder-' + i).hidden = false; this.pause(); this.ready(); }
      });
      clip.video.addEventListener('timeupdate', () => this.updateClipTime(i));
      clip.video.addEventListener('play', () => { $('.clip-play', clip.box).textContent = 'Ⅱ'; $('.clip-play', clip.box).setAttribute('aria-label', `Pause clip ${i ? 'B' : 'A'}`); });
      clip.video.addEventListener('pause', () => { $('.clip-play', clip.box).textContent = '▶'; $('.clip-play', clip.box).setAttribute('aria-label', `Play clip ${i ? 'B' : 'A'}`); });
      $('.clip-prev', clip.box).onclick = () => this.stepClip(i, -1);
      $('.clip-next', clip.box).onclick = () => this.stepClip(i, 1);
      $('.clip-play', clip.box).onclick = () => { if (clip.video.paused) { if (clip.video.ended) clip.video.currentTime = 0; clip.video.play().catch(() => this.mediaError()); } else clip.video.pause(); };
      $('.clip-range', clip.box).oninput = e => { clip.video.pause(); this.seekClip(clip, Number(e.target.value)); this.updateClipTime(i); };
      this.poll(i);
    });
    $('#link-playback').onchange = e => { this.pause(); this.linked = e.target.checked; this.ready(); if (this.linked && this.allReady()) this.seek(this.clips[this.master()].video.currentTime); };
    $('#compare-rate').onchange = e => { this.rate = +e.target.value; this.clips.forEach(c => c.video.playbackRate = this.rate); };
    $('#pair-start').onclick = () => { this.pause(); this.seek(0); };
    $('#pair-prev').onclick = () => this.stepPair(-1); $('#pair-next').onclick = () => this.stepPair(1);
    $('#pair-play').onclick = () => this.playing ? this.pause() : this.play();
    $('#pair-range').oninput = e => { this.pause(); this.seek(+e.target.value); };
    this.key = e => { if (e.defaultPrevented || /INPUT|SELECT|TEXTAREA|BUTTON|A|SUMMARY/.test(document.activeElement.tagName) || !this.linked || !this.allReady()) return; if (e.key === ' ') {e.preventDefault(); this.playing ? this.pause() : this.play();} if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {e.preventDefault(); this.stepPair((e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1));} };
    document.addEventListener('keydown', this.key);
  }
  destroy() { this.dead = true; this.controller.abort(); this.timers.forEach(clearTimeout); cancelAnimationFrame(this.raf); document.removeEventListener('keydown',this.key); this.clips.forEach(c => {c.video.pause(); c.video.removeAttribute('src'); c.video.load();}); }
  async poll(i) {
    const clip = this.clips[i];
    try {
      const data = await api(`/api/jobs/${clip.id}`, {signal:this.controller.signal});
      if (this.dead) return;
      clip.data = data; document.getElementById('clip-name-' + i).textContent = data.job.filename || 'Untitled clip';
      if (data.frames?.times?.length && !clip.frames) {
        clip.frames = data.frames;
        const ts = clip.frames.times; clip.dt = ts.length > 1 ? ts[ts.length-1] - ts[ts.length-2] : 1/30;
        clip.end = ts[ts.length-1] + clip.dt;
        $('.clip-range', clip.box).max = ts.length-1;
        clip.video.src = `/media/${clip.id}/preview.mp4`;
      }
      this.renderResult(i);
      if (data.job.status !== 'done' && data.job.status !== 'error') this.timers[i] = setTimeout(() => this.poll(i), 1200);
      if (data.job.status === 'error' && !clip.frames) { document.getElementById('clip-placeholder-' + i).innerHTML = `<span>Video preview unavailable</span>`; }
    } catch(e) {
      if (this.dead) return;
      document.getElementById('clip-result-' + i).innerHTML = `<p class="err">Couldn't load this analysis.</p><p class="hint">${esc(e.message)}</p><button class="btn ghost small" id="retry-clip-${i}">Retry</button>`;
      document.getElementById('retry-clip-' + i).onclick = () => this.poll(i);
      document.getElementById('clip-placeholder-' + i).innerHTML = '<span>Analysis unavailable</span>';
    }
  }
  renderResult(i) {
    const clip = this.clips[i], j = clip.data.job, v = clip.data.result?.verdict || {};
    const box = document.getElementById('clip-result-' + i), st = j.status === 'done' ? (v.verdict || 'UNSURE') : j.status;
    box.innerHTML = `<span class="pill ${esc(st)}">${esc(stateLabel({...j, verdict:v.verdict || j.verdict}))}</span>
      <h2>${j.status === 'done' ? esc(VERDICT[v.verdict] || v.verdict || 'Inconclusive') : j.status === 'error' ? 'A closer look is needed.' : 'Following the physics…'}</h2>
      <p>${j.status === 'done' ? esc(summaryText(v)) : j.status === 'error' ? 'This clip could not be analyzed. Open its details or try another clip.' : 'You can explore the video while the measurement finishes.'}</p>
      ${j.status === 'done' ? `<div class="compare-measure"><span>Measured gravity</span><b>${v.g != null ? `${num(v.g, 2)} <small>m/s²</small>` : 'Not available'}</b></div>` : ''}
      <a class="result-link" href="#/job/${clip.id}">${j.status === 'done' ? 'Open full analysis' : 'View analysis'} ${icon('arrow')}</a>`;
  }
  allReady() { return this.clips.every(c => c.loaded && c.frames?.times?.length && !c.failed); }
  master() { return (this.clips[1].end || 0) > (this.clips[0].end || 0) ? 1 : 0; }
  ready() {
    if (this.dead) return;
    const ready = this.allReady();
    ['pair-start','pair-prev','pair-play','pair-next','pair-range'].forEach(id => document.getElementById(id).disabled = !ready || !this.linked);
    this.clips.forEach(c => c.box.querySelectorAll('.clip-controls button,.clip-range').forEach(b => b.disabled = !c.loaded || this.linked));
    $('.comparison-transport').classList.toggle('unlinked', !this.linked);
    $('#sync-note').textContent = !this.linked ? 'Independent playback. Use the controls under each clip to inspect it separately.' : ready ? `Linked by elapsed video time. Frame stepping follows Clip ${this.master() ? 'B' : 'A'}; a shorter clip holds on its last frame.` : 'Playback is available when both videos are ready.';
    if (ready) { $('#pair-range').max = Math.max(...this.clips.map(c => c.frames.times.at(-1))); this.seek(this.time); }
  }
  frameAt(clip,t) { const ts = clip.frames.times; let lo=0,hi=ts.length-1; while(lo<hi) { const m=(lo+hi+1)>>1; if(ts[m] <= t+1e-6) lo=m; else hi=m-1; } return lo; }
  seekClip(clip, i) { const ts=clip.frames.times; i=Math.max(0,Math.min(ts.length-1,i)); const t=ts[i]+((ts[i+1] ?? ts[i]+clip.dt)-ts[i])/2; try {clip.video.currentTime=t;} catch(_) {} }
  seek(t) {
    if (!this.allReady()) return;
    this.time = Math.max(0,Math.min(Number($('#pair-range').max),t));
    this.clips.forEach((clip,i) => {this.seekClip(clip,this.frameAt(clip,this.time));this.updateClipTime(i);});
    $('#pair-range').value=this.time; $('#pair-time').textContent=`${num(this.time,2)} s`;
  }
  stepPair(delta) { if (!this.allReady() || !this.linked) return; this.pause(); const clip=this.clips[this.master()], i=Math.max(0,Math.min(clip.frames.times.length-1,this.frameAt(clip,this.time)+delta)); this.seek(clip.frames.times[i]); }
  stepClip(i,delta) { if (this.linked) return; const clip=this.clips[i]; if (!clip.loaded) return; clip.video.pause(); this.seekClip(clip,this.frameAt(clip,clip.video.currentTime)+delta); }
  updateClipTime(i) { if(this.dead) return; const clip=this.clips[i]; if(!clip.frames) return; const index=this.frameAt(clip,clip.video.currentTime); document.getElementById('clip-time-'+i).textContent=`${num(clip.frames.times[index],2)} s · ${index+1}/${clip.frames.times.length}`; $('.clip-range',clip.box).value=index; }
  async play() {
    if (!this.allReady() || !this.linked || this.playing) return;
    if (this.time >= +$('#pair-range').max-0.001) this.seek(0);
    this.playing=true; $('#pair-error').hidden=true; $('#pair-play').textContent='Ⅱ Pause both';
    try {
      await Promise.all(this.clips.map(c => {c.video.playbackRate=this.rate; return this.time < c.frames.times.at(-1) ? c.video.play() : Promise.resolve();}));
      if (!this.dead && this.playing) this.tick();
    } catch(_) { if(!this.dead) {this.pause();this.mediaError();} }
  }
  pause() { this.playing=false;cancelAnimationFrame(this.raf);this.clips.forEach(c=>c.video.pause());if(!this.dead) $('#pair-play').innerHTML=`${icon('play')} Play both`; }
  mediaError() { if(this.dead) return; $('#pair-error').textContent='Playback could not start. Try Play again after the videos have loaded.';$('#pair-error').hidden=false; }
  tick() {
    if (!this.playing || this.dead) return;
    const master=this.clips[this.master()], other=this.clips[1-this.master()], t=master.video.currentTime;
    this.time=Math.min(t,+$('#pair-range').max); $('#pair-range').value=this.time; $('#pair-time').textContent=`${num(this.time,2)} s`;
    if(t >= other.frames.times.at(-1)) {other.video.pause();this.seekClip(other,other.frames.times.length-1);}
    else if(master.video.readyState < 3) other.video.pause();
    else {if(Math.abs(other.video.currentTime-t) > 0.075) other.video.currentTime=t;if(other.video.paused) other.video.play().catch(()=>{});}
    if(master.video.ended || t >= master.frames.times.at(-1)) { this.pause();this.seek(+$('#pair-range').max);return; }
    this.raf=requestAnimationFrame(()=>this.tick());
  }
}
