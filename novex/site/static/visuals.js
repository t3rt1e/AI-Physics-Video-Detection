"use strict";
/* Presentation only: render the core's measurements, never infer a verdict. */
const SiteVisuals = (() => {
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = matchMedia('(pointer: fine)');
  const tracks = new WeakMap();

  function prepared(points) {
    if (tracks.has(points)) return tracks.get(points);
    const clean = points.filter(p => Number.isFinite(p.t) && Number.isFinite(p.x) && Number.isFinite(p.y)).slice().sort((a,b) => a.t-b.t);
    const gaps = clean.slice(1).map((p,i) => p.t-clean[i].t).filter(t => t>0).sort((a,b) => a-b);
    const result = {points:clean, gap:(gaps[gaps.length>>1] || 1/30)*3};
    tracks.set(points,result); return result;
  }

  function drawTracking(canvas, {points=[], time=0, tolerance=1/60, width=1, height=1, ball=true, trail=true, ghost=null}) {
    if (!canvas) return;
    const b=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio || 1;
    const cw=Math.round(b.width*dpr), ch=Math.round(b.height*dpr);
    if(canvas.width!==cw) canvas.width=cw;
    if(canvas.height!==ch) canvas.height=ch;
    const c=canvas.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0); c.clearRect(0,0,b.width,b.height);
    if(!b.width || !b.height || !width || !height) return;
    // Both players use object-fit:contain. Include letterboxing in the mapping.
    const scale=Math.min(b.width/width,b.height/height), W=width*scale,H=height*scale;
    const left=(b.width-W)/2,top=(b.height-H)/2, track=prepared(points), P=track.points;
    c.save(); c.translate(left,top); c.beginPath();c.rect(0,0,W,H);c.clip();
    const path = (elapsed) => {
      c.beginPath(); let prev=null;
      for(const p of P) {
        if(elapsed && p.t>time+tolerance) break;
        if(!prev || p.t-prev.t>track.gap) c.moveTo(p.x*W,p.y*H); else c.lineTo(p.x*W,p.y*H);
        prev=p;
      }
      c.stroke();
    };
    if(trail) {
      c.lineJoin='round'; c.lineCap='round'; c.setLineDash([3,5]);
      c.lineWidth=3; c.strokeStyle='rgba(0,12,30,.45)';path(false);
      c.lineWidth=1.5; c.strokeStyle='rgba(134,199,255,.8)';path(false);
      c.setLineDash([]); c.lineWidth=3.5;c.strokeStyle='rgba(0,12,30,.55)';path(true);
      c.lineWidth=2;c.strokeStyle='#88caff';path(true);
    }
    let lo=0,hi=P.length-1;
    while(lo<hi) {const mid=(lo+hi)>>1;if(P[mid].t<time)lo=mid+1;else hi=mid;}
    let p=P[lo]; if(lo>0 && Math.abs(P[lo-1].t-time)<Math.abs(p.t-time)) p=P[lo-1];
    if(p && Math.abs(p.t-time)>tolerance+1e-6) p=null;
    const radius = p && Number.isFinite(p.r) ? Math.max(p.r*W,4)+3 : 9;
    if(ghost && Number.isFinite(ghost.x) && Number.isFinite(ghost.y)) {
      c.strokeStyle='#ffb86b';c.lineWidth=2;c.setLineDash([5,3]);c.beginPath();c.arc(ghost.x*W,ghost.y*H,radius,0,Math.PI*2);c.stroke();c.setLineDash([]);
    }
    if(ball && p) {
      c.beginPath();c.arc(p.x*W,p.y*H,radius,0,Math.PI*2);
      c.lineWidth=5;c.strokeStyle='rgba(0,12,30,.65)';c.stroke();
      c.lineWidth=2.5;c.strokeStyle='#62b6ff';c.stroke();
      c.fillStyle='rgba(70,160,255,.08)';c.fill();
      c.beginPath();
      for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {c.moveTo(p.x*W+dx*(radius+3),p.y*H+dy*(radius+3));c.lineTo(p.x*W+dx*(radius+8),p.y*H+dy*(radius+8));}
      c.lineWidth=1.5;c.stroke();
    }
    c.restore();
    return {tracked:!!p, count:P.length};
  }

  function gauge(v={}, reference=9.8) {
    if(!Number.isFinite(v.g)) return '<div class="nog">No reliable gravity measurement yet. Open the technical details to see what the detector found.</div>';
    const g=v.g, sigma=Number.isFinite(v.sigma) ? Math.max(0,v.sigma) : 0;
    const ref=Number.isFinite(v.g_ref) && v.g_ref>0 ? v.g_ref : reference;
    const tol=Number.isFinite(v.tol) && v.tol>=0 ? v.tol : .15;
    const lower=Math.max(0,ref*(1-tol)),upper=ref*(1+tol);
    const minimum=Math.min(0,g-sigma), maximum=Math.max(14,upper*1.1,(g+sigma)*1.1);
    const raw=(maximum-minimum)/6, power=10**Math.floor(Math.log10(raw)), fraction=raw/power;
    const step=(fraction<=1?1:fraction<=2?2:fraction<=5?5:10)*power;
    const min=Math.floor(minimum/step)*step,max=Math.ceil(maximum/step)*step;
    const X=n=>16+(368*(n-min)/(max-min));
    let ticks='';
    for(let i=0;i<=Math.round((max-min)/step);i++) {const n=min+i*step;ticks+=`<g class="gauge-tick"><path d="M${X(n)} 40v6"/><text x="${X(n)}" y="63" text-anchor="middle">${Number(n.toPrecision(4))}</text></g>`;}
    const note=v.k && Math.abs(v.k-1)>.02 ? `<p class="gauge-note">Adjusted for ${Number(v.k).toFixed(1)}× slow motion</p>` : '';
    return `<svg viewBox="0 0 400 76" role="img" aria-label="Measured gravity ${g.toFixed(2)} metres per second squared. Earth comparison range ${lower.toFixed(2)} to ${upper.toFixed(2)}.">
      <rect x="16" y="24" width="368" height="16" rx="2" fill="var(--line)"/>
      <rect x="${X(lower)}" y="19" width="${Math.max(1,X(upper)-X(lower))}" height="26" rx="2" fill="var(--orangeSoft)" stroke="var(--orange)"/>
      <path d="M${X(ref)} 19v26" stroke="var(--orange)" stroke-dasharray="2 3"/>
      ${ticks}
      <path d="M${X(g-sigma)} 32H${X(g+sigma)}" stroke="var(--blue)" stroke-width="6" opacity=".35"/>
      <path d="M${X(g)} 12v9" stroke="var(--blue)" stroke-width="2"/>
      <circle cx="${X(g)}" cy="32" r="7" fill="var(--blue)" stroke="var(--card)" stroke-width="3"/>
    </svg><div class="gauge-key"><span><i class="measure-dot"></i>Measured${sigma ? ` ± ${sigma.toFixed(2)}` : ''}</span><span><i class="earth-band"></i>Earth range · ±${Number((tol*100).toFixed(1))}%</span></div>${note}`;
  }

  // Pointer light and one brief text decode. No continuous render loop or dependency.
  let pointerFrame=0, active=null;
  document.addEventListener('pointermove', e => {
    if(!finePointer.matches || motion.matches) return;
    const card=e.target.closest('[data-light]');
    if(active && active!==card) active.removeAttribute('data-lit');
    active=card; cancelAnimationFrame(pointerFrame);
    if(!card) return;
    pointerFrame=requestAnimationFrame(() => {
      if(!card.isConnected) return;
      const r=card.getBoundingClientRect();card.style.setProperty('--mx',`${e.clientX-r.left}px`);card.style.setProperty('--my',`${e.clientY-r.top}px`);card.dataset.lit='';
    });
  },{passive:true});
  document.addEventListener('pointerout', e => { if(!e.relatedTarget && active) {cancelAnimationFrame(pointerFrame);active.removeAttribute('data-lit');active=null;} });
  const decoding=new WeakSet();
  document.addEventListener('pointerover', e => {
    const target=e.target.closest('[data-decode]');
    if(!target || motion.matches || !finePointer.matches || decoding.has(target) || target.contains(e.relatedTarget)) return;
    const original=target.dataset.decode, alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';let start;
    decoding.add(target);
    const frame=t => {
      start??=t;const progress=Math.min(1,(t-start)/520);
      if(!target.isConnected || progress===1 || motion.matches) {target.textContent=original;decoding.delete(target);return;}
      const resolved=Math.floor(progress*original.length);
      target.textContent=Array.from(original,(letter,i)=>i<resolved || letter===' '?letter:alphabet[Math.floor((t/45+i*7)%alphabet.length)]).join('');
      requestAnimationFrame(frame);
    };requestAnimationFrame(frame);
  });
  return {drawTracking,gauge};
})();
