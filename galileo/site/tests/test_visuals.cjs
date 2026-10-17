// Run with: node --test galileo/site/tests/test_visuals.cjs
// No browser or new production dependency required.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {runInNewContext} = require('node:vm');
const visuals = runInNewContext(readFileSync(join(__dirname,'../static/visuals.js'),'utf8') + '\nSiteVisuals', {
  matchMedia:()=>({matches:false}), document:{addEventListener(){}}, window:{devicePixelRatio:1},
});
function surface(width=300,height=300) {
  const calls=[], context={};
  for(const name of ['setTransform','clearRect','save','translate','beginPath','rect','clip','moveTo','lineTo','stroke','setLineDash','arc','fill','restore'])
    context[name]=(...args)=>calls.push([name,...args]);
  return {canvas:{width:0,height:0,getBoundingClientRect:()=>({width,height}),getContext:()=>context},calls};
}
test('highlight aligns with the video inside a letterboxed comparison pane',()=>{
  const {canvas,calls}=surface();
  const state=visuals.drawTracking(canvas,{points:[{t:0,x:.5,y:.5,r:.05}],time:0,width:400,height:200,trail:false});
  assert.equal(state.tracked,true);
  assert.deepEqual(calls.find(c=>c[0]==='translate'),['translate',0,75]);
  const arc=calls.find(c=>c[0]==='arc');assert.equal(arc[1],150);assert.equal(arc[2],75);
});
test('the full measured path is visible before playback reaches the last point',()=>{
  const {canvas,calls}=surface(100,100);
  visuals.drawTracking(canvas,{points:[{t:0,x:.2,y:.2},{t:.1,x:.3,y:.4},{t:.2,x:.4,y:.8}],time:0,width:100,height:100,ball:false});
  assert(calls.some(c=>c[0]==='lineTo' && c[1]===40 && c[2]===80));
});
test('no highlight is invented on a frame without a nearby detection',()=>{
  const {canvas,calls}=surface();
  const state=visuals.drawTracking(canvas,{points:[{t:0,x:.5,y:.5,r:.05}],time:1,tolerance:.02,width:300,height:300});
  assert.equal(state.tracked,false);assert(!calls.some(c=>c[0]==='arc'));
});
test('the path does not bridge a long gap in recorded tracking',()=>{
  const {canvas,calls}=surface(100,100);
  visuals.drawTracking(canvas,{points:[{t:0,x:.1,y:.1},{t:.1,x:.2,y:.2},{t:.2,x:.3,y:.3},{t:2,x:.8,y:.8},{t:2.1,x:.9,y:.9}],time:0,width:100,height:100,ball:false});
  assert(calls.some(c=>c[0]==='moveTo' && c[1]===80));assert(!calls.some(c=>c[0]==='lineTo' && c[1]===80));
});
test('empty tracks and disabled overlays leave the source image unobstructed',()=>{
  const {canvas,calls}=surface();
  const state=visuals.drawTracking(canvas,{points:[],time:0,width:100,height:100,ball:false,trail:false});
  assert.equal(state.count,0);assert.equal(state.tracked,false);assert(!calls.some(c=>c[0]==='stroke'));
});
test('the gravity range uses the reference and tolerance returned by the core',()=>{
  const html=visuals.gauge({g:11,g_ref:10,tol:.2});
  assert.match(html,/range 8\.00 to 12\.00/);assert.match(html,/Earth range · ±20%/);
});
test('missing measurements show an explicit empty state, never a fabricated zero',()=>{
  for(const g of [null,undefined,NaN,Infinity]) {
    const html=visuals.gauge({g});assert.match(html,/No reliable/);assert(!html.includes('<svg'));
  }
});
test('extreme measured values retain a finite scale and a bounded number of ticks',()=>{
  for(const g of [5000000,-23,0,.0001]) {
    const html=visuals.gauge({g,sigma:1});assert(!/NaN|Infinity/.test(html));
    const ticks=[...html.matchAll(/class="gauge-tick"/g)].length;assert(ticks>=3 && ticks<=10);
  }
});
