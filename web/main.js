// Demo host: the page owns the canvas and the frame clock, the engine draws and sculpts.
import init, { WebEngine } from './pkg/sculpt_engine.js';

const canvas = document.getElementById('c');
const status = document.getElementById('status');
const fit = () => {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
  return dpr;
};
let dpr = fit();

await init();
const t0create = performance.now();
const engine = await WebEngine.create(canvas, new URLSearchParams(location.search).has('webgl'));
window.createMs = performance.now() - t0create; // GPU setup, IBL bake, first mesh and BVH
const backend = engine.backend();
window.engine = engine; // for tests and screenshots

let kind = 'draw';
const brush = () => engine.setBrush(kind, +document.getElementById('radius').value, 0.6, document.getElementById('sym').checked);
brush();
document.querySelectorAll('[data-kind]').forEach(b => b.onclick = () => {
  kind = b.dataset.kind;
  document.querySelectorAll('[data-kind]').forEach(x => x.setAttribute('aria-pressed', x === b));
  brush();
});
document.getElementById('radius').oninput = brush;
document.getElementById('sym').onchange = brush;
document.getElementById('look').onchange = e => engine.setLook(e.target.value, e.target.value !== 'linear', e.target.value !== 'linear');
document.getElementById('undo').onclick = () => engine.undo();
document.getElementById('redo').onclick = () => engine.redo();
document.getElementById('file').onchange = async e => {
  const f = e.target.files[0];
  if (f) try { engine.openProject(new Uint8Array(await f.arrayBuffer())); } catch (err) { alert(err); }
};

let mode = null, last = null;
canvas.onpointerdown = e => {
  canvas.setPointerCapture(e.pointerId);
  const x = e.offsetX * dpr, y = e.offsetY * dpr, p = e.pressure || 0.5;
  mode = e.button === 0 && !e.shiftKey && engine.sculptBegin(x, y, e.pointerType === 'mouse' ? 1 : p) ? 'sculpt' : (e.button === 1 || e.shiftKey ? 'pan' : 'orbit');
  last = [e.offsetX, e.offsetY];
};
canvas.onpointermove = e => {
  if (!mode) return;
  const dx = e.offsetX - last[0], dy = e.offsetY - last[1];
  last = [e.offsetX, e.offsetY];
  if (mode === 'sculpt') engine.sculptMove(e.offsetX * dpr, e.offsetY * dpr, e.pointerType === 'mouse' ? 1 : (e.pressure || 0.5));
  else if (mode === 'orbit') engine.orbit(dx, dy);
  else engine.pan(dx * dpr, dy * dpr);
};
canvas.onpointerup = canvas.onpointercancel = () => { if (mode === 'sculpt') engine.sculptEnd(); mode = null; };
canvas.onwheel = e => { e.preventDefault(); engine.zoom(Math.exp(-e.deltaY * 0.002)); };
window.onkeydown = e => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? engine.redo() : engine.undo(); }
};
window.onresize = () => { dpr = fit(); engine.resize(canvas.width, canvas.height); };

let frames = 0, t0 = performance.now();
const loop = () => {
  engine.render();
  if (++frames === 30) {
    const ms = (performance.now() - t0) / 30;
    status.textContent = `${backend} | ${ms.toFixed(1)} ms/frame`;
    frames = 0; t0 = performance.now();
  }
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
