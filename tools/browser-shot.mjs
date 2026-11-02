// Opens a page in headless Chrome over the DevTools protocol, waits until `ready` (a JS expression)
// is truthy, runs optional `actions`, then saves a screenshot and prints the console.
// node tools/browser-shot.mjs <url> <out.png> [--sculpt] [--headed]   (add ?webgl to the URL for WebGL2)
import { spawn } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [url, out, ...flags] = process.argv.slice(2);
const port = 9300 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), 'sculpt-shot-'));
const args = [`--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--window-size=1280,800',
  '--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check'];
if (!flags.includes('--headed')) args.push('--headless=new');
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [...args, 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws, id = 0;
const pending = new Map(), logs = [];
const send = (method, params = {}) => new Promise((res, rej) => {
  const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params }));
});
try {
  let target;
  for (let k = 0; k < 50 && !target; k++) {
    await sleep(200);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(t => t.type === 'page'); } catch {}
  }
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise(r => ws.onopen = r);
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.rej(new Error(m.error.message)) : p.res(m.result); }
    if (m.method === 'Runtime.consoleAPICalled') logs.push(m.params.args.map(a => a.value ?? a.description).join(' '));
    if (m.method === 'Runtime.exceptionThrown') logs.push('EXCEPTION ' + JSON.stringify(m.params.exceptionDetails).slice(0, 400));
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.navigate', { url });
  const evaluate = async expr => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result.value;
  let status = '';
  for (let k = 0; k < 150; k++) {
    await sleep(200);
    status = await evaluate("document.getElementById('status')?.textContent || ''");
    if (status.includes('ms/frame')) break;
  }
  if (flags.includes('--sculpt')) {
    // A stroke across the sphere through the engine's own API, as pointer events would drive it.
    await evaluate(`(() => { const e = window.engine, c = document.getElementById('c');
      const w = c.width, h = c.height; e.setBrush('draw', 5, 0.8, true);
      const t = performance.now();
      e.sculptBegin(w * 0.55, h * 0.35, 1);
      for (let i = 0; i <= 40; i++) e.sculptMove(w * (0.55 + 0.04 * Math.sin(i / 6)), h * (0.35 + i * 0.008), 1);
      const len = e.sculptEnd()?.length;
      window.strokeMs = performance.now() - t;
      return len; })()`);
    await sleep(800);
  }
  status = await evaluate("document.getElementById('status')?.textContent || ''");
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(out, Buffer.from(shot.data, 'base64'));
  console.log(`status: ${status}`);
  console.log(`create: ${(await evaluate('window.createMs'))?.toFixed(0)} ms, stroke (begin, 41 moves, end): ${(await evaluate('window.strokeMs'))?.toFixed(1)} ms`);
  console.log(logs.slice(0, 20).join('\n'));
} finally {
  try { ws?.close(); } catch {}
  chrome.kill();
  await sleep(300);
  rmSync(profile, { recursive: true, force: true });
}
