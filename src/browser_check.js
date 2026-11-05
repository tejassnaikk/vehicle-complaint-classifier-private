// Drive headless Chrome through the DevTools protocol (Node 22+ has a built-in WebSocket client).
// Used to test and screenshot the interactive Explore pages at real viewport widths.
//   node src/browser_check.js <page.html> <scenario.js> [widths=1440,390]
// scenario.js exports: async function run(p, width) with p.eval(expr), p.waitFor(expr, ms), p.click(sel), p.type(sel, text),
// p.shot(name), p.log(...). Screenshots go to results/site_screenshots/<name>_<width>.png. Serves docs/ on a local port.
const { spawn } = require("child_process");
const http = require("http"), fs = require("fs"), path = require("path"), os = require("os");
const root = path.resolve(__dirname, "..");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml", ".bin": "application/octet-stream" };

function serve() {
  return new Promise((res) => {
    const srv = http.createServer((q, r) => {
      let f = path.join(root, "docs", decodeURIComponent(q.url.split("?")[0]));
      if (f.endsWith("/")) f += "index.html";
      fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { "content-type": MIME[path.extname(f)] || "application/octet-stream" }); r.end(d); } });
    }).listen(0, "127.0.0.1", () => res(srv));
  });
}

async function main() {
  const [page, scen, widthsArg] = process.argv.slice(2);
  const widths = (widthsArg || "1440,390").split(",").map(Number);
  const scenario = scen && scen !== "-" ? require(path.resolve(scen)) : { run: async (p) => { await p.sleep(1500); await p.shot(path.basename(page, ".html")); } };
  const srv = await serve(), port = srv.address().port;
  const dbg = 9300 + Math.floor(Math.random() * 500), prof = fs.mkdtempSync(path.join(os.tmpdir(), "chr-"));
  const chrome = spawn(CHROME, ["--headless=new", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${dbg}`, `--user-data-dir=${prof}`, "about:blank"], { stdio: "ignore" });
  let targets;
  for (let i = 0; i < 50; i++) { try { targets = await (await fetch(`http://127.0.0.1:${dbg}/json/list`)).json(); if (targets.find((t) => t.type === "page")) break; } catch (e) {} await new Promise((r) => setTimeout(r, 200)); }
  const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0; const pend = new Map(), errors = [];
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pend.has(d.id)) { pend.get(d.id)(d); pend.delete(d.id); return; }
    if (d.method === "Runtime.exceptionThrown") errors.push("exception: " + (d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text));
    if (d.method === "Runtime.consoleAPICalled" && d.params.type === "error") errors.push("console.error: " + d.params.args.map((a) => a.value ?? a.description).join(" "));
    if (d.method === "Log.entryAdded" && d.params.entry.level === "error" && !/favicon/.test(d.params.entry.url || "")) errors.push("log: " + d.params.entry.text + " " + (d.params.entry.url || ""));
  };
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  await send("Runtime.enable"); await send("Page.enable"); await send("Log.enable");
  const out = path.join(root, "results", "site_screenshots"); fs.mkdirSync(out, { recursive: true });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let failed = 0;
  for (const w of widths) {
    await send("Emulation.setDeviceMetricsOverride", { width: w, height: 900, deviceScaleFactor: 1, mobile: w < 600 });
    await send("Page.navigate", { url: `http://127.0.0.1:${port}/${page}` });
    await sleep(800);
    const p = {
      sleep,
      eval: async (expr) => { const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true }); if (r.result.exceptionDetails) throw new Error("eval failed: " + expr + " -> " + JSON.stringify(r.result.exceptionDetails.exception?.description)); return r.result.result.value; },
      waitFor: async (expr, ms = 20000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { if (await p.eval(expr)) return; await sleep(100); } throw new Error("timeout waiting for " + expr); },
      click: (sel) => p.eval(`document.querySelector(${JSON.stringify(sel)}).click()`),
      type: (sel, text) => p.eval(`(()=>{const e=document.querySelector(${JSON.stringify(sel)});e.value=${JSON.stringify(text)};e.dispatchEvent(new Event('input',{bubbles:true}));})()`),
      log: (...a) => console.log(`[${w}]`, ...a),
      shot: async (name) => {
        const m = (await send("Page.getLayoutMetrics")).result.cssContentSize;
        const h = Math.min(Math.ceil(m.height), 16000);
        await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile: w < 600 });
        await sleep(300);
        const r = await send("Page.captureScreenshot", { format: "png" });
        fs.writeFileSync(path.join(out, `${name}_${w}.png`), Buffer.from(r.result.data, "base64"));
        await send("Emulation.setDeviceMetricsOverride", { width: w, height: 900, deviceScaleFactor: 1, mobile: w < 600 });
        console.log(`[${w}] screenshot ${name}_${w}.png ${w}x${h}`);
      },
      overflowX: () => p.eval("document.documentElement.scrollWidth > document.documentElement.clientWidth + 1"),
    };
    try { await scenario.run(p, w); if (await p.overflowX()) { console.log(`[${w}] FAIL horizontal overflow: scrollWidth ${await p.eval("document.documentElement.scrollWidth")}`); failed++; } }
    catch (e) { console.log(`[${w}] FAIL`, e.message); failed++; }
  }
  if (errors.length) { console.log("browser errors:\n  " + [...new Set(errors)].join("\n  ")); failed++; }
  ws.close(); chrome.kill(); srv.close();
  console.log(failed ? "RESULT: FAILED" : "RESULT: OK");
  process.exit(failed ? 1 : 0);
}
main();
