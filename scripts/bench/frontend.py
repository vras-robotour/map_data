#!/usr/bin/env python3
# ruff: noqa: E501, RUF001
"""Frontend benchmark for the map_data viewer (headless Chromium via Playwright).

    uv run --with playwright python scripts/bench/frontend.py --quick
    uv run --with playwright python scripts/bench/frontend.py --map strom.mapdata --reps 7 --out base.json
    uv run --with playwright python scripts/bench/frontend.py --map strom.mapdata --compare base.json
    uv run --with playwright python scripts/bench/frontend.py --map strom.mapdata --profile
    uv run --with playwright python scripts/bench/frontend.py --flows

Starts its own viewer on a throw-away copy of the data (maps are looked up in
--src, then demo/), serves CDN assets and tiles from a local cache so the
numbers do not depend on the network, and reports median and spread
(min-max half range) over N repetitions, each in a fresh browser context.

Latency = input event timeStamp -> first frame after the UI reached the
expected state (rAF + setTimeout), the usual INP approximation. "client" is
the part after the last /api response, i.e. excluding server time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "map_data" / "viewer" / "static"
CACHE = Path(os.environ.get("MAPDATA_BENCH_CACHE", Path.home() / ".cache" / "map_data_bench"))
DEFAULT_SRC = Path("/home/vlkjan6/Documents/mapdata_ws/src/map_data/data")
PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8ffff3f0005fe02fea7d6050d0000000049454e44ae426082"
)
CDN_HOSTS = (
    "unpkg.com",
    "cdn.jsdelivr.net",
    "cdn.socket.io",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
)

# Installed before any page script: long-task log + one-shot latency probes.
INIT_JS = r"""
(() => {
  const b = window.__b = { long: [], samples: [], frames: [], lcp: null };
  new PerformanceObserver(l => l.getEntries().forEach(e => b.long.push([e.startTime, e.duration])))
    .observe({ type: 'longtask', buffered: true });
  new PerformanceObserver(l => l.getEntries().forEach(e => { b.lcp = e.startTime; }))
    .observe({ type: 'largest-contentful-paint', buffered: true });
  // arm(label, events, cond): on the next matching input event start a clock; stop it on the first
  // frame after cond() holds.
  b.arm = (label, events, cond) => {
    const fn = cond ? new Function('return (' + cond + ')') : () => true;
    const h = ev => {
      events.forEach(n => window.removeEventListener(n, h, true));
      const t0 = ev.timeStamp;
      const tick = () => {
        let ok = false;
        try { ok = fn(); } catch (_) { }
        if (!ok) {
          if (performance.now() - t0 > 10000) { b.samples.push({ label, t0, ms: null }); return; }
          return requestAnimationFrame(tick);
        }
        setTimeout(() => {
          const t1 = performance.now();
          const net = performance.getEntriesByType('resource')
            .filter(r => r.name.includes('/api/') && r.responseEnd >= t0).map(r => r.responseEnd);
          const lastNet = net.length ? Math.max(...net) : t0;
          b.samples.push({ label, t0, t1, ms: t1 - t0, client: t1 - Math.max(t0, lastNet) });
        }, 0);
      };
      requestAnimationFrame(tick);
    };
    events.forEach(n => window.addEventListener(n, h, true));
  };
  b.frameLog = ms => {
    b.frames = [];
    let last = performance.now();
    const end = last + ms;
    const loop = t => { b.frames.push(t - last); last = t; if (t < end) requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  };
})();
"""


# ── server / data ─────────────────────────────────────────────────────────────
def prepare_data(maps: list[str], src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in maps:
        found = next((d / name for d in (src, ROOT / "demo") if (d / name).exists()), None)
        if found is None:
            sys.exit(f"map {name!r} not found in {src} or demo/")
        shutil.copy(found, dest / name)
        ann = found.with_suffix(".annotations.json")
        if ann.exists():
            shutil.copy(ann, dest / ann.name)


def start_server(data_dir: Path, port: int) -> subprocess.Popen:
    with socket.socket() as sk:  # a leftover server on the port would serve stale data
        if sk.connect_ex(("127.0.0.1", port)) == 0:
            sys.exit(f"port {port} is already in use")
    proc = subprocess.Popen(
        ["uv", "run", "map_data_viewer", "--data-dir", str(data_dir), "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/files", timeout=1).read()
            return proc
        except Exception:
            time.sleep(0.5)
    stop_server(proc)
    sys.exit("viewer did not start")


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        pass


def reset_annotations(maps: list[str], src: Path, data_dir: Path) -> None:
    for name in maps:
        found = next((d / name for d in (src, ROOT / "demo") if (d / name).exists()), None)
        ann = found.with_suffix(".annotations.json") if found else None
        target = data_dir / Path(name).with_suffix(".annotations.json").name
        if ann and ann.exists():
            shutil.copy(ann, target)
        else:
            target.unlink(missing_ok=True)


# ── browser plumbing ──────────────────────────────────────────────────────────
def install_routes(ctx, base: str, patches: dict[str, tuple[str, str]]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)

    def handler(route):
        url = route.request.url
        if url.startswith(base):
            for fname, (old, new) in patches.items():
                if url.split("?")[0].endswith(fname):
                    body = route.fetch().text().replace(old, new)
                    return route.fulfill(body=body, content_type="text/javascript")
            return route.continue_()
        if any(h in url for h in CDN_HOSTS):
            f = CACHE / hashlib.sha1(url.encode()).hexdigest()
            meta = f.with_suffix(".ct")
            if not f.exists():
                r = route.fetch()
                f.write_bytes(r.body())
                meta.write_text(r.headers.get("content-type", "application/octet-stream"))
            return route.fulfill(body=f.read_bytes(), content_type=meta.read_text())
        return route.fulfill(body=PNG_1PX, content_type="image/png")  # map tiles

    ctx.route("**/*", handler)


class Rep:
    """One fresh page; helpers for timed interactions."""

    def __init__(self, browser, base: str, patches, profile: bool, trace: bool = False):
        self.browser, self.trace, self.traces = browser, trace, {}
        self.ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        install_routes(self.ctx, base, patches)
        self.ctx.add_init_script(INIT_JS)
        self.page = self.ctx.new_page()
        self.cdp = self.ctx.new_cdp_session(self.page)
        self.errors: list[str] = []
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.on(
            "console", lambda m: m.type == "error" and self.errors.append(f"console: {m.text}")
        )
        self.base, self.metrics = base, {}
        self.profile = profile
        self.profiles: dict[str, list] = {}
        self._n = 0
        self.cdp.send("Performance.enable")
        if profile:
            self.cdp.send("Profiler.enable")
            self.cdp.send("Profiler.setSamplingInterval", {"interval": 200})

    def ev(self, js, arg=None):
        return self.page.evaluate(js, arg)

    def prof_start(self):
        if self.profile:
            self.cdp.send("Profiler.start")
        if self.trace:
            self.browser.start_tracing(page=self.page, categories=TRACE_CATS)

    def prof_stop(self, name):
        if self.profile:
            self.profiles[name] = self.cdp.send("Profiler.stop")["profile"]
        if self.trace:
            self.traces[name] = trace_buckets(json.loads(self.browser.stop_tracing()))

    def timed(self, label, events, cond, action, *, timeout=30000):
        """Arm a probe, run `action`, wait for the sample, record it under `label`."""
        self.ev("([l, e, c]) => __b.arm(l, e, c)", [label, events, cond])
        self._n = self.ev("__b.samples.length")
        self.prof_start()
        action()
        self.page.wait_for_function(f"__b.samples.length > {self._n}", timeout=timeout)
        self.prof_stop(label)
        s = self.ev(f"__b.samples[{self._n}]")
        if s["ms"] is None:
            print(f"  WARN: {label}: condition never reached", file=sys.stderr)
            self.errors.append(f"timeout: {label}")
        self.metrics.setdefault(label, []).append(s["ms"])
        self.metrics.setdefault(label + ":client", []).append(s.get("client"))
        return s

    def hit(self, w):
        pt = self.ev(_HIT_JS, w)
        if pt is None:
            raise RuntimeError(f"no clickable point for way {w['id']}")
        return pt

    def pick(self, candidates):
        """First way in `candidates` that a click can actually hit -> (way, page point)."""
        for w in candidates:
            pt = self.ev(_HIT_JS, w)
            if pt is not None:
                return w, pt
        raise RuntimeError("no clickable way found")

    def settle(self, ms=400):
        self.page.wait_for_timeout(ms)

    def tbt(self, t0, t1):
        long = self.ev("__b.long")
        in_win = [d for s, d in long if t0 <= s <= t1]
        return sum(max(0, d - 50) for d in in_win), max(in_win, default=0), len(in_win)


TRACE_CATS = ["devtools.timeline", "v8", "disabled-by-default-devtools.timeline"]
BUCKETS = {
    "script": (
        "FunctionCall",
        "EvaluateScript",
        "EventDispatch",
        "TimerFire",
        "FireAnimationFrame",
        "v8.callFunction",
        "RunMicrotasks",
        "HitTest",
    ),
    "gc": ("MinorGC", "MajorGC", "V8.GC_", "BlinkGC"),
    "style": ("UpdateLayoutTree", "RecalculateStyles", "ScheduleStyleRecalculation"),
    "layout": ("Layout",),
    "paint": (
        "Paint",
        "PrePaint",
        "Layerize",
        "UpdateLayer",
        "Commit",
        "CompositeLayers",
        "RasterTask",
        "ImageDecodeTask",
    ),
    "parse": ("ParseHTML", "v8.compile", "v8.parseOnBackground", "EvaluateScript"),
}


def trace_buckets(trace) -> dict[str, float]:
    """Exclusive main-thread ms per bucket (script / gc / style / layout / paint / other) of a Chrome trace."""
    evs = trace["traceEvents"] if isinstance(trace, dict) else trace
    main = {
        (e["pid"], e["tid"])
        for e in evs
        if e.get("name") == "thread_name" and e["args"]["name"] == "CrRendererMain"
    }
    xs = sorted(
        (e for e in evs if e.get("ph") == "X" and (e["pid"], e["tid"]) in main),
        key=lambda e: (e["ts"], -e["dur"]),
    )
    out: dict[str, float] = {}
    # Exclusive time: walk events in start order and subtract nested children.
    excl: list[tuple[str, float]] = []
    active: list[dict] = []
    for e in xs:
        while active and active[-1]["end"] <= e["ts"]:
            a = active.pop()
            excl.append((a["name"], a["dur"] - a["child"]))
        if active:
            active[-1]["child"] += e["dur"]
        active.append({"name": e["name"], "end": e["ts"] + e["dur"], "dur": e["dur"], "child": 0})
    while active:
        a = active.pop()
        excl.append((a["name"], a["dur"] - a["child"]))
    skip = {
        "RunTask",
        "ThreadControllerImpl::RunTask",
        "ThreadControllerImpl::DoWork",
        "RunLoop::Run",
    }
    for name, d in excl:
        if name in skip:
            continue
        b = "other"
        for k, pats in BUCKETS.items():
            if k != "parse" and any(name.startswith(p) for p in pats):
                b = k
                break
        out[b] = out.get(b, 0) + d / 1000
    return out


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def run_rep(browser, base, mapname, patches, args) -> Rep:
    r = Rep(browser, base, patches, args.profile, args.trace)
    p, m = r.page, r.metrics
    p.goto(base + "/", wait_until="load")
    r.settle(800)
    m["page_lcp"] = [r.ev("__b.lcp")]
    nav = r.ev("performance.getEntriesByType('navigation')[0].toJSON()")
    m["page_dcl"] = [nav["domContentLoadedEventEnd"]]
    m["page_load"] = [nav["loadEventEnd"]]
    p.wait_for_function("document.getElementById('file-select').options.length > 1")

    # (a) load the big map: Load click -> all layers rendered, UI responsive
    p.select_option("#file-select", mapname)
    s = r.timed(
        "load",
        ["pointerdown"],
        "document.getElementById('status').textContent.startsWith('Loaded')",
        lambda: p.click("#load-btn"),
    )
    r.settle(500)
    blocking, longest, _ = r.tbt(s["t0"], s["t1"])
    m["load:tbt"] = [blocking]
    m["load:longest_task"] = [longest]
    net = r.ev(
        "performance.getEntriesByType('resource').filter(x => x.name.includes('/api/mapdata'))"
        ".map(x => [x.responseEnd - x.startTime, x.transferSize, x.decodedBodySize])[0]"
    )
    m["load:fetch_mapdata"] = [net[0]]
    m["payload_kb"] = [net[2] / 1024]
    # (c) memory after load
    r.cdp.send("HeapProfiler.collectGarbage")
    pm = {x["name"]: x["value"] for x in r.cdp.send("Performance.getMetrics")["metrics"]}
    m["heap_mb"] = [pm["JSHeapUsedSize"] / 1e6]
    m["dom_nodes"] = [pm["Nodes"]]
    info = r.ev(_PICK_JS)
    m["n_layers"] = [info["nLayers"]]
    r.info = info

    # click a way (view mode)
    way, pt = r.pick(info["mids"])
    r.settle(500)
    p.mouse.move(pt["x"], pt["y"])
    r.timed(
        "click_way",
        ["pointerdown"],
        f"currentClickedFeature && currentClickedFeature.properties.id === {json.dumps(way['id'])}",
        lambda: p.mouse.click(pt["x"], pt["y"]),
    )
    r.settle()

    # pan/zoom frame cost (renderer comparison): drag the map, log rAF deltas
    r.ev("__b.frameLog(1500)")
    cx, cy = 720, 450
    p.mouse.move(cx, cy)
    p.mouse.down()
    for i in range(40):
        p.mouse.move(cx + i * 6, cy + i * 3)
        p.wait_for_timeout(16)
    p.mouse.up()
    p.wait_for_timeout(900)
    fr = r.ev("__b.frames.slice(2)")
    m["pan:frame_p95"] = [pctl(fr, 0.95)]
    m["pan:slow_frames"] = [sum(1 for f in fr if f > 33)]
    r.ev(_HIT_JS, way)
    r.settle(500)

    # search box, per keystroke
    p.click("#feature-search")
    for ch in info["query"]:
        r.timed("search_key", ["keydown"], None, lambda c=ch: p.keyboard.press(c))
        r.settle(250)
    p.keyboard.press("Control+a")
    r.timed("search_clear", ["keydown"], None, lambda: p.keyboard.press("Backspace"))
    r.settle()
    p.evaluate("document.activeElement.blur()")

    # layer checkbox toggle (off, on)
    for want in ("off", "on"):
        r.timed(
            f"toggle_layer_{want}",
            ["pointerdown"],
            None,
            lambda: p.click(f'input[data-layer="{info["toggle"]}"]'),
        )
        r.settle()

    # edit mode: enter, select the long way, drag one node, undo
    r.timed(
        "edit_enter",
        ["pointerdown"],
        "currentMode === 'edit'",
        lambda: p.click('.mode-btn[data-mode="edit"]'),
    )
    r.settle()
    longway, pt = r.pick(info["longs"])
    m["edit_way_vertices"] = [longway["n"]]
    r.settle(500)
    p.mouse.move(pt["x"], pt["y"])
    r.timed(
        "edit_select_long_way",
        ["pointerdown"],
        "typeof nodeLayer !== 'undefined' && nodeLayer && currentNodes.length > 0 && map.hasLayer(nodeLayer)",
        lambda: p.mouse.click(pt["x"], pt["y"]),
    )
    m["edit_nodes"] = [r.ev("currentNodes.length")]
    r.settle(600)
    # drag the middle node
    r.ev(
        "(() => { const n = currentNodes[currentNodes.length >> 1]; map.setView([n.lat, n.lon], map.getZoom(), {animate: false}); })()"
    )
    r.settle(500)
    npt = r.ev(
        "(() => { const n = currentNodes[currentNodes.length >> 1]; const q = map.latLngToContainerPoint([n.lat, n.lon]); const r = map.getContainer().getBoundingClientRect(); return {x: r.left + q.x, y: r.top + q.y}; })()"
    )
    p.mouse.move(npt["x"], npt["y"])
    p.mouse.down()
    n0 = r.ev("__b.samples.length")
    for i in range(1, 21):
        r.ev("([l]) => __b.arm(l, ['mousemove'], null)", ["drag_move"])
        p.mouse.move(npt["x"] + i * 2, npt["y"] + i)
        p.wait_for_function(f"__b.samples.length > {n0 + i - 1}")
        p.wait_for_timeout(20)
    for s in r.ev(f"__b.samples.slice({n0})"):
        m.setdefault("drag_move", []).append(s["ms"])
    r.timed(
        "drag_end",
        ["mouseup"],
        "changeLog.some(c => c.type === 'move')",
        lambda: p.mouse.up(),
    )
    r.settle(600)
    p.mouse.move(5, 5)
    r.timed(
        "undo",
        ["keydown"],
        "document.getElementById('status').textContent.startsWith('Undid')",
        lambda: p.keyboard.press("Control+z"),
    )
    r.settle(300)
    return r


# Picks, in page: a mid-size way to click, the longest way (most vertices) to edit, a search term,
# a layer with many sub-layers to toggle.
_PICK_JS = r"""
(() => {
  const ways = [];
  let nLayers = 0;
  ['road', 'footway', 'barrier'].forEach(cat => {
    if (!geoLayers[cat]) return;
    geoLayers[cat].eachLayer(l => {
      nLayers++;
      const ll = l.getLatLngs();
      const ring = Array.isArray(ll[0]) ? (Array.isArray(ll[0][0]) ? ll[0][0] : ll[0]) : ll;
      const geomType = l._featureRef.geometry.type;
      ways.push({ id: l._featureId, cat, n: ring.length, geomType,
                  lat: ring[0].lat, lng: ring[0].lng, tags: l._featureRef.properties.tags || {} });
    });
  });
  const editable = ways.filter(w => w.cat !== 'crossroad');
  const sorted = [...editable].sort((a, b) => b.n - a.n);
  const third = Math.floor(sorted.length / 3);
  const toggle = ['footway', 'road', 'barrier']
    .map(c => [c, geoLayers[c] ? geoLayers[c].getLayers().length : 0]).sort((a, b) => b[1] - a[1])[0][0];
  return { longs: sorted.slice(0, 40), mids: sorted.slice(third, third + 40), toggle, query: 'footway', nLayers };
})()
"""
# Centre the map on a point where a click hits exactly `w` (topmost, as Leaflet's canvas hit-test
# does) and return its page coordinates, or null.
_HIT_JS = r"""
w => {
  let target = null;
  ['road', 'footway', 'barrier', 'crossroad'].forEach(c => geoLayers[c] && geoLayers[c].eachLayer(l => {
    if (l._featureId === w.id) target = l;
  }));
  if (!target) return null;
  const flat = a => (Array.isArray(a) ? a.flatMap(flat) : [a]);
  const ring = flat(target.getLatLngs());
  const n = ring.length;
  // Buffered ways are polygons: the mean of opposite outline vertices lies on the centre line, which
  // overlaps fewer neighbours than the outline itself.
  const cand = [];
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 80))) {
    const o = ring[(i + (n >> 1)) % n];
    cand.push(L.latLng((ring[i].lat + o.lat) / 2, (ring[i].lng + o.lng) / 2), ring[i]);
  }
  for (const c of cand) {
    map.setView(c, 18, { animate: false });
    const pt = map.latLngToLayerPoint(c).round();
    const q = map.latLngToContainerPoint(c);
    const b = map.getContainer().getBoundingClientRect();
    let hit = null;
    if (target._renderer._drawFirst !== undefined) {  // canvas: topmost interactive layer containing the point
      for (let o = target._renderer._drawFirst; o; o = o.next) {
        if (o.layer.options.interactive && o.layer._containsPoint(pt)) hit = o.layer;
      }
    } else {  // svg: the path element under the pointer
      const el = document.elementFromPoint(b.left + q.x, b.top + q.y);
      if (el === target._path) hit = target;
    }
    if (hit === target) return { x: b.left + q.x, y: b.top + q.y };
  }
  return null;
}
"""


# ── aggregation / reporting ──────────────────────────────────────────────────
def summarize(reps: list[dict]) -> dict:
    keys: dict[str, list] = {}
    for rep in reps:
        for k, v in rep.items():
            keys.setdefault(k, []).extend(x for x in v if x is not None)
    out = {}
    for k, v in keys.items():
        if not v:
            continue
        out[k] = {
            "median": statistics.median(v),
            "min": min(v),
            "max": max(v),
            "spread": (max(v) - min(v)) / 2,
            "n": len(v),
        }
    return out


def fmt(s):
    return f"{s['median']:9.1f} ±{s['spread']:7.1f}  [{s['min']:.1f}–{s['max']:.1f}] n={s['n']}"


def report(title, summ, compare=None):
    print(f"\n== {title}")
    for k, s in summ.items():
        line = f"  {k:28s} {fmt(s)}"
        if compare and k in compare:
            b = compare[k]
            d = s["median"] - b["median"]
            beyond = abs(d) > max(s["spread"], b["spread"])
            line += (
                f"   Δ{d:+.1f} ({d / b['median'] * 100:+.0f}%) {'BEYOND spread' if beyond else 'within spread'}"
                if b["median"]
                else ""
            )
        print(line)


def top_functions(profile, n=10):
    nodes = {x["id"]: x for x in profile["nodes"]}
    self_us: dict[tuple, float] = {}
    for nid, dt in zip(profile["samples"], profile["timeDeltas"]):
        cf = nodes[nid]["callFrame"]
        key = (cf["functionName"] or "(anon)", cf["url"].rsplit("/", 1)[-1], cf["lineNumber"] + 1)
        self_us[key] = self_us.get(key, 0) + dt
    tot = sum(self_us.values()) or 1
    return [
        (k, v / 1000, v / tot * 100) for k, v in sorted(self_us.items(), key=lambda kv: -kv[1])[:n]
    ]


# ── functional flows (guard against behaviour change) ───────────────────────
def run_flows(browser, base, mapname, patches) -> list[str]:
    r = Rep(browser, base, patches, False)
    p = r.page
    fails = []

    def check(name, cond):
        ok = bool(cond)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            fails.append(name)

    p.goto(base + "/", wait_until="load")
    p.wait_for_function("document.getElementById('file-select').options.length > 1")
    p.select_option("#file-select", mapname)
    p.click("#load-btn")
    p.wait_for_function("document.getElementById('status').textContent.startsWith('Loaded')")
    info = r.ev(_PICK_JS)
    way, pt = r.pick(info["mids"])

    pt = r.hit(way)
    r.settle(500)
    p.mouse.click(pt["x"], pt["y"])
    check("select way by click", r.ev("!!currentClickedFeature && !!currentClickedLayer"))
    sel_id = r.ev("String(currentClickedFeature.properties.id)")
    check("props panel shows way", sel_id in p.inner_text("#props-content"))

    # search narrows and restores
    total = r.ev(
        "['road','footway','barrier'].reduce((n, c) => n + geoLayers[c].getLayers().length, 0)"
    )
    p.fill("#feature-search", "zzzzzz-nothing")
    check(
        "search hides non-matches",
        r.ev(
            "['road','footway','barrier'].reduce((n, c) => n + geoLayers[c].getLayers().length, 0)"
        )
        == 0,
    )
    p.fill("#feature-search", "")
    check(
        "search clear restores",
        r.ev(
            "['road','footway','barrier'].reduce((n, c) => n + geoLayers[c].getLayers().length, 0)"
        )
        == total,
    )
    p.fill("#feature-search", info["query"])
    n_q = r.ev(
        "['road','footway','barrier'].reduce((n, c) => n + geoLayers[c].getLayers().length, 0)"
    )
    check("search keeps matches only", 0 < n_q <= total)
    p.fill("#feature-search", "")

    # layer toggle
    lay = info["toggle"]
    p.click(f'input[data-layer="{lay}"]')
    check("layer toggle off", r.ev(f"!map.hasLayer(geoLayers['{lay}'])"))
    p.click(f'input[data-layer="{lay}"]')
    check("layer toggle on", r.ev(f"map.hasLayer(geoLayers['{lay}'])"))

    # hide / show the selected way
    pt = r.hit(way)
    p.wait_for_timeout(300)
    p.mouse.click(pt["x"], pt["y"])
    p.keyboard.press("h")
    p.wait_for_function(f"hiddenWayIds.has({json.dumps(way['id'])})", timeout=10000)
    check(
        "hide way",
        r.ev(
            f"!geoLayers['{way['cat']}'].getLayers().some(l => l._featureId === {json.dumps(way['id'])})"
        ),
    )
    r.ev(f"showWay({json.dumps(way['id'])})")
    p.wait_for_function(f"!hiddenWayIds.has({json.dumps(way['id'])})", timeout=10000)
    p.wait_for_timeout(500)
    check(
        "show way",
        r.ev(
            f"geoLayers['{way['cat']}'].getLayers().some(l => l._featureId === {json.dumps(way['id'])})"
        ),
    )

    # edit mode: select way, drag a node, undo
    p.click('.mode-btn[data-mode="edit"]')
    longway, _ = r.pick(info["longs"])
    pt = r.hit(longway)
    r.settle(500)
    p.mouse.click(pt["x"], pt["y"])
    p.wait_for_function(
        "currentNodes.length > 0 && nodeMarkers.length === currentNodes.length", timeout=15000
    )
    check(
        "edit: node handles shown",
        r.ev(
            "nodeMarkers.length === currentNodes.length && midpointMarkers.length === currentNodes.length - 1"
        ),
    )
    mid = r.ev("currentNodes.length >> 1")
    before = r.ev(f"[currentNodes[{mid}].lat, currentNodes[{mid}].lon]")
    r.ev(
        f"(() => {{ const n = currentNodes[{mid}]; map.setView([n.lat, n.lon], 18, {{animate: false}}); }})()"
    )
    p.wait_for_timeout(300)
    npt = r.ev(
        f"(() => {{ const n = currentNodes[{mid}]; const q = map.latLngToContainerPoint([n.lat, n.lon]); const r = map.getContainer().getBoundingClientRect(); return {{x: r.left + q.x, y: r.top + q.y}}; }})()"
    )
    p.mouse.move(npt["x"], npt["y"])
    p.mouse.down()
    for i in range(1, 11):
        p.mouse.move(npt["x"] + i * 3, npt["y"] + i * 2)
    p.mouse.up()
    p.wait_for_function(
        "document.getElementById('status').textContent === 'Way updated' && currentNodes.length > 0",
        timeout=15000,
    )
    after = r.ev(f"[currentNodes[{mid}].lat, currentNodes[{mid}].lon]")
    check("edit: node drag moves node + logs change", after != before)
    p.mouse.move(5, 5)
    p.keyboard.press("Control+z")
    p.wait_for_function(
        "document.getElementById('status').textContent.startsWith('Undid')", timeout=20000
    )
    check("undo reverts move", r.ev("!changeLog.some(c => c.type === 'move')"))
    p.wait_for_timeout(300)

    # split at an inner node of an editable way, then undo the split
    p.click('.mode-btn[data-mode="view"]')
    pt = r.hit(longway)
    r.settle(500)
    p.mouse.click(pt["x"], pt["y"])
    p.keyboard.press("n")
    try:
        p.wait_for_function("nodeLayer && currentNodes.length > 2", timeout=10000)
        ok = r.ev("(() => { const i = 1; return _canSplit(i); })()")
        if ok:
            n_split = r.ev("changeLog.filter(c => c.type === 'split').length")
            r.ev("splitCurrentWay(currentClickedFeature.properties.id, currentNodes[1].id)")
            p.wait_for_function(
                f"changeLog.filter(c => c.type === 'split').length > {n_split}", timeout=15000
            )
            check("split way", True)
            p.keyboard.press("Control+z")
            p.wait_for_function(
                f"changeLog.filter(c => c.type === 'split').length === {n_split}", timeout=20000
            )
            check("undo split", True)
        else:
            print("  skip split (selected way not splittable)")
    except Exception as e:
        check(f"split flow ({type(e).__name__})", False)

    # annotation edit: add, select, edit-mode drag, modal save, delete
    r.ev("""(() => { const c = map.getCenter(); const d = 0.0003;
      window.__annTmp = {type: 'Polygon', coordinates: [[[c.lng - d, c.lat - d], [c.lng + d, c.lat - d], [c.lng + d, c.lat + d], [c.lng - d, c.lat + d], [c.lng - d, c.lat - d]]]};
      return createAnnotationApi(currentFile, 'obstacle', window.__annTmp, {barrier: 'wall'}).then(a => { annotations.push(a); annBaselineGeoms[a.id] = structuredClone(a.geometry); addAnnotationToLayer(a); renderAnnotationList(); window.__annId = a.id; }); })()""")
    check("annotation added", r.ev("annotations.some(a => a.id === window.__annId)"))
    cpt = r.ev(
        "(() => { const q = map.latLngToContainerPoint(map.getCenter()); const r = map.getContainer().getBoundingClientRect(); return {x: r.left + q.x, y: r.top + q.y}; })()"
    )
    p.mouse.click(cpt["x"], cpt["y"])
    check(
        "select annotation by click",
        r.ev("!!currentClickedLayer && currentClickedLayer.options._ann_id === window.__annId"),
    )
    p.click('.mode-btn[data-mode="edit"]')
    old_geom = r.ev("JSON.stringify(annotations.find(a => a.id === window.__annId).geometry)")
    p.mouse.move(cpt["x"], cpt["y"])
    p.mouse.down()
    for i in range(1, 9):
        p.mouse.move(cpt["x"] + i * 4, cpt["y"] + i * 2)
    p.mouse.up()
    p.wait_for_timeout(500)
    p.click('.mode-btn[data-mode="view"]')  # commits the drag
    p.wait_for_function(
        f"JSON.stringify(annotations.find(a => a.id === window.__annId).geometry) !== {json.dumps(old_geom)}",
        timeout=10000,
    )
    check("annotation drag saved", True)
    r.ev("openAnnEditModal(window.__annId)")
    p.wait_for_selector("#ann-detail-modal.show")
    p.select_option("#ann-barrier-sel", "fence")
    p.click("#ann-detail-save")
    p.wait_for_function(
        "annotations.find(a => a.id === window.__annId).properties.barrier === 'fence'",
        timeout=10000,
    )
    check("annotation properties edit", True)
    r.ev("removeAnnotationById(window.__annId)")
    p.wait_for_function("!annotations.some(a => a.id === window.__annId)", timeout=10000)
    check("annotation removed", True)

    check("no JS errors", not r.errors)
    for e in r.errors:
        print("   ", e)
    r.ctx.close()
    return fails


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--map", action="append", help="mapdata file name (repeatable); default demo map"
    )
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="uncounted leading reps (default 1)")
    ap.add_argument("--quick", action="store_true", help="demo map, 3 reps, then the flow checks")
    ap.add_argument(
        "--src", type=Path, default=DEFAULT_SRC, help="dir with big maps (falls back to demo/)"
    )
    ap.add_argument("--port", type=int, default=0, help="viewer port (default: a free one)")
    ap.add_argument("--out", type=Path, help="write summary JSON here")
    ap.add_argument("--compare", type=Path, help="summary JSON from an earlier run")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="1 rep with the V8 sampler; print top self-time per step",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="1 rep with Chrome tracing; print main-thread ms per bucket (script/gc/style/layout/paint) per step",
    )
    ap.add_argument("--flows", action="store_true", help="only run the behaviour checks")
    ap.add_argument(
        "--svg", action="store_true", help="experiment: SVG renderer instead of preferCanvas"
    )
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    maps = args.map or ["buchlovice.mapdata"]
    if args.quick:
        maps, args.reps = ["buchlovice.mapdata"], 2
    patches = {"map_setup.js": ("preferCanvas: true", "preferCanvas: false")} if args.svg else {}
    if args.profile or args.trace:
        args.reps, args.warmup = 1, 0

    tmp = Path(tempfile.mkdtemp(prefix="mdbench-"))
    data = tmp / "data"
    prepare_data(maps, args.src, data)
    if not args.port:
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            args.port = sk.getsockname()[1]
    proc = start_server(data, args.port)
    base = f"http://127.0.0.1:{args.port}"
    rc = 0
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            if not args.flows:
                compare = json.loads(args.compare.read_text()) if args.compare else {}
                all_summ = {}
                for name in maps:
                    urllib.request.urlopen(
                        f"{base}/api/mapdata?file={name}"
                    ).read()  # warm server cache
                    reps = []
                    for i in range(args.warmup + args.reps):
                        reset_annotations([name], args.src, data)
                        rep = run_rep(browser, base, name, patches, args)
                        if rep.errors:
                            rc = 1
                            print(f"JS errors in rep {i}: {rep.errors[:3]}", file=sys.stderr)
                        if (
                            i >= args.warmup
                        ):  # first rep(s) run cold (JIT, server caches): not counted
                            reps.append(rep.metrics)
                        if args.profile:
                            for step, prof in rep.profiles.items():
                                print(f"\n-- profile: {step}")
                                for (fn, f, line), ms, pct in top_functions(prof):
                                    print(f"   {ms:8.1f} ms {pct:5.1f}%  {fn} ({f}:{line})")
                        for step, bk in rep.traces.items():
                            print(
                                f"  trace {step:22s} "
                                + "  ".join(
                                    f"{k}={v:.0f}"
                                    for k, v in sorted(bk.items(), key=lambda kv: -kv[1])
                                )
                            )
                        rep.ctx.close()
                        print(
                            f"  rep {i + 1}/{args.warmup + args.reps} done ({name})",
                            file=sys.stderr,
                        )
                    summ = summarize(reps)
                    all_summ[name] = summ
                    report(
                        f"{name} ({'svg' if args.svg else 'canvas'}, {args.reps} reps)",
                        summ,
                        compare.get(name),
                    )
                if args.out:
                    args.out.write_text(json.dumps(all_summ, indent=1))
            if args.flows or args.quick:
                print("\n== flows")
                reset_annotations(maps, args.src, data)
                fails = run_flows(browser, base, maps[0], patches)
                if fails:
                    rc = 1
                    print("FAILED:", fails)
            browser.close()
    finally:
        stop_server(proc)
        shutil.rmtree(tmp, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
