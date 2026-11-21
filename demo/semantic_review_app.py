#!/usr/bin/env python3
"""Interactive local UI to label Alexandria semantic ``maybe`` pairs.

Plain HTML over ``http.server`` (no Gradio). Shows query vs candidate crops;
Yes / No / Skip writes ``human`` into ``datasets/alexandria/semantic_pairs.csv``.

Usage::

    python demo/semantic_review_app.py
    # open http://127.0.0.1:7861
"""

from __future__ import annotations

import base64
import csv
import io
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.retrieval.datasets import load_splits  # noqa: E402
from shawaf_vlm.semantic_labels import parse_attributes, relevance  # noqa: E402
from shawaf_vlm.semantic_review import unknown_attrs  # noqa: E402

PAIRS = ROOT / "datasets" / "alexandria" / "semantic_pairs.csv"
ACTIONS = ROOT / "datasets" / "alexandria" / "actions.csv"
QUEUE = ROOT / "datasets" / "alexandria" / "maybe_queue.csv"
FIELDS = ("query", "candidate", "auto", "score", "human", "query_caption", "candidate_caption")
HOST, PORT = "127.0.0.1", 7861


class Store:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []
        self.queue: list[int] = []
        self.pos = 0
        self.history: list[tuple[int, str]] = []
        self.crops: dict[str, list[Path]] = {}
        self.caps: dict[str, dict[str, str]] = {}
        self.mode = "open"
        self.reload()

    def reload(self, mode: str | None = None) -> None:
        if mode:
            self.mode = mode
        with ACTIONS.open(encoding="utf-8", newline="") as handle:
            self.caps = {r["track_id"]: r for r in csv.DictReader(handle)}
        with PAIRS.open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        splits = load_splits("alexandria:all", download=False)
        self.crops = {g.track_id: list(g.crop_paths) for g in splits.gallery}
        self.queue = []
        for i, row in enumerate(self.rows):
            if row.get("auto") != "maybe":
                continue
            human = (row.get("human") or "").strip().lower()
            if self.mode == "open" and human:
                continue
            if self.mode == "yes" and human != "yes":
                continue
            if self.mode == "no" and human != "no":
                continue
            self.queue.append(i)
        self.pos = 0
        self.history.clear()
        self._write_queue()

    def save(self) -> None:
        with PAIRS.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        self._write_queue()

    def _write_queue(self) -> None:
        remaining = []
        for row in self.rows:
            if row.get("auto") != "maybe" or (row.get("human") or "").strip():
                continue
            q = parse_attributes(
                self.caps[row["query"]]["caption_fixed"],
                self.caps[row["query"]]["action_class"],
            )
            g = parse_attributes(
                self.caps[row["candidate"]]["caption_fixed"],
                self.caps[row["candidate"]]["action_class"],
            )
            remaining.append({**row, "unknown": "|".join(unknown_attrs(q, g))})
        with QUEUE.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FIELDS) + ["unknown"])
            writer.writeheader()
            writer.writerows(remaining)

    def current(self) -> dict | None:
        if self.pos >= len(self.queue):
            return None
        return self.payload(self.queue[self.pos])

    def payload(self, idx: int) -> dict:
        row = self.rows[idx]
        q = parse_attributes(
            self.caps[row["query"]]["caption_fixed"],
            self.caps[row["query"]]["action_class"],
        )
        g = parse_attributes(
            self.caps[row["candidate"]]["caption_fixed"],
            self.caps[row["candidate"]]["action_class"],
        )
        _, score = relevance(q, g)
        open_n = sum(
            1 for r in self.rows if r.get("auto") == "maybe" and not (r.get("human") or "").strip()
        )
        return {
            "pos": self.pos + 1,
            "queue_len": len(self.queue),
            "open_overall": open_n,
            "query": row["query"],
            "candidate": row["candidate"],
            "query_caption": row["query_caption"],
            "candidate_caption": row["candidate_caption"],
            "score": row["score"],
            "recomputed": f"{score:.2f}",
            "human": row.get("human") or "",
            "unknown": unknown_attrs(q, g),
            "query_images": _data_uris(self.crops.get(row["query"], [])),
            "candidate_images": _data_uris(self.crops.get(row["candidate"], [])),
            "mode": self.mode,
        }

    def decide(self, label: str) -> dict | None:
        if self.pos >= len(self.queue):
            return None
        idx = self.queue[self.pos]
        self.history.append((idx, self.rows[idx].get("human") or ""))
        self.rows[idx]["human"] = label
        self.save()
        self.pos += 1
        return self.current()

    def skip(self) -> dict | None:
        if self.pos < len(self.queue):
            self.pos += 1
        return self.current()

    def undo(self) -> dict | None:
        if not self.history:
            return self.current()
        idx, prev = self.history.pop()
        self.rows[idx]["human"] = prev
        self.save()
        if idx in self.queue:
            self.pos = self.queue.index(idx)
        return self.current()

    def clear_maybe_human(self) -> dict | None:
        for row in self.rows:
            if row.get("auto") == "maybe":
                row["human"] = ""
        self.save()
        self.reload("open")
        return self.current()


def _data_uris(paths: list[Path], n: int = 3) -> list[str]:
    if not paths:
        return []
    idxs = sorted({0, len(paths) // 2, len(paths) - 1})[:n]
    out = []
    for i in idxs:
        try:
            img = Image.open(paths[i]).convert("RGB")
        except OSError:
            continue
        img.thumbnail((160, 320))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


STORE = Store()


def page_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alexandria maybe review</title>
<style>
body { margin:0; font: 15px/1.45 system-ui,Segoe UI,sans-serif; background:#f4f6f8; color:#12181f; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 1.4rem; margin: 0 0 8px; }
.sub { color:#5b6570; margin: 0 0 16px; }
.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom: 14px; }
select, button { font: inherit; padding: 10px 14px; border-radius: 8px; border: 1px solid #c9d0d6; background:#fff; cursor:pointer; }
button.yes { background:#0f7a6c; color:#fff; border-color:#0f7a6c; font-weight:600; }
button.no { background:#b42318; color:#fff; border-color:#b42318; font-weight:600; }
.status { background:#fff; border:1px solid #d9dee3; border-radius:10px; padding:12px 14px; margin-bottom:14px; white-space:pre-wrap; }
.grid { display:grid; gap:14px; grid-template-columns: 1fr 1fr; }
@media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
.card { background:#fff; border:1px solid #d9dee3; border-radius:12px; padding:14px; }
.card h2 { margin:0 0 8px; font-size:1rem; }
.meta { color:#5b6570; font-size:.9rem; margin:0 0 8px; word-break:break-all; }
.crops { display:flex; gap:6px; flex-wrap:wrap; }
.crops img { width: 96px; height: 200px; object-fit: contain; background:#111; border-radius:6px; }
.prompt { margin: 14px 0; font-weight: 600; }
.hint { color:#5b6570; font-size:.9rem; }
kbd { background:#eef1f3; border-radius:4px; padding:1px 5px; font-size:.85em; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Alexandria maybe review</h1>
  <p class="sub">Does the candidate also match the query (appearance + action)? Saves to <code>semantic_pairs.csv</code>.</p>
  <div class="bar">
    <label>Queue
      <select id="mode">
        <option value="open">Open maybes only</option>
        <option value="all">All maybes</option>
        <option value="yes">Already human=yes</option>
        <option value="no">Already human=no</option>
      </select>
    </label>
    <button class="yes" id="yes">Yes — match</button>
    <button class="no" id="no">No — not a match</button>
    <button id="skip">Skip</button>
    <button id="undo">Undo</button>
    <button id="clear">Clear maybe human labels</button>
  </div>
  <p class="hint">Keys: <kbd>Y</kbd> yes · <kbd>N</kbd> no · <kbd>S</kbd> skip · <kbd>U</kbd> undo</p>
  <div class="status" id="status">Loading…</div>
  <p class="prompt" id="prompt"></p>
  <div class="grid">
    <div class="card">
      <h2>Query (labelled person)</h2>
      <p class="meta" id="qid"></p>
      <p id="qcap"></p>
      <div class="crops" id="qimgs"></div>
    </div>
    <div class="card">
      <h2>Candidate</h2>
      <p class="meta" id="cid"></p>
      <p id="ccap"></p>
      <div class="crops" id="cimgs"></div>
    </div>
  </div>
</div>
<script>
async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  };
  const r = await fetch(path, opts);
  return r.json();
}
function imgs(el, list) {
  el.innerHTML = (list || []).map(function (src) {
    return '<img src="' + src + '" alt="">';
  }).join('') || '<span class="meta">(no crops)</span>';
}
function render(d) {
  if (!d || d.done) {
    document.getElementById('status').textContent = (d && d.message) || 'Queue finished.';
    document.getElementById('prompt').textContent = '';
    imgs(document.getElementById('qimgs'), []);
    imgs(document.getElementById('cimgs'), []);
    ['qid','qcap','cid','ccap'].forEach(function (id) {
      document.getElementById(id).textContent = '';
    });
    return;
  }
  var unk = (d.unknown && d.unknown.length) ? d.unknown.join(', ') : '(none)';
  document.getElementById('status').textContent =
    'Item ' + d.pos + '/' + d.queue_len + ' · open overall ' + d.open_overall + '\\n' +
    'score ' + d.score + ' (recomputed ' + d.recomputed + ') · human=' + (d.human || '(empty)') + '\\n' +
    'unknown: ' + unk;
  document.getElementById('prompt').textContent = 'Is the candidate also a valid match for the query caption?';
  document.getElementById('qid').textContent = d.query;
  document.getElementById('qcap').textContent = d.query_caption;
  document.getElementById('cid').textContent = d.candidate;
  document.getElementById('ccap').textContent = d.candidate_caption;
  imgs(document.getElementById('qimgs'), d.query_images);
  imgs(document.getElementById('cimgs'), d.candidate_images);
  document.getElementById('mode').value = d.mode;
}
async function refresh() { render(await api('/api/current')); }
document.getElementById('yes').onclick = async function () { render(await api('/api/decide', {label: 'yes'})); };
document.getElementById('no').onclick = async function () { render(await api('/api/decide', {label: 'no'})); };
document.getElementById('skip').onclick = async function () { render(await api('/api/skip', {})); };
document.getElementById('undo').onclick = async function () { render(await api('/api/undo', {})); };
document.getElementById('clear').onclick = async function () {
  if (!confirm('Clear ALL human labels on maybe rows (including soft auto-labels)?')) return;
  render(await api('/api/clear', {}));
};
document.getElementById('mode').onchange = async function (e) {
  render(await api('/api/mode', {mode: e.target.value}));
};
window.addEventListener('keydown', function (e) {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  var k = e.key.toLowerCase();
  if (k === 'y') document.getElementById('yes').click();
  if (k === 'n') document.getElementById('no').click();
  if (k === 's') document.getElementById('skip').click();
  if (k === 'u') document.getElementById('undo').click();
});
refresh();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[review] {args[0] if args else fmt}", flush=True)

    def _send(self, payload: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(json.dumps(obj).encode("utf-8"), "application/json", code)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(page_html().encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/current":
            cur = STORE.current()
            self._json(cur or {"done": True, "message": "Queue finished. Switch mode or clear labels."})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        body = self._read_json()
        if path == "/api/decide":
            cur = STORE.decide(str(body.get("label") or ""))
        elif path == "/api/skip":
            cur = STORE.skip()
        elif path == "/api/undo":
            cur = STORE.undo()
        elif path == "/api/clear":
            cur = STORE.clear_maybe_human()
        elif path == "/api/mode":
            STORE.reload(str(body.get("mode") or "open"))
            cur = STORE.current()
        else:
            self.send_error(404)
            return
        self._json(cur or {"done": True, "message": "Queue finished."})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}", flush=True)
    print(f"Labels: {PAIRS}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
