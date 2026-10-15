"""Blind human rating of replies: 100 A/B pairs, v3 against the untrained model, v2, a two-call Qwen3-1.7B and the
dataset's reference (25 each). The page never sees which system wrote which reply; the mapping stays in
data/v3/human_items.json. Ratings are appended to out/human_ratings.jsonl as you go, so you can stop and resume.

Run:  uv run rate_server.py   → http://127.0.0.1:8778
"""
import argparse
import errno
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
ITEMS = ROOT / "data/v3/human_items.json"
RATINGS = ROOT / "out/human_ratings.jsonl"
OTHERS = ["base", "v2", "two_call_1.7b", "reference"]


def build_items(per_system=25, seed=20261015):
    rows = [json.loads(l) for l in open(ROOT / "data/v3/judge_rows.jsonl")]
    rep = {s: [json.loads(l)["text"] or "" for l in open(ROOT / f"out/replies/{s}.jsonl")] for s in ["v3"] + OTHERS}
    rng = random.Random(seed)
    items = []
    for s in OTHERS:
        for i in rng.sample(range(len(rows)), per_system):
            v3_first = rng.random() < 0.5
            a, b = ("v3", s) if v3_first else (s, "v3")
            items.append({"row": i, "set": rows[i]["set"], "state": rows[i]["state"], "a_sys": a, "b_sys": b,
                          "a": rep[a][i], "b": rep[b][i]})
    rng.shuffle(items)
    for k, it in enumerate(items):
        it["id"] = k
    ITEMS.write_text(json.dumps(items, indent=1, ensure_ascii=False))
    return items


def load():
    items = json.loads(ITEMS.read_text()) if ITEMS.exists() else build_items()
    done = {}
    if RATINGS.exists():
        for line in open(RATINGS):
            r = json.loads(line)
            done[r["id"]] = r  # the latest rating of an item wins
    return items, done


def handler(items):
    public = [{"id": it["id"], "state": it["state"], "a": it["a"], "b": it["b"]} for it in items]

    class H(BaseHTTPRequestHandler):
        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/rate.html"):
                return self._send(200, (ROOT / "web/rate.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/items":
                _, done = load()
                return self._send(200, {"items": public, "done": done})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/rate":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if body.get("id") not in range(len(items)) or body.get("better") not in ("A", "B", "tie"):
                    raise ValueError("need an item id and better = A, B or tie")
                rec = {"id": body["id"], "better": body["better"], "a_ok": bool(body.get("a_ok")),
                       "b_ok": bool(body.get("b_ok"))}
                RATINGS.parent.mkdir(exist_ok=True)
                with open(RATINGS, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                self._send(200, {"ok": True})
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *a):
            pass

    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8778)
    args = ap.parse_args()
    items, _ = load()
    try:
        server = HTTPServer(("127.0.0.1", args.port), handler(items))
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        sys.exit(f"Port {args.port} is in use; try --port {args.port + 1}")
    print(f"{len(items)} items → http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
