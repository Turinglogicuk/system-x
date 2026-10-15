"""Local web console for TL_decision_plus.

Run:  uv run server.py            → http://127.0.0.1:8777
      uv run server.py --no-base  (only load the trained model)
"""
import argparse
import errno
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from tl_decision_plus.engine import REPLY_NAME, DecisionEngine
from tl_decision_plus.options import CATALOG, make_options

ROOT = Path(__file__).parent
MAX_OPTIONS = 9  # the console offers at most the 8 catalogue buttons; the engine itself has no limit
PRESETS = [
    {"title": "Account hacked", "state": {"customer_msg": "Someone in another country just logged into my account and bought three gift cards. That wasn't me!", "risk_score": 0.96},
     "options": ["AUTO_REFUND", "BLOCK_FRAUD", "ESCALATE_HUMAN", REPLY_NAME]},
    {"title": "Bill question", "state": {"customer_msg": "Why is my bill $14 this month instead of $10?", "invoice_note": "$4 add-on: extra storage"},
     "options": ["AUTO_REFUND", "BLOCK_FRAUD", "ESCALATE_HUMAN", REPLY_NAME]},
    {"title": "Vague", "state": {"customer_msg": "something is wrong with my order"},
     "options": ["AUTO_REFUND", "BLOCK_FRAUD", "ESCALATE_HUMAN", REPLY_NAME]},
    {"title": "Double charge", "state": {"customer_msg": "I was charged twice for my headphones", "order_id": "H-210", "charges": [59.0, 59.0], "order_total": 59.0},
     "options": ["AUTO_REFUND", "TRACK_ORDER", "RESEND_RECEIPT", REPLY_NAME]},
    {"title": "Address, already shipped", "state": {"customer_msg": "Please send my order to my new flat at 9 Elm Close", "order_id": "K-77", "order_status": "shipped"},
     "options": ["UPDATE_ADDRESS", "TRACK_ORDER", REPLY_NAME]},
    {"title": "Address, not shipped", "state": {"customer_msg": "Please send my order to my new flat at 9 Elm Close", "order_id": "K-77", "order_status": "processing"},
     "options": ["UPDATE_ADDRESS", "TRACK_ORDER", REPLY_NAME]},
    {"title": "Right action missing", "state": {"customer_msg": "I need the receipt for order X-19 for my expense report", "order_id": "X-19", "email_on_file": True},
     "options": ["AUTO_REFUND", "TRACK_ORDER", "BLOCK_FRAUD", REPLY_NAME]},
    {"title": "Angry, wants manager", "state": {"customer_msg": "Fifth time contacting you. Put me through to a manager.", "previous_contacts": 5, "sentiment": "very_negative"},
     "options": ["ESCALATE_HUMAN", "AUTO_REFUND", "CANCEL_SUBSCRIPTION", REPLY_NAME]},
]


class Api:
    def __init__(self, load_base: bool, adapter: str):
        self.engines = {}
        if Path(adapter).exists():
            print(f"loading trained model ({adapter}) ...")
            self.engines["lora"] = DecisionEngine(adapter_path=adapter)
        else:
            print(f"no adapter at {adapter}; skipping the trained model")
        if load_base or not self.engines:
            print("loading base model ...")
            self.engines["base"] = DecisionEngine()
        for e in self.engines.values():  # warm-up: first mps call compiles kernels
            e.decide({"customer_msg": "hi"}, make_options(["TRACK_ORDER", REPLY_NAME]), max_reply_tokens=2)
        self.tests = [json.loads(line) for line in open(ROOT / "data/test_handwritten.jsonl")]

    def meta(self):
        eng = next(iter(self.engines.values()))
        return {"device": eng.device, "models": list(self.engines), "catalog": CATALOG,
                "reply_name": REPLY_NAME, "max_options": MAX_OPTIONS, "presets": PRESETS, "tests": self.tests}

    def decide(self, body):
        state, names = body.get("state"), body.get("options")
        if not isinstance(state, dict) or not str(state.get("customer_msg", "")).strip():
            raise ValueError("state must be a JSON object with a non-empty customer_msg")
        if not isinstance(names, list) or REPLY_NAME not in names:
            raise ValueError(f"options must be a list that includes {REPLY_NAME}")
        if not 2 <= len(names) <= MAX_OPTIONS:
            raise ValueError(f"pick 2 to {MAX_OPTIONS} options")
        unknown = [n for n in names if n not in CATALOG]
        if unknown:
            raise ValueError(f"unknown options: {unknown}")
        threshold = float(body.get("threshold", 0.65))
        max_tokens = max(0, min(80, int(body.get("max_reply_tokens", 40))))
        wanted = [m for m in body.get("models", list(self.engines)) if m in self.engines]
        opts = make_options(names)
        return {"results": {m: self.engines[m].decide(state, opts, threshold=threshold, max_reply_tokens=max_tokens)
                            for m in wanted}}


def make_handler(ref: dict):
    """ref['api'] is filled in after the port is bound, so a busy port fails fast."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")  # always show the latest page after edits
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (ROOT / "web/index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/meta":
                self._send(200, ref["api"].meta())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/decide":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self._send(200, ref["api"].decide(body))
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, fmt, *args):
            pass

    return Handler


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--adapter", default="out/v2")
    ap.add_argument("--no-base", action="store_true", help="don't load the untrained model")
    args = ap.parse_args()
    ref = {}
    try:
        # Single-threaded on purpose: one request at a time on the GPU.
        server = HTTPServer(("127.0.0.1", args.port), make_handler(ref))
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        sys.exit(f"Port {args.port} is already in use, so the console is probably already running: "
                 f"open http://127.0.0.1:{args.port}\n"
                 f"To stop the old one: kill $(lsof -ti tcp:{args.port})   or pick another port: --port {args.port + 1}")
    ref["api"] = Api(load_base=not args.no_base, adapter=args.adapter)
    print(f"TL_decision_plus console → http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
