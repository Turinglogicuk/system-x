"""Controlled latency test: one model (reply continues from the decision's KV cache) vs two calls
(the reply call reads the whole prompt again). Both write exactly the same number of reply tokens with
the same generate() loop, so the only difference is the second prefill. Prompt length is varied.

Run: uv run latency.py --adapter out/v2      → out/latency.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from tl_decision_plus.device import sync
from tl_decision_plus.engine import DecisionEngine
from tl_decision_plus.options import make_options

HISTORY = ("Order {n}: placed on day {n}, shipped by courier, delivered to the front door. "
           "Customer note: please call before delivery. Payment by card ending 4{n:03d}. ")


def state_with_tokens(tok, target):
    s = {"customer_msg": "Can you explain the extra fee on my last bill?", "account_history": ""}
    n = 0
    while len(tok.encode(json.dumps(s))) < target:
        n += 1
        s["account_history"] += HISTORY.format(n=n)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=None, help="base model (default: the adapter's base, else Qwen3-0.6B)")
    ap.add_argument("--tokens", type=int, default=32, help="reply length forced in both paths")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    eng = DecisionEngine(model_name=args.base, adapter_path=args.adapter)
    opts = make_options(["AUTO_REFUND", "TRACK_ORDER", "ESCALATE_HUMAN", "REPLY_WITH_TEXT"])
    t = args.tokens
    eng.decide({"customer_msg": "warm-up"}, opts, threshold=1.01, max_reply_tokens=4)
    rows = []
    for target in [200, 500, 1000, 2000, 4000]:
        st = state_with_tokens(eng.tokenizer, target)
        one, two, dec = [], [], []
        for _ in range(args.repeats):
            r = eng.decide(st, opts, threshold=1.01, max_reply_tokens=t, min_reply_tokens=t)  # always replies
            assert r["reply_tokens"] == t, r["reply_tokens"]
            one.append(r["total_ms"]); dec.append(r["decide_ms"])
            sync(eng.device); t0 = time.perf_counter()
            eng.decide(st, opts, threshold=1.01, max_reply_tokens=0)  # decision call
            eng.reference_reply(st, opts, max_reply_tokens=t, min_new_tokens=t)  # reply call re-reads the prompt
            sync(eng.device); two.append((time.perf_counter() - t0) * 1000)
        row = {"prompt_tokens": r["prompt_tokens"], "decide_ms": round(statistics.median(dec), 1),
               "one_model_ms": round(statistics.median(one), 1), "two_calls_ms": round(statistics.median(two), 1)}
        row["saved_ms"] = round(row["two_calls_ms"] - row["one_model_ms"], 1)
        row["saved_pct"] = round(100 * row["saved_ms"] / row["two_calls_ms"], 1)
        rows.append(row)
        print(row, flush=True)
    out = "out/latency.json" if eng.model_name.endswith("0.6B") else f"out/latency_{eng.model_name.split('/')[-1]}.json"
    Path(out).write_text(json.dumps({"reply_tokens": t, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
