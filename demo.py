"""Three cases: clear decision, explicit reply, ambiguous input. Run: uv run demo.py [--adapter out/lora]"""
import argparse
import json

import torch

from tl_decision_plus.engine import DecisionEngine
from tl_decision_plus.options import make_options

OPTS = make_options(["AUTO_REFUND", "BLOCK_FRAUD", "ESCALATE_HUMAN", "REPLY_WITH_TEXT"])

CASES = [
    ("A: clear decision",
     {"customer_msg": "Someone in another country just logged into my account and bought three gift cards. That wasn't me!",
      "risk_score": 0.96}),
    ("B: needs a short reply",
     {"customer_msg": "Why is my bill $14 this month instead of $10?", "invoice_note": "$4 add-on: extra storage"}),
    ("C: ambiguous",
     {"customer_msg": "something is wrong with my order"}),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--fp32", action="store_true", help="float32: KV-cache reply should match generate() exactly")
    args = ap.parse_args()

    eng = DecisionEngine(adapter_path=args.adapter, dtype=torch.float32 if args.fp32 else torch.bfloat16)
    print(f"device={eng.device}")
    eng.decide(*CASES[0][1:], OPTS)  # warm-up (first mps call compiles kernels)

    for name, state in CASES:
        r = eng.decide(state, OPTS, threshold=args.threshold)
        print(f"\n--- {name} ---")
        print(json.dumps(r, indent=2))
        if r["text"] is not None:
            ref = eng.reference_reply(state, OPTS)
            same = "same" if ref == r["text"] else f"DIFFERS (float rounding?): {ref!r}"
            print(f"KV-cache reply vs full re-read generate(): {same}")
