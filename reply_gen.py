"""Replies from several systems on the same rows, for the judge and the human rating page.

Rows: 150 held-out rows whose gold answer is a reply: 50 old Bitext, 50 Bitext verticals, 50 ABCD
(data/v3/judge_rows.jsonl, fixed seed).

Systems:
  base, v2, v3     our engine, forced to reply (threshold above 1, so it always writes from the decision cache)
  two_call_<size>  a separate Qwen3 model writing the reply in a second call that reads the State again,
                   with a plain prompt (no options): the "decision model + reply LLM" pipeline we compare against
  reference        the dataset's own reply (Bitext target sentences, or ABCD's agent turn)

Run: uv run reply_gen.py --system v3 --adapter out/v3
     uv run reply_gen.py --system two_call_1.7b --model Qwen/Qwen3-1.7B
     uv run reply_gen.py --system reference
"""
import argparse
import json
import random
from pathlib import Path

import torch

ROWS = Path("data/v3/judge_rows.jsonl")
PLAIN = ("You are a customer support agent. Read the State and write a 1-2 sentence reply to the customer. "
         "Use facts from the State when they help; do not invent details.")


def judge_rows():
    if ROWS.exists():
        return [json.loads(l) for l in open(ROWS)]
    rng = random.Random(20261015)
    out = []
    for src, path in [("bitext", "data/v2/heldout_reply.jsonl"), ("verticals", "data/v3/heldout_verticals.jsonl"),
                      ("abcd", "data/v3/heldout_abcd.jsonl")]:
        rows = [json.loads(l) for l in open(path)]
        gold_reply = [dict(r, set=src, idx=i) for i, r in enumerate(rows)
                      if r["options"][r["label"]]["name"] == "REPLY_WITH_TEXT" and r.get("reply")]
        out += rng.sample(gold_reply, 50)
    with open(ROWS, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


@torch.no_grad()
def plain_replies(model_name, rows, max_new_tokens=64):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tl_decision_plus.device import pick_device
    from tl_decision_plus.prompt import render_state

    dev = pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(dev).eval()
    out = []
    for r in rows:
        msgs = [{"role": "system", "content": PLAIN}, {"role": "user", "content": f"State: {render_state(r['state'])}"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
        g = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        out.append(tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=None, help="base model (default: the adapter's base, else Qwen3-0.6B)")
    ap.add_argument("--model", default=None, help="a plain Qwen3 model for a two-call baseline")
    args = ap.parse_args()
    rows = judge_rows()
    if args.system == "reference":
        texts = [r["reply"] for r in rows]
    elif args.model:
        texts = plain_replies(args.model, rows)
    else:
        from tl_decision_plus.engine import DecisionEngine

        eng = DecisionEngine(model_name=args.base, adapter_path=args.adapter)
        texts = [eng.decide(r["state"], r["options"], threshold=1.01, max_reply_tokens=48)["text"] for r in rows]
    Path("out/replies").mkdir(parents=True, exist_ok=True)
    with open(f"out/replies/{args.system}.jsonl", "w") as f:
        for r, t in zip(rows, texts):
            f.write(json.dumps({"set": r["set"], "idx": r["idx"], "text": t}, ensure_ascii=False) + "\n")
    print(f"{args.system}: {len(texts)} replies → out/replies/{args.system}.jsonl")


if __name__ == "__main__":
    main()
