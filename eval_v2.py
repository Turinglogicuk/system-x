"""v0.2 evaluation of the reply option (JevBench / Banking77 are in bench.py).

Sets:
  heldout   300 Bitext reply-task rows whose customer messages were never trained on
  handwritten  our 97 hand-written cases (now out of distribution: different wording, options and domain details)
Reports decision accuracy / ECE / Brier, reply quality (ROUGE-L F1 vs Bitext's reply, length, finished cleanly),
and latency of the one-model path vs a two-call pipeline that re-reads the prompt to write the reply.

Run:  uv run eval_v2.py --adapter out/v2           (add --name base and no adapter for the untrained model)
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from bench import ece
from tl_decision_plus.device import sync
from tl_decision_plus.engine import DecisionEngine
from tl_decision_plus.prompt import REPLY_NAME, option_name


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    l = lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def run(eng, rows, threshold, max_tokens, pipeline_n):
    out, lat_one, lat_two = [], [], []
    for n, r in enumerate(rows):
        res = eng.decide(r["state"], r["options"], threshold=threshold, max_reply_tokens=max_tokens)
        names = [option_name(o) for o in r["options"]]
        probs = [res["probabilities"][x] for x in names]
        gold = r["label"]
        item = {"src": r["src"], "kind": r["kind"], "gold": names[gold], "pred": res["decision"],
                "argmax": names[max(range(len(probs)), key=probs.__getitem__)], "probs": probs, "gold_idx": gold,
                "text": res["text"], "ref": r.get("reply"), "mode": res["mode"], "total_ms": res["total_ms"],
                "decide_ms": res["decide_ms"], "reply_tokens": res["reply_tokens"]}
        if res["text"] is not None and REPLY_NAME in names and n < pipeline_n:
            # same reply, but written by a second call that re-reads the whole prompt (a two-model pipeline)
            sync(eng.device); t0 = time.perf_counter()
            eng.decide(r["state"], r["options"], threshold=0.0, max_reply_tokens=0)
            eng.reference_reply(r["state"], r["options"], max_reply_tokens=max_tokens)
            sync(eng.device)
            lat_two.append((time.perf_counter() - t0) * 1000)
            lat_one.append(res["total_ms"])
        out.append(item)
    return out, lat_one, lat_two


def summarize(items):
    n = len(items)
    acc_arg = statistics.mean(i["argmax"] == i["gold"] for i in items)
    acc_sys = statistics.mean(i["pred"] == i["gold"] for i in items)
    confs = [max(i["probs"]) for i in items]
    corr = [i["argmax"] == i["gold"] for i in items]
    brier = statistics.mean(sum((p - (k == i["gold_idx"])) ** 2 for k, p in enumerate(i["probs"])) for i in items)
    by_kind = {}
    for i in items:
        by_kind.setdefault(i["kind"], []).append(i["pred"] == i["gold"])
    s = {"n": n, "accuracy_argmax": round(100 * acc_arg, 1), "accuracy_with_fallback": round(100 * acc_sys, 1),
         "ece": round(ece(confs, corr), 3), "brier": round(brier, 3),
         "accuracy_by_kind": {k: f"{100 * statistics.mean(v):.1f} ({sum(v)}/{len(v)})" for k, v in sorted(by_kind.items())},
         "reply_share": round(100 * statistics.mean(i["text"] is not None for i in items), 1)}
    scored = [i for i in items if i["text"] and i["ref"] and i["gold"] == REPLY_NAME]
    if scored:
        s["reply_rouge_l"] = round(statistics.mean(rouge_l(i["text"], i["ref"]) for i in scored), 3)
        s["reply_words_median"] = statistics.median(len(i["text"].split()) for i in scored)
    replied = [i for i in items if i["text"] is not None]
    if replied:
        s["reply_finished_cleanly"] = round(100 * statistics.mean(i["reply_tokens"] < 48 for i in replied), 1)
    s["decide_ms_p50"] = round(statistics.median(i["decide_ms"] for i in items), 1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=None, help="base model (default: the adapter's base, else Qwen3-0.6B)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--pipeline-n", type=int, default=60, help="rows on which to time the two-call pipeline")
    ap.add_argument("--sets", nargs="+", default=["heldout=data/v2/heldout_reply.jsonl", "handwritten=data/v2/handwritten.jsonl"],
                    help="name=path pairs; v0.3 adds abcd=data/v3/heldout_abcd.jsonl verticals=... cfpb=...")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    args = ap.parse_args()
    name = args.name or (Path(args.adapter).name if args.adapter else "base")

    eng = DecisionEngine(model_name=args.base, adapter_path=args.adapter)
    eng.decide({"customer_msg": "hi"}, ["A", REPLY_NAME], max_reply_tokens=2)  # warm-up
    report = {"model": name, "threshold": args.threshold}
    for set_name, path in (x.split("=", 1) for x in args.sets):
        rows = [json.loads(l) for l in open(path)]
        items, one, two = run(eng, rows, args.threshold, 48, args.pipeline_n if set_name == "heldout" else 0)
        report[set_name] = summarize(items)
        if one:
            report[set_name]["latency_reply_one_model_ms_p50"] = round(statistics.median(one), 1)
            report[set_name]["latency_reply_two_calls_ms_p50"] = round(statistics.median(two), 1)
        report[f"{set_name}_items"] = items
    Path("out").mkdir(exist_ok=True)
    Path(f"out/eval_v2_{name}{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_items")}, indent=2))


if __name__ == "__main__":
    main()
