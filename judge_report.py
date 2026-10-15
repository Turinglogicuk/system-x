"""Summarise judge.py output: mean 1-5 scores and send rate per system, and v3's wins/ties/losses against each.

Run: uv run judge_report.py out/judge_gemma4_31b-cloud.json
"""
import json
import math
import statistics
import sys
from collections import defaultdict


def sign_p(w, l):
    n, k = w + l, min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main(path):
    d = json.load(open(path))
    rows = [json.loads(l) for l in open("data/v3/judge_rows.jsonl")]
    rate, pair = defaultdict(list), defaultdict(dict)
    for r in d["results"]:
        if "error" in r["out"]:
            continue
        if r["kind"] == "rate":
            rate[r["system"]].append((rows[r["i"]]["set"], r["out"]))
        else:
            v = str(r["out"].get("better", "tie")).strip().upper()
            # from the focus system's point of view: +1 better, -1 worse, 0 tie
            score = {"A": 1, "B": -1}.get(v, 0) * (1 if r["order"] == "fa" else -1)
            pair[r["system"]].setdefault(r["i"], []).append(score)
    report = {"judge": d["judge"], "rate": {}, "pair_vs_focus": {}}
    for s, xs in rate.items():
        num = lambda k, sub=None: round(statistics.mean(float(o.get(k, 0)) for st, o in xs if sub in (None, st)), 2)
        report["rate"][s] = {"n": len(xs), "helpful": num("helpful"), "grounded": num("grounded"), "clean": num("clean"),
                             "send_pct": round(100 * statistics.mean(str(o.get("send", "")).lower() == "yes" for _, o in xs), 1),
                             "send_pct_by_set": {sub: round(100 * statistics.mean(str(o.get("send", "")).lower() == "yes"
                                                                                  for st, o in xs if st == sub), 1)
                                                 for sub in sorted({st for st, _ in xs})}}
    for s, per_row in pair.items():
        w = l = t = 0
        for scores in per_row.values():
            if len(scores) == 2 and scores[0] == scores[1] != 0:  # same verdict in both orders
                w, l = (w + 1, l) if scores[0] > 0 else (w, l + 1)
            else:
                t += 1
        report["pair_vs_focus"][s] = {"focus_wins": w, "ties_or_order_flips": t, "focus_losses": l,
                                      "sign_p": round(sign_p(w, l), 3)}
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(sys.argv[1])
