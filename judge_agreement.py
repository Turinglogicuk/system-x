"""Second-judge check: Claude's blind A/B grades of the 100 items in data/v3/human_items.json, against Gemma's.

Claude graded the items twice (out/claude_grades_pass1.json as shown, pass2 with A and B swapped). As with Gemma,
a verdict counts only if both orders agree; otherwise it is a tie. Verdicts are read from v3's point of view
(win / tie / loss) and compared with Gemma's verdict on the same row and opponent (out/judge_gemma4_31b-cloud.json).

Run: uv run judge_agreement.py
     uv run judge_agreement.py --items data/v3/human_items_v4.json --grades out/claude_grades_v4 \
         --gemma out/judge_gemma4_31b-cloud_v4.json --focus v4 --out out/judge_agreement_v4.json
"""
import argparse
import json
from collections import Counter, defaultdict

from judge_report import sign_p


def kappa(pairs, labels=("win", "tie", "loss")):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/v3/human_items.json")
    ap.add_argument("--grades", default="out/claude_grades", help="prefix of the _pass1.json / _pass2.json files")
    ap.add_argument("--gemma", default="out/judge_gemma4_31b-cloud.json")
    ap.add_argument("--focus", default="v3")
    ap.add_argument("--out", default="out/judge_agreement.json")
    args = ap.parse_args()
    items = json.load(open(args.items))
    g1 = {g["id"]: g for g in json.load(open(f"{args.grades}_pass1.json"))}
    g2 = {g["id"]: g for g in json.load(open(f"{args.grades}_pass2.json"))}
    gem = defaultdict(list)
    for r in json.load(open(args.gemma))["results"]:
        if r["kind"] == "pair" and "error" not in r["out"]:
            v = str(r["out"].get("better", "tie")).strip().upper()
            gem[(r["i"], r["system"])].append({"A": 1, "B": -1}.get(v, 0) * (1 if r["order"] == "fa" else -1))

    def gemma_verdict(row, other):
        s = gem[(row, other)]
        return ("win" if s[0] > 0 else "loss") if len(s) == 2 and s[0] == s[1] != 0 else "tie"

    per_sys, pairs, ok = defaultdict(Counter), [], defaultdict(list)
    for it in items:
        v3_is_a = it["a_sys"] == args.focus
        other = it["b_sys"] if v3_is_a else it["a_sys"]
        p1 = {"A": 1, "B": -1}.get(g1[it["id"]]["better"], 0) * (1 if v3_is_a else -1)
        p2 = {"A": 1, "B": -1}.get(g2[it["id"]]["better"], 0) * (-1 if v3_is_a else 1)  # pass 2 swapped A and B
        claude = ("win" if p1 > 0 else "loss") if p1 == p2 != 0 else "tie"
        per_sys[other][claude] += 1
        pairs.append((claude, gemma_verdict(it["row"], other)))
        # "would send" from pass 1 (A/B as stored)
        ok[it["a_sys"]].append(g1[it["id"]]["a_ok"]); ok[it["b_sys"]].append(g1[it["id"]]["b_ok"])
    report = {
        "claude_vs_opponent": {s: {"focus_wins": c["win"], "ties_or_flips": c["tie"], "focus_losses": c["loss"],
                                   "sign_p": round(sign_p(c["win"], c["loss"]), 3)} for s, c in per_sys.items()},
        "claude_send_pct": {s: round(100 * sum(v) / len(v), 1) for s, v in ok.items()},
        "agreement_with_gemma": {"n": len(pairs), "raw": round(sum(a == b for a, b in pairs) / len(pairs), 3),
                                 "cohen_kappa": round(kappa(pairs), 3),
                                 "excluding_ties": round(sum(a == b for a, b in pairs if "tie" not in (a, b)) /
                                                         max(1, sum("tie" not in (a, b) for a, b in pairs)), 3),
                                 "confusion": Counter(f"claude={a}/gemma={b}" for a, b in pairs)},
        "claude_order_consistency": round(sum(
            ({"A": 1, "B": -1}.get(g1[i]["better"], 0) == -{"A": 1, "B": -1}.get(g2[i]["better"], 0)) for i in g1) / len(g1), 3),
    }
    print(json.dumps(report, indent=1))
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
