"""Automatic reply checks on the outputs of eval_v2.py (no generation, no judge).

For every row where the model wrote a reply:
  grounded      the reference reply uses a fact from the State (website, phone, order number, ...) and the model's
                reply contains that same fact. Reported over the rows whose reference needs a State fact.
  hallucinated  the reply contains a URL, email, phone number, dollar amount or order-style ID that appears nowhere
                in the State (the customer message and conversation included)
  placeholder   an unfilled template slot such as {{Order Number}} (counted apart from hallucinations)
  echo          the reply repeats the prompt's instruction ("1-2 sentence reply")
  long          more than 2 sentences

Run: uv run reply_checks.py out/eval_v2_v3.json=data/v2/heldout_reply.jsonl,data/v2/handwritten.jsonl ...
     (each eval file is followed by the jsonl files of its sets, in the order eval_v2.py ran them)
"""
import json
import re
import sys

PATTERNS = {
    "url": r"\b(?:https?://\S+|www\.[a-z0-9-]+\.[a-z.]+|[a-z0-9-]+\.(?:com|org|net|io)\b)",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "phone": r"(?:\+?\d[\d\s().-]{7,}\d)",
    "money": r"\$\s?\d[\d,]*(?:\.\d+)?",
    "order_id": r"\b(?=[A-Z0-9-]*\d)[A-Z]{1,3}-?\d{3,}\b|\b\d{6,}\b",
}


def facts_in(state) -> list[str]:
    """String values of the State's fact blocks (company / customer / account), long enough to be distinctive."""
    out = []
    for part in ("company", "customer", "account"):
        for v in (state.get(part) or {}).values() if isinstance(state, dict) else []:
            if isinstance(v, str) and len(v) >= 4:
                out.append(v)
    return out


def norm(s: str) -> str:
    return re.sub(r"[\s().,$-]", "", s.lower())


def check(text: str, state, ref: str | None) -> dict:
    st = norm(json.dumps(state, ensure_ascii=False))
    found = {k: re.findall(p, text, re.I) for k, p in PATTERNS.items()}
    halluc = [x for xs in found.values() for x in xs if norm(x) not in st]
    needed = [f for f in facts_in(state) if ref and f.lower() in ref.lower()]
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return {
        "needs_fact": bool(needed),
        "grounded": bool(needed) and all(f.lower() in text.lower() for f in needed),
        "hallucinated": bool(halluc), "hallucinations": halluc,
        "placeholder": "{{" in text, "echo": "1-2 sentence" in text, "long": len(sents) > 2,
    }


def summarize(results):
    n = len(results)
    if not n:
        return {"replies": 0}
    pct = lambda k: round(100 * sum(r[k] for r in results) / n, 1)
    need = [r for r in results if r["needs_fact"]]
    return {"replies": n,
            "grounded_pct_of_needed": round(100 * sum(r["grounded"] for r in need) / len(need), 1) if need else None,
            "needed_n": len(need), "hallucinated_pct": pct("hallucinated"), "placeholder_pct": pct("placeholder"),
            "echo_pct": pct("echo"), "over_2_sentences_pct": pct("long")}


def main():
    report = {}
    for arg in sys.argv[1:]:
        path, sets = arg.split("=", 1)
        d = json.load(open(path))
        names = [k[: -len("_items")] for k in d if k.endswith("_items")]
        for name, jl in zip(names, sets.split(",")):
            rows = [json.loads(l) for l in open(jl)]
            items = d[f"{name}_items"]
            assert len(rows) == len(items), f"{path}:{name} has {len(items)} items but {jl} has {len(rows)} rows"
            res = [check(i["text"], r["state"], r.get("reply")) for i, r in zip(items, rows) if i["text"]]
            report[f"{d['model']}/{name}"] = summarize(res)
            report[f"{d['model']}/{name}"]["examples"] = [x["hallucinations"] for x in res if x["hallucinated"]][:5]
    for k, v in report.items():
        print(k, json.dumps(v, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
