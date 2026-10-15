"""Build the v0.4 mixture: more reasoning data, less support data, and reply targets rewritten by a teacher.

Mixture (24,500 rows; customer support ≈ 43%, down from 66% in v0.3):
  intent     CLINC150, MASSIVE, Bitext intents: the same 1,000 rows each as v0.2 (data/v2/train.jsonl)
  reasoning  ReClor, LogiQA 2.0, CosmosQA, PIQA, αNLI, ProofWriter: 2,000 fresh rows each (twice v0.2)
  support    Bitext reply task 2,000 · ABCD 3,500 · Bitext verticals 3,000 · CFPB 1,000, sampled from the
             v0.2 / v0.3 training rows, keeping each source's mix of kinds

Reply targets: every Bitext and Bitext-vertical row with a reply gets a new target written by a teacher (Qwen3-14B through Ollama,
thinking off, temperature 0). The teacher sees the State and the full original reference reply (Bitext's whole
response with its {{slots}}) and rewrites it into one or two sentences that keep its
intent, use State facts, and add nothing that is not in the State or the reference. Each output is checked with
reply_checks.check(); a row whose new target fails (invented fact, {{slot}}, more than 2 sentences or 55 words)
is dropped. Teacher outputs are cached in data/v4/teacher_cache.jsonl, so reruns don't call the teacher again.
ABCD replies are real agent messages and stay as written. Held-out sets are unchanged (their references stay
the original ones).

Run: ssh -f -N -L 11435:localhost:11434 <gpu-host>   # the teacher runs on a GPU machine with Ollama
     uv run data/prepare_v4.py --teacher-url http://localhost:11435/api/chat
"""
import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import data.prepare_v2 as v2  # noqa: E402
from data.prepare_v3 import VERTICALS, leak_check  # noqa: E402
from reply_checks import check  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "v4"
CACHE = OUT / "teacher_cache.jsonl"
TEACHER = "qwen3:14b"
PROMPT = """You write replies for a customer-support agent.

State (everything the agent knows):
{state}

Reference reply (a template; it may be long, generic, or contain {{{{slots}}}}):
\"\"\"{ref}\"\"\"

Rewrite the reference into ONE or TWO sentences the agent can send to the customer now.
Rules:
- Keep the reference's intent and its main step. Do not add steps, offers or explanations that the reference does not contain.
- Use facts from the State where they help (name, order number, website, phone, email, hours).
- Fill a {{{{slot}}}} only with a fact from the State; otherwise drop that detail. Never write {{{{ or }}}}.
- Never invent specifics: no URLs, phone numbers, emails, amounts, dates, durations, policies or IDs, and no names of website sections, pages, buttons, menus, apps or documents, unless they appear in the State or as plain text (not a {{{{slot}}}}) in the reference.
- Do not ask for information that is already in the State.
- No greeting, apology, empathy phrase or filler. Plain text, no lists, at most 40 words.
Reply with the message only."""


def key(state, ref):
    return hashlib.sha1(json.dumps([state, ref], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def teach(url, state, ref):
    body = json.dumps({"model": TEACHER, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": 100},
                       "messages": [{"role": "user", "content": PROMPT.format(
                           state=json.dumps(state, ensure_ascii=False), ref=ref)}]}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=180))["message"]["content"].strip().strip('"')
        except Exception:
            if attempt == 3:
                return None


LEAD = re.compile(r"^(sure|certainly|absolutely|of course|no problem|great|indeed|assuredly|definitely|"
                  r"understood|got it|okay|ok)[,!.]\s*", re.I)
EXTRA_FLUFF = re.compile(r"^(i (understand|grasp|see|hear|appreciate|realize)|thank you|thanks|i'm (here|happy|glad|sorry)|"
                         r"we (understand|appreciate|apologi)|if you (need|have) (any )?(further|more|other|additional))", re.I)


def strip_filler(text):
    """Drop empathy / sign-off sentences the teacher still adds (Bitext's own filler patterns, plus a few more),
    then keep at most the first two sentences."""
    sents = [LEAD.sub("", x).strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    keep = [x for x in sents if not (v2.FILLER.match(x) or EXTRA_FLUFF.match(x) or v2.FLUFF.search(x))]
    out = " ".join(keep[:2]).strip()  # one or two sentences, as the prompt asks
    return (out[0].upper() + out[1:]) if out else ""


def full_references():
    """customer message → the full original Bitext response, for the Bitext reply rows and the verticals."""
    refs = {}
    df = pd.read_csv(v2.dl("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                           "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"))
    refs["bitext_reply"] = dict(zip(df.instruction, df.response))
    for v, f in VERTICALS.items():
        d = pd.read_csv(v2.dl(f"bitext/Bitext-{v}-llm-chatbot-training-dataset", f))
        refs[f"vert_{v}"] = dict(zip(d.instruction, d.response))
    return refs


def stratified(rows, n, rng, by=lambda r: r["kind"]):
    """n rows keeping the share of each group."""
    groups = defaultdict(list)
    for r in rows:
        groups[by(r)].append(r)
    out = []
    for g, rs in sorted(groups.items()):
        take = round(n * len(rs) / len(rows))
        out += rng.sample(rs, min(take, len(rs)))
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-url", default="http://localhost:11435/api/chat")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(20261016)
    OUT.mkdir(exist_ok=True)

    old2 = [json.loads(l) for l in open(HERE / "v2/train.jsonl")]
    old3 = [json.loads(l) for l in open(HERE / "v3/train.jsonl")]
    parts = {s: [r for r in old2 if r["src"] == s] for s in ("clinc150", "massive", "bitext")}
    v2.N = 2000
    for name, fn in [("reclor", v2.reclor), ("logiqa2", v2.logiqa2), ("cosmosqa", v2.cosmosqa), ("piqa", v2.piqa),
                     ("anli", v2.anli), ("proofwriter", v2.proofwriter)]:
        parts[name] = fn(rng)
    parts["bitext_reply"] = stratified([r for r in old2 if r["src"] == "bitext_reply"], 2000, rng)
    parts["abcd"] = stratified([r for r in old3 if r["src"] == "abcd"], 3500, rng)
    parts["verticals"] = stratified([r for r in old3 if r["src"].startswith("vert_")], 3000, rng,
                                    by=lambda r: (r["src"], r["kind"]))
    parts["cfpb"] = stratified([r for r in old3 if r["src"] == "cfpb"], 1000, rng)

    # ---- teacher rewrites of every reply target
    refs = full_references()
    cache = {}
    if CACHE.exists():
        for line in open(CACHE):
            c = json.loads(line)
            cache[c["key"]] = c["text"]
    jobs = []
    for p in ("bitext_reply", "verticals"):  # ABCD replies are real agent messages and stay as written
        for r in parts[p]:
            if r.get("reply"):
                full = refs.get(r["src"], {}).get(r["state"].get("customer_msg", ""), r["reply"])
                r["_ref"] = str(full)
                r["_key"] = key(r["state"], r["_ref"])
                if r["_key"] not in cache:
                    jobs.append(r)
    print(f"teacher calls needed: {len(jobs)} (cached: {len(cache)})", flush=True)
    with ThreadPoolExecutor(args.workers) as ex, open(CACHE, "a") as f:
        for r, text in zip(jobs, ex.map(lambda r: teach(args.teacher_url, r["state"], r["_ref"]), jobs)):
            if text is not None:
                cache[r["_key"]] = text
                f.write(json.dumps({"key": r["_key"], "text": text}, ensure_ascii=False) + "\n")

    drops = Counter()
    for p in ("bitext_reply", "verticals"):
        kept = []
        for r in parts[p]:
            if r.get("reply"):
                text = strip_filler(cache.get(r.get("_key")) or "")
                c = check(text, {"state": r["state"], "reference": r["_ref"]}, None) if text else None
                if (not text or c["hallucinated"] or c["placeholder"] or c["long"] or len(text.split()) > 55
                        or "1-2 sentence" in text):
                    drops[p] += 1
                    continue
                r["reply_original"], r["reply"] = r["reply"], text
            r.pop("_ref", None), r.pop("_key", None)
            kept.append(r)
        parts[p] = kept

    leak_check(parts["abcd"] + parts["verticals"] + parts["cfpb"])  # the v0.2 rows are reused unchanged
    train = [r for p in parts.values() for r in p]
    rng.shuffle(train)
    v2.write(OUT / "train.jsonl", train)
    support = sum(len(parts[p]) for p in ("bitext", "bitext_reply", "abcd", "verticals", "cfpb"))
    for k, v in parts.items():
        print(f"{k:14} {len(v):6}" + (f"   (dropped {drops[k]} failed rewrites)" if drops[k] else ""))
    print(f"train total {len(train)}, support share {support / len(train):.1%}")


if __name__ == "__main__":
    main()
