"""LLM judge for reply quality (blind; standard library only, talks to a local Ollama server).

Two protocols on the 150 rows of data/v3/judge_rows.jsonl:
  rate   each system's reply alone, 1-5 on helpful / grounded / clean, plus "would a support lead send it" (yes/no).
         The reference reply is NOT shown, so the judge doesn't reward Bitext's template style.
  pair   v3 against each other system. The judge picks the better reply to send. Every pair is judged twice
         with A and B swapped; a verdict that flips with the order counts as a tie.
System names are never shown to the judge.

Run (on the machine with Ollama):  python3 judge.py --judge gemma4:31b-cloud --systems base v2 v3 two_call_1.7b reference
"""
import argparse
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RATE = """You are evaluating a customer-support reply written by an AI agent.

State (everything the agent knows; the customer's message is inside it):
{state}

Agent's reply:
\"\"\"{reply}\"\"\"

Score the reply from 1 (very poor) to 5 (excellent) on:
- helpful: addresses what the customer needs and moves the issue forward
- grounded: uses facts from the State correctly and does not invent facts (URLs, numbers, policies, order details) that are not in the State
- clean: reads as a finished 1-2 sentence message to a customer, with no template slots like {{{{X}}}}, no repeated instructions, no cut-off text
Also say whether a support team lead would send it as is (send: yes or no).

Answer with JSON only: {{"helpful": n, "grounded": n, "clean": n, "send": "yes" or "no"}}"""

PAIR = """You are comparing two customer-support replies written for the same situation.

State (everything the agent knows; the customer's message is inside it):
{state}

Reply A:
\"\"\"{a}\"\"\"

Reply B:
\"\"\"{b}\"\"\"

Which reply would you rather send to this customer? Consider helpfulness, correct use of the State without invented facts, and whether it reads as a finished 1-2 sentence message.
Answer with JSON only: {{"better": "A" or "B" or "tie"}}"""


def ask(model, prompt, url, tries=6):
    body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            out = json.load(urllib.request.urlopen(req, timeout=120))
            if "error" in out:
                raise RuntimeError(out["error"])
            text = out["message"]["content"]
            return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        except Exception as e:  # rate limits, timeouts, malformed JSON: back off and retry
            if k == tries - 1:
                return {"error": str(e)[:200]}
            time.sleep(2 ** k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gemma4:31b-cloud")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--focus", default="v3", help="the system compared against every other one")
    ap.add_argument("--url", default="http://localhost:11434/api/chat")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix for the output file, to add systems without overwriting")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open("data/v3/judge_rows.jsonl")]
    rep = {s: [json.loads(l)["text"] or "" for l in open(f"out/replies/{s}.jsonl")] for s in args.systems}
    st = [json.dumps(r["state"], ensure_ascii=False) for r in rows]
    jobs = [("rate", s, i, RATE.format(state=st[i], reply=rep[s][i])) for s in args.systems for i in range(len(rows))]
    for s in args.systems:
        if s == args.focus:
            continue
        for i in range(len(rows)):
            for order in ("fa", "af"):  # focus first, then focus second
                a, b = (rep[args.focus][i], rep[s][i]) if order == "fa" else (rep[s][i], rep[args.focus][i])
                jobs.append(("pair", s, i, order, PAIR.format(state=st[i], a=a, b=b)))
    random.Random(0).shuffle(jobs)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        answers = list(ex.map(lambda j: ask(args.judge, j[-1], args.url), jobs))
    results = [dict(kind=j[0], system=j[1], i=j[2], order=(j[3] if j[0] == "pair" else None), out=a)
               for j, a in zip(jobs, answers)]
    out = Path(f"out/judge_{args.judge.replace(':', '_').replace('/', '_')}{args.tag}.json")
    out.write_text(json.dumps({"judge": args.judge, "rows": len(rows), "results": results}, indent=1))
    errors = sum("error" in r["out"] for r in results)
    print(f"{len(results)} judgments in {(time.time() - t0) / 60:.1f} min, {errors} errors → {out}")


if __name__ == "__main__":
    main()
