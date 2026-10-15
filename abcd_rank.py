"""ABCD utterance ranking (Chen et al. 2021, the "Utterance Ranking" subtask of Cascading Dialogue Success).

At each agent turn whose next step is retrieve_utterance, the gold reply is hidden among ABCD's 100 candidate
utterances (data/raw_v3/utterances.json, indexed by the turn's `candidates`). We rank the candidates by the model's
likelihood of writing them as the reply, s = log p(candidate + <|im_end|> | prompt, "r] Reply:"), three ways:
  pmi (primary)   s minus the same log-prob with an empty conversation. Raw likelihood favours generic lines
                  ("is there anything else i can help you with?") at every turn; subtracting the context-free prior
                  rewards fit to this conversation (surface-form competition, Holtzman et al. 2021).
  mean / sum      s per token, and s itself.
Recall@1/5/10 and MRR.

Context and candidates are ABCD's delexicalised text, as the benchmark specifies. Our v0.3 training used the
original text, so this is slightly out of format for it.

Run: uv run abcd_rank.py --adapter out/v3 --n 500     (leave out --adapter for the untrained model)
"""
import argparse
import gzip
import json
import random
import statistics
import time
from pathlib import Path

import torch

from data.prepare_v3 import ABCD_ACTIONS, REPLY_OPT, opts
from tl_decision_plus.engine import DecisionEngine
from tl_decision_plus.prompt import REPLY_LEAD, build_prompt
from tl_decision_plus.scoring import score_suffixes


def turns(convos, history=10):
    """(conversation so far, actions taken, candidate ids, gold position) for every ranking turn."""
    out = []
    for c in convos:
        conv, done, seen_customer = [], [], False
        for d in c["delexed"]:
            sp, txt, tg = d["speaker"], d["text"], d["targets"]
            if sp == "agent":
                if seen_customer and tg[1] == "retrieve_utterance" and d["candidates"]:
                    out.append((list(conv[-history:]), list(done), d["candidates"], tg[4]))
                conv.append(f"agent: {txt}")
            elif sp == "customer":
                seen_customer = True
                conv.append(f"customer: {txt}")
            elif sp == "action":
                conv.append(f"system: {txt}")
                done.append(ABCD_ACTIONS[tg[2]][0] if tg[2] in ABCD_ACTIONS else tg[2])
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=None, help="base model (default: the adapter's base, else Qwen3-0.6B)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    name = args.name or (Path(args.adapter).name if args.adapter else "base")

    data = json.load(gzip.open("data/raw_v3/abcd_v1.1.json.gz"))["test"]
    utts = json.load(open("data/raw_v3/utterances.json"))
    rng = random.Random(20261015)
    sample = rng.sample(turns(data), args.n)

    eng = DecisionEngine(model_name=args.base, adapter_path=args.adapter)
    tok, dev = eng.tokenizer, eng.device
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    ranks_pmi, ranks_mean, ranks_sum, per_turn, t0 = [], [], [], [], time.time()
    for conv, done, cands, gold in sample:
        actions = rng.sample(list(ABCD_ACTIONS.values()), rng.randint(3, 6)) + [REPLY_OPT]
        rng.shuffle(actions)
        options = opts(actions)
        r = [o["name"] for o in options].index(REPLY_OPT[0]) + 1
        prompt = build_prompt(tok, {"conversation": conv, "actions_taken": done}, options)
        head = prompt + f"{r}]{REPLY_LEAD}"
        hid = tok.encode(head, add_special_tokens=False)
        sufs = []
        for cid in cands:
            full = tok.encode(head + " " + utts[cid], add_special_tokens=False)
            assert full[: len(hid)] == hid, "prompt + lead is not a token prefix of the candidate reply"
            sufs.append(full[len(hid):] + [eos])
        _, scores, _, _ = score_suffixes(eng.model, torch.tensor([hid], device=dev), sufs, tok.pad_token_id, chunk=16)
        null = build_prompt(tok, {"conversation": [], "actions_taken": []}, options) + f"{r}]{REPLY_LEAD}"
        nid = tok.encode(null, add_special_tokens=False)
        nsufs = []
        for cid in cands:
            full = tok.encode(null + " " + utts[cid], add_special_tokens=False)
            assert full[: len(nid)] == nid
            nsufs.append(full[len(nid):] + [eos])
        _, prior, _, _ = score_suffixes(eng.model, torch.tensor([nid], device=dev), nsufs, tok.pad_token_id, chunk=16)
        mean = scores / torch.tensor([len(s) for s in sufs], dtype=scores.dtype)
        for sc, ranks in ((scores - prior, ranks_pmi), (mean, ranks_mean), (scores, ranks_sum)):
            ranks.append(int((sc > sc[gold]).sum()) + 1)
        per_turn.append({"gold_utt_id": cands[gold], "candidates": cands, "rank_pmi": ranks_pmi[-1],
                         "rank_mean": ranks_mean[-1], "rank_sum": ranks_sum[-1]})

    def metrics(ranks):
        return {"R@1": round(100 * statistics.mean(x <= 1 for x in ranks), 1),
                "R@5": round(100 * statistics.mean(x <= 5 for x in ranks), 1),
                "R@10": round(100 * statistics.mean(x <= 10 for x in ranks), 1),
                "MRR": round(statistics.mean(1 / x for x in ranks), 3)}

    report = {"model": name, "n": len(sample), "candidates": 100, "pmi": metrics(ranks_pmi), "per_token_mean": metrics(ranks_mean),
              "sum": metrics(ranks_sum), "minutes": round((time.time() - t0) / 60, 1)}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/abcd_rank_{name}.json").write_text(json.dumps(report, indent=1))
    Path(f"out/abcd_rank_{name}_turns.jsonl").write_text("".join(json.dumps(t) + "\n" for t in per_turn))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
