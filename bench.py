"""Benchmarks in the LLM-as-Jev format: JevBench public items (231) and Banking77 test (3,080, 77 options).

Run:  uv run bench.py --task jevbench                    # base model, training-free
      uv run bench.py --task jevbench --adapter out/v2   # trained adapter
      uv run bench.py --task jevbench --check 3          # also verify the fast scorer on 3 items
Results → out/bench_<task>_<name>.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from tl_decision_plus.device import pick_device, sync
from tl_decision_plus.prompt import build_prompt, suffix_token_ids
from tl_decision_plus.scoring import score_suffixes

ROOT = Path(__file__).parent


# ---------------------------------------------------------------- tasks
def jevbench_items():
    """Each JevBench item → (state, options, gold index). The item's instructions go first in the state."""
    items = []
    for split in ("original", "easy", "hard"):
        for line in open(ROOT / f"data/bench/jevbench/{split}.jsonl"):
            r = json.loads(line)
            q = r["question"]
            labels = r["labels"]
            if q["type"] == "choice":
                opts = [f"{l}: {q['criteria'][l]}" for l in labels]
            elif q["type"] == "noul":
                crit = {"yes": q["criteria"]["true"], "no": q["criteria"]["false"]}
                opts = [f"{l}: {crit[l]}" for l in labels]
            else:  # score: ordered grades, criteria is a list aligned with labels
                opts = [f"{l}: {c}" for l, c in zip(labels, q["criteria"])]
            state = f"{q['instructions']}\n\n{r['state']}"
            items.append({"id": r["id"], "tier": split, "type": q["type"], "family": r["family"],
                          "state": state, "options": opts, "gold": labels.index(str(r["expected"]))})
    return items


def banking77_items():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo = "legacy-datasets/banking77"
    test = pd.read_parquet(hf_hub_download(repo, "data/test-00000-of-00001.parquet", repo_type="dataset"))
    labels = banking77_label_names()
    opts = [l.replace("_", " ") for l in labels]
    return [{"id": f"b77-{i}", "tier": "test", "type": "choice", "family": "banking77",
             "state": row.text, "options": opts, "gold": int(row.label)} for i, row in enumerate(test.itertuples())]


def banking77_label_names():
    """The 77 intent names in label-id order (from the dataset card's class_label list)."""
    from huggingface_hub import hf_hub_download

    card = Path(hf_hub_download("legacy-datasets/banking77", "README.md", repo_type="dataset")).read_text()
    block = card.split("class_label:")[1].split("names:")[1]
    names = []
    for line in block.splitlines()[1:]:
        line = line.strip()
        if not line or not line[0].isdigit() and not line.startswith("'"):
            if names:
                break
            continue
        if ":" in line:
            names.append(line.split(":", 1)[1].strip().strip("'\""))
    assert len(names) == 77, f"expected 77 Banking77 labels, got {len(names)}"
    return names


TASKS = {"jevbench": jevbench_items, "banking77": banking77_items}


# ---------------------------------------------------------------- metrics
def ece(confs, correct, bins=15):
    n = len(confs)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0)]
        if sel:
            total += len(sel) / n * abs(statistics.mean(confs[i] for i in sel) - statistics.mean(correct[i] for i in sel))
    return total


def summarize(rows):
    correct = [r["pred"] == r["gold"] for r in rows]
    confs = [max(r["probs"]) for r in rows]
    brier = statistics.mean(sum((p - (k == r["gold"])) ** 2 for k, p in enumerate(r["probs"])) for r in rows)
    out = {"n": len(rows), "accuracy": round(100 * statistics.mean(correct), 1), "ece": round(ece(confs, correct), 3),
           "brier": round(brier, 3), "root_legal_mass": round(statistics.mean(r["root_mass"] for r in rows), 4)}
    for key in ("tier", "type"):
        groups = {}
        for r, c in zip(rows, correct):
            groups.setdefault(r[key], []).append(c)
        out[f"accuracy_by_{key}"] = {k: f"{100 * statistics.mean(v):.1f} ({sum(v)}/{len(v)})" for k, v in sorted(groups.items())}
    return out


# ---------------------------------------------------------------- main
@torch.no_grad()
def brute_force(model, tok, prompt, n, device):
    """Score each candidate by a full forward over prompt + suffix (slow; used only to check the fast path)."""
    p = tok.encode(prompt, add_special_tokens=False)
    scores = []
    for k in range(1, n + 1):
        full = tok.encode(prompt + f"{k}]", add_special_tokens=False)
        logp = F.log_softmax(model(input_ids=torch.tensor([full], device=device)).logits[0].float(), -1)
        scores.append(sum(float(logp[t - 1, full[t]]) for t in range(len(p), len(full))))
    return F.softmax(torch.tensor(scores), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="jevbench")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--fp32", action="store_true", help="the paper reports FP32 inference")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", type=int, default=0, help="verify the fast scorer against brute force on N items")
    ap.add_argument("--name", default=None)
    ap.add_argument("--base", default=None, help="base model (default: the adapter's base, else Qwen3-0.6B)")
    args = ap.parse_args()
    name = args.name or (Path(args.adapter).name if args.adapter else "base")

    device = pick_device()
    from tl_decision_plus.engine import base_of

    base_name = base_of(args.adapter, args.base)
    tok = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32 if args.fp32 else torch.bfloat16)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to(device).eval()

    items = TASKS[args.task]()
    if args.limit:
        items = items[: args.limit]

    rows, lat = [], []
    t_start = time.time()
    for i, it in enumerate(items):
        prompt = build_prompt(tok, it["state"], it["options"])
        pids = tok.encode(prompt, add_special_tokens=False)
        sufs = suffix_token_ids(tok, prompt, len(it["options"]), pids)
        sync(device); t0 = time.perf_counter()
        probs, _, root_mass, _ = score_suffixes(model, torch.tensor([pids], device=device), sufs, tok.pad_token_id)
        sync(device); lat.append((time.perf_counter() - t0) * 1000)
        if i < args.check:
            ref = brute_force(model, tok, prompt, len(it["options"]), device)
            diff = float((ref - probs).abs().max())
            print(f"check item {i} ({len(it['options'])} options): max |fast - brute force| = {diff:.2e}")
        p = probs.tolist()
        rows.append({**{k: it[k] for k in ("id", "tier", "type", "family", "gold")}, "probs": p,
                     "pred": max(range(len(p)), key=p.__getitem__), "root_mass": root_mass, "prompt_tokens": len(pids)})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(items)} ({time.time() - t_start:.0f}s)", flush=True)

    summary = summarize(rows)
    summary["latency_ms_p50"] = round(sorted(lat)[len(lat) // 2], 1)
    summary["dtype"] = "fp32" if args.fp32 else "bf16"
    Path("out").mkdir(exist_ok=True)
    Path(f"out/bench_{args.task}_{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps({"task": args.task, "model": name, **summary}, indent=2))


if __name__ == "__main__":
    main()
