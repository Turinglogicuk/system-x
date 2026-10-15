"""v0.2 training: the LLM-as-Jev recipe (§3.4) plus our reply option.

Per row, the decision loss is the tree-factorized listwise loss on the gold suffix "k]":
    L_tree = -sum_t [log p(c_t) - log m(node_t)],   m = probability of the legal next tokens at that trie node
KL anchors to the frozen base model (paper Eq. 3, reference statistics cached once, top-64 + remainder):
    L_mass  binary KL of legal mass at gold-path nodes
    L_out   KL of the illegal-token distribution at gold-path nodes
    L_pos   full-distribution KL at the reply start and right after "]" (decision rows)
Our additions:
    Brier on the option distribution (rows with <= 9 options, where it is exact from the root logits)
    reply rows: LM loss on " Reply: <text><|im_end|>" + KL to the base model on those tokens
Deviation from the paper (documented in the README): off-path trie nodes are not anchored.

Run:  uv run train_v2.py --ref-only            # cache base-model statistics (once)
      uv run train_v2.py --max-steps 5         # dry run, prints s/step
      uv run train_v2.py                       # full run → out/v2
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tl_decision_plus.device import pick_device
from tl_decision_plus.prompt import PREFILL, build_prompt, reply_token_ids, suffix_token_ids

TOPK = 64
REF_PATH = Path("out/v2_ref.pt")


# ---------------------------------------------------------------- encoding
def encode(tok, r):
    """Token ids + the positions/targets every loss term needs. Positions index the full sequence."""
    prompt = build_prompt(tok, r["state"], r["options"])
    p = tok.encode(prompt, add_special_tokens=False)
    sufs = suffix_token_ids(tok, prompt, len(r["options"]), p)
    g = r["label"]
    gold = sufs[g]
    nodes = []  # (position predicting the token, gold token, legal tokens at that node)
    for t in range(len(gold)):
        legal = sorted({s[t] for s in sufs if len(s) > t and s[:t] == gold[:t]})
        nodes.append((len(p) - 1 + t, gold[t], legal))
    tmpl = tok.encode(prompt[: -len(PREFILL)], add_special_tokens=False)
    assert p[: len(tmpl)] == tmpl, "chat template tokens are not a prefix of the prompt"
    ex = {"nodes": nodes, "K": len(sufs), "gold": g, "first_ids": [s[0] for s in sufs],
          "reply_start": len(tmpl) - 1, "after": None, "reply_span": None}  # paper: the assistant reply start
    if r.get("reply"):
        tail = reply_token_ids(tok, prompt, g + 1, r["reply"], p) + [tok.eos_token_id]
        assert tail[: len(gold)] == gold, "reply continuation must start with the gold suffix"
        ids = p + tail
        lead = len(reply_token_ids(tok, prompt, g + 1, None, p))  # "k] Reply:" tokens
        ex["reply_span"] = (len(p) + lead, len(ids))  # predict these tokens (reply text + <|im_end|>)
    else:
        ids = p + gold
        ex["after"] = len(ids) - 1  # position right after "]"
    ex["ids"] = ids
    return ex


def sites(ex):
    """All positions whose logits we need, in a fixed order shared by the reference pass and training."""
    pos = [n[0] for n in ex["nodes"]] + [ex["reply_start"]]
    if ex["after"] is not None:
        pos.append(ex["after"])
    if ex["reply_span"]:
        s, e = ex["reply_span"]
        pos += list(range(s - 1, e - 1))
    return pos


# ---------------------------------------------------------------- reference statistics
def topk_stats(logp, k=TOPK):
    """[n, V] log-probs → (ids [n,k], probs [n,k], remainder [n])."""
    p = logp.exp()
    v, i = p.topk(k, dim=-1)
    return i.to(torch.int32).cpu(), v.half().cpu(), (1 - v.sum(-1)).clamp_min(0).half().cpu()


@torch.no_grad()
def build_reference(model, body, lm_head, data, device):
    """One forward-only pass of the frozen base model (paper App. A): per site top-64 + remainder,
    legal mass and top-64 illegal tokens at gold-path nodes."""
    refs = []
    t0 = time.time()
    for n, ex in enumerate(data):
        ids = torch.tensor([ex["ids"]], device=device)
        h = body(input_ids=ids).last_hidden_state[0]
        pos = torch.tensor(sites(ex), device=device)
        logp = F.log_softmax(lm_head(h[pos]).float(), -1)
        full = topk_stats(logp)
        nodes = []
        for j, (_, _, legal) in enumerate(ex["nodes"]):
            lp = logp[j].clone()
            m0 = float(lp[legal].exp().sum())
            lp[legal] = -float("inf")
            ii = lp.topk(TOPK).indices
            pi = lp[ii].exp()
            nodes.append({"m0": m0, "ill_ids": ii.to(torch.int32).cpu(), "ill_p": pi.half().cpu()})
        refs.append({"full": full, "nodes": nodes})
        if (n + 1) % 500 == 0:
            print(f"  reference {n + 1}/{len(data)} ({time.time() - t0:.0f}s)", flush=True)
    return refs


def kl_top(ref_ids, ref_p, ref_rem, logp):
    """KL(p0 || p_theta) over the reference top-k tokens plus one remainder bucket (paper App. I)."""
    p0 = ref_p.float()
    lp = logp.gather(-1, ref_ids.long())
    q_rem = (1 - lp.exp().sum(-1)).clamp_min(1e-8)
    r0 = ref_rem.float().clamp_min(1e-8)
    return (p0 * (p0.clamp_min(1e-8).log() - lp)).sum(-1) + r0 * (r0.log() - q_rem.log())


# ---------------------------------------------------------------- loss
def row_loss(ex, ref, logp, args):
    """logp: [n_sites, V] log-probs at sites(ex), same order. Returns (loss, parts)."""
    nn = len(ex["nodes"])
    dev = logp.device
    tree = mass = out = torch.zeros((), device=dev)
    for j, (_, tgt, legal) in enumerate(ex["nodes"]):
        lp = logp[j]
        lm = torch.logsumexp(lp[legal], 0)
        tree = tree - (lp[tgt] - lm)
        r = ref["nodes"][j]
        m0 = min(max(r["m0"], 1e-6), 1 - 1e-6)
        mt = lm.exp().clamp(1e-6, 1 - 1e-6)
        mass = mass + 0.5 / nn * (m0 * math.log(m0) - m0 * mt.log() + (1 - m0) * math.log(1 - m0) - (1 - m0) * (1 - mt).log())
        # illegal-token distribution, renormalised by the illegal mass on both sides
        ill = r["ill_ids"].to(dev).long()
        p0 = r["ill_p"].to(dev).float() / (1 - m0)
        r0 = (1 - p0.sum()).clamp_min(1e-8)
        lq = lp[ill] - torch.log1p(-mt)
        q_rem = (1 - lq.exp().sum()).clamp_min(1e-8)
        out = out + 0.5 / nn * ((p0 * (p0.clamp_min(1e-8).log() - lq)).sum() + r0 * (r0.log() - q_rem.log()))
    fi, fp, fr = (t.to(dev) for t in ref["full"])
    # reply start (always) and after "]" (decision rows) — 0.5 each
    pos_kl = 0.5 * kl_top(fi[nn], fp[nn], fr[nn], logp[nn])
    if ex["after"] is not None:
        pos_kl = pos_kl + 0.5 * kl_top(fi[nn + 1], fp[nn + 1], fr[nn + 1], logp[nn + 1])
    anchor = mass + out + pos_kl

    brier = torch.zeros((), device=dev)
    if ex["K"] <= 9 and args.gamma > 0:  # single-digit options: q = root distribution renormalised over the K digits
        q = F.softmax(logp[0][ex["first_ids"]], -1)
        brier = ((q - F.one_hot(torch.tensor(ex["gold"], device=dev), ex["K"]).float()) ** 2).sum()

    lm_loss = rkl = torch.zeros((), device=dev)
    if ex["reply_span"]:
        s, e = ex["reply_span"]
        tgt = torch.tensor(ex["ids"][s:e], device=dev)
        base = nn + 1  # reply sites start after the node sites and the reply-start site
        lp = logp[base : base + len(tgt)]
        lm_loss = F.nll_loss(lp, tgt)
        rkl = kl_top(fi[base : base + len(tgt)], fp[base : base + len(tgt)], fr[base : base + len(tgt)], lp).mean()

    loss = tree + args.lam * anchor + args.gamma * brier + args.beta * lm_loss + args.lam_reply * rkl
    return loss, {"tree": tree, "anchor": anchor, "brier": brier, "lm": lm_loss, "rkl": rkl}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/v2/train.jsonl")
    ap.add_argument("--out", default="out/v2")
    ap.add_argument("--batch", type=int, default=32, help="examples per optimizer step (paper: 32)")
    ap.add_argument("--token-budget", type=int, default=4000, help="max tokens per micro-batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lam-reply", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--ref-only", action="store_true")
    ap.add_argument("--longest-first", action="store_true", help="dry run on the longest rows (memory check)")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast for the forward/backward")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B", help="base model to adapt")
    ap.add_argument("--ref", default=str(REF_PATH), help="cached base-model statistics for --train (one file per dataset)")
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(20260922)
    tok = AutoTokenizer.from_pretrained(args.base)
    rows = [json.loads(l) for l in open(args.train)]
    data = [encode(tok, r) for r in rows]
    print(f"{len(data)} rows, median {sorted(len(d['ids']) for d in data)[len(data) // 2]} tokens, "
          f"max {max(len(d['ids']) for d in data)}", flush=True)

    order = list(range(len(data)))
    random.Random(0).shuffle(order)
    if args.longest_first:
        order.sort(key=lambda i: -len(data[i]["ids"]))
    ref_path = Path(args.ref)
    if args.max_steps:  # dry run: only the rows it will touch, with its own reference file
        keep = order[: args.max_steps * args.batch]
        data = [data[i] for i in keep]
        order = list(range(len(data)))
        ref_path = Path("out/v2_ref_dry.pt")
        if ref_path.exists():
            ref_path.unlink()

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(device)
    if not ref_path.exists():
        print("building reference statistics (one pass of the frozen model) ...", flush=True)
        base.eval()
        refs = build_reference(base, base.model, base.lm_head, data, device)
        ref_path.parent.mkdir(exist_ok=True)
        torch.save(refs, ref_path)
    if args.ref_only:
        return
    refs = torch.load(ref_path)
    assert len(refs) == len(data), f"reference cache does not match the training data; delete {ref_path}"

    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # paper Table 2
    model.enable_input_require_grads()
    model.train()
    body, lm_head = model.base_model.model.model, model.base_model.model.lm_head

    steps = math.ceil(len(order) / args.batch)
    total = args.max_steps or steps
    warm = max(1, int(0.1 * total))
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warmup 10%, linear decay to 10% of peak (paper Table 2)
        optim, lambda s: (s + 1) / warm if s < warm else 1 - 0.9 * (s - warm) / max(1, total - warm))

    t0 = time.time()
    for step in range(total):
        batch = order[step * args.batch : (step + 1) * args.batch]
        # micro-batches by token budget; each row's loss is divided by the batch size
        # sorted by length so each micro-batch pads to similar lengths; the budget counts padded tokens
        micro, cur = [], []
        for i in sorted(batch, key=lambda i: -len(data[i]["ids"])):
            if cur and (len(cur) + 1) * len(data[cur[0]]["ids"]) > args.token_budget:
                micro.append(cur); cur = []
            cur.append(i)
        if cur:
            micro.append(cur)
        agg = {}
        for mb in micro:
            Lmax = max(len(data[i]["ids"]) for i in mb)
            ids = torch.full((len(mb), Lmax), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(mb), Lmax), dtype=torch.long)
            for r, i in enumerate(mb):
                ids[r, : len(data[i]["ids"])] = torch.tensor(data[i]["ids"]); att[r, : len(data[i]["ids"])] = 1
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=args.bf16):
                h = body(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
            loss_mb = 0
            for r, i in enumerate(mb):
                pos = torch.tensor(sites(data[i]), device=device)
                logp = F.log_softmax(lm_head(h[r, pos].float()).float(), -1)
                l, parts = row_loss(data[i], refs[i], logp, args)
                loss_mb = loss_mb + l / len(batch)
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0) + v.detach() / len(batch)
            loss_mb.backward()
            del h, loss_mb
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)

        el = time.time() - t0
        if step < 3 or (step + 1) % 10 == 0:
            print(f"step {step + 1}/{total} " + " ".join(f"{k} {float(v):.4f}" for k, v in agg.items())
                  + f" | {el / (step + 1):.1f}s/step, eta {el / (step + 1) * (total - step - 1) / 60:.0f} min", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0 and not args.max_steps:
            model.save_pretrained(f"{args.out}_ckpt")
    model.save_pretrained(args.out if not args.max_steps else f"{args.out}_dry")
    print(f"saved after {total} steps in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
