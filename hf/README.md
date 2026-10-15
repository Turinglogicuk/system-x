---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
language:
  - en
tags:
  - lora
  - peft
  - decision-model
  - calibration
  - customer-support
  - kv-cache
  - qwen3
---

# System X: An LLM Decision Model That Replies from Its Own KV Cache

**Authors:** Dr. Syed Usama Khalid Bukhari (Turing Logic LTD), Dr. Syed Safwan Khalid (Idrak AI LTD)

**Code and paper:** [github.com/Turinglogicuk/system-x](https://github.com/Turinglogicuk/system-x)

This repository holds the **System X v4** LoRA adapter (17.4M parameters) for [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). The base model is not included; it downloads separately under its own Apache-2.0 license.

> **Non-commercial use only.** The adapter is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You may use and adapt it with attribution, but not for commercial purposes. For commercial licensing, contact Turing Logic LTD.

## What it does

You give System X a situation (the *State*) and a numbered list of options.
- **Deciding:** with one prefill it returns a calibrated probability for every option.
- **Replying:** one option is special: `REPLY_WITH_TEXT`. When it wins, or no option is confident enough, the same model **continues from the key–value cache it already built for the decision** and writes a one- to two-sentence reply, without processing the prompt a second time.
- **No architecture change:** it is the unchanged Qwen3 transformer plus a LoRA adapter.

## Prompt format

Training and inference must use exactly this format:
- **Chat template:** Qwen3's, with `enable_thinking=False` and no system message.
- **User message:** `State: {json}`, then `Options:`, then one line per option (`[1] NAME: description` …), then the instruction:
  - `Select one option. Answer only with its bracketed numeric identifier.`
  - When `REPLY_WITH_TEXT` is offered, followed by: ` If you select REPLY_WITH_TEXT, write a 1-2 sentence reply to the customer after the identifier.`
- **Assistant turn:** pre-filled with `Best answer: [`.
- **Decision:** option *k* is the token suffix `k]`. Its score is the summed log-probability of those tokens, and a softmax over the scores gives the probabilities.
- **Reply:** append `k] Reply:` to the same cache and decode greedily until `<|im_end|>`.

```python
import json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32)
model = PeftModel.from_pretrained(model, "TuringLogic/system-x").merge_and_unload().eval()

state = {"customer_msg": "Someone in another country just logged into my account and bought three gift cards. That wasn't me!",
         "risk_score": 0.96}
options = ["AUTO_REFUND: Issue an instant refund", "BLOCK_FRAUD: Lock the account now",
           "ESCALATE_HUMAN: Hand to a human agent", "REPLY_WITH_TEXT: Write a 1-2 sentence reply"]
user = "\n".join([f"State: {json.dumps(state)}", "Options:"] + [f"[{i+1}] {o}" for i, o in enumerate(options)]
                 + ["Select one option. Answer only with its bracketed numeric identifier. If you select "
                    "REPLY_WITH_TEXT, write a 1-2 sentence reply to the customer after the identifier."])
prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False) + "Best answer: ["

p = tok.encode(prompt, add_special_tokens=False)
scores = []
with torch.no_grad():
    for k in range(1, len(options) + 1):                       # score each suffix "k]"
        full = tok.encode(prompt + f"{k}]", add_special_tokens=False)
        logp = model(torch.tensor([full])).logits[0].log_softmax(-1)
        scores.append(sum(logp[t - 1, full[t]] for t in range(len(p), len(full))))
probs = torch.stack(scores).softmax(-1)
print({o.split(":")[0]: round(float(x), 3) for o, x in zip(options, probs)})
```

This loop is the simple way to score, with one forward pass per option. The GitHub code (`DecisionEngine`) scores all options from **one** prefill and writes the reply from that cache.

## Results

Evaluated against the untrained Qwen3-1.7B:

| | Qwen3-1.7B (untrained) | System X v4 |
|---|---|---|
| JevBench public (231 items, FP32) | 55.0% (ECE 0.404) | **61.0%** (ECE 0.123), p = 0.04 |
| Banking77 (77 options) | 28.3% | **65.2%** |
| Hand-written support cases (97) | 72.2% | **86.6%** |
| ABCD next-step decisions (held out) | 48.3% | **86.0%** |
| IFEval (prompt-level strict) | 66.0% | 64.5% (not significant, p = 0.71) |

**Latency:** replying from the decision's cache saves 17–39% of end-to-end latency (367 to 4,161 prompt tokens, Apple M4 Pro). In FP32 the cached reply is token-for-token identical to a full re-read.

**Reply quality:** of two LLM judges, one prefers this model's replies to a separate, unadapted Qwen3-1.7B reply call (31 to 11 pairs, p = 0.003). The other shows no significant preference (p = 0.11).

## Training

- **Objective:** the LLM-as-Jev tree-factorized listwise loss with KL anchors, plus a Brier term and, on reply rows, a language-model loss and a KL term.
- **LoRA:** r = 16, α = 32, on all attention and MLP projections.
- **Schedule:** one epoch (758 steps of 32 rows), lr 1e-4, FP32.
- **Data:** 24,256 rows:
  - the LLM-as-Jev reasoning and intent sets (CLINC150, MASSIVE, ReClor, LogiQA 2.0, CosmosQA, PIQA, αNLI, ProofWriter);
  - customer-support rows (42%) from Bitext, ABCD and the CFPB Consumer Complaint Database;
  - Bitext reply targets rewritten by Qwen3-14B into short replies grounded in the State.

## Limitations

- **State conflicts:** the model still acts when the account state rules the action out, e.g. an address change on an order that has already shipped. It got 2 of 8 such cases right.
- **Invented details:** when asked for a phone number that is not in the State, it can invent a placeholder one.
- **Evaluation scope:** a single training seed; reply quality judged by LLMs only, with no human evaluation; English only.
- **Not for unattended high-stakes actions:** keep a human or a rule check in front of refunds, account locks and similar actions.

## License

- **The adapter weights in this repository:** **CC BY-NC 4.0** (see `LICENSE`).
- **The base model Qwen3-1.7B:** Apache-2.0, not distributed here.
- **Training data:** each source keeps its own licence. ReClor is research-only and LogiQA 2.0 is CC BY-NC-SA 4.0, which is one reason the adapter is non-commercial.

## Citation

```bibtex
@misc{bukhari2026systemx,
  title  = {System X: An LLM Decision Model That Replies from Its Own KV Cache},
  author = {Bukhari, Syed Usama Khalid and Khalid, Syed Safwan},
  year   = {2026},
  note   = {Turing Logic LTD; https://github.com/Turinglogicuk/system-x}
}
```
