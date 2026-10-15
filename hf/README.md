---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3-0.6B
library_name: peft
pipeline_tag: text-generation
language:
  - en
tags:
  - lora
  - peft
  - decision-model
  - calibration
  - classification
  - customer-support
---

# TL_decision_plus (v0.2, experimental)

A small **decision model** that can also write a reply.

You give it a situation (State) and a numbered list of options. With **one prefill** of the prompt it gives every option a probability. One option is special: `REPLY_WITH_TEXT`. If that option wins, or the model isn't confident enough about any action, the **same model** continues from the KV cache it already built and writes a 1–2 sentence reply. It never reads the prompt a second time and never needs a second model.

```
State + options ──► 1 prefill + 1 batched step ──► P([1]) P([2]) P([3]) ...
                                                     │
             action wins, confident ─────────────────┴──► return the decision, no text
             REPLY wins / not confident ────────────────► continue from the prompt's KV cache,
                                                          write 1–2 lines
```

This repository holds a **LoRA adapter** (40 MB, r = 16) for [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). The base model is not included. It downloads separately under its own license.

> **Non-commercial use only.** The adapter weights are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You may not use them, or anything derived from them, for commercial purposes. For commercial licensing, contact the author.

**Versions:**
- **`main`:** v0.2, trained on public datasets in the LLM-as-Jev format. This card describes v0.2.
- **`v0.1`:** the first, template-trained adapter. It uses a different prompt format (a system message, `Decision: [`, digits only). Load it with `revision="v0.1"`.

## Credits

- **Base model:** [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) by the Qwen team (Alibaba), Apache-2.0.
- **Decision method, prompt format and training objective:** *LLM-as-Jev* (Yinheng Li and Justin Wagle, arXiv:2610.02076). This covers the bracketed-identifier readout, the tree-factorized listwise loss and the KL anchors.
- **Inspiration:** Jev, the decision model from TypeSafe AI. This project is not affiliated with TypeSafe AI.
- **This work adds:**
  - the `REPLY_WITH_TEXT` option inside the same decision pass;
  - writing the reply from the decision's KV cache;
  - the low-confidence fallback to a reply;
  - a Brier calibration term;
  - a reply-training set built from Bitext customer-support messages.

## Prompt format

Training and inference must use exactly this format.

- **Chat template:** Qwen3's, with `enable_thinking=False` and **no system message**.
- **User message:**
  - `State: {json}`
  - `Options:`
  - one line per option: `[1] NAME: description`, `[2] ...`
  - `Select one option. Answer only with its bracketed numeric identifier.`
  - When `REPLY_WITH_TEXT` is among the options, the same line continues: ` If you select REPLY_WITH_TEXT, write a 1-2 sentence reply to the customer after the identifier.`
- **Assistant turn:** pre-filled with `Best answer: [`.
- **Decision:**
  - Option *k* is the token suffix `k]`. Its score is the sum of the log-probabilities of those tokens, and a softmax over the scores gives the option probabilities.
  - The `]` makes the suffixes prefix-free, so any number of options works.
- **Reply:** if `REPLY_WITH_TEXT` wins (or confidence < τ), append `k] Reply:` to the same KV cache and decode greedily until `<|im_end|>`.

```python
import json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
model = PeftModel.from_pretrained(model, "drusama1979/decision_plus").merge_and_unload().eval()

state = {"customer_msg": "I want to change the delivery address on order 4417.", "order_status": "processing"}
options = ["UPDATE_ADDRESS: Change the shipping address", "CANCEL_ORDER: Cancel the order",
           "TRACK_ORDER: Send tracking details", "REPLY_WITH_TEXT: Write a 1-2 sentence reply"]
user = "\n".join([f"State: {json.dumps(state)}", "Options:"] + [f"[{i+1}] {o}" for i, o in enumerate(options)]
                 + ["Select one option. Answer only with its bracketed numeric identifier. If you select "
                    "REPLY_WITH_TEXT, write a 1-2 sentence reply to the customer after the identifier."])
prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False) + "Best answer: ["

p = tok.encode(prompt, add_special_tokens=False)
scores = []
with torch.no_grad():
    for k in range(1, len(options) + 1):                      # score each suffix "k]"
        full = tok.encode(prompt + f"{k}]", add_special_tokens=False)
        logp = model(torch.tensor([full])).logits[0].log_softmax(-1)
        scores.append(sum(logp[t - 1, full[t]] for t in range(len(p), len(full))))
probs = torch.stack(scores).softmax(-1)
print({o.split(":")[0]: round(float(x), 3) for o, x in zip(options, probs)})
```

The loop above is the simple way to score: one forward pass per option. The GitHub code does the same with **one** prefill plus one batched step on copies of the prompt cache, and writes the reply from that cache.

## Results

- **Hardware:** Apple M4 Pro, `mps`.
- **Precision:** JevBench in FP32, as in the paper. Everything else in bf16.

**Decisions on public benchmarks:**

| | Base Qwen3-0.6B | + this adapter |
|---|---|---|
| JevBench public items (231), accuracy | 55.4% | 54.5% |
| JevBench, expected calibration error (ECE) | 0.292 | **0.169** |
| JevBench, Brier | 0.674 | **0.575** |
| Banking77 test (3,080 rows × 77 options), accuracy | 18.5% | **53.5%** |
| Banking77, ECE | 0.460 | **0.079** |

On JevBench, accuracy does not change significantly: 18 items won, 20 lost, sign test p = 0.87.

**The reply option:**

| | Base | + this adapter |
|---|---|---|
| 300 held-out Bitext messages, accuracy | 58.7% | **98.3%** |
| Held-out Bitext, ECE | 0.389 | **0.020** |
| Held-out Bitext, reply ROUGE-L vs Bitext's reply | 0.099 | **0.231** |
| 97 hand-written cases (out of distribution), accuracy | 47.4% | **78.4%** |

**Latency** (one model vs a second call that re-reads the prompt, 32 reply tokens, median of 5):

| Prompt tokens | One model | Two calls | Saved |
|---|---|---|---|
| 367 | 414 ms | 488 ms | 15% |
| 1,159 | 671 ms | 901 ms | 26% |
| 4,161 | 1,824 ms | 2,891 ms | 37% |

## Training

- **Recipe:** the LLM-as-Jev objective:
  - tree-factorized listwise loss on the gold suffix;
  - KL anchors to the frozen base model (legal mass, illegal-token distribution, reply start, after `]`), λ = 1, top-64 cached reference statistics.
- **Our additions to the objective:**
  - a Brier term on the option distribution;
  - for reply rows, a language-model loss on ` Reply: <text>` plus a KL term to the base model on the reply tokens (weight 0.5).
- **LoRA:**
  - r = 16, α = 32, on all attention and MLP projections;
  - lr 1e-4 with 10% warm-up and linear decay;
  - batch 32, one epoch (375 steps), FP32, about 90 minutes on the M4 Pro.
- **Data (12,000 rows):**
  - **9,000 rows:** the paper's Reason-9k mix, 1,000 each from CLINC150, MASSIVE, Bitext, ReClor, LogiQA 2.0, CosmosQA, PIQA, αNLI and ProofWriter.
  - **3,000 reply-task rows** built from Bitext customer-support messages:
    - 50%: the right action is offered;
    - 35%: an information question, so the answer is a reply;
    - 15%: the right action is not offered, so the answer is a reply.

## Limitations

- **The reply option attracts unfamiliar actions.** Fraud, safety and legal escalations often go to `REPLY_WITH_TEXT`, at 80–96% confidence. Those concepts are absent from the training data.
- **Replies don't use the account state.** The model asks for details that are already in the State. It also chose an address change on an already-shipped order: accuracy on state conflicts is 37.5% (3 of 8).
- **Reply style comes from Bitext.** Expect `{{Order Number}}`-style placeholders and some list fragments.
- **Narrow evaluation:**
  - English only, a single seed, 0.6B only;
  - the paper's long-context training sets are not included;
  - the JevBench public set has 231 items, too few to detect small changes.
- **Not for unattended high-stakes actions.** Keep a human or a rule check in front of anything like refunds or account locks.

## License

- **The adapter weights in this repository:** **CC BY-NC 4.0** (see `LICENSE`). Non-commercial use only, with attribution.
- **The base model Qwen3-0.6B:** not distributed here. It is licensed separately by its authors under Apache-2.0.
- **Training-data licenses:**
  - CLINC150: CC BY 3.0
  - MASSIVE: CC BY 4.0
  - Bitext: CDLA-Sharing-1.0
  - ReClor: research use only
  - LogiQA 2.0: CC BY-NC-SA 4.0
  - CosmosQA: CC BY 4.0
  - PIQA: AFL-3.0
  - αNLI: license not stated
  - ProofWriter: CC BY 4.0

  Because ReClor and LogiQA 2.0 are in the training mix, the adapter is for research and non-commercial use only.
