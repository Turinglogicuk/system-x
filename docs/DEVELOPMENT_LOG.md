# TL_decision_plus

A small decision model in the style of [Jev](https://yourstory.com/ai-story/what-is-jev-typesafe-ai-decision-model), with one extra option: it can also reply.

Give it a situation and a numbered list of options. In **one prefill** it scores every option and returns calibrated probabilities, like Jev. One option is special: `REPLY_WITH_TEXT`. If that option wins, or the model isn't confident enough about any action, the **same model** continues from the KV cache it already built and writes a 1–2 sentence reply. It never reads the prompt twice, and you never need a second model.

```
State + options ──► 1 prefill + 1 batched step ──► P([1]) P([2]) P([3]) ...
                                                     │
             action wins, confident ─────────────────┴──► return the decision, no text
             REPLY wins / not confident ────────────────► continue from the prompt's KV cache,
                                                          write 1–2 lines
```

**v0.2 builds on "LLM-as-Jev"** (Li & Wagle, arXiv:2610.02076):
- the same prompt (`State / Options / Select one option…`, assistant pre-filled with `Best answer: [`);
- the same readout: the full suffix `k]` is scored, so any number of options works;
- the same training objective: tree-factorized listwise loss + KL anchors;
- the same public training datasets.

What this project adds:
- the reply option, with its reply decoded from the decision's KV cache;
- a confidence fallback to that reply;
- a Brier calibration term;
- a reply-training set built from real Bitext customer-support messages.

## Run it

```bash
uv sync                                        # Python 3.12, torch, transformers, peft, pandas
uv run data/prepare_v2.py                      # → data/v2/train.jsonl (12k), heldout_reply.jsonl (300), handwritten.jsonl (97)
uv run train_v2.py                             # base-model reference pass (~18 min) + LoRA training (~90 min on an M4 Pro) → out/v2
uv run bench.py --task jevbench --fp32 --adapter out/v2   # JevBench public items (231)
uv run bench.py --task banking77 --adapter out/v2         # Banking77 test (3,080 × 77 options)
uv run eval_v2.py --adapter out/v2             # reply task: held-out Bitext + hand-written cases
uv run latency.py --adapter out/v2             # one model vs two calls, by prompt length
uv run demo.py --adapter out/v2 --fp32         # 3 cases; checks the cached reply equals a full re-read
uv run server.py                               # web console → http://localhost:8777
```

Leave out `--adapter` to run the untrained model.

## Use it in code

```python
from tl_decision_plus.engine import DecisionEngine
from tl_decision_plus.options import make_options

eng = DecisionEngine(adapter_path="out/v2")
opts = make_options(["AUTO_REFUND", "BLOCK_FRAUD", "ESCALATE_HUMAN", "REPLY_WITH_TEXT"])
r = eng.decide({"customer_msg": "I was charged twice", "charges": [20, 20]}, opts, threshold=0.65)
r["decision"], r["probability"], r["text"]   # text is None unless it replied
```

Options can be strings or `{"name", "desc"}` dicts, and any number of them is allowed. Without `REPLY_WITH_TEXT` among the options, the prompt is exactly the paper's and the engine only decides.

## Results (v0.2)

- **Setup:** Apple M4 Pro (24 GB), `mps`, Qwen3-0.6B + LoRA (r16, α32).
- **Training:** 375 steps × 32 rows, one epoch, lr 1e-4, λ = 1.
- **Precision:** JevBench in FP32, as in the paper. Everything else in bf16.

**Decisions on the paper's benchmarks** (the paper's numbers are for Qwen3-0.6B):

| | Untrained | v0.2 | Paper: untrained | Paper: Reason-9k (our mix, full FT) | Paper: best JevBench run (LoRA, λ=0.01) |
|---|---|---|---|---|---|
| JevBench public, accuracy | 55.4% | 54.5% | 56.7% | 55.0% | 61.0% |
| JevBench, calibration error (ECE) | 0.292 | **0.169** | 0.278 | 0.118 | 0.121 |
| JevBench, Brier | 0.674 | **0.575** | — | — | — |
| Banking77, accuracy (77 options) | 18.5% | **53.5%** | 22.2% | 60.4% | 59.0% |
| Banking77, ECE | 0.460 | **0.079** | — | — | — |

- **Comparing like with like:** our Reason-9k run used LoRA; the paper's Reason-9k row is full fine-tuning. The paper finds the two within one JevBench item of each other at 0.6B (λ = 1), so the comparison holds. The paper's lowest 0.6B ECE is 0.088 (Reason-30k), and its highest 0.6B Banking77 is 64.0% (Intent-1ep).
- **JevBench accuracy does not move:** 18 items won, 20 lost against the untrained model, sign test p = 0.87. This matches the paper's finding that this training barely changes broad JevBench accuracy at 0.6B.
- **Calibration and Banking77 improve strongly.**
- **Adding the reply task did not hurt general decisions.**

**The reply option**, measured on 300 Bitext messages never seen in training and on our 97 hand-written cases:

| | Untrained | v0.2 |
|---|---|---|
| Held-out Bitext, decision accuracy | 58.7% | **98.3%** |
| Held-out Bitext, ECE | 0.389 | **0.020** |
| Held-out Bitext, reply ROUGE-L vs Bitext's reply | 0.099 | **0.231** |
| Hand-written (out of distribution), accuracy | 47.4% | **78.4%** |
| Hand-written: replies / vague / action-not-offered | 100% | 100% |
| Hand-written: actions | 10.7% | 67.9% |
| Hand-written: state conflicts (e.g. shipped order) | 87.5% | 37.5% |

Without training, the model almost always picks the reply option (91% of rows).

**Latency: one model vs two calls.**
- Both paths write exactly 32 reply tokens with the same `generate()` loop.
- "Two calls" re-reads the prompt for the reply, as a pipeline of a decision model plus a separate reply call does.
- Medians of 5 runs.

| Prompt tokens | One model | Two calls | Saved |
|---|---|---|---|
| 367 | 414 ms | 488 ms | 74 ms (15%) |
| 679 | 508 ms | 639 ms | 131 ms (21%) |
| 1,159 | 671 ms | 901 ms | 230 ms (26%) |
| 2,159 | 1,019 ms | 1,494 ms | 476 ms (32%) |
| 4,161 | 1,824 ms | 2,891 ms | 1,067 ms (37%) |

- **The saving is one prefill, so it grows with prompt length.** A real pipeline would also keep a second model in memory.
- **On the real held-out Bitext messages** (short prompts, natural reply lengths): 292 ms with one model vs 325 ms with two calls (median, 60 rows).
- **The cached reply is exact.** In FP32 it is token-for-token identical to a full re-read (`demo.py --fp32`).

## Honest limits (v0.2)

- **The reply option attracts unfamiliar actions.** On the hand-written set, almost all errors go *to* REPLY_WITH_TEXT: fraud, safety and legal escalations, and refunds. These concepts are absent from Bitext's 27 intents, so the model treats replying as the safe default, often at 80–96% confidence. This mirrors the paper's "avoids none-of-these" failure.
- **Replies don't use the account state.** Bitext replies never refer to account facts, so the model asks for details instead of reading them from the State (for example, an `invoice_note`). It also chose UPDATE_ADDRESS on an already-shipped order. That is the state-conflict drop: 87.5% → 37.5%.
- **Reply style.** Replies inherit Bitext's style: `{{Order Number}}`-style placeholders, some list fragments, and an occasional echo of the instruction ("With a 1-2 sentence reply, …").
- **Deviations from the paper:**
  - off-path trie nodes are not anchored;
  - no long-context datasets (ContractNLI, MuSiQue, QuALITY), so v0.2 is "Reason-9k + reply", not "Reason+Long-12k";
  - single seed;
  - Banking77 run in bf16.
- **The comparison with v0.1 is not like-for-like.** v0.1 scored 95.9% on the hand-written set, but its templated training data was written in the same style as those tests. v0.2's 78.4% is a fairer out-of-distribution number.

## v0.3: broader support data

**What changed:** v0.3 keeps v0.2's 12,000 rows and adds 11,496 rows from three public sources.
- **ABCD** (MIT; 5,000 rows): the next step in real support dialogues. Account facts appear in the State only after the lookup that reveals them.
- **10 Bitext industry sets** (CDLA-Sharing; 5,000 rows). Template slots in the replies are filled from facts placed in the State.
- **CFPB consumer complaints** (CC0; 1,496 rows): fraud plus 7 non-fraud issues as negatives.

**Training:** same recipe as v0.2, 735 steps, 32 minutes on an RTX 5090.

**Build and run:**
```bash
# ABCD data: github.com/asappresearch/abcd → data/raw_v3/{abcd_v1.1.json.gz,utterances.json}
uv run data/prepare_v3.py                       # → data/v3/train.jsonl (23,496) + held-out abcd/verticals/cfpb
uv run train_v2.py --train data/v3/train.jsonl --out out/v3 --ref out/v3_ref.pt --token-budget 16000
uv run eval_v2.py --adapter out/v3 --tag _v3sets --sets abcd=data/v3/heldout_abcd.jsonl verticals=data/v3/heldout_verticals.jsonl cfpb=data/v3/heldout_cfpb.jsonl
```

**Decisions** (argmax accuracy; same machine and settings as v0.2):

| | Untrained | v0.2 | v0.3 |
|---|---|---|---|
| ABCD next step (300 held-out) | 52.3% | 51.7% | **88.7%** (ECE 0.037) |
| Bitext industries (300) | 58.3% | 86.3% | **97.0%** |
| CFPB complaints (199) / fraud only (80) | 39.2% / 30.0% | 64.3% / 57.5% | **86.9% / 90.0%** |
| Banking77 | 18.5% | 53.5% | **57.2%** |
| JevBench (FP32) | 55.4% | 54.5% | 52.4% (ECE 0.201; vs untrained p = 0.35) |
| Hand-written (97) | 47.4% | 78.4% | 76.3% (vs v0.2 p = 0.69) |

Gains are large on everything the new data covers. JevBench and the hand-written cases don't improve, and the hand-written fraud cases still mostly go to "reply". Customer-support rows are now 66% of the mix (v0.2: 33%).

**Reply quality.** Public decision benchmarks don't score replies, so we use four checks:
- **Automatic checks** (`reply_checks.py`):
  - on the industry set, v0.3 copies the needed State fact in **46%** of cases (v0.2: 29%);
  - instruction echo is down to 1–3% on the new sets;
  - invented facts are rare.
- **ABCD utterance ranking** (`abcd_rank.py`; 100 candidates, PMI scoring): right reply ranked first **15.0%** (v0.2 8.0%, untrained 7.2%); in the top 10 49.4%. ABCD's own trained retrieval baselines score 21.7–22.0% top-1.
- **LLM judges** (`reply_gen.py`, `judge.py`, `judge_report.py`):
  - **Gemma 4 31B, 150 rows:** v0.3 beats the untrained model (101–13) and the same 0.6B model replying in a separate second call (98–23). It ties v0.2 (60–49), and loses to a Qwen3-1.7B second call (32–88) and to the reference replies (35–94).
  - **Claude, as a second blind judge on 100 pairs** (`judge_agreement.py`): agrees with Gemma on 69% of pairs (κ 0.51) and gives the same ordering.
- **General abilities** (lm-evaluation-harness, the paper's suite; our untrained IFEval 55.5% matches the paper's 55.0%):
  - GSM8K, TriviaQA and LAMBADA are flat, and WikiText perplexity improves (26.1 → 24.1).
  - **IFEval drops from 55.5% to 45.0%** (6 prompts won, 27 lost, p = 0.0003). The extra support data costs instruction following; v0.2's 52.0% is within noise.

**Known data issue:** 113 industry reply targets (0.5%) say "app app" because of slot filling.

## v0.4: Qwen3-1.7B, rebalanced data, teacher-rewritten replies

**What changed:**
- **Backbone:** Qwen3-1.7B.
- **Data:** twice the reasoning data and a smaller support share (42%).
- **Reply targets:** every Bitext and industry reply target was rewritten by Qwen3-14B (local Ollama, thinking off). The teacher kept each reference's intent, used State facts and added nothing new. ABCD replies are real agent messages and were kept as written.

**Training** (RTX 5090): 1.7B in 57 min; a 0.6B control on the same data (`out/v4_06`) in 29 min.

**Build and run:**
```bash
ssh -f -N -L 11435:localhost:11434 <gpu-host>     # teacher served by Ollama on a GPU machine
uv run data/prepare_v4.py --teacher-url http://localhost:11435/api/chat   # → data/v4/train.jsonl (24,256)
uv run train_v2.py --base Qwen/Qwen3-1.7B --train data/v4/train.jsonl --out out/v4 --ref out/v4_ref.pt --token-budget 16000
```

`DecisionEngine` and every evaluation script read the base model from the adapter; use `--base` for an untrained model.

| | Untrained 1.7B | **v0.4 (1.7B)** | v0.4-0.6B control | v0.3 (0.6B) |
|---|---|---|---|---|
| JevBench (FP32) | 55.0% (ECE .404) | **61.0% (.123)**, p = 0.04 | 51.5% | 52.4% |
| Banking77 | 28.3% | **65.2%** | 55.5% | 57.2% |
| Hand-written (97) | 72.2% | **86.6%** (ECE .070) | 84.5% | 76.3% |
| ABCD / industries / CFPB | 48.3 / 74.7 / 53.3% | 86.0 / 97.0 / 85.9% | 88.3 / 96.7 / 87.9% | 88.7 / 97.0 / 86.9% |
| IFEval (chat) | 66.0% | 64.5% (no significant change) | 49.5% | 45.0% |
| ABCD reply ranking, top-1 | 2.2% | **17.6%** | 13.2% | 15.0% |

**Notes on the decisions:**
- The hand-written fraud, escalation and refund cases are now mostly right (9/10, 9/9, 10/10).
- **State conflicts are still the main miss** (2/8).

**Reply quality** (150 rows):
- **Gemma judge:** v0.4 has the best helpfulness (3.92/5) and is judged sendable most often (57%). It beats the references (79–44, p = 0.002). It beats the unadapted 1.7B second call 66–48, which is not significant (p = 0.11).
- **Claude (`claude-opus-5-5`), blind:** prefers v0.4 to that second call, 31–11 (p = 0.003).
- **Caveat:** these comparisons differ in training and prompts. Cache reuse itself is quality-neutral (token-identical in FP32), and the quality gain comes from training on the teacher's targets.

**Latency at 1.7B** (M4 Pro): the cached reply saves 17% at 367 tokens and 39% at 4,161 tokens.

**Known issue:** when a customer asks for a phone number the State doesn't contain, the model sometimes invents a placeholder one ("1-800-123-4567").

## Trained weights

The released model is **System X v4** (Qwen3-1.7B) at [TuringLogic/system-x](https://huggingface.co/TuringLogic/system-x), under **CC BY-NC 4.0 (non-commercial only)**. Its model card is `hf/README.md`. Earlier versions (v0.1–v0.3) were development checkpoints and are not released.

Training-data licenses (needed for any release):
- CLINC150: CC BY 3.0
- MASSIVE: CC BY 4.0
- Bitext: CDLA-Sharing-1.0
- ReClor: research use only
- LogiQA 2.0: CC BY-NC-SA 4.0
- CosmosQA: CC BY 4.0
- PIQA: AFL-3.0
- αNLI: not stated
- ProofWriter: CC BY 4.0
- JevBench public items: MIT

## Files

| File | What it does |
|---|---|
| `tl_decision_plus/prompt.py` | The one prompt format (the paper's, plus the reply sentence), used everywhere |
| `tl_decision_plus/scoring.py` | Training-free readout: one prefill, batched scoring of every suffix `k]` from the prompt cache |
| `tl_decision_plus/engine.py` | `DecisionEngine.decide()`: Mode 1 decision, Mode 2 reply from the same cache |
| `data/prepare_v2.py` | Builds the training mixture from the public datasets plus the Bitext reply task |
| `train_v2.py` | Tree-factorized listwise loss + KL anchors (cached top-64) + Brier + reply LM/KL |
| `bench.py` | JevBench and Banking77 in the paper's format (accuracy, 15-bin ECE, Brier) |
| `eval_v2.py`, `latency.py` | Reply-task evaluation, and the one-model vs two-call latency test |
| `data/prepare_v4.py` | v0.4 mixture: more reasoning, less support, teacher-rewritten reply targets (cached) with automatic checks |
| `data/prepare_v3.py` | v0.3 mixture: v0.2 rows + ABCD + Bitext industry sets + CFPB, with a leak check against the hand-written set |
| `reply_checks.py` | Automatic reply checks: grounding in the State, invented facts, template slots, instruction echo |
| `abcd_rank.py` | ABCD utterance ranking (100 candidates, PMI / mean / sum scoring) |
| `reply_gen.py`, `judge.py`, `judge_report.py`, `judge_agreement.py` | Replies from every system on the same rows, the blind LLM judge, its summary, and the second-judge agreement |
| `rate_server.py`, `web/rate.html` | Optional blind A/B rating page for human raters (port 8778) |
| `server.py`, `web/index.html` | Local web console |
