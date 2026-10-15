"""DecisionEngine: one prefill scores every option's suffix "k]" (Mode 1). Only if REPLY_WITH_TEXT wins,
or confidence is below the threshold, it continues from the same prompt KV cache to write 1-2 lines (Mode 2)."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tl_decision_plus.device import pick_device, sync
from tl_decision_plus.prompt import REPLY_NAME, build_prompt, option_name, reply_token_ids, suffix_token_ids
from tl_decision_plus.scoring import score_suffixes


DEFAULT_BASE = "Qwen/Qwen3-0.6B"


def base_of(adapter_path, model_name=None):
    """The base model to load: the explicit name, else the one the adapter was trained on, else Qwen3-0.6B."""
    if model_name:
        return model_name
    if adapter_path:
        from peft import PeftConfig

        return PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    return DEFAULT_BASE


class DecisionEngine:
    def __init__(self, model_name=None, adapter_path=None, device=None, dtype=torch.bfloat16):
        model_name = base_of(adapter_path, model_name)
        self.model_name = model_name
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|endoftext|>")}

    @torch.no_grad()
    def decide(self, state, options, threshold: float = 0.65, max_reply_tokens: int = 48, min_reply_tokens: int = 0):
        tok = self.tokenizer
        names = [option_name(o) for o in options]
        reply_idx = names.index(REPLY_NAME) + 1 if REPLY_NAME in names else None

        sync(self.device)
        t0 = time.perf_counter()

        # ---- Mode 1: one prefill + one batched step scores every suffix "k]" -------------
        prompt = build_prompt(tok, state, options)
        pids = tok.encode(prompt, add_special_tokens=False)
        sufs = suffix_token_ids(tok, prompt, len(options), pids)
        probs, _, root_mass, out = score_suffixes(self.model, torch.tensor([pids], device=self.device), sufs, tok.pad_token_id)
        probs = probs.tolist()
        best = max(range(len(probs)), key=probs.__getitem__)

        sync(self.device)
        decide_ms = (time.perf_counter() - t0) * 1000
        result = {
            "decision": names[best],
            "probability": round(probs[best], 4),
            "probabilities": {n: round(p, 4) for n, p in zip(names, probs)},
            "mass_on_options": round(root_mass, 4),
            "prompt_tokens": len(pids),
            "decide_ms": round(decide_ms, 1),
        }
        reply_won = reply_idx is not None and best + 1 == reply_idx
        if reply_idx is None or (not reply_won and probs[best] >= threshold):
            result.update(mode="MODE_1_DECISION", text=None, reply_tokens=0, total_ms=round(decide_ms, 1))
            return result

        # ---- Mode 2: continue from the SAME prompt cache; only "k] Reply:" and the reply are processed ----
        trigger = "REPLY_OPTION_WON" if reply_won else "LOW_CONFIDENCE_FALLBACK"
        kv = out.past_key_values
        assert kv.get_seq_length() == len(pids), "KV cache does not hold exactly the prompt"
        generated = self._generate(pids, reply_token_ids(tok, prompt, reply_idx, None, pids), kv, max_reply_tokens,
                                   min_reply_tokens)

        sync(self.device)
        result.update(
            mode=f"MODE_2_REPLY ({trigger})",
            decision=REPLY_NAME,
            text=tok.decode(generated, skip_special_tokens=True).strip(),
            reply_tokens=len(generated),
            total_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return result

    def _generate(self, pids, cont, kv, max_new_tokens, min_new_tokens=0):
        """Greedy reply after prompt + "k] Reply:". With kv (the prompt cache) only the new tokens are processed;
        with kv=None the whole prompt is read again. Both paths use the same generate() loop."""
        if max_new_tokens <= 0:
            return []
        ids = torch.tensor([pids + cont], device=self.device)
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=kv,
                                  max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, do_sample=False,
                                  eos_token_id=list(self.stop_ids), pad_token_id=self.tokenizer.pad_token_id)
        return [t for t in out[0, ids.shape[1]:].tolist() if t not in self.stop_ids]

    @torch.no_grad()
    def reference_reply(self, state, options, max_reply_tokens: int = 48, min_new_tokens: int = 0) -> str:
        """The same reply written by a second call that re-reads the whole prompt (a two-call pipeline)."""
        reply_idx = [option_name(o) for o in options].index(REPLY_NAME) + 1
        prompt = build_prompt(self.tokenizer, state, options)
        pids = self.tokenizer.encode(prompt, add_special_tokens=False)
        cont = reply_token_ids(self.tokenizer, prompt, reply_idx, None, pids)
        return self.tokenizer.decode(self._generate(pids, cont, None, max_reply_tokens, min_new_tokens),
                                     skip_special_tokens=True).strip()
