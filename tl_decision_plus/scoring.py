"""Training-free readout (LLM-as-Jev Eq. 1): s_k = sum_t log p(c_k,t | x, c_k,<t), P = softmax(s).

The prompt is prefilled once. The first suffix token of every candidate is read from the prefill logits;
the remaining suffix tokens are scored in one batched teacher-forced step on copies of the prompt KV cache
(chunked so memory stays bounded for 100+ options). The original cache is left untouched, so the reply
path can continue from it.
"""
import torch
import torch.nn.functional as F
from transformers import DynamicCache


def expanded_cache(cache, n: int) -> DynamicCache:
    """A new cache holding n copies of the prompt cache along the batch axis (on-device; the original is untouched)."""
    new = DynamicCache()
    for i, layer in enumerate(cache.layers):
        new.update(layer.keys.repeat_interleave(n, 0), layer.values.repeat_interleave(n, 0), i)
    return new


@torch.no_grad()
def score_suffixes(model, prompt_ids: torch.Tensor, suffixes: list[list[int]], pad_id: int, chunk: int = 32):
    """prompt_ids: [1, T] on the model device. Returns (probs[K], scores[K], root_legal_mass, prefill_out)."""
    out = model(input_ids=prompt_ids, use_cache=True)
    root_logp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
    first = torch.tensor([s[0] for s in suffixes], device=prompt_ids.device)
    scores = root_logp[first].clone()
    root_mass = float(root_logp[torch.unique(first)].exp().sum())

    rest = [i for i, s in enumerate(suffixes) if len(s) > 1]
    for c in range(0, len(rest), chunk):
        idx = rest[c : c + chunk]
        L = max(len(suffixes[i]) for i in idx) - 1
        inp = torch.full((len(idx), L), pad_id, dtype=torch.long, device=prompt_ids.device)
        for r, i in enumerate(idx):
            inp[r, : len(suffixes[i]) - 1] = torch.tensor(suffixes[i][:-1], device=prompt_ids.device)
        kv = expanded_cache(out.past_key_values, len(idx))
        T = prompt_ids.shape[1]
        mask = torch.ones((len(idx), T + L), dtype=torch.long, device=prompt_ids.device)  # right padding: harmless
        step = model(input_ids=inp, attention_mask=mask, past_key_values=kv, use_cache=False)
        logp = F.log_softmax(step.logits.float(), dim=-1)  # [n, L, V]
        for r, i in enumerate(idx):
            s = suffixes[i]
            for t in range(1, len(s)):  # token t is predicted at position t-1 of the step
                scores[i] += logp[r, t - 1, s[t]]
        del kv, step, logp

    probs = F.softmax(scores, dim=-1)
    return probs.cpu(), scores.cpu(), root_mass, out
