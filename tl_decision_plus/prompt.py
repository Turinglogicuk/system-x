"""One prompt format, shared by the engine, training and every evaluation.

It follows LLM-as-Jev (arXiv:2610.02076, Fig. 1a and Appendix A): default chat template, thinking off,
user message "State / Options / Select one option...", assistant turn pre-filled with "Best answer: [".
Option k is the suffix "k]". The closing bracket makes the suffixes prefix-free (1] is not a prefix of 12]),
so there is no limit on the number of options.

Our addition: when REPLY_WITH_TEXT is among the options, one extra sentence tells the model it may
write a short reply after that identifier. Without it, the prompt is exactly the paper's.
"""
import json

REPLY_NAME = "REPLY_WITH_TEXT"
INSTRUCTION = "Select one option. Answer only with its bracketed numeric identifier."
REPLY_INSTRUCTION = f" If you select {REPLY_NAME}, write a 1-2 sentence reply to the customer after the identifier."
PREFILL = "Best answer: ["
REPLY_LEAD = " Reply:"  # what follows "k]" when the reply option is chosen; the reply text starts with a space


def render_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def option_text(o) -> str:
    """Options are plain strings, or {"name", "desc"} dicts rendered as "NAME: desc"."""
    if isinstance(o, str):
        return o
    return f"{o['name']}: {o['desc']}" if o.get("desc") else o["name"]


def option_name(o) -> str:
    return o if isinstance(o, str) else o["name"]


def build_prompt(tokenizer, state, options) -> str:
    """Full prompt string ending in 'Best answer: ['."""
    assert len(options) >= 2, "need at least 2 options"
    lines = [f"State: {render_state(state)}", "Options:"]
    lines += [f"[{i + 1}] {option_text(o)}" for i, o in enumerate(options)]
    has_reply = any(option_name(o) == REPLY_NAME for o in options)
    lines.append(INSTRUCTION + (REPLY_INSTRUCTION if has_reply else ""))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "\n".join(lines)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return text + PREFILL


def suffix_token_ids(tokenizer, prompt: str, n_options: int, prompt_ids: list[int] | None = None) -> list[list[int]]:
    """Token ids of each candidate suffix "k]", taken from the tokenization of prompt + suffix (paper App. C)."""
    p = prompt_ids if prompt_ids is not None else tokenizer.encode(prompt, add_special_tokens=False)
    out = []
    for k in range(1, n_options + 1):
        full = tokenizer.encode(prompt + f"{k}]", add_special_tokens=False)
        assert full[: len(p)] == p, f"prompt tokens are not a prefix of prompt + '{k}]'"
        out.append(full[len(p):])
    return out


def reply_token_ids(tokenizer, prompt: str, reply_idx: int, reply_text: str | None = None,
                    prompt_ids: list[int] | None = None) -> list[int]:
    """Tokens after the prompt for 'k] Reply:' (+ ' text' if given), checked to extend the prompt cleanly."""
    p = prompt_ids if prompt_ids is not None else tokenizer.encode(prompt, add_special_tokens=False)
    s = f"{reply_idx}]{REPLY_LEAD}" + (f" {reply_text}" if reply_text else "")
    full = tokenizer.encode(prompt + s, add_special_tokens=False)
    assert full[: len(p)] == p, "prompt tokens are not a prefix of the reply continuation"
    return full[len(p):]
