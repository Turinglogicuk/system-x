"""Build the v0.2 training mixture from the public, human-annotated datasets LLM-as-Jev uses.

Decision rows (the paper's "Reason-9k" mixture, Table 3): 1,000 examples each from
  intent:    CLINC150 (151 options), MASSIVE en-US (60), Bitext customer support (27)
  reasoning: ReClor, LogiQA 2.0, CosmosQA (none-of-the-above upsampled to 16%), PIQA, αNLI, ProofWriter (balanced)
Every option list is shuffled (paper §3.4). No reply option on these rows, so their prompt is exactly the paper's.

Reply-task rows (our addition), built from Bitext's real customer messages:
  action intent + its action offered      → that action
  information intent (refund policy, ...)  → REPLY_WITH_TEXT, target = 1-2 informative sentences of Bitext's reply
  action intent + its action NOT offered   → REPLY_WITH_TEXT, target = same
Bitext instructions are split into train / held-out by normalised text, so no held-out message is trained on.

Run: uv run data/prepare_v2.py   → data/v2/train.jsonl, data/v2/heldout_reply.jsonl, data/v2/handwritten.jsonl
"""
import json
import random
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OUT = Path(__file__).parent / "v2"
N = 1000
REPLY = "REPLY_WITH_TEXT"
REPLY_DESC = "Write a 1-2 sentence reply (answer a question, or explain what the customer can do)"


def dl(repo, f, rev=None):
    return hf_hub_download(repo, f, repo_type="dataset", revision=rev)


def label_names(path, col):
    meta = json.loads(pq.read_schema(path).metadata[b"huggingface"])
    return meta["info"]["features"][col]["names"]


def shuffled(options, gold, rng):
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(gold)


def row(src, state, options, gold, rng, reply=None, kind="decision"):
    opts, g = shuffled(options, gold, rng)
    return {"src": src, "kind": kind, "state": state, "options": opts, "label": g, "reply": reply}


# ---------------------------------------------------------------- intent sets
def intent_rows(src, texts, labels, names, rng, n=N):
    names_h = [x.replace("_", " ") for x in names]
    idx = rng.sample(range(len(texts)), n)
    return [row(src, texts[i], names_h, labels[i], rng) for i in idx]


def clinc(rng):
    p = dl("clinc/clinc_oos", "plus/train-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    return intent_rows("clinc150", df.text.tolist(), df.intent.astype(int).tolist(), label_names(p, "intent"), rng)


def massive(rng):
    p = dl("AmazonScience/massive", "en-US/train/0000.parquet", "refs/convert/parquet")
    df = pd.read_parquet(p)
    return intent_rows("massive", df.utt.tolist(), df.intent.astype(int).tolist(), label_names(p, "intent"), rng)


# ---------------------------------------------------------------- reasoning sets
def reclor(rng):
    df = pd.read_json(dl("metaeval/reclor", "train.json"))
    idx = rng.sample(range(len(df)), N)
    return [row("reclor", f"{df.context[i]}\n\nQuestion: {df.question[i]}", list(df.answers[i]), int(df.label[i]), rng) for i in idx]


def logiqa2(rng):
    rows = [json.loads(l) for l in open(dl("datatune/LogiQA2.0", "MRC/train.txt"))]
    out = []
    for r in rng.sample(rows, N):
        out.append(row("logiqa2", f"{r['text']}\n\nQuestion: {r['question']}", r["options"], int(r["answer"]), rng))
    return out


def cosmosqa(rng):
    df = pd.read_parquet(dl("allenai/cosmos_qa", "default/train/0000.parquet", "refs/convert/parquet"))
    ans = df[["answer0", "answer1", "answer2", "answer3"]].values.tolist()
    lab = df.label.astype(int).tolist()
    none = [i for i in range(len(df)) if "none of the above" in ans[i][lab[i]].lower()]
    other = [i for i in range(len(df)) if i not in set(none)]
    k = int(0.16 * N)  # paper: "none of the above" upsampled to 16%
    idx = rng.sample(none, k) + rng.sample(other, N - k)
    return [row("cosmosqa", f"{df.context[i]}\n\nQuestion: {df.question[i]}", ans[i], lab[i], rng) for i in idx]


def piqa(rng):
    df = pd.read_parquet(dl("ybisk/piqa", "plain_text/train/0000.parquet", "refs/convert/parquet"))
    idx = rng.sample(range(len(df)), N)
    return [row("piqa", f"Goal: {df.goal[i]}\n\nWhich solution achieves the goal?", [df.sol1[i], df.sol2[i]], int(df.label[i]), rng) for i in idx]


def anli(rng):
    df = pd.read_parquet(dl("allenai/art", "anli/train-00000-of-00001.parquet"))
    idx = rng.sample(range(len(df)), N)
    return [row("anli", f"Beginning: {df.observation_1[i]}\nEnding: {df.observation_2[i]}\n\nWhich explanation of what happened in between is most plausible?",
                [df.hypothesis_1[i], df.hypothesis_2[i]], int(df.label[i]) - 1, rng) for i in idx]


def proofwriter(rng):
    df = pd.read_parquet(dl("tasksource/proofwriter", "data/train-00000-of-00002-6176cf3d78f48858.parquet"))
    labels = ["True", "False", "Unknown"]
    out = []
    for j, lab in enumerate(labels):  # balanced true / false / unknown
        sub = df[df.answer == lab]
        take = N // 3 + (1 if j < N % 3 else 0)
        for i in rng.sample(range(len(sub)), take):
            r = sub.iloc[i]
            out.append(row("proofwriter", f"{r.theory}\n\nBased only on these facts and rules, is the following statement true, false, or unknown?\n{r.question}",
                           labels, labels.index(lab), rng))
    return out


# ---------------------------------------------------------------- Bitext: intent rows + reply task
ACTION_OF = {
    "cancel_order": ("CANCEL_ORDER", "Cancel the customer's order"),
    "change_order": ("CHANGE_ORDER", "Change the items in an existing order"),
    "change_shipping_address": ("UPDATE_ADDRESS", "Add or change the shipping address"),
    "set_up_shipping_address": ("UPDATE_ADDRESS", "Add or change the shipping address"),
    "contact_human_agent": ("ESCALATE_HUMAN", "Transfer the customer to a human agent"),
    "get_invoice": ("SEND_INVOICE", "Send the customer their invoice"),
    "check_invoice": ("SEND_INVOICE", "Send the customer their invoice"),
    "get_refund": ("START_REFUND", "Open a refund request"),
    "track_order": ("TRACK_ORDER", "Send the order tracking link"),
    "track_refund": ("TRACK_REFUND", "Send the status of a refund"),
    "recover_password": ("SEND_PASSWORD_RESET", "Send a password-reset link"),
    "delete_account": ("DELETE_ACCOUNT", "Start closing the customer's account"),
    "create_account": ("CREATE_ACCOUNT", "Send a sign-up link to create an account"),
    "edit_account": ("EDIT_ACCOUNT", "Open the account settings to edit details"),
    "newsletter_subscription": ("MANAGE_NEWSLETTER", "Subscribe or unsubscribe the customer from the newsletter"),
    "place_order": ("PLACE_ORDER", "Start a new order for the customer"),
    "complaint": ("LOG_COMPLAINT", "Log a formal complaint for review"),
}
INFO_INTENTS = ["check_cancellation_fee", "check_payment_methods", "check_refund_policy", "delivery_options",
                "delivery_period", "contact_customer_service", "payment_issue", "registration_problems",
                "switch_account", "review"]
ACTIONS = sorted({v for v in ACTION_OF.values()})

FILLER = re.compile(
    r"^(of course|i'm on it|absolutely|definitely|certainly|sure|great|no problem|thank you|thanks|i understand|we understand|"
    r"i realized|i've understood|i hear you|i'm sorry|we apologize|rest assured|allow me|let's dive|i'm here to|i'm thrilled|"
    r"your satisfaction|i appreciate|we appreciate|i completely understand|i can see|i get it|i'll get right|how wonderful|"
    r"i'm aware|we truly|we're here|i'm delighted|i'm glad|i'm excited|it's great|what a|got it|ah,|i've taken note|"
    r"i've got|i'd be (happy|glad|delighted)|i can help you with|our dedicated team is here|i'm happy|happy to help|"
    r"we value|i'm fully|i acknowledge|i'm truly|it's wonderful|wonderful|excellent|fantastic|i'm really)", re.I)


USEFUL = re.compile(r"you can|please|could you|can you|visit|log ?in|sign in|navigate|provide|contact|reach|available|"
                    r"\{\{|click|select|go to|follow|section|page|policy|fee|within|days|hours|we offer|we accept|include", re.I)
FLUFF = re.compile(r"feedback is|valuable|appreciate|understand|take note|note that|sensitive|clued|glad|happy|delighted|"
                   r"enthusiasm|passion|our goal|ensure your|shed some light|got you covered|here to (help|assist)|"
                   r"don't hesitate|anything else|let me know|thrilled|wonderful|apologi|inconvenience", re.I)


def reply_target(response: str) -> str | None:
    """1-2 informative sentences from a Bitext reply: openers, empathy filler and list fragments removed,
    remaining sentences ranked by useful content, kept in their original order."""
    text = re.sub(r"\*\*", "", response.strip())
    sents = []
    for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split("\n\n")[:3])):
        s = re.split(r"\s\d+\.\s", s.strip())[0].strip()  # drop an inline list that starts mid-sentence
        if not s or s.endswith(":") or FILLER.match(s) or re.match(r"^(\d+\.|-|•)", s) or len(s.split()) < 6:
            continue
        score = 2 * bool(USEFUL.search(s)) - 2 * bool(FLUFF.search(s))
        if score > 0:
            sents.append(s)
    keep = []
    for s in sents:
        if keep and sum(len(k.split()) for k in keep) + len(s.split()) > 45:
            break
        keep.append(s)
        if len(keep) == 2:
            break
    out = " ".join(keep)
    return out if 8 <= len(out.split()) <= 55 else None


def norm(t: str) -> str:
    return re.sub(r"[^a-z ]", "", re.sub(r"\{\{[^}]+\}\}", "", t.lower())).strip()


def bitext(rng):
    df = pd.read_csv(dl("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                        "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"))
    keys = sorted({norm(t) for t in df.instruction})
    rng.shuffle(keys)
    held = set(keys[: len(keys) // 10])  # 10% of distinct messages held out
    df["held"] = [norm(t) in held for t in df.instruction]
    train, test = df[~df.held], df[df.held]

    intents = sorted(df.intent.unique())
    names_h = [x.replace("_", " ") for x in intents]
    intent_part = [row("bitext", r.instruction, names_h, intents.index(r.intent), rng)
                   for r in train.sample(N, random_state=rng.randint(0, 10**6)).itertuples()]

    def reply_rows(part, n, seed):
        r2 = random.Random(seed)
        out = []
        rows = list(part.itertuples())
        r2.shuffle(rows)
        want = {"action": int(0.5 * n), "info": int(0.35 * n), "not_offered": n - int(0.5 * n) - int(0.35 * n)}
        for r in rows:
            if not any(want.values()):
                break
            state = {"customer_msg": r.instruction}
            if r.intent in ACTION_OF:
                act = ACTION_OF[r.intent]
                kind = "action" if want["action"] and (not want["not_offered"] or r2.random() < 0.7) else "not_offered"
                if not want[kind]:
                    continue
                pool = [a for a in ACTIONS if a != act]
                distract = r2.sample(pool, r2.randint(2, 5))
                if kind == "action":
                    opts = [act] + distract + [(REPLY, REPLY_DESC)]
                    out.append(row("bitext_reply", state, [{"name": a, "desc": d} for a, d in opts], 0, r2, kind="action"))
                else:
                    tgt = reply_target(r.response)
                    if not tgt:
                        continue
                    opts = distract + [(REPLY, REPLY_DESC)]
                    out.append(row("bitext_reply", state, [{"name": a, "desc": d} for a, d in opts], len(opts) - 1, r2,
                                   reply=tgt, kind="not_offered"))
                want[kind] -= 1
            elif r.intent in INFO_INTENTS and want["info"]:
                tgt = reply_target(r.response)
                if not tgt:
                    continue
                opts = r2.sample(ACTIONS, r2.randint(2, 5)) + [(REPLY, REPLY_DESC)]
                out.append(row("bitext_reply", state, [{"name": a, "desc": d} for a, d in opts], len(opts) - 1, r2,
                               reply=tgt, kind="info"))
                want["info"] -= 1
        return out

    return intent_part, reply_rows(train, 3000, rng.randint(0, 10**6)), reply_rows(test, 300, rng.randint(0, 10**6))


def handwritten():
    """Our 97 hand-written cases, converted to the new option format (an out-of-distribution test now)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tl_decision_plus.options import CATALOG

    out = []
    for line in open(Path(__file__).parent / "test_handwritten.jsonl"):
        r = json.loads(line)
        opts = [{"name": n, "desc": CATALOG[n]} for n in r["options"]]
        out.append({"src": "handwritten", "kind": r["kind"], "state": r["state"], "options": opts,
                    "label": r["options"].index(r["label"]), "reply": None})
    return out


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    rng = random.Random(20261015)
    OUT.mkdir(exist_ok=True)
    parts = {}
    for name, fn in [("clinc150", clinc), ("massive", massive), ("reclor", reclor), ("logiqa2", logiqa2),
                     ("cosmosqa", cosmosqa), ("piqa", piqa), ("anli", anli), ("proofwriter", proofwriter)]:
        parts[name] = fn(rng)
    parts["bitext"], parts["bitext_reply"], held = bitext(rng)
    train = [r for p in parts.values() for r in p]
    rng.shuffle(train)
    write(OUT / "train.jsonl", train)
    write(OUT / "heldout_reply.jsonl", held)
    write(OUT / "handwritten.jsonl", handwritten())
    for k, v in parts.items():
        print(f"{k:14} {len(v):5}")
    print(f"train total {len(train)}, held-out reply {len(held)}")
