"""Synthetic customer-support data for TL_decision_plus.

Each row: {"state": {...}, "options": [names], "label": name, "reply": str|None, "kind": str}
The label depends on the message AND the state fields (e.g. an address change on a shipped
order is a reply, not UPDATE_ADDRESS), and on which options are offered (if the right action
is not offered, the answer is a reply).

Run: uv run data/make_dataset.py   → data/train.jsonl, data/val.jsonl
"""
import json
import random
from pathlib import Path

ACTIONS = ["AUTO_REFUND", "BLOCK_FRAUD", "ESCALATE_HUMAN", "RESEND_RECEIPT",
           "CANCEL_SUBSCRIPTION", "TRACK_ORDER", "UPDATE_ADDRESS"]
REPLY = "REPLY_WITH_TEXT"

OPEN = ["", "", "", "Hi, ", "Hello, ", "Hey, ", "Hi there, ", "Good morning, ", "Excuse me, ", "Um, "]
CLOSE = ["", "", "", " Thanks.", " Thank you!", " Please help.", " Asap please.", " Thx", " Cheers."]
NAMES = ["Sam", "Priya", "Omar", "Lena", "Diego", "Mei", "Tom", "Aisha", "Jonas", "Fatima", "Ken", "Rosa"]
ITEMS = ["headphones", "a desk lamp", "running shoes", "a phone case", "a coffee grinder", "a backpack",
         "a yoga mat", "a keyboard", "a winter jacket", "a blender", "sunglasses", "a water bottle"]
CITIES = ["Leeds", "Austin", "Lyon", "Pune", "Osaka", "Toronto", "Dublin", "Lagos", "Porto", "Denver"]
PLANS = [("Basic", 10), ("Plus", 15), ("Pro", 25), ("Family", 30)]


def oid(r):
    return f"{r.choice('ABCDEFGHJK')}-{r.randint(100, 9999)}"


def money(r, lo=5, hi=250):
    return round(r.uniform(lo, hi), 2)


def style(r, s):
    """Light noise so the model doesn't key on exact casing/punctuation."""
    s = r.choice(OPEN) + s + r.choice(CLOSE)
    x = r.random()
    if x < 0.15:
        s = s.lower()
    elif x < 0.22:
        s = s.rstrip(".!") + "!!"
    return s.strip()


# ---------------------------------------------------------------- action intents
def fraud(r):
    city = r.choice(CITIES)
    msg = r.choice([
        f"someone logged into my account from {city} and it wasn't me",
        f"there are {r.randint(2, 6)} purchases on my account I never made",
        "my card was stolen yesterday and I can see new orders on my account",
        f"I just got a login alert from {city}, I've never been there. Lock it please",
        f"who bought {r.choice(ITEMS)} on my account?? not me. I think I've been hacked",
        "I got an email saying my password was changed but I didn't change it",
        "my account is placing orders on its own, I think someone has my password",
        f"there's a gift card purchase for ${money(r, 50, 500)} I didn't do",
    ])
    state = {"customer_msg": style(r, msg), "risk_score": round(r.uniform(0.75, 0.99), 2),
             "last_login_city": city, "account_age_days": r.randint(30, 3000)}
    return state, "BLOCK_FRAUD"


def refund(r):
    amt = money(r)
    o = oid(r)
    kind = r.randrange(3)
    if kind == 0:
        msg = r.choice([
            f"I was charged twice for order {o}",
            f"my card shows two charges of ${amt} for the same order",
            f"double charge on {o}, please fix",
            f"you took ${amt} from me two times",
        ])
        state = {"customer_msg": style(r, msg), "order_id": o, "charges": [amt, amt], "order_total": amt}
    elif kind == 1:
        msg = r.choice([
            f"order {o} never arrived and it was due {r.randint(8, 30)} days ago, I want my money back",
            f"still no package for {o}, it's weeks late. refund please",
            f"the {r.choice(ITEMS)} I paid for never came, refund me",
        ])
        state = {"customer_msg": style(r, msg), "order_id": o, "order_status": "lost_in_transit",
                 "days_past_due": r.randint(8, 30), "order_total": amt}
    else:
        wrong = round(amt + r.choice([10, 20, 50, 100]), 2)
        msg = r.choice([
            f"I was charged ${wrong} but the price was ${amt}",
            f"the checkout said ${amt} and my bank shows ${wrong}, that's wrong",
            "you overcharged me on my last order",
        ])
        state = {"customer_msg": style(r, msg), "order_id": o, "listed_price": amt, "charged": wrong}
    return state, "AUTO_REFUND"


def escalate(r):
    msg = r.choice([
        "this is the fourth time I'm contacting you. I want to speak to a manager NOW",
        "if this isn't fixed today my lawyer will be in touch",
        "the charger you sold me started smoking and burned my desk",
        "I want a real person, not a bot. get me a supervisor",
        "I'm filing a complaint with consumer protection, this is unacceptable",
        "the heater I bought caught fire last night, my kid was nearby",
        "I've been passed around for two weeks, escalate this to someone senior",
        "my solicitor says you are breaking the law by holding my refund",
    ])
    state = {"customer_msg": style(r, msg), "previous_contacts": r.randint(0, 6),
             "sentiment": r.choice(["very_negative", "negative"])}
    return state, "ESCALATE_HUMAN"


def receipt(r):
    o = oid(r)
    msg = r.choice([
        f"can you send me the invoice for {o} again?",
        "I need a receipt for my last order for my expenses",
        f"please re-send the receipt for order {o}, I deleted the email",
        "my accountant needs the invoice for my March purchase",
        "can I get a copy of my receipt",
        "the receipt email never came, can you send it",
    ])
    state = {"customer_msg": style(r, msg), "order_id": o, "email_on_file": True}
    return state, "RESEND_RECEIPT"


def cancel(r):
    plan, price = r.choice(PLANS)
    msg = r.choice([
        "please cancel my subscription",
        f"I want to stop my {plan} plan",
        "cancel my membership, I don't use it anymore",
        "how do I unsubscribe? actually just do it for me",
        f"end my {plan} subscription today please",
        "I'd like to cancel, it's too expensive for me now",
    ])
    state = {"customer_msg": style(r, msg), "plan": plan, "monthly_price": price, "subscription_active": True}
    return state, "CANCEL_SUBSCRIPTION"


def track(r):
    o = oid(r)
    msg = r.choice([
        f"where is my order {o}?",
        f"has {o} shipped yet?",
        "when will my package arrive",
        f"can I get the tracking for {o}",
        f"what's the status of my {r.choice(ITEMS)}?",
        "any update on my delivery?",
    ])
    state = {"customer_msg": style(r, msg), "order_id": o,
             "order_status": r.choice(["shipped", "out_for_delivery", "in_transit"]),
             "eta_days": r.randint(1, 5)}
    return state, "TRACK_ORDER"


def address(r):
    o = oid(r)
    city = r.choice(CITIES)
    msg = r.choice([
        f"I moved! please send order {o} to my new address in {city}",
        f"can you change the delivery address on {o}",
        f"wrong address on my order, it should go to {r.randint(1, 200)} Park Road, {city}",
        "I need to update the shipping address for the order I just placed",
    ])
    state = {"customer_msg": style(r, msg), "order_id": o, "order_status": "processing"}
    return state, "UPDATE_ADDRESS"


ACTION_GEN = {"BLOCK_FRAUD": fraud, "AUTO_REFUND": refund, "ESCALATE_HUMAN": escalate,
              "RESEND_RECEIPT": receipt, "CANCEL_SUBSCRIPTION": cancel, "TRACK_ORDER": track,
              "UPDATE_ADDRESS": address}


# ---------------------------------------------------------------- reply intents
def info_question(r):
    k = r.randrange(5)
    if k == 0:
        plan, price = r.choice(PLANS)
        extra = r.choice([3, 4, 5, 8])
        addon = r.choice(["extra storage", "a second user", "premium support", "HD streaming"])
        msg = r.choice([f"why is my bill ${price + extra} instead of ${price}?",
                        f"why did I pay ${price + extra} this month?",
                        "my bill went up, why?"])
        state = {"customer_msg": style(r, msg), "plan": plan, "plan_price": price,
                 "invoice_note": f"${extra} add-on: {addon}"}
        reply = r.choice([
            f"Your bill includes a ${extra} add-on for {addon} on top of your ${price} {plan} plan. You can remove it in Settings if you don't need it.",
            f"The extra ${extra} is for the {addon} add-on added to your {plan} plan. Let me know if you'd like it removed.",
        ])
    elif k == 1:
        days = r.choice([14, 30, 60])
        msg = r.choice(["what is your return policy?", "how long do I have to return something?",
                        "can I still return an item I bought last week?"])
        state = {"customer_msg": style(r, msg), "return_window_days": days}
        reply = r.choice([
            f"You can return most items within {days} days of delivery for a full refund. Just start the return from your Orders page.",
            f"Returns are accepted for {days} days after delivery. Start one from your Orders page and we'll email a label.",
        ])
    elif k == 2:
        hours = r.choice(["9am-6pm Mon-Fri", "8am-8pm every day", "10am-5pm Mon-Sat"])
        msg = r.choice(["when is your support open?", "what are your opening hours?",
                        "can I call you on the weekend?"])
        state = {"customer_msg": style(r, msg), "support_hours": hours}
        reply = f"Our support team is available {hours}. You can also message us here any time and we'll reply when we're open."
    elif k == 3:
        msg = r.choice(["do you ship internationally?", "can you deliver to Canada?",
                        "do you ship outside the country?"])
        intl = r.random() < 0.5
        state = {"customer_msg": style(r, msg), "ships_international": intl}
        reply = ("Yes, we ship to most countries; shipping costs and times are shown at checkout."
                 if intl else "Sorry, we currently only ship within the country. We hope to add international shipping soon.")
    else:
        msg = r.choice(["how do I reset my password?", "I forgot my password",
                        "how can I change my password?"])
        state = {"customer_msg": style(r, msg), "risk_score": round(r.uniform(0.01, 0.2), 2)}
        reply = "Tap 'Forgot password' on the login page and we'll email you a reset link. It expires after one hour."
    return state, reply, "info"


def vague(r):
    msg = r.choice([
        "something is wrong with my order",
        "it's not working",
        "I have a problem",
        "help",
        "this is broken",
        "I need help with my account",
        "there's an issue with my payment",
        "can you fix it",
        "my thing didn't work again",
    ])
    state = {"customer_msg": style(r, msg)}
    if r.random() < 0.5:
        state["order_id"] = oid(r)
    reply = r.choice([
        "Sorry to hear that! Could you tell me a bit more about what went wrong so I can help?",
        "I'd like to help. Can you describe the problem and share your order number if it's about an order?",
        "Thanks for reaching out. What exactly isn't working, and when did it start?",
    ])
    return state, reply, "vague"


def conflict(r):
    o = oid(r)
    k = r.randrange(3)
    if k == 0:
        msg = f"please change the address on {o} to my new place"
        state = {"customer_msg": style(r, msg), "order_id": o, "order_status": "shipped"}
        reply = f"Order {o} has already shipped, so I can't change its address now. You can redirect it with the courier using the tracking link in your email."
    elif k == 1:
        msg = r.choice(["where is my order?", "has my package shipped?", "tracking please"])
        state = {"customer_msg": style(r, msg), "orders_on_account": 0}
        reply = "I can't find an order on this account. Could you share your order number or the email you used at checkout?"
    else:
        msg = r.choice(["cancel my subscription", "stop my membership please"])
        state = {"customer_msg": style(r, msg), "subscription_active": False}
        reply = "You don't have an active subscription right now, so there's nothing to cancel and you won't be charged."
    return state, reply, "conflict"


NOT_OFFERED_REPLY = {
    "BLOCK_FRAUD": "I'm sorry, that sounds serious. Please change your password now, and I'm flagging this so our security team can look at it right away.",
    "AUTO_REFUND": "Sorry about that! I can't issue the refund from here, but I've noted the charge and our billing team will follow up by email.",
    "ESCALATE_HUMAN": "I'm really sorry about this. I'm noting everything you've told me so a senior team member can follow up with you.",
    "RESEND_RECEIPT": "You can download any receipt from your Orders page by opening the order and tapping 'Invoice'.",
    "CANCEL_SUBSCRIPTION": "You can cancel any time from Settings > Subscription, and you'll keep access until the end of the billing period.",
    "TRACK_ORDER": "Your tracking link is in the shipping confirmation email, and you can also see it on your Orders page.",
    "UPDATE_ADDRESS": "I can't change the address from here, but if you reply with the new address our team will update it before it ships.",
}


def build_options(r, must: list[str], exclude: set[str]) -> list[str]:
    pool = [a for a in ACTIONS if a not in must and a not in exclude]
    n_total = r.randint(3, 6)
    picks = r.sample(pool, k=max(0, min(len(pool), n_total - len(must))))
    opts = list(dict.fromkeys(must + picks))
    r.shuffle(opts)
    return opts


def make_row(r):
    x = r.random()
    if x < 0.72:  # action is offered and correct
        label = r.choice(ACTIONS)
        state, label = ACTION_GEN[label](r)
        return {"state": state, "options": build_options(r, [label, REPLY], set()),
                "label": label, "reply": None, "kind": "action"}
    if x < 0.79:  # the right action is NOT on the menu → reply instead
        act = r.choice(ACTIONS)
        state, _ = ACTION_GEN[act](r)
        return {"state": state, "options": build_options(r, [REPLY], {act}),
                "label": REPLY, "reply": NOT_OFFERED_REPLY[act], "kind": "not_offered"}
    gen = r.choices([info_question, vague, conflict], weights=[4, 3, 3])[0]
    state, reply, kind = gen(r)
    return {"state": state, "options": build_options(r, [REPLY], set()),
            "label": REPLY, "reply": reply, "kind": kind}


def write(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    here = Path(__file__).parent
    write(here / "train.jsonl", [make_row(random.Random(i)) for i in range(2000)])
    write(here / "val.jsonl", [make_row(random.Random(10_000 + i)) for i in range(200)])
    print("wrote train.jsonl (2000) and val.jsonl (200)")
