"""Customer-support option catalogue. Examples show a varying subset, like Jev's dynamic choices."""

CATALOG = {
    "AUTO_REFUND": "Issue an instant refund (double charge, clear billing error, undelivered order past its date)",
    "BLOCK_FRAUD": "Lock the account now (stolen card, unknown login, purchases the customer did not make)",
    "ESCALATE_HUMAN": "Hand to a human agent (legal threat, angry repeat complaint, safety issue, request for a manager)",
    "RESEND_RECEIPT": "Email the invoice or receipt again",
    "CANCEL_SUBSCRIPTION": "Cancel the customer's subscription",
    "TRACK_ORDER": "Send the live tracking link for an order",
    "UPDATE_ADDRESS": "Change the shipping address on an order that has not shipped",
    "REPLY_WITH_TEXT": "Write a 1-2 sentence reply (answer a question, or ask for missing details)",
}

ACTIONS = [k for k in CATALOG if k != "REPLY_WITH_TEXT"]


def make_options(names: list[str]) -> list[dict]:
    return [{"name": n, "desc": CATALOG[n]} for n in names]
