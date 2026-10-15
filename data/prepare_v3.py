"""Build the v0.3 training mixture: v0.2's rows unchanged, plus three new public sources.

New sources (fix v0.2's two weaknesses: unfamiliar actions go to REPLY, and replies ignore the State):
  ABCD (ASAPP, MIT)            next step of real support dialogues: an action, or the agent's reply. Account and
                               order facts enter the State only after the lookup that reveals them, as in ABCD.
  Bitext verticals (CDLA-S.)   10 industries (banking, insurance, telco, e-commerce, ...): action offered / info
                               question / action not offered. Placeholders in the reply ({{WEBSITE_URL}}, ...)
                               are filled from facts we put in the State, so replies learn to copy from it.
  CFPB complaints (CC0)        real fraud / unauthorized-transaction messages, plus non-fraud issues as negatives.
                               REPLY_WITH_TEXT is offered as a distractor in half the rows and is never correct.

Option names and descriptions are new: none of the hand-written test set's names or descriptions are used
(checked at the end), so that set stays out of distribution.

Run: uv run data/prepare_v3.py   → data/v3/train.jsonl, heldout_{abcd,verticals,cfpb}.jsonl
Needs data/v2/train.jsonl (uv run data/prepare_v2.py) and data/raw_v3/abcd_v1.1.json.gz (from github.com/asappresearch/abcd).
"""
import ast
import gzip
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.prepare_v2 import REPLY, dl, norm, reply_target, row, write  # noqa: E402
from tl_decision_plus.options import CATALOG  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "v3"
RAW = HERE / "raw_v3"
REPLY_OPT = (REPLY, "Write a 1-2 sentence reply (answer a question, or explain what the customer can do)")


def opts(pairs):
    return [{"name": n, "desc": d} for n, d in pairs]


# ================================================================= ABCD
ABCD_ACTIONS = {
    "pull-up-account": ("LOOKUP_ACCOUNT", "Pull up the customer's account (needs their full name or account ID)"),
    "verify-identity": ("VERIFY_IDENTITY", "Verify the customer's identity (needs name, account ID, order ID and zip code)"),
    "validate-purchase": ("VALIDATE_PURCHASE", "Confirm the purchase is valid (needs username, email and order ID)"),
    "make-password": ("GENERATE_TEMP_PASSWORD", "Create a new temporary password for the customer"),
    "promo-code": ("SEND_PROMO_CODE", "Give the customer a promo code"),
    "subscription-status": ("CHECK_SUBSCRIPTION", "Look up the customer's subscription status and due amount"),
    "offer-refund": ("ISSUE_REFUND", "Refund an amount to the customer"),
    "make-purchase": ("BUY_FOR_CUSTOMER", "Purchase an item on the customer's behalf"),
    "shipping-status": ("CHECK_SHIPPING_STATUS", "Look up where the order is in shipping"),
    "update-order": ("UPDATE_ORDER", "Change the order: shipping address, items, shipping method or payment"),
    "update-account": ("UPDATE_ACCOUNT", "Change account details: address, name, payment method or phone"),
    "send-link": ("SEND_HELP_LINK", "Send the customer a link with step-by-step instructions"),
    "notify-team": ("NOTIFY_TEAM", "Escalate the case to another team (manager, website team, purchasing)"),
}
# facts an agent sees once the lookup that reveals them has run
REVEALS = {"pull-up-account": ("personal", ["customer_name", "username"]),
           "verify-identity": ("order", ["order_id", "purchase_date", "products", "full_address", "payment_method"]),
           "validate-purchase": ("order", ["order_id", "purchase_date", "products", "full_address", "payment_method"])}


def abcd_products(raw: str):
    try:
        return [f"{p['brand'].replace('_', ' ')} {p['product_type']} (${p['amount']})" for p in ast.literal_eval(raw)]
    except (ValueError, SyntaxError, KeyError):
        return raw


def abcd_rows(convos, rng, n, history=10):
    """One row per decision point after the customer's first message: the next action, or the next agent reply."""
    cands = []
    for c in convos:
        sc = c["scenario"]
        turns = [(sp, txt.strip(), d["targets"]) for (sp, txt), d in zip(c["original"], c["delexed"])]
        known, done, conv, seen_customer = {}, [], [], False
        i = 0
        while i < len(turns):
            sp, txt, tg = turns[i]
            if sp == "agent":  # merge consecutive agent messages into one reply
                j, parts = i, []
                while j < len(turns) and turns[j][0] == "agent":
                    parts.append(turns[j][1]); j += 1
                reply = " ".join(parts)
                if seen_customer and 4 <= len(reply.split()) <= 50:
                    cands.append(("reply", sc["flow"], dict(known), list(done), list(conv[-history:]), reply))
                conv.append(f"agent: {reply}")
                i = j
                continue
            if sp == "customer":
                seen_customer = True
                conv.append(f"customer: {txt}")
            elif sp == "action":
                act = tg[2]
                if seen_customer and act in ABCD_ACTIONS:
                    cands.append(("action", sc["flow"], dict(known), list(done), list(conv[-history:]), act))
                conv.append(f"system: {txt}")
                done.append(ABCD_ACTIONS[act][0] if act in ABCD_ACTIONS else act)
                if act in REVEALS:
                    part, keys = REVEALS[act]
                    for k in keys:
                        if k in sc[part]:
                            known[k] = abcd_products(sc[part][k]) if k == "products" else sc[part][k]
            i += 1
    acts = [x for x in cands if x[0] == "action"]
    reps = [x for x in cands if x[0] == "reply"]
    n_act = int(0.45 * n)
    pick = rng.sample(acts, min(n_act, len(acts))) + rng.sample(reps, n - min(n_act, len(acts)))
    out = []
    for kind, flow, known, done, conv, gold in pick:
        state = {"conversation": conv, "actions_taken": done}
        if known:
            state["account"] = known
        others = [v for k, v in ABCD_ACTIONS.items() if k != gold]
        distract = rng.sample(others, rng.randint(3, 6))
        if kind == "action":
            options = [ABCD_ACTIONS[gold]] + distract + [REPLY_OPT]
            out.append(row("abcd", state, opts(options), 0, rng, kind="action"))
        else:
            options = distract + [REPLY_OPT]
            out.append(row("abcd", state, opts(options), len(options) - 1, rng, reply=gold, kind="reply"))
    return out


# ================================================================= Bitext verticals
VERTICALS = {
    "retail-banking": "bitext-retail-banking-llm-chatbot-training-dataset.csv",
    "insurance": "bitext-insurance-llm-chatbot-training-dataset.csv",
    "telco": "bitext-telco-llm-chatbot-training-dataset.csv",
    "retail-ecommerce": "bitext-retail-ecommerce-llm-chatbot-training-dataset.csv",
    "travel": "bitext-travel-llm-chatbot-training-dataset.csv",
    "mortgage-loans": "bitext-mortgage-loans-llm-chatbot-training-dataset.csv",
    "wealth-management": "bitext-wealth-management-llm-chatbot-training-dataset.csv",
    "hospitality": "bitext-hospitality-llm-chatbot-training-dataset.csv",
    "events-ticketing": "bitext-events-ticketing-llm-chatbot-training-dataset .csv",
    "media": "bitext-media-llm-chatbot-training-dataset.csv",
}
A = {  # shared action vocabulary (name → description)
    "TRANSFER_TO_AGENT": "Connect the customer with a live agent",
    "TAKE_PAYMENT": "Take a payment from the customer now",
    "SCHEDULE_PAYMENT": "Set up a scheduled payment",
    "OPEN_PAYMENT_INVESTIGATION": "Open a case for a failed or wrong payment",
    "OPEN_BILLING_DISPUTE": "Open a dispute about a bill or invoice",
    "SEND_INVOICE_COPY": "Email a copy of the bill or invoice",
    "ISSUE_REFUND": "Refund the customer",
    "SEND_REFUND_STATUS": "Send the status of a refund or cancellation",
    "SEND_PASSWORD_RESET": "Send a password-reset link",
    "OPEN_ACCOUNT": "Start opening a new account",
    "CLOSE_ACCOUNT": "Start closing the customer's account",
    "UPDATE_PERSONAL_DETAILS": "Change the customer's name, address or contact details",
    "LOG_FORMAL_COMPLAINT": "Record a formal complaint for the complaints team",
    "RECORD_FEEDBACK": "Record the customer's feedback or review",
    "OPEN_TECH_TICKET": "Open a technical support ticket",
}
VA = {  # vertical-specific actions: intent → (name, description); intents not listed are information questions
    "retail-banking": {
        "activate_card": ("ACTIVATE_CARD", "Activate the customer's new card"),
        "activate_card_international_usage": ("ENABLE_INTERNATIONAL_USE", "Allow the card to be used abroad"),
        "apply_for_loan": ("START_LOAN_APPLICATION", "Start a loan application"),
        "apply_for_mortgage": ("START_MORTGAGE_APPLICATION", "Start a mortgage application"),
        "block_card": ("FREEZE_CARD", "Block the card at once so no new payments go through"),
        "cancel_card": ("CANCEL_CARD", "Cancel the card permanently"),
        "cancel_loan": ("CANCEL_LOAN", "Cancel the loan"),
        "cancel_mortgage": ("CANCEL_MORTGAGE", "Cancel the mortgage"),
        "cancel_transfer": ("CANCEL_TRANSFER", "Stop a pending transfer"),
        "close_account": "CLOSE_ACCOUNT", "create_account": "OPEN_ACCOUNT",
        "dispute_ATM_withdrawal": ("OPEN_ATM_DISPUTE", "Open a dispute about an ATM withdrawal"),
        "get_password": "SEND_PASSWORD_RESET",
        "set_up_password": ("SEND_PASSWORD_SETUP_LINK", "Send a link to set up a password"),
        "human_agent": "TRANSFER_TO_AGENT",
        "make_transfer": ("START_TRANSFER", "Start a money transfer"),
        "recover_swallowed_card": ("REPORT_RETAINED_CARD", "Report a card kept by an ATM and order a new one"),
    },
    "insurance": {
        "accept_settlement": ("ACCEPT_SETTLEMENT", "Accept the settlement offer on the claim"),
        "appeal_denied_insurance_claim": ("START_CLAIM_APPEAL", "Start an appeal of a denied claim"),
        "buy_insurance_policy": ("START_POLICY_PURCHASE", "Start buying a new policy"),
        "cancel_insurance_policy": ("CANCEL_POLICY", "Cancel the insurance policy"),
        "change_coverage": ("CHANGE_COVERAGE", "Change the policy's coverage level"),
        "upgrade_coverage": ("CHANGE_COVERAGE", "Change the policy's coverage level"),
        "downgrade_coverage": ("CHANGE_COVERAGE", "Change the policy's coverage level"),
        "change_personal_details": "UPDATE_PERSONAL_DETAILS",
        "dispute_invoice": "OPEN_BILLING_DISPUTE",
        "file_claim": ("START_CLAIM", "Open a new insurance claim"),
        "report_incident": ("START_CLAIM", "Open a new insurance claim"),
        "file_complaint": "LOG_FORMAL_COMPLAINT",
        "human_agent": "TRANSFER_TO_AGENT", "agent": "TRANSFER_TO_AGENT", "insurance_representative": "TRANSFER_TO_AGENT",
        "negotiate_settlement": ("REFER_TO_ADJUSTER", "Refer the claim to a claims adjuster"),
        "reject_settlement": ("REFER_TO_ADJUSTER", "Refer the claim to a claims adjuster"),
        "pay": "TAKE_PAYMENT", "schedule_payments": "SCHEDULE_PAYMENT",
        "renew_insurance_policy": ("RENEW_POLICY", "Renew the policy"),
        "report_payment_issue": "OPEN_PAYMENT_INVESTIGATION",
        "schedule_appointment": ("BOOK_APPOINTMENT", "Book an appointment with an agent"),
        "track_claim": ("SEND_CLAIM_STATUS", "Send the current status of a claim"),
    },
    "telco": {
        "activate_call_management_services": ("ACTIVATE_LINE_FEATURE", "Turn on a call feature (voicemail, forwarding, ...)"),
        "deactivate_call_management_services": ("DEACTIVATE_LINE_FEATURE", "Turn off a call feature"),
        "activate_phone": ("ACTIVATE_LINE", "Activate a phone line or SIM"),
        "deactivate_phone": ("SUSPEND_LINE", "Suspend or deactivate a phone line"),
        "activate_roaming": ("ENABLE_ROAMING", "Turn on roaming"),
        "cancel_plan": ("CANCEL_PLAN", "Cancel the mobile or internet plan"),
        "change_plan": ("CHANGE_PLAN", "Move the customer to a different plan"),
        "change_provider": ("START_NUMBER_PORT", "Start moving the number to another provider"),
        "dispute_invoice": "OPEN_BILLING_DISPUTE",
        "get_compensation": ("REQUEST_SERVICE_CREDIT", "Request a credit for a service outage"),
        "human_agent": "TRANSFER_TO_AGENT",
        "install_internet": ("BOOK_INSTALLATION", "Book an internet installation"),
        "invoices": "SEND_INVOICE_COPY", "pay": "TAKE_PAYMENT", "schedule_payments": "SCHEDULE_PAYMENT",
        "report_poor_signal_coverage": "OPEN_TECH_TICKET", "report_problem": "OPEN_TECH_TICKET",
        "set_usage_limits": ("SET_USAGE_LIMIT", "Set a data or spending limit"),
        "sign_up_for_plan": ("START_PLAN_SIGNUP", "Sign the customer up for a plan"),
    },
    "retail-ecommerce": {
        "add_product": ("ADD_TO_CART", "Add an item to the customer's cart"),
        "remove_product": ("REMOVE_FROM_CART", "Remove an item from the customer's cart"),
        "cancel_order": ("CANCEL_THE_ORDER", "Cancel an order that has not shipped"),
        "change_order": ("MODIFY_ORDER", "Change the items in an order"),
        "change_account": "UPDATE_PERSONAL_DETAILS", "close_account": "CLOSE_ACCOUNT", "open_account": "OPEN_ACCOUNT",
        "damaged_delivery": ("REPORT_ORDER_PROBLEM", "Open a case for a damaged, wrong or missing item"),
        "wrong_item": ("REPORT_ORDER_PROBLEM", "Open a case for a damaged, wrong or missing item"),
        "missing_item": ("REPORT_ORDER_PROBLEM", "Open a case for a damaged, wrong or missing item"),
        "product_issue": ("REPORT_ORDER_PROBLEM", "Open a case for a damaged, wrong or missing item"),
        "delivery_issue": ("OPEN_DELIVERY_INVESTIGATION", "Ask the carrier to investigate a delivery problem"),
        "exchange_product": ("START_EXCHANGE", "Start an exchange for a different item or size"),
        "exchange_product_in_store": ("START_EXCHANGE", "Start an exchange for a different item or size"),
        "human_agent": "TRANSFER_TO_AGENT", "recover_password": "SEND_PASSWORD_RESET",
        "request_invoice": "SEND_INVOICE_COPY", "request_refund": "ISSUE_REFUND", "refund_status": "SEND_REFUND_STATUS",
        "request_right_to_rectification": ("OPEN_DATA_CORRECTION", "Open a request to correct the customer's personal data"),
        "return_product": ("START_RETURN", "Start a return and send a return label"),
        "return_product_in_store": ("START_RETURN", "Start a return and send a return label"),
        "return_product_online": ("START_RETURN", "Start a return and send a return label"),
        "submit_feedback": "RECORD_FEEDBACK", "submit_product_feedback": "RECORD_FEEDBACK",
        "submit_product_idea": "RECORD_FEEDBACK",
        "track_delivery": ("SEND_TRACKING_INFO", "Send the delivery tracking details"),
        "track_order": ("SEND_TRACKING_INFO", "Send the delivery tracking details"),
        "pay": "TAKE_PAYMENT", "payment_issue": "OPEN_PAYMENT_INVESTIGATION", "technical_issue": "OPEN_TECH_TICKET",
    },
    "travel": {
        "book_flight": ("START_FLIGHT_BOOKING", "Start booking a flight"),
        "book_trip": ("START_TRIP_BOOKING", "Start booking a trip package"),
        "cancel_flight": ("CANCEL_FLIGHT", "Cancel the flight booking"),
        "cancel_trip": ("CANCEL_TRIP", "Cancel the trip booking"),
        "change_flight": ("CHANGE_FLIGHT", "Move the booking to another flight"),
        "change_trip": ("CHANGE_TRIP", "Change the trip booking"),
        "change_seat": ("ASSIGN_SEAT", "Choose or change the passenger's seat"),
        "choose_seat": ("ASSIGN_SEAT", "Choose or change the passenger's seat"),
        "check_in": ("CHECK_IN_PASSENGER", "Check the passenger in for the flight"),
        "get_boarding_pass": ("SEND_BOARDING_PASS", "Send the boarding pass"),
        "print_boarding_pass": ("SEND_BOARDING_PASS", "Send the boarding pass"),
        "get_refund": "ISSUE_REFUND", "human_agent": "TRANSFER_TO_AGENT",
        "purchase_flight_insurance": ("ADD_TRAVEL_INSURANCE", "Add travel insurance to the booking"),
        "purchase_trip_insurance": ("ADD_TRAVEL_INSURANCE", "Add travel insurance to the booking"),
    },
    "mortgage-loans": {
        "add_coborrower": ("ADD_COBORROWER", "Add a co-borrower to the loan"),
        "apply_for_joint_loan": ("START_LOAN_APPLICATION", "Start a loan application"),
        "apply_for_loan": ("START_LOAN_APPLICATION", "Start a loan application"),
        "reapply_for_loan": ("START_LOAN_APPLICATION", "Start a loan application"),
        "change_application": ("UPDATE_APPLICATION", "Change details on a pending application"),
        "withdraw_application": ("WITHDRAW_APPLICATION", "Withdraw a pending application"),
        "change_due_date": ("CHANGE_DUE_DATE", "Move the monthly payment due date"),
        "change_personal_data": "UPDATE_PERSONAL_DETAILS",
        "change_preferred_bank_account": ("UPDATE_PAYMENT_ACCOUNT", "Change the bank account payments come from"),
        "consolidate_debt": ("START_CONSOLIDATION", "Start a debt-consolidation application"),
        "contact_agent": "TRANSFER_TO_AGENT", "human_agent": "TRANSFER_TO_AGENT",
        "extend_loan": ("REQUEST_LOAN_EXTENSION", "Request a longer loan term"),
        "lock_interest_rate": ("LOCK_RATE", "Lock the current interest rate"),
        "make_additional_payments": "TAKE_PAYMENT", "pay_off_loan": "TAKE_PAYMENT",
        "refinance_loan": ("START_REFINANCE", "Start a refinance application"),
        "request_payment_arrangement": ("OFFER_HARDSHIP_PLAN", "Set up a payment plan for a customer who is struggling to pay"),
        "split_payment": ("SET_UP_SPLIT_PAYMENTS", "Split the monthly payment into smaller payments"),
        "submit_documentation": ("REQUEST_DOCUMENT_UPLOAD", "Send a secure link to upload documents"),
        "turn_off_recurring_payments": ("STOP_AUTOPAY", "Turn off automatic payments"),
        "turn_on_recurring_payments": ("START_AUTOPAY", "Turn on automatic payments"),
    },
    "wealth-management": {
        "arrange_meeting": ("BOOK_ADVISOR_MEETING", "Book a meeting with a financial advisor"),
        "become_client": "OPEN_ACCOUNT", "create_account": "OPEN_ACCOUNT",
        "contact_manager": ("CONNECT_ACCOUNT_MANAGER", "Put the customer through to their account manager"),
        "get_manager": ("CONNECT_ACCOUNT_MANAGER", "Put the customer through to their account manager"),
        "human_agent": "TRANSFER_TO_AGENT", "delete_account": "CLOSE_ACCOUNT", "recover_password": "SEND_PASSWORD_RESET",
        "set_price_alert": ("SET_PRICE_ALERT", "Set a stock price alert"),
        "transfer_money_to_account": ("START_DEPOSIT", "Start a deposit into the investment account"),
        "withdraw_money_from_account": ("START_WITHDRAWAL", "Start a withdrawal from the investment account"),
    },
    "hospitality": {
        "add_night": ("EXTEND_STAY", "Add nights to the reservation"),
        "book_hotel": ("START_HOTEL_BOOKING", "Start a hotel booking"),
        "book_parking_space": ("RESERVE_PARKING", "Reserve a parking space"),
        "cancel_hotel_reservation": ("CANCEL_RESERVATION", "Cancel the hotel reservation"),
        "change_hotel_reservation": ("MODIFY_RESERVATION", "Change the hotel reservation"),
        "file_complaint": "LOG_FORMAL_COMPLAINT", "get_refund": "ISSUE_REFUND", "human_agent": "TRANSFER_TO_AGENT",
        "invoices": "SEND_INVOICE_COPY", "leave_review": "RECORD_FEEDBACK",
        "redeem_points": ("REDEEM_POINTS", "Redeem loyalty points"),
        "shuttle_service": ("BOOK_SHUTTLE", "Book the airport shuttle"),
    },
    "events-ticketing": {
        "buy_ticket": ("START_TICKET_PURCHASE", "Start buying tickets"),
        "cancel_ticket": ("CANCEL_TICKET", "Cancel the tickets"),
        "change_personal_details_on_ticket": ("UPDATE_TICKET_HOLDER", "Change the name or details on a ticket"),
        "get_refund": "ISSUE_REFUND", "human_agent": "TRANSFER_TO_AGENT",
        "report_payment_issue": "OPEN_PAYMENT_INVESTIGATION", "pay": "TAKE_PAYMENT",
        "sell_ticket": ("LIST_TICKET_FOR_RESALE", "List the customer's ticket for resale"),
        "track_cancellation": "SEND_REFUND_STATUS", "track_refund": "SEND_REFUND_STATUS",
        "transfer_ticket": ("TRANSFER_TICKET", "Transfer a ticket to another person"),
        "upgrade_ticket": ("UPGRADE_TICKET", "Upgrade the ticket to a better seat or tier"),
    },
    "media": {
        "cancel_subscription": ("END_SUBSCRIPTION", "End the customer's streaming subscription"),
        "change_subscription": ("CHANGE_SUBSCRIPTION_PLAN", "Move the customer to another subscription plan"),
        "premium_subscription": ("START_SUBSCRIPTION", "Start a new subscription"),
        "subscribe": ("START_SUBSCRIPTION", "Start a new subscription"),
        "renew_subscription": ("RENEW_SUBSCRIPTION", "Renew the subscription"),
        "human_agent": "TRANSFER_TO_AGENT", "recover_password": "SEND_PASSWORD_RESET", "pay": "TAKE_PAYMENT",
        "report_copyright_infringement": ("OPEN_COPYRIGHT_REPORT", "Open a copyright-infringement report"),
        "report_inappropiate_content": ("FLAG_CONTENT_FOR_REVIEW", "Send content to the moderation team for review"),
        "report_payment_issue": "OPEN_PAYMENT_INVESTIGATION",
    },
}


def vertical_actions(v):
    out = {}
    for intent, a in VA[v].items():
        out[intent] = (a, A[a]) if isinstance(a, str) else a
    return out


# placeholders filled from facts we add to the State
FACT = {
    "WEBSITE_URL": ("company", "website"), "Company Website URL": ("company", "website"),
    "Company Website": ("company", "website"),
    "Customer Support Phone Number": ("company", "support_phone"), "SUPPORT_NUMBER": ("company", "support_phone"),
    "Customer Service Phone Number": ("company", "support_phone"),
    "Customer Support Working Hours": ("company", "support_hours"),
    "Customer Support Email Address": ("company", "support_email"), "Customer Support Email": ("company", "support_email"),
    "APP_NAME": ("company", "app"), "APP": ("company", "app"),
    "Order Number": ("customer", "order_number"), "Name": ("customer", "name"),
    "Email Address": ("customer", "email"), "Credit Card": ("customer", "card"),
}
LITERAL = {"CUSTOMER_SUPPORT": "customer support", "Customer Support Team": "our support team",
           "Customer Support": "customer support", "SUPPORT_TEAM_CONTACT": "our support team",
           "Account Settings": "Account Settings", "Order History": "Order History", "My Orders": "My Orders",
           "Profile": "Profile", "Password": "password"}
UI = re.compile(r"^([A-Z_]+)_(SECTION|BUTTON|OPTION)$")
BRANDS = ["Northwind", "Bluepeak", "Orbit", "Larkspur", "Meridian", "Fernway", "Corvid", "Tallow", "Quillon", "Harbor & Co"]
FIRST = ["Maya", "Omar", "Lena", "Diego", "Aisha", "Tom", "Priya", "Jonas", "Chloe", "Kenji", "Fatima", "Sam"]
LAST = ["Okafor", "Lindqvist", "Haddad", "Moreno", "Chen", "Walsh", "Novak", "Ibrahim", "Rossi", "Kato"]


def fake_facts(rng):
    b = rng.choice(BRANDS)
    slug = re.sub(r"[^a-z]", "", b.lower())
    first, last = rng.choice(FIRST), rng.choice(LAST)
    open_h, close_h = rng.choice([7, 8, 9]), rng.choice([17, 18, 20, 22])
    days = rng.choice(["Monday to Friday", "Monday to Saturday", "every day"])
    return {
        "company": {"name": b, "website": f"www.{slug}.com", "support_phone": f"1-800-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}",
                    "support_hours": f"{days}, {open_h}am to {close_h - 12}pm", "support_email": f"help@{slug}.com",
                    "app": f"{b} app"},
        "customer": {"name": f"{first} {last}", "email": f"{first.lower()}.{last.lower()}@example.com",
                     "order_number": f"{rng.choice('ABCDKMRTW')}{rng.randint(10000, 99999)}",
                     "card": f"{rng.choice(['Visa', 'Mastercard', 'Amex'])} ending {rng.randint(1000, 9999)}"},
    }


def tidy(t):
    """Drop list scaffolding reply_target() can leave: a lead-in ending in ':' and '1.'-style numbers."""
    if not t:
        return None
    t = re.sub(r"[^.!?]*:\s*\d+\.\s*", "", t)
    t = re.sub(r"(^|\s)\d+\.\s+", " ", t).strip()
    return t if len(t.split()) >= 6 else None


def fill(text, facts, used):
    """Replace placeholders; None if any placeholder can't be filled. Records which facts were used."""
    def sub(m):
        k = m.group(1).strip()
        if k in FACT:
            part, key = FACT[k]
            used.add((part, key))
            return facts[part][key]
        if k in LITERAL:
            return LITERAL[k]
        u = UI.match(k)
        if u:
            return f"{u.group(1).replace('_', ' ').title()} {u.group(2).lower()}"
        raise KeyError(k)
    try:
        out = re.sub(r"\{\{([^}]+)\}\}", sub, text)
        return re.sub(r"\b(section|button|option) \1\b", r"\1", out)
    except KeyError:
        return None


def vertical_state(msg, facts, used, rng):
    """customer_msg + the facts the reply needs + a few unrelated ones (so the model has to pick)."""
    state = {"customer_msg": msg}
    for part in ("company", "customer"):
        keys = {k for p, k in used if p == part}
        extra = [k for k in facts[part] if k not in keys and rng.random() < 0.3]
        if part == "company" and (keys or extra):
            keys.add("name")
        if keys or extra:
            state[part] = {k: facts[part][k] for k in facts[part] if k in keys or k in extra}
    return state


def vertical_rows(rng, n_train, n_held):
    train, held = [], []
    per_t, per_h = n_train // len(VERTICALS), n_held // len(VERTICALS)
    for v, f in VERTICALS.items():
        df = pd.read_csv(dl(f"bitext/Bitext-{v}-llm-chatbot-training-dataset", f))
        df = df[~df.instruction.str.contains(r"\{\{", regex=True)]  # messages with unfilled placeholders
        acts = vertical_actions(v)
        names = {}  # name → description, deduplicated
        for a in acts.values():
            names[a[0]] = a[1]
        keys = sorted({norm(t) for t in df.instruction})
        rng.shuffle(keys)
        hk = set(keys[: len(keys) // 10])
        df["held"] = [norm(t) in hk for t in df.instruction]
        for part, n, sink in [(df[~df.held], per_t, train), (df[df.held], per_h, held)]:
            r2 = random.Random(rng.randint(0, 10**6))
            rows = list(part.itertuples())
            r2.shuffle(rows)
            want = {"action": int(0.5 * n), "info": int(0.35 * n), "not_offered": n - int(0.5 * n) - int(0.35 * n)}
            for r in rows:
                if not any(want.values()):
                    break
                facts, used = fake_facts(r2), set()
                if r.intent in acts:
                    act = acts[r.intent]
                    kind = "action" if want["action"] and (not want["not_offered"] or r2.random() < 0.7) else "not_offered"
                    if not want[kind]:
                        continue
                    pool = [(k, d) for k, d in names.items() if k != act[0]]
                    distract = r2.sample(pool, min(len(pool), r2.randint(2, 5)))
                    if kind == "action":
                        state = vertical_state(r.instruction, facts, used, r2)
                        sink.append(row(f"vert_{v}", state, opts([act] + distract + [REPLY_OPT]), 0, r2, kind="action"))
                    else:
                        tgt = tidy(reply_target(str(r.response)))
                        tgt = fill(tgt, facts, used) if tgt else None
                        if not tgt:
                            continue
                        state = vertical_state(r.instruction, facts, used, r2)
                        o = distract + [REPLY_OPT]
                        sink.append(row(f"vert_{v}", state, opts(o), len(o) - 1, r2, reply=tgt, kind="not_offered"))
                    want[kind] -= 1
                elif want["info"]:
                    tgt = tidy(reply_target(str(r.response)))
                    tgt = fill(tgt, facts, used) if tgt else None
                    if not tgt:
                        continue
                    state = vertical_state(r.instruction, facts, used, r2)
                    o = r2.sample(list(names.items()), min(len(names), r2.randint(2, 5))) + [REPLY_OPT]
                    sink.append(row(f"vert_{v}", state, opts(o), len(o) - 1, r2, reply=tgt, kind="info"))
                    want["info"] -= 1
    return train, held


# ================================================================= CFPB
CFPB_CLASSES = {  # class → name/description variants (one is picked per row)
    "fraud": [("LOCK_ACCOUNT_OPEN_FRAUD_CASE", "Freeze the account and open a fraud investigation"),
              ("REPORT_FRAUD", "Treat as fraud: block the card or account and start a fraud case"),
              ("FREEZE_AND_INVESTIGATE", "Freeze the account now and send it to the fraud team")],
    "dispute": [("ESCALATE_DISPUTE", "Escalate an unresolved charge dispute to the disputes team"),
                ("OPEN_CHARGE_DISPUTE", "Open a dispute about a charge on the statement")],
    "fees": [("REVIEW_FEES", "Review the fees or interest and refund any charged in error"),
             ("FEE_REFUND_REVIEW", "Check the fees charged and reverse any that were wrong")],
    "card_use": [("CARD_TROUBLESHOOTING", "Help with a card that is declined or not working"),
                 ("FIX_CARD_PROBLEM", "Investigate why the card or ATM card is not working")],
    "close": [("PROCESS_ACCOUNT_CLOSURE", "Close the account the customer has asked to close"),
              ("CLOSE_ACCOUNT_REQUEST", "Complete the customer's request to close their account")],
    "payments": [("PAYMENT_SUPPORT", "Investigate a payment that failed, was late or was misapplied"),
                 ("TRACE_PAYMENT", "Trace a payment that did not go through or went to the wrong place")],
    "hardship": [("OFFER_HARDSHIP_OPTIONS", "Offer a hardship plan to a customer who can't keep up with payments"),
                 ("PAYMENT_RELIEF_REVIEW", "Review the customer for payment relief or a modified plan")],
    "access": [("RESTORE_ACCOUNT_ACCESS", "Help the customer get back into an account they can't access"),
               ("ACCOUNT_ACCESS_SUPPORT", "Fix a login or online-banking access problem")],
}


CUES = {  # the kept part of the message must mention its class, or the row is dropped (CFPB labels are noisy)
    "fraud": r"fraud|scam|unauthori[sz]ed|did ?n[o']t (make|authori[sz]e|recogni[sz]e)|not recogni[sz]e|stolen|identity theft|hack",
    "dispute": r"disput",
    "fees": r"\bfees?\b|interest|overdraft|charged",
    "card_use": r"\bcard\b|declin|\bATM\b",
    "close": r"\bclos",
    "payments": r"payment",
    "hardship": r"afford|hardship|struggl|forbearance|modif|unemploy|lost my job|behind on",
    "access": r"access|log ?in|locked|password",
}


def cfpb_class(issue, sub):
    if issue == "Fraud or scam" or sub in ("Transaction was not authorized",
                                           "Card was charged for something you did not purchase with the card",
                                           "Card opened as result of identity theft or fraud"):
        return "fraud"
    if sub == "Credit card company isn't resolving a dispute about a purchase on your statement":
        return "dispute"
    if issue in ("Fees or interest", "Charged fees or interest you didn't expect", "Problem caused by your funds being low"):
        return "fees"
    if issue == "Trouble using your card" or sub == "Problem using a debit or ATM card":
        return "card_use"
    if sub == "Can't close your account":
        return "close"
    if issue in ("Trouble during payment process", "Problem when making payments"):
        return "payments"
    if issue in ("Struggling to pay mortgage", "Struggling to pay your loan", "Struggling to repay your loan"):
        return "hardship"
    if sub == "Problem accessing account":
        return "access"
    return None


def first_words(text, max_words=70):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(XX/){1,2}(XX|XXXX|\d{2,4})", "[date]", text)
    text = re.sub(r"\bX{2,}\b", "[redacted]", text)
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if out and len(" ".join(out + [s]).split()) > max_words:
            break
        out.append(s)
    msg = " ".join(out)
    words = msg.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def cfpb_rows(rng, n_train, n_held):
    p = dl("BEE-spoke-data/consumer-finance-complaints", "has-text/train-00000-of-00003.parquet")
    df = pq.read_table(p, columns=["Product", "Issue", "Sub-issue", "Consumer complaint narrative", "Complaint ID"]).to_pandas()
    df = df[~df.Product.str.startswith("Credit reporting")]
    df["cls"] = [cfpb_class(i, s) for i, s in zip(df.Issue, df["Sub-issue"])]
    df = df[df.cls.notna() & (df["Consumer complaint narrative"].str.split().str.len() >= 25)].copy()
    df["msg"] = [first_words(t) for t in df["Consumer complaint narrative"]]
    df = df[[bool(re.search(CUES[c], m, re.I)) for c, m in zip(df.cls, df.msg)]]
    ids = sorted(df["Complaint ID"].unique())
    rng.shuffle(ids)
    held_ids = set(ids[: len(ids) // 10])
    out = {}
    for split, n in [("train", n_train), ("held", n_held)]:
        part = df[df["Complaint ID"].isin(held_ids) == (split == "held")]
        r2 = random.Random(rng.randint(0, 10**6))
        n_fraud = int(0.4 * n)
        others = [c for c in CFPB_CLASSES if c != "fraud"]
        quota = {"fraud": n_fraud, **{c: (n - n_fraud) // len(others) for c in others}}
        rows = []
        for cls, q in quota.items():
            sub = part[part.cls == cls]
            for r in sub.sample(min(q, len(sub)), random_state=r2.randint(0, 10**6)).itertuples():
                gold = r2.choice(CFPB_CLASSES[cls])
                distract = [r2.choice(CFPB_CLASSES[c]) for c in r2.sample([c for c in CFPB_CLASSES if c != cls], r2.randint(2, 4))]
                o = [gold] + distract + ([REPLY_OPT] if r2.random() < 0.5 else [])
                state = {"customer_msg": r.msg, "product": r.Product}
                rows.append(row("cfpb", state, opts(o), 0, r2, kind=cls))
        out[split] = rows
    return out["train"], out["held"]


# ================================================================= main
def leak_check(rows):
    """None of the hand-written test set's option names or descriptions may appear in the new rows."""
    names = set(CATALOG) - {REPLY}
    descs = set(CATALOG.values())
    bad = [(r["src"], o["name"]) for r in rows for o in r["options"] if o["name"] in names or o.get("desc") in descs]
    assert not bad, f"hand-written option names/descriptions leaked into new rows: {bad[:5]}"


if __name__ == "__main__":
    rng = random.Random(20261016)
    OUT.mkdir(exist_ok=True)
    v2 = [json.loads(l) for l in open(HERE / "v2" / "train.jsonl")]
    abcd = json.load(gzip.open(RAW / "abcd_v1.1.json.gz"))
    parts = {"abcd": abcd_rows(abcd["train"], rng, 5000)}
    held_abcd = abcd_rows(abcd["test"], rng, 300)
    parts["verticals"], held_vert = vertical_rows(rng, 5000, 300)
    parts["cfpb"], held_cfpb = cfpb_rows(rng, 1500, 200)
    new = [r for p in parts.values() for r in p]
    leak_check(new + held_abcd + held_vert + held_cfpb)
    train = v2 + new
    rng.shuffle(train)
    write(OUT / "train.jsonl", train)
    write(OUT / "heldout_abcd.jsonl", held_abcd)
    write(OUT / "heldout_verticals.jsonl", held_vert)
    write(OUT / "heldout_cfpb.jsonl", held_cfpb)
    print(f"{'v2 (unchanged)':16} {len(v2):6}")
    for k, v in parts.items():
        print(f"{k:16} {len(v):6}")
    print(f"train total {len(train)}; held-out abcd {len(held_abcd)}, verticals {len(held_vert)}, cfpb {len(held_cfpb)}")
