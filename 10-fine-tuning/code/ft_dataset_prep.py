"""Prepare a chat fine-tuning dataset: validate, de-duplicate, scrub PII, split without leakage,
measure lengths, and build loss-masked labels. Standard library only, deterministic.

Pipeline order matters:  validate -> dedupe -> scrub PII -> split by group -> length stats.

Run:  python ft_dataset_prep.py
"""
import random
import re
import statistics

ROLES = {"system", "user", "assistant"}


# ---------------------------------------------------------------- 1. validation
def validate_record(rec):
    """Return a list of problems (empty list = valid) for one chat-format record.

    Shape: {"id": str, "group": str, "messages": [{"role": ..., "content": ...}, ...]}
    Rules: optional system message first, then user/assistant alternating, ending with assistant,
    every content a non-empty string.
    """
    problems = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    if not isinstance(rec.get("group"), str) or not rec["group"]:
        problems.append("missing group")
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return problems + ["messages missing or empty"]
    roles = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            problems.append(f"message {i}: bad role")
            continue
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            problems.append(f"message {i}: empty content")
        roles.append(m["role"])
    body = roles[1:] if roles and roles[0] == "system" else roles
    if "system" in body:
        problems.append("system message not first")
    elif body:
        if body[-1] != "assistant":
            problems.append("last message is not assistant")
        for j, role in enumerate(body):
            if role != ("user" if j % 2 == 0 else "assistant"):
                problems.append("roles do not alternate user/assistant")
                break
    return problems


def validate_all(records):
    """Return (valid_records, [(id, problems)])."""
    ok, bad = [], []
    for rec in records:
        p = validate_record(rec)
        if p:
            bad.append((rec.get("id", "?") if isinstance(rec, dict) else "?", p))
        else:
            ok.append(rec)
    return ok, bad


# ---------------------------------------------------------------- 2. de-duplication
def normalize(text):
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def record_text(rec):
    return " ".join(m["content"] for m in rec["messages"] if m["role"] != "system")


def jaccard(a, b):
    """Overlap of two word sets: |A and B| / |A or B|."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def dedupe(records, threshold=0.8):
    """Drop exact duplicates (same normalized text) and near duplicates (Jaccard >= threshold).

    Returns (kept, dropped) where dropped is [(id, kept_id, reason, score)]. First occurrence wins.
    Near-duplicate search is O(n^2): fine for thousands of rows; use MinHash/LSH beyond that.
    """
    kept, dropped, seen = [], [], {}
    norms = []
    for rec in records:
        n = normalize(record_text(rec))
        if n in seen:
            dropped.append((rec["id"], seen[n], "exact", 1.0))
            continue
        hit = None
        for other, on in zip(kept, norms):
            s = jaccard(n, on)
            if s >= threshold:
                hit = (other["id"], s)
                break
        if hit:
            dropped.append((rec["id"], hit[0], "near", round(hit[1], 3)))
            continue
        seen[n] = rec["id"]
        kept.append(rec)
        norms.append(n)
    return kept, dropped


# ---------------------------------------------------------------- 3. PII scrub
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,5}[\s-]\d{3,4}(?:[\s-]\d{2,4})?(?![\w-])")


def scrub_pii(text):
    """Replace emails and phone-like numbers with placeholders. Returns (clean_text, counts).

    A regex scrub is a first pass only: it misses names, addresses and anything unusual.
    """
    text, n_email = EMAIL.subn("[EMAIL]", text)
    text, n_phone = PHONE.subn("[PHONE]", text)
    return text, {"email": n_email, "phone": n_phone}


def scrub_record(rec):
    out = {**rec, "messages": []}
    total = {"email": 0, "phone": 0}
    for m in rec["messages"]:
        c, counts = scrub_pii(m["content"])
        total["email"] += counts["email"]
        total["phone"] += counts["phone"]
        out["messages"].append({**m, "content": c})
    return out, total


# ---------------------------------------------------------------- 4. leakage-free split
def split_by_group(records, fractions=(0.7, 0.15, 0.15), seed=0):
    """Split so that every group id lands in exactly one of train / val / test.

    Records from one group (same customer, same document, same conversation) share vocabulary and
    facts; if they straddle the split, the test score is inflated. Whole groups move together.
    """
    groups = {}
    for rec in records:
        groups.setdefault(rec["group"], []).append(rec)
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    total = len(records)
    targets = [f * total for f in fractions]
    splits = [[], [], []]
    for g in ids:
        # put the group into the split that is furthest below its target
        k = max(range(3), key=lambda i: targets[i] - len(splits[i]))
        splits[k].extend(groups[g])
    return splits[0], splits[1], splits[2]


def group_overlap(a, b):
    return sorted({r["group"] for r in a} & {r["group"] for r in b})


def text_overlap(a, b):
    """Normalized texts that appear in both sets (should be none)."""
    return sorted({normalize(record_text(r)) for r in a} & {normalize(record_text(r)) for r in b})


# ---------------------------------------------------------------- 5. token-length stats
def whitespace_tokens(text):
    """A crude tokenizer: real tokenizers (BPE) produce roughly 1.3 tokens per English word."""
    return text.split()


def length_stats(records):
    lens = sorted(len(whitespace_tokens(record_text(r))) for r in records)
    n = len(lens)

    def pct(p):
        return lens[min(n - 1, int(p * n))]

    return {"n": n, "min": lens[0], "mean": round(statistics.mean(lens), 2), "median": statistics.median(lens),
            "p90": pct(0.9), "max": lens[-1]}


# ---------------------------------------------------------------- 6. chat template + loss masking
IGNORE_INDEX = -100


def render_chat(messages):
    """Turn messages into (tokens, assistant_mask) with a simple made-up template.

    Template:  <|role|> word word ... <|end|>   (one block per message)
    The mask is 1 for the words of assistant messages AND their <|end|> token (so the model learns
    to stop), and 0 for everything else, including the "<|assistant|>" header that we supply at
    inference time.
    """
    tokens, mask = [], []
    for m in messages:
        header = f"<|{m['role']}|>"
        body = whitespace_tokens(m["content"])
        tokens += [header] + body + ["<|end|>"]
        is_a = 1 if m["role"] == "assistant" else 0
        mask += [0] + [is_a] * len(body) + [is_a]
    return tokens, mask


def build_labels(tokens, assistant_mask, vocab=None):
    """labels[i] = token id (or the token itself) where mask is 1, else -100 so the loss ignores it.

    If `vocab` (token -> id) is given, labels are ids; otherwise they are the tokens themselves.
    """
    if len(tokens) != len(assistant_mask):
        raise ValueError("tokens and mask must have the same length")
    out = []
    for t, m in zip(tokens, assistant_mask):
        if not m:
            out.append(IGNORE_INDEX)
        else:
            out.append(vocab[t] if vocab else t)
    return out


def masked_mean_loss(token_losses, labels):
    """Average the per-token loss over positions whose label is not -100."""
    kept = [l for l, y in zip(token_losses, labels) if y != IGNORE_INDEX]
    return sum(kept) / len(kept) if kept else 0.0


# ---------------------------------------------------------------- built-in sample (30 records)
def _rec(i, group, user, assistant, system=None):
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"id": f"r{i:02d}", "group": group, "messages": msgs}


SYSTEM = "You are a concise support assistant for Acme Shop."

SAMPLE = [
    _rec(1, "t01", "How do I reset my password?", "Open Settings, choose Security, then press Reset password. We email you a link.", SYSTEM),
    _rec(2, "t01", "The reset link has expired. What now?", "Links last 30 minutes. Request a new one from the login page.", SYSTEM),
    _rec(3, "t02", "Can I cancel my order?", "Yes, before it ships. Go to Orders and press Cancel.", SYSTEM),
    _rec(4, "t02", "Can I cancel my order?!", "Yes, before it ships. Go to Orders and press Cancel.", SYSTEM),        # exact (normalized) duplicate of r03
    _rec(5, "t03", "Where is my parcel?", "Open Orders and press Track. You will see the carrier and the date.", SYSTEM),
    _rec(6, "t03", "Where is my parcel right now?", "Open Orders and press Track. You will see the carrier and the date.", SYSTEM),  # near duplicate of r05
    _rec(7, "t04", "Do you ship abroad?", "We ship to 12 countries. The list is on the Shipping page.", SYSTEM),
    _rec(8, "t04", "What are your opening hours?", "Support is open Monday to Friday, 9 to 5.", SYSTEM),
    _rec(9, "t05", "Email me the invoice please, I am jane.doe@example.com", "Done. We sent the invoice to jane.doe@example.com.", SYSTEM),
    _rec(10, "t05", "Call me on 020 7946 0123 about my refund", "We will call 020 7946 0123 within one working day.", SYSTEM),
    _rec(11, "t06", "How long do refunds take?", "Refunds reach your card in 5 to 7 working days.", SYSTEM),
    _rec(12, "t06", "Can I get a refund after 60 days?", "Refunds are accepted within 30 days of delivery only.", SYSTEM),
    _rec(13, "t07", "Do you sell gift cards?", "Yes. Gift cards from 10 to 200 are on the Gifts page.", SYSTEM),
    _rec(14, "t07", "Can I pay with a gift card and a credit card?", "Yes. Apply the gift card first, then pay the rest by card.", SYSTEM),
    _rec(15, "t08", "How do I change my delivery address?", "Before dispatch, open the order and press Edit address.", SYSTEM),
    _rec(16, "t08", "My item arrived damaged.", "Sorry about that. Send a photo through Help and we will replace it.", SYSTEM),
    _rec(17, "t09", "Is there a student discount?", "Yes, 10 percent with a verified student email.", SYSTEM),
    _rec(18, "t09", "How do I delete my account?", "Open Settings, then Privacy, then press Delete account.", SYSTEM),
    _rec(19, "t10", "Which payment methods do you accept?", "Visa, Mastercard, PayPal and bank transfer.", SYSTEM),
    _rec(20, "t10", "Is my card data stored?", "No. Payments are handled by our payment partner and we keep no card numbers.", SYSTEM),
    _rec(21, "t11", "Can I change the size of an item I ordered?", "Before dispatch yes. After that, return it and order the new size.", SYSTEM),
    _rec(22, "t11", "Do you offer next-day delivery?", "Yes, on orders placed before 2 pm in selected cities.", SYSTEM),
    _rec(23, "t12", "Why was my payment declined?", "Banks decline for many reasons. Check the card details, then contact your bank.", SYSTEM),
    _rec(24, "t12", "How do I contact a human?", "Press Talk to a person in Help. Wait times are shown on the button.", SYSTEM),
    _rec(25, "t13", "Do you have a loyalty scheme?", "Yes. You earn 1 point per 1 spent, and 100 points give 5 off.", SYSTEM),
    _rec(26, "t13", "How do I unsubscribe from emails?", "Use the unsubscribe link at the bottom of any email.", SYSTEM),
    _rec(27, "t14", "Where is your warehouse?", "", SYSTEM),                                                           # invalid: empty reply
    {"id": "r28", "group": "t14", "messages": [{"role": "user", "content": "Hello?"},
                                                {"role": "bot", "content": "Hi there!"}]},                           # invalid: bad role
    _rec(29, "t15", "Can I track my parcel from my phone?", "Yes. The Acme app shows live tracking for every order.", SYSTEM),
    _rec(30, "t15", "Does the app work offline?", "You can browse saved items offline. Orders need a connection.", SYSTEM),
]


def run_pipeline(records, seed=0):
    """Run the whole pipeline and return a dict of every intermediate result."""
    valid, invalid = validate_all(records)
    kept, dropped = dedupe(valid)
    scrubbed, totals = [], {"email": 0, "phone": 0}
    for r in kept:
        s, c = scrub_record(r)
        scrubbed.append(s)
        totals["email"] += c["email"]
        totals["phone"] += c["phone"]
    train, val, test = split_by_group(scrubbed, seed=seed)
    return dict(valid=valid, invalid=invalid, kept=kept, dropped=dropped, scrubbed=scrubbed, pii=totals,
                train=train, val=val, test=test)


if __name__ == "__main__":
    out = run_pipeline(SAMPLE)
    print(f"records in: {len(SAMPLE)}")
    print(f"invalid   : {len(out['invalid'])} -> {out['invalid']}")
    print(f"duplicates: {len(out['dropped'])} -> {out['dropped']}")
    print(f"kept      : {len(out['kept'])}")
    print(f"PII found : {out['pii']}")
    print(f"split     : train {len(out['train'])} | val {len(out['val'])} | test {len(out['test'])}")
    print(f"group overlap train/val {group_overlap(out['train'], out['val'])}, train/test "
          f"{group_overlap(out['train'], out['test'])}, val/test {group_overlap(out['val'], out['test'])}")
    print(f"text overlap train/test: {len(text_overlap(out['train'], out['test']))}")
    print("length stats (whitespace tokens, all kept):", length_stats(out["kept"]))
    ex = [r for r in out["scrubbed"] if r["id"] == "r09"][0]
    print("scrubbed r09:", ex["messages"][1]["content"], "|", ex["messages"][2]["content"])

    # a random split would leak: count groups that straddle train and test
    rng = random.Random(0)
    shuffled = out["scrubbed"][:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.7)
    print("random (non-group) split: groups shared between train and test =",
          len(group_overlap(shuffled[:cut], shuffled[cut:])))

    toks, mask = render_chat(out["kept"][1]["messages"][1:])
    print("\ntokens:", toks)
    print("mask  :", mask)
    labels = build_labels(toks, mask)
    print("labels:", labels)
    print(f"trained positions: {sum(mask)} of {len(mask)}")

    short_toks, short_mask = render_chat([{"role": "user", "content": "hi there"},
                                          {"role": "assistant", "content": "hello friend"}])
    short_labels = build_labels(short_toks, short_mask)
    losses = [4.0, 3.5, 3.0, 2.5, 2.0, 1.0, 3.0, 2.0]   # made-up per-token losses for the 8 tokens
    print("\nshort chat tokens:", short_toks)
    print("short chat labels:", short_labels)
    print(f"mean loss over all 8 tokens: {sum(losses) / len(losses):.3f} | masked (reply only): "
          f"{masked_mean_loss(losses, short_labels):.3f}")
