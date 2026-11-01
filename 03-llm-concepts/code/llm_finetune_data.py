"""Check a tiny fine-tuning dataset before you pay to train on it. Standard library only.

Most fine-tuning failures are data problems: empty answers, duplicates, and
no held-out examples to measure with. This script finds the first two and
makes the third.
"""
import json
import random


def clean(rows):
    """Return (kept_rows, problems). A problem is (row_index, reason)."""
    seen, kept, problems = set(), [], []
    for i, r in enumerate(rows):
        if not r.get("prompt", "").strip() or not r.get("completion", "").strip():
            problems.append((i, "empty field"))
            continue
        key = (r["prompt"], r["completion"])
        if key in seen:
            problems.append((i, "duplicate"))
            continue
        seen.add(key)
        kept.append(r)
    return kept, problems


def split(rows, valid_fraction=0.2, seed=0):
    """Shuffle with a fixed seed and hold some rows out for validation."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_valid = max(1, round(len(rows) * valid_fraction)) if len(rows) > 1 else 0
    return rows[n_valid:], rows[:n_valid]


def to_jsonl(rows):
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


if __name__ == "__main__":
    data = [
        {"prompt": "Refund for order 1001?", "completion": "Refunds are issued within 5 days."},
        {"prompt": "Refund for order 1002?", "completion": "Refunds are issued within 5 days."},
        {"prompt": "Can I cancel?", "completion": "Yes, before it ships."},
        {"prompt": "Can I cancel?", "completion": "Yes, before it ships."},
        {"prompt": "Where is my parcel?", "completion": ""},
        {"prompt": "Change address?", "completion": "Yes, before it ships."},
        {"prompt": "Do you ship abroad?", "completion": "We ship to 12 countries."},
        {"prompt": "Opening hours?", "completion": "Mon to Fri, 9 to 5."},
    ]
    kept, problems = clean(data)
    train, valid = split(kept)
    print(f"kept {len(kept)} of {len(data)} | problems: {problems}")
    print(f"train {len(train)} | validation {len(valid)}")
    print("first line of the file you would upload:")
    print(to_jsonl(train[:1]))
