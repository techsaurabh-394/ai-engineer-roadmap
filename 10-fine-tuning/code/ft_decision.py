"""Prompting, RAG or fine-tuning? A break-even cost calculator. Standard library only.

Fine-tuning costs money up front. It can save money later if it lets you shrink a long prompt
(instructions plus examples) that you send on every request. This script answers one question:
after how many months does the saving pay back the cost of the training project?

IMPORTANT: every price below is an ILLUSTRATIVE parameter in made-up "currency units", not a
real vendor price. Replace them with your provider's current price list before deciding anything.

Run:  python ft_decision.py
"""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Scenario:
    requests_per_month: int = 200_000
    prompt_tokens_before: int = 1_200     # long instructions + few-shot examples sent every call
    prompt_tokens_after: int = 150        # a fine-tuned model needs only the user's question
    output_tokens: int = 150
    price_in_per_million: float = 3.00    # ILLUSTRATIVE units per 1M input tokens
    price_out_per_million: float = 15.00  # ILLUSTRATIVE units per 1M output tokens
    tuned_price_multiplier: float = 1.5   # some providers charge more per token for a tuned model
    training_cost: float = 400.00         # ILLUSTRATIVE one-off compute / provider training fee
    data_cost: float = 2_000.00           # ILLUSTRATIVE one-off: collecting, cleaning, labelling
    upkeep_per_month: float = 50.00       # ILLUSTRATIVE: re-evaluation and re-training amortised


def tokens_per_month(requests, prompt_tokens, output_tokens):
    """Return (input_tokens, output_tokens) per month."""
    return requests * prompt_tokens, requests * output_tokens


def monthly_cost(requests, prompt_tokens, output_tokens, price_in, price_out, multiplier=1.0):
    t_in, t_out = tokens_per_month(requests, prompt_tokens, output_tokens)
    return multiplier * (t_in * price_in + t_out * price_out) / 1_000_000


def costs(s: Scenario):
    """Return (monthly cost with the long prompt, monthly cost after fine-tuning incl. upkeep)."""
    before = monthly_cost(s.requests_per_month, s.prompt_tokens_before, s.output_tokens,
                          s.price_in_per_million, s.price_out_per_million)
    after = monthly_cost(s.requests_per_month, s.prompt_tokens_after, s.output_tokens,
                         s.price_in_per_million, s.price_out_per_million, s.tuned_price_multiplier)
    return before, after + s.upkeep_per_month


def months_to_break_even(s: Scenario):
    """Months until the cumulative saving covers the one-off cost. None if it never does."""
    before, after = costs(s)
    saving = before - after
    if saving <= 0:
        return None
    return (s.training_cost + s.data_cost) / saving


def break_even_requests(s: Scenario, months=12):
    """Smallest monthly request volume that pays back within `months` (found by bisection)."""
    lo, hi = 1, 10_000_000_000
    while lo < hi:
        mid = (lo + hi) // 2
        m = months_to_break_even(replace(s, requests_per_month=mid))
        if m is not None and m <= months:
            hi = mid
        else:
            lo = mid + 1
    return lo


def recommend(needs_new_facts, facts_change_often, needs_citations, prompt_fix_works,
              has_clean_examples, high_volume):
    """A small rule table. Returns the cheapest approach that fits, as a string."""
    if prompt_fix_works:
        return "prompting"
    if needs_new_facts or facts_change_often or needs_citations:
        return "RAG"
    if has_clean_examples and high_volume:
        return "fine-tuning"
    if has_clean_examples:
        return "prompting with examples first, then consider fine-tuning"
    return "prompting; collect examples before fine-tuning"


if __name__ == "__main__":
    s = Scenario()
    before, after = costs(s)
    t_in, t_out = tokens_per_month(s.requests_per_month, s.prompt_tokens_before, s.output_tokens)
    print("ILLUSTRATIVE prices (not real vendor prices)")
    print(f"tokens per month before: {t_in:,} in, {t_out:,} out")
    t_in2, _ = tokens_per_month(s.requests_per_month, s.prompt_tokens_after, s.output_tokens)
    print(f"tokens per month after : {t_in2:,} in, {t_out:,} out")
    print(f"monthly cost, long prompt : {before:,.2f}")
    print(f"monthly cost, fine-tuned  : {after:,.2f} (incl. upkeep {s.upkeep_per_month:.0f})")
    print(f"monthly saving            : {before - after:,.2f}")
    print(f"one-off cost              : {s.training_cost + s.data_cost:,.2f}")
    m = months_to_break_even(s)
    assert m is not None
    print(f"months to break even      : {m:.2f}  (about {math.ceil(m)} months)")

    small = replace(s, requests_per_month=10_000)
    b2, a2 = costs(small)
    m2 = months_to_break_even(small)
    print(f"At 10,000 requests/month: long prompt {b2:,.2f} vs fine-tuned {a2:,.2f}; "
          f"months to break even: {'never' if m2 is None else round(m2, 1)}")
    print(f"Volume needed to pay back within 12 months: {break_even_requests(s, 12):,} requests/month")

    nosave = replace(s, prompt_tokens_before=200, prompt_tokens_after=150)
    print(f"\nIf the prompt was already short (200 -> 150 tokens): break even = {months_to_break_even(nosave)}")

    print("\nDecision rules:")
    print(" facts change often          ->", recommend(False, True, False, False, True, True))
    print(" style, clean data, big scale ->", recommend(False, False, False, False, True, True))
    print(" a clearer prompt fixes it    ->", recommend(False, False, False, True, True, True))
