"""Context engineering and prompt caching: cost arithmetic, stdlib only.

Prices are ILLUSTRATIVE (made up, per million input tokens). Real providers
differ: check your own price page.

Run:  python fa_context.py
"""
PRICE_INPUT = 3.00        # $ per 1M uncached input tokens (illustrative)
PRICE_CACHED = 0.30       # $ per 1M cached input tokens (illustrative, 10 percent)
PRICE_CACHE_WRITE = 3.75  # $ per 1M tokens when writing the cache (illustrative)


def prefix_cache_hits(prev_tokens, new_tokens):
    """Number of leading tokens two requests share. A cache reuses only a PREFIX."""
    n = 0
    for a, b in zip(prev_tokens, new_tokens):
        if a != b:
            break
        n += 1
    return n


def request_cost(total_tokens, cached_tokens, write_tokens=0):
    """Dollar cost of the input side of one request."""
    fresh = total_tokens - cached_tokens - write_tokens
    return (fresh * PRICE_INPUT + cached_tokens * PRICE_CACHED + write_tokens * PRICE_CACHE_WRITE) / 1e6


def conversation_cost(system_tokens, turn_tokens, turns, cached):
    """Input cost of a chat where every turn resends system prompt + all history.
    With caching, each turn reads the previous prefix from cache and writes the new part."""
    total = 0.0
    history = system_tokens
    for t in range(turns):
        if cached and t > 0:
            total += request_cost(history + turn_tokens, cached_tokens=history, write_tokens=turn_tokens)
        else:
            total += request_cost(history + turn_tokens, 0)
        history += turn_tokens
    return total


def break_even_reads(price_input=PRICE_INPUT, price_cached=PRICE_CACHED, price_write=PRICE_CACHE_WRITE):
    """Reads of a cached block needed before caching beats not caching (write once, then read)."""
    extra_write = price_write - price_input
    saving_per_read = price_input - price_cached
    return extra_write / saving_per_read


if __name__ == "__main__":
    system = list(range(2000))
    a = system + [9001, 9002]
    b_ok = system + [9001, 9003]
    b_bad = [777] + system[1:] + [9001]   # a timestamp changed the very first token
    print("shared prefix, edit at the end :", prefix_cache_hits(a, b_ok), "of", len(a))
    print("shared prefix, edit at the start:", prefix_cache_hits(a, b_bad), "of", len(a))
    for cached in (False, True):
        c = conversation_cost(system_tokens=6000, turn_tokens=400, turns=20, cached=cached)
        print(f"20-turn chat, 6000-token system prompt, cached={cached}: ${c:.4f} input cost")
    print("break-even reads of a cached block:", round(break_even_reads(), 3))
