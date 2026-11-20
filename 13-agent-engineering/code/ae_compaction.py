"""Context compaction for long agent runs: three strategies, one measurable score.

A run is a list of messages with a token size. Some messages carry a FACT the
agent will need later. We compact to a token budget and count how many facts
survive. Summaries are simulated: a summary keeps `ratio` of the tokens and
keeps the facts only if they were flagged as "kept by the summary" (a real
summariser may drop them, which is exactly the risk).

Run:  python ae_compaction.py
"""


def make_run(turns=30, tool_every=2, tool_tokens=900, chat_tokens=120):
    """Deterministic run. Fact messages appear at turns 4, 11 and 24; the fact at
    turn 11 is in a tool result, the other two are in chat."""
    msgs = []
    for t in range(turns):
        is_tool = t % tool_every == 1
        m = {"turn": t, "role": "tool" if is_tool else "chat",
             "tokens": tool_tokens if is_tool else chat_tokens, "fact": None, "pinned": False}
        msgs.append(m)
    msgs[4]["fact"] = "order id is 8841"
    msgs[11]["fact"] = "refund limit is 50 dollars"
    msgs[24]["fact"] = "customer prefers email"
    msgs[0]["pinned"] = True          # the system prompt / task statement
    return msgs


def total(msgs):
    return sum(m["tokens"] for m in msgs)


def facts(msgs):
    return {m["fact"] for m in msgs if m["fact"]}


def truncate_oldest(msgs, budget):
    """Drop oldest unpinned messages until the run fits."""
    out = list(msgs)
    i = 0
    while total(out) > budget and i < len(out):
        if out[i]["pinned"]:
            i += 1
            continue
        out.pop(i)
    return out


def clear_old_tool_results(msgs, keep_recent=4, stub_tokens=12):
    """Replace old tool outputs with a short stub, keep chat. Facts inside a cleared
    tool result are lost unless the agent wrote them down elsewhere."""
    tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    old = set(tool_idx[:-keep_recent]) if keep_recent else set(tool_idx)
    out = []
    for i, m in enumerate(msgs):
        if i in old:
            out.append({**m, "tokens": stub_tokens, "fact": None})
        else:
            out.append(dict(m))
    return out


def summarise_old(msgs, keep_recent=6, ratio=0.1, summary_keeps=()):
    """Collapse everything except pinned and the last `keep_recent` messages into one
    summary of ratio * tokens. `summary_keeps` lists facts the summary happened to keep."""
    recent = msgs[-keep_recent:]
    head = [m for m in msgs[:-keep_recent]]
    pinned = [m for m in head if m["pinned"]]
    rest = [m for m in head if not m["pinned"]]
    kept_facts = [m["fact"] for m in rest if m["fact"] in summary_keeps]
    summary = {"turn": -1, "role": "summary", "tokens": max(1, int(sum(m["tokens"] for m in rest) * ratio)),
               "fact": None, "pinned": False, "facts": kept_facts}
    return pinned + [summary] + recent


def surviving_facts(msgs):
    got = set()
    for m in msgs:
        if m.get("fact"):
            got.add(m["fact"])
        got.update(m.get("facts", []))
    return got


def notes_pass(msgs, notes):
    """Agent-written notes: a tiny pinned message that restates facts as they are found.
    `notes` is a list of fact strings; each costs 15 tokens."""
    note = {"turn": -2, "role": "notes", "tokens": 15 * len(notes), "fact": None, "pinned": True, "facts": list(notes)}
    return [msgs[0], note] + msgs[1:]


if __name__ == "__main__":
    run = make_run()
    allfacts = facts(run)
    print("full run:", total(run), "tokens,", len(allfacts), "facts")
    budget = 6000
    print("budget:", budget)
    t = truncate_oldest(run, budget)
    print(f"truncate oldest      : {total(t):>5} tokens, facts kept {len(surviving_facts(t))}/{len(allfacts)}")
    c = clear_old_tool_results(run)
    print(f"clear old tool output: {total(c):>5} tokens, facts kept {len(surviving_facts(c))}/{len(allfacts)}")
    s = summarise_old(run, summary_keeps={"order id is 8841"})
    print(f"summarise (keeps 1)  : {total(s):>5} tokens, facts kept {len(surviving_facts(s))}/{len(allfacts)}")
    n = clear_old_tool_results(notes_pass(run, ["refund limit is 50 dollars"]))
    print(f"notes + clear tools  : {total(n):>5} tokens, facts kept {len(surviving_facts(n))}/{len(allfacts)}")
