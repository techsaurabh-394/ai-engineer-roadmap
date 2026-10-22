# Short-term Memory
> **Level:** Beginner · **Reading time:** 5 min · **Phase:** 4 · Agentic AI

**In one line:** Short-term memory is the recent conversation (and tool results) that you re-send to the model on every turn, usually trimmed to a window so it fits the context.

## Explain it like I'm new
Think of a sticky note on your desk during a phone call. You jot down the last few things the caller said so you can follow along. When the note fills up, you erase the oldest lines. You don't keep a full transcript on the note, just what you need right now.

In an agent, short-term memory (also called **working memory** or the **conversation buffer**) is the list of messages for the current session: user messages, assistant replies, tool calls, and tool results. On each turn, you send this list to the model again. That's how it "remembers" what you said two messages ago.

The problem: the list keeps growing. Each turn costs more tokens, and eventually it doesn't fit in the [context window](../../03-llm-concepts/topics/08-context-window.md). So you need a policy for trimming.

## How it works
Common policies:

1. **Sliding window (buffer window):** keep only the last N messages (or last N tokens). Simple and predictable.
2. **Summary + window:** keep the last N messages in full, plus a running summary of older ones written by the model.
3. **Token budget:** drop the oldest messages until the total fits a budget.
4. **Always keep:** the system prompt, the original goal, and any pinned facts, no matter what.

```text
window size = 3
turn 1: [m0]
turn 2: [m0, m1]
turn 3: [m0, m1, m2]
turn 4: [m1, m2, m3]        <- m0 falls off
turn 5: [m2, m3, m4]        <- m1 falls off
```

**Tool-call pairs.** If you trim in the middle of a tool call and its result, some APIs reject the history (a result without its call). Trim at turn boundaries.

## Code example
From [`agent_memory.py`](../code/agent_memory.py), using `collections.deque` with `maxlen`, which drops the oldest item automatically:

```python
from collections import deque

class ShortTermMemory:
    def __init__(self, max_messages=4):
        self.buffer = deque(maxlen=max_messages)

    def add(self, role, content):
        self.buffer.append({"role": role, "content": content})

    def messages(self):
        return list(self.buffer)

stm = ShortTermMemory(max_messages=3)
for i in range(5):
    stm.add("user", f"message {i}")
print([m["content"] for m in stm.messages()])   # ['message 2', 'message 3', 'message 4']
```

A summary + window version (the summarizer would be an LLM call in practice):

```python
def compact(messages, keep_last=6, summarize=lambda msgs: f"{len(msgs)} earlier messages about deployment"):
    if len(messages) <= keep_last:
        return messages
    old, recent = messages[:-keep_last], messages[-keep_last:]
    return [{"role": "user", "content": f"Summary of earlier conversation: {summarize(old)}"}] + recent
```

## When to use it / when not to
| Policy | Pros | Cons |
|---|---|---|
| Full history | nothing lost | cost grows, overflows eventually |
| Sliding window | simple, bounded cost | forgets older details |
| Summary + window | keeps gist of long chats | summary can lose or distort details, extra call |
| Token budget | precise fit | needs token counting |

- For short sessions, keeping the full history is fine.
- For long sessions, use summary + window, and move durable facts into [semantic memory](10-semantic-memory.md).

## Common mistakes
- Dropping the system prompt or the original goal when trimming.
- Splitting a tool call from its result.
- Counting messages instead of tokens when one message is huge (a pasted document).
- Never summarizing, so the agent forgets a decision made 30 turns ago.

## Interview questions
**1. What is a buffer window memory?**
<details><summary>Answer</summary>
A short-term memory that keeps only the last N messages (or tokens) of the conversation, dropping the oldest, so the prompt stays within budget.
</details>

**2. How do you keep a long chat coherent without exceeding the context window?**
<details><summary>Answer</summary>
Keep the system prompt and recent turns verbatim, replace older turns with a running summary, and store durable facts in long-term or semantic memory to retrieve when relevant.
</details>

**3. Why must tool calls and results be trimmed together?**
<details><summary>Answer</summary>
APIs expect each tool result to follow its matching tool call. Removing one but not the other produces an invalid message history or confuses the model.
</details>

## Go deeper
- [Anthropic docs: context windows](https://docs.anthropic.com/en/docs/build-with-claude/context-windows)
- [MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)](https://arxiv.org/abs/2310.08560)

**Prev:** [Memory](07-memory.md) · **Next:** [Long-term Memory](09-long-term-memory.md) · **Back to phase:** [Phase 4](../README.md)
