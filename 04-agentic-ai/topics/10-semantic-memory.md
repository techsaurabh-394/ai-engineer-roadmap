# Semantic Memory
> **Level:** Intermediate · **Reading time:** 5 min · **Phase:** 4 · Agentic AI

**In one line:** Semantic memory stores stable facts (about the user, the world, or the domain) that stay true across conversations, such as "Asha's timezone is IST".

## Explain it like I'm new
You know that Paris is the capital of France, but you probably don't remember the day you learned it. That kind of knowledge, facts without the story of how you got them, is what psychologists call **semantic memory**. Remembering a specific event ("my trip to Paris last May") is **episodic memory** ([next topic](11-episodic-memory.md)).

For an agent, semantic memory is a profile or fact sheet:
- About the user: name, role, language, preferences ("prefers short answers").
- About the domain: "Invoice IDs start with INV-", "Our refund window is 30 days".
- About the environment: "The prod database is read-only for this agent".

These facts are small, structured, and useful on almost every turn, so many systems load them into every prompt rather than searching for them.

## How it works
1. **Represent** facts in a structured form, for example `(subject, relation) -> value`: `("user", "timezone") -> "IST"`.
2. **Write:** add facts when the user states them or when an extraction step finds them. If a fact already exists, **overwrite** (or keep history with timestamps).
3. **Read:** load all facts about the current user (usually small), or query by subject.
4. **Resolve conflicts:** newer explicit statements beat older or inferred ones. Keep the source.
5. **Expose controls:** let users view, correct, and delete what's stored.

```text
subject   relation    value          source          updated
-------   ---------   ------------   -------------   ----------
user      name        Asha           user said       2026-10-01
user      timezone    IST            user said       2026-10-01
user      city        Delhi          user said       2026-10-20   <- replaced "Pune"
domain    refund_days 30             policy doc      2026-09-15
```

## Code example
From [`agent_memory.py`](../code/agent_memory.py):

```python
class SemanticMemory:
    def __init__(self):
        self.facts = {}

    def set(self, subject, relation, value):
        self.facts[(subject.lower(), relation.lower())] = value   # newer overwrites older

    def get(self, subject, relation, default=None):
        return self.facts.get((subject.lower(), relation.lower()), default)

    def about(self, subject):
        s = subject.lower()
        return {rel: v for (subj, rel), v in self.facts.items() if subj == s}

sem = SemanticMemory()
sem.set("User", "city", "Pune")
sem.set("user", "city", "Delhi")          # the user moved
sem.set("user", "timezone", "IST")
print(sem.get("USER", "City"))            # Delhi
print(sem.about("user"))                  # {'city': 'Delhi', 'timezone': 'IST'}
```

To let the model write facts itself, give it a tool like `remember_fact(subject, relation, value)` with a strict schema, and validate it in code ([Function Calling](03-function-calling.md)).

## When to use it / when not to
- **Use** for personalization (preferences, profile) and domain rules the agent needs constantly.
- **Use** structured storage (key-value or a table), not free text, so facts can be updated and deleted cleanly.
- **Don't** store sensitive personal data unless you need it and have consent and a retention policy.

| | Semantic | Episodic | Long-term notes |
|---|---|---|---|
| Content | facts | events + outcomes | free text |
| Shape | key → value | timestamped records | documents/chunks |
| Retrieval | load all for the user, or exact lookup | similar past tasks | keyword/vector search |

## Common mistakes
- Storing facts as free text that can't be updated ("user lives in Pune" and later "user lives in Delhi" both remain).
- Letting the model infer and save guesses ("user is probably a manager") as facts.
- No way for users to see or delete their stored facts.
- Loading a huge fact sheet into every prompt.

## Interview questions
**1. What is the difference between semantic and episodic memory in agents?**
<details><summary>Answer</summary>
Semantic memory holds general facts that stay true (preferences, domain rules). Episodic memory holds records of specific past events: what was attempted, when, and how it turned out.
</details>

**2. How do you handle a fact that changes?**
<details><summary>Answer</summary>
Store facts with a unique key so a new value overwrites the old one (or versions it with a timestamp and source), and prefer explicit user statements over inferred values.
</details>

**3. Why prefer structured storage for semantic memory?**
<details><summary>Answer</summary>
Structured keys make facts easy to update, delete, audit, and load exactly. Free-text memories duplicate and contradict each other over time.
</details>

## Go deeper
- [Cognitive Architectures for Language Agents (CoALA, Sumers et al., 2023)](https://arxiv.org/abs/2309.02427)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)

**Prev:** [Long-term Memory](09-long-term-memory.md) · **Next:** [Episodic Memory](11-episodic-memory.md) · **Back to phase:** [Phase 4](../README.md)
