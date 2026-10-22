"""Four kinds of agent memory, as small stdlib classes.

  ShortTermMemory  - a sliding window of the last N messages (what fits in the prompt)
  LongTermMemory   - a persistent keyword-searchable store (stand-in for a vector DB)
  EpisodicMemory   - a timestamped log of what happened in past runs (events + outcomes)
  SemanticMemory   - stable facts as (subject, relation) -> value

Run:  python agent_memory.py
"""
import json
import re
from collections import deque

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = {"the", "a", "an", "is", "of", "to", "and", "in", "what", "my", "i", "do", "does", "for", "on"}


def tokenize(text):
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class ShortTermMemory:
    """Keeps only the last `max_messages` messages. Older ones fall off the front."""

    def __init__(self, max_messages=4):
        self.buffer = deque(maxlen=max_messages)

    def add(self, role, content):
        self.buffer.append({"role": role, "content": content})

    def messages(self):
        return list(self.buffer)


class LongTermMemory:
    """Stores text notes and returns the ones sharing the most keywords with a query."""

    def __init__(self):
        self.items = []

    def save(self, text, **metadata):
        self.items.append({"text": text, "words": set(tokenize(text)), **metadata})

    def search(self, query, k=2):
        q = set(tokenize(query))
        scored = [(len(q & it["words"]), i, it) for i, it in enumerate(self.items)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], -s[1]))  # more overlap first, then newer first
        return [it["text"] for _, _, it in scored[:k]]

    def dump(self):
        return json.dumps([{"text": it["text"]} for it in self.items])

    @classmethod
    def load(cls, data):
        mem = cls()
        for it in json.loads(data):
            mem.save(it["text"])
        return mem


class EpisodicMemory:
    """An append-only log of episodes: what the agent tried and how it went."""

    def __init__(self):
        self.episodes = []
        self._clock = 0

    def record(self, task, actions, outcome, lesson=""):
        self._clock += 1
        self.episodes.append({"t": self._clock, "task": task, "actions": list(actions),
                              "outcome": outcome, "lesson": lesson})

    def recall(self, task, k=1):
        q = set(tokenize(task))
        ranked = sorted(self.episodes, key=lambda e: (len(q & set(tokenize(e["task"]))), e["t"]), reverse=True)
        return [e for e in ranked if q & set(tokenize(e["task"]))][:k]

    def failures(self):
        return [e for e in self.episodes if e["outcome"] == "failure"]


class SemanticMemory:
    """Facts that stay true across conversations. Newer facts overwrite older ones."""

    def __init__(self):
        self.facts = {}

    def set(self, subject, relation, value):
        self.facts[(subject.lower(), relation.lower())] = value

    def get(self, subject, relation, default=None):
        return self.facts.get((subject.lower(), relation.lower()), default)

    def about(self, subject):
        s = subject.lower()
        return {rel: v for (subj, rel), v in self.facts.items() if subj == s}


def build_context(question, stm, ltm, sem, user="user"):
    """Assemble what an agent would put in its prompt before calling the model."""
    lines = [f"Known facts about {user}: {sem.about(user)}"]
    for note in ltm.search(question):
        lines.append(f"Relevant note: {note}")
    for m in stm.messages():
        lines.append(f"{m['role']}: {m['content']}")
    lines.append(f"user: {question}")
    return "\n".join(lines)


if __name__ == "__main__":
    stm, ltm, epi, sem = ShortTermMemory(3), LongTermMemory(), EpisodicMemory(), SemanticMemory()
    for i in range(5):
        stm.add("user", f"message {i}")
    print("short-term window:", [m["content"] for m in stm.messages()])

    ltm.save("The deploy script lives in scripts/deploy.sh")
    ltm.save("Staging database password rotates every 30 days")
    print("long-term search 'where is the deploy script':", ltm.search("where is the deploy script"))

    epi.record("deploy to staging", ["run tests", "deploy"], "failure", "run migrations before deploy")
    epi.record("deploy to staging", ["run tests", "migrate", "deploy"], "success")
    print("episodic recall:", epi.recall("deploy staging"))

    sem.set("user", "name", "Asha")
    sem.set("user", "timezone", "IST")
    print("semantic facts:", sem.about("user"))
    print("\n--- prompt context ---\n" + build_context("How do I deploy?", stm, ltm, sem))
