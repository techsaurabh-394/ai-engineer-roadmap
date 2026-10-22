"""Versioned prompts with content hash, rollback and A/B selection. Stdlib only."""
import hashlib
import random


class PromptRegistry:
    def __init__(self):
        self._versions = {}  # name -> list of {"version", "text", "hash"}
        self._active = {}    # name -> index into list

    def register(self, name, text):
        """Add a new version and make it active. Same text as the active one is a no-op."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        versions = self._versions.setdefault(name, [])
        if name in self._active and versions[self._active[name]]["hash"] == digest:
            return versions[self._active[name]]["version"]
        version = len(versions) + 1
        versions.append({"version": version, "text": text, "hash": digest})
        self._active[name] = len(versions) - 1
        return version

    def get(self, name, version=None):
        versions = self._versions[name]
        entry = versions[self._active[name]] if version is None else versions[version - 1]
        return dict(entry)

    def rollback(self, name):
        """Make the previous version active. Returns its version number."""
        idx = self._active[name]
        if idx == 0:
            raise ValueError("no earlier version to roll back to")
        self._active[name] = idx - 1
        return self._versions[name][idx - 1]["version"]

    def history(self, name):
        return [(v["version"], v["hash"]) for v in self._versions[name]]

    def pick_ab(self, name, version_a, version_b, user_id, percent_b=50):
        """Deterministic split: the same user always gets the same version."""
        h = int(hashlib.sha256(f"{name}:{user_id}".encode()).hexdigest(), 16) % 100
        return self.get(name, version_b if h < percent_b else version_a)

    def pick_random(self, name, version_a, version_b, percent_b=50, rng=random):
        return self.get(name, version_b if rng.random() * 100 < percent_b else version_a)


if __name__ == "__main__":
    r = PromptRegistry()
    r.register("qa", "Answer briefly.")
    r.register("qa", "Answer briefly and cite sources.")
    print(r.history("qa"), r.pick_ab("qa", 1, 2, "user-42")["version"])
    print("rolled back to", r.rollback("qa"))
