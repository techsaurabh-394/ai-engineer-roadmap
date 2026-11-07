#!/usr/bin/env python3
"""Put (or refresh) a "Watch it" block in every topic page that has an animation.

Run after gen_animations.py:  python scripts/embed_animations.py
The block goes just before "## How it works" and is rewritten on each run.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import anim_core as core  # noqa: E402
import anim_specs_a, anim_specs_b, anim_specs_c  # noqa: E402,F401

SITE = "https://techsaurabh-394.github.io/ai-engineer-roadmap/"
OLD = re.compile(r"<!-- anim:[^\n]*-->\n## Watch it\n.*?\n\n(?=## How it works)", re.S)


def main():
    by_topic = {}
    for a in core.REGISTRY.values():
        for t in a["topics"]:
            if t in by_topic:
                raise SystemExit(f"two animations for {t}")
            by_topic[t] = a
    changed = 0
    for rel, a in sorted(by_topic.items()):
        p = core.ROOT / rel
        text = p.read_text(encoding="utf-8")
        text = OLD.sub("", text)
        img = "../" * rel.count("/") + f"assets/anim/{a['name']}.svg"
        block = (f"<!-- anim:{a['name']} -->\n## Watch it\n"
                 f"![{a['title']}: {a['subtitle']}]({img})\n\n"
                 f"*It plays on a loop. Read the caption in the box, one step at a time. "
                 f"Want to go at your own pace? [Step through it with Previous and Next]({SITE}#{a['name']}).*\n\n")
        new, n = re.subn(r"^## How it works", lambda m: block + "## How it works", text, count=1, flags=re.M)
        if n == 0:
            raise SystemExit(f"no '## How it works' in {rel}")
        if new != p.read_text(encoding="utf-8"):
            p.write_text(new, encoding="utf-8")
            changed += 1
    print(f"{len(by_topic)} topic pages have an animation; {changed} files changed")


if __name__ == "__main__":
    main()
