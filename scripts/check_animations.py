#!/usr/bin/env python3
"""Check that every animation really shows the right caption at the right time.

For each SVG and each step, sample the middle of that step's time slot, work out
every animated opacity by linear interpolation of keyTimes/values, and require:
  * the caption line 1 of THIS step is fully visible (opacity 1)
  * the caption line 1 of every OTHER step is fully hidden (opacity 0)
Exit code 1 on any failure.   python scripts/check_animations.py
"""
import pathlib
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_animations as g  # noqa: E402

NS = "{http://www.w3.org/2000/svg}"
CUSTOM = {"attention", "chunking", "cosine-similarity"}  # hand-drawn: no caption text to match


def opacity_at(el, frac):
    anim = el.find(NS + "animate")
    if anim is None or anim.get("attributeName") != "opacity":
        return None
    kt = [float(x) for x in anim.get("keyTimes").split(";")]
    vs = [float(x) for x in anim.get("values").split(";")]
    for i in range(len(kt) - 1):
        if kt[i] <= frac <= kt[i + 1]:
            if kt[i + 1] == kt[i]:
                return vs[i + 1]
            t = (frac - kt[i]) / (kt[i + 1] - kt[i])
            return vs[i] + (vs[i + 1] - vs[i]) * t
    return vs[-1]


def main():
    bad = 0
    anims = g.build()
    for a in anims:
        root = ET.fromstring(a["svg"])
        if a["name"] in CUSTOM:
            continue
        texts = [t for t in root.iter(NS + "text") if t.find(NS + "animate") is not None]
        n = len(a["steps"])
        for i, mark in enumerate(a["marks"]):
            frac = mark / a["dur"]
            # every animated text belongs to exactly one step: match by first words of c1
            owner = {}
            for j, s in enumerate(a["steps"]):
                owner[j] = s["c1"].split()[:2]
            seen = False
            for t in texts:
                label = "".join(t.itertext()).strip()
                for j, words in owner.items():
                    if label.split()[:2] == words:
                        o = opacity_at(t, frac)
                        expect = 1.0 if j == i else 0.0
                        seen = seen or j == i
                        if o is None or abs(o - expect) > 0.01:
                            bad += 1
                            print(f"FAIL {a['name']} step {i + 1}/{n}: {label[:50]!r} opacity {o} expected {expect}")
            if not seen:
                bad += 1
                print(f"FAIL {a['name']} step {i + 1}: caption missing from svg")
    print(f"checked {len(anims)} animations, {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
