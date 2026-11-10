#!/usr/bin/env python3
"""Click through the built site in headless Chrome and fail on any console error (needs selenium).

    python scripts/build_site.py
    python scripts/check_site.py                       # serves _site/ itself on a free port
    python scripts/check_site.py --url https://techsaurabh-394.github.io/ai-engineer-roadmap/
    python scripts/check_site.py --shots shots/        # also saves screenshots (375 and 1280 px, light and dark)

Options: --chrome PATH (Chrome/Chromium binary), --driver PATH (chromedriver). Selenium 4 finds both itself
when they are installed normally. Not part of pytest: it needs a browser.
"""
import argparse
import functools
import http.server
import pathlib
import socketserver
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LABS = ["sampling", "attention", "toyllm", "chunking", "similarity", "search"]


def serve(directory):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(directory)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/"


def driver(args):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    o = Options()
    for a in ("--headless=new", "--no-sandbox", "--hide-scrollbars", "--disable-gpu"):
        o.add_argument(a)
    if args.chrome:
        o.binary_location = args.chrome
    o.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    return webdriver.Chrome(service=Service(args.driver) if args.driver else Service(), options=o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url"); ap.add_argument("--shots"); ap.add_argument("--chrome"); ap.add_argument("--driver")
    args = ap.parse_args()
    base = args.url or serve(ROOT / "_site")
    if not base.endswith("/"):
        base += "/"
    d, problems = driver(args), []
    shots = pathlib.Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    def setup(width, height, theme):
        d.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 700})
        d.execute_cdp_cmd("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-color-scheme", "value": theme}]})

    def visit(path, label, script=None, width=1280, theme="light"):
        d.get(base + path)
        time.sleep(0.9)
        if script:
            r = d.execute_script(script)
            time.sleep(0.4)
            if r:
                problems.append(f"{label}: {r}")
        sw = d.execute_script("return document.documentElement.scrollWidth")
        if sw > width + 1:
            problems.append(f"{label} @{width}: page scrolls sideways ({sw}px wide)")
        for entry in d.get_log("browser"):
            if entry["level"] == "SEVERE":
                problems.append(f"{label}: console error: {entry['message']}")
        if shots:
            d.save_screenshot(str(shots / f"{label}_{width}_{theme}.png"))

    click_all = """
      const out = [];
      const lab = document.querySelector('.lab'); if (!lab) return 'no lab on the page';
      lab.querySelectorAll('input[type=range]').forEach(r => { r.value = r.min; r.dispatchEvent(new Event('input')); r.value = r.max; r.dispatchEvent(new Event('input')); });
      lab.querySelectorAll('.seg button, .wordchips button, .row-btns button, .att-token, .pool button').forEach(b => { try { b.click(); } catch (e) { out.push(e.message); } });
      lab.querySelectorAll('details').forEach(x => x.open = true);
      return out.join('; ');"""
    lesson_steps = """
      const n = document.querySelectorAll('.steps button').length; if (!n) return 'no steps';
      for (let i = 0; i < n; i++) document.querySelector('[data-act=next]').click();
      return document.querySelector('.step-panel .k').textContent.includes(n + ' of ' + n) ? '' : 'step text did not follow Next';"""
    quiz_flow = """
      document.querySelector('#show').click();
      if (document.querySelector('.answer').hidden) return 'answer did not open';
      if (!document.querySelector('.watch a[href^="../#"]')) return 'no animation link on the answer';
      document.querySelector('#notyet').click(); document.querySelector('#show').click(); document.querySelector('#knew').click();
      return JSON.parse(localStorage.getItem('aier-quiz-v2')) ? '' : 'progress not saved';"""
    try:
        for width, height in ((1280, 900), (375, 812)):
            for theme in ("light", "dark"):
                setup(width, height, theme)
                visit("", "home", width=width, theme=theme)
                visit("#rag-flow", "lesson-rag-flow", lesson_steps, width, theme)
                visit("#attention", "lesson-attention", lesson_steps, width, theme)
                visit("quiz/", "quiz", quiz_flow, width, theme)
                visit("quiz/#view=strengths", "quiz-strengths", width=width, theme=theme)
                if theme == "light":
                    for lab in LABS:
                        visit(f"#lab/{lab}", f"lab-{lab}", click_all, width, theme)
        setup(1280, 900, "light")
        names = d.execute_script("return AIER_DATA.lessons.map(l => l.name)") if d.get(base) is None else []
        for n in names:                                    # every lesson opens without errors
            d.get(base + "#" + n); time.sleep(0.15)
        for entry in d.get_log("browser"):
            if entry["level"] == "SEVERE":
                problems.append(f"lessons: console error: {entry['message']}")
        print(f"checked {base}: home, {len(names)} lessons, {len(LABS)} labs, quiz")
    finally:
        d.quit()
    for p in problems:
        print("PROBLEM", p)
    print(f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
