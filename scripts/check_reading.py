"""Browser check of the lesson text on the Pages site. Usage: python browser_check.py BASE_URL [shot_dir]"""
import sys, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

base = sys.argv[1].rstrip("/") + "/"
shots = sys.argv[2] if len(sys.argv) > 2 else None
o = Options()
o.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for a in ("--headless=new", "--window-size=1280,1800", "--hide-scrollbars"):
    o.add_argument(a)
o.set_capability("goog:loggingPrefs", {"browser": "ALL"})
d = webdriver.Chrome(options=o)
fails = []
names = d.execute_script("return 0") or 0
d.get(base)
time.sleep(2)
names = d.execute_script("return AIER_DATA.lessons.map(l=>l.name)")
print("lessons", len(names))
for i, n in enumerate(names):
    d.get(base + "#" + n)
    time.sleep(0.6)
    r = d.execute_script("""var b=document.querySelectorAll('.reading-body');
      var out={n:b.length,len:0,h3:0,busy:0,raw:0};
      b.forEach(function(x){out.len+=x.textContent.length;out.h3+=x.querySelectorAll('h3').length;out.busy+=x.getAttribute('aria-busy')?1:0;
        if(/\\*\\*|```|\\]\\(/.test(x.textContent.replace(/\\s+/g,' ').split('Code')[0]))out.raw++;});
      return out;""")
    if r["n"] == 0 or r["busy"] or r["len"] < 1500 or r["h3"] < 4:
        fails.append((n, r))
    if shots and n in ("cosine-similarity", "rag-flow", "attention"):
        d.execute_script("document.getElementById('read').scrollIntoView()")
        time.sleep(0.4)
        d.save_screenshot(f"{shots}/{n}.png")
sev = [l for l in d.get_log("browser") if l["level"] == "SEVERE"]
print("fails", len(fails), fails[:5])
print("severe console", len(sev), sev[:3])
d.quit()
