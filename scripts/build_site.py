#!/usr/bin/env python3
"""Build the GitHub Pages site into _site/ (standard library only).

    python scripts/build_site.py
The site is one static page: a searchable list of every animation and a viewer
with Previous / Play / Next, a step slider, a speed control and arrow-key support.
"""
import html
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_animations as g  # noqa: E402

ROOT = g.core.ROOT
OUT = ROOT / "_site"
REPO = "https://github.com/techsaurabh-394/ai-engineer-roadmap"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Engineer Roadmap: animated lessons</title>
<meta name="description" content="Step through RAG, LLMs, agents, MCP and production AI with animated explainers. Previous, Play and Next controls. Free and open source.">
<meta property="og:title" content="AI Engineer Roadmap: animated lessons">
<meta property="og:description" content="__COUNT__ animated explainers for RAG, LLMs, agents, MCP and production AI.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%232DD4BF'/%3E%3Ctext x='16' y='23' font-size='20' text-anchor='middle' font-family='sans-serif' font-weight='700' fill='%230D1117'%3EAI%3C/text%3E%3C/svg%3E">
<style>
:root{--bg:#0D1117;--card:#161B22;--line:#30363D;--text:#E6EDF3;--muted:#9DA7B3;--teal:#2DD4BF;--amber:#F5B042}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--text);font:16px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;display:flex;flex-direction:column}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header h1{font-size:18px;margin:0;flex:1;min-width:200px}header a{color:var(--teal);text-decoration:none;font-size:14px}
header a:hover{text-decoration:underline}
.wrap{flex:1;display:flex;min-height:0}
nav{width:300px;border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
nav input{margin:12px;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--text);font-size:15px}
nav input:focus{outline:2px solid var(--teal)}
#list{overflow:auto;padding:0 8px 16px;flex:1}
#list h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:16px 8px 6px}
#list a{display:block;padding:7px 10px;border-radius:8px;color:var(--text);text-decoration:none;font-size:14.5px}
#list a:hover{background:var(--card)}#list a.on{background:var(--card);box-shadow:inset 3px 0 var(--teal)}
main{flex:1;overflow:auto;padding:20px;min-width:0}
.card{max-width:960px;margin:0 auto}
h2#title{margin:0 0 4px;font-size:24px}#sub{color:var(--muted);margin:0 0 14px}
#stage{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:8px;min-height:200px}
#stage svg{width:100%;height:auto;display:block}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}
button,select{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:9px 16px;font-size:15px;cursor:pointer}
button:hover:not(:disabled){border-color:var(--teal)}button:focus-visible,select:focus-visible,a:focus-visible,input:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
button:disabled{opacity:.4;cursor:default}button.primary{background:var(--teal);color:#0D1117;border-color:var(--teal);font-weight:700;min-width:96px}
#slider{flex:1;min-width:140px;accent-color:var(--teal)}#count{color:var(--muted);font-size:14px;min-width:92px;text-align:right}
.links{margin-top:6px;font-size:15px}.links a{color:var(--teal)}
.hint{color:var(--muted);font-size:13px;margin-top:10px}
@media(max-width:760px){.wrap{flex-direction:column}nav{width:100%;max-height:34vh;border-right:0;border-bottom:1px solid var(--line)}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body>
<header><h1>AI Engineer Roadmap: animated lessons</h1>
<a href="__REPO__">GitHub repo</a><a href="__REPO__/blob/main/README.md#start-in-5-minutes">Start here</a><a href="quiz/">Quizzes</a></header>
<div class="wrap">
<nav aria-label="Animations"><input id="q" type="search" placeholder="Search __COUNT__ animations" aria-label="Search animations"><div id="list"></div></nav>
<main><div class="card">
<h2 id="title"></h2><p id="sub"></p>
<div id="stage" aria-live="polite"></div>
<div class="bar">
<button id="prev" aria-label="Previous step">&larr; Previous</button>
<button id="play" class="primary" aria-label="Play or pause">Play</button>
<button id="next" aria-label="Next step">Next &rarr;</button>
<input id="slider" type="range" min="0" value="0" aria-label="Step">
<span id="count"></span>
<label for="speed" style="color:var(--muted);font-size:14px">Speed</label>
<select id="speed"><option value="1.6">Slow</option><option value="1" selected>Normal</option><option value="0.6">Fast</option></select>
</div>
<div class="links" id="links"></div>
<p class="hint">Keys: left and right arrows move one step, space plays or pauses. The picture is a plain SVG, so it works offline once loaded.</p>
</div></main></div>
<script>
const M=__MANIFEST__;
const $=s=>document.querySelector(s);
const PH=__PHASES__;
let cur=null,step=0,timer=null,svg=null;
const list=$('#list');
function renderList(f){
  list.innerHTML='';const q=(f||'').toLowerCase();
  for(const ph of Object.keys(PH)){
    const items=M.filter(a=>a.phase===ph&&(a.title+' '+a.subtitle).toLowerCase().includes(q));
    if(!items.length)continue;
    const h=document.createElement('h2');h.textContent=PH[ph];list.appendChild(h);
    for(const a of items){const l=document.createElement('a');l.href='#'+a.name;l.textContent=a.title;l.dataset.n=a.name;
      if(cur&&cur.name===a.name)l.className='on';list.appendChild(l);}
  }
}
$('#q').addEventListener('input',e=>renderList(e.target.value));
function pause(){clearInterval(timer);timer=null;$('#play').textContent='Play';}
function show(i){
  step=Math.max(0,Math.min(cur.marks.length-1,i));
  if(svg){svg.pauseAnimations();svg.setCurrentTime(cur.marks[step]);}
  $('#slider').value=step;$('#count').textContent='Step '+(step+1)+' of '+cur.marks.length;
  $('#prev').disabled=step===0;
}
function play(){
  if(timer){pause();return;}
  if(step>=cur.marks.length-1)show(0);
  $('#play').textContent='Pause';
  timer=setInterval(()=>{if(step>=cur.marks.length-1){pause();return;}show(step+1);},2600*parseFloat($('#speed').value));
}
async function open(name){
  const a=M.find(x=>x.name===name)||M[0];cur=a;pause();
  document.title=a.title+' | AI Engineer Roadmap';
  $('#title').textContent=a.title;$('#sub').textContent=a.subtitle;
  $('#slider').max=a.marks.length-1;
  $('#links').innerHTML='Full lesson: '+a.topics.map(t=>'<a href="__REPO__/blob/main/'+t.path+'">'+t.title+'</a>').join(' &middot; ');
  const st=$('#stage');st.textContent='Loading...';svg=null;
  try{
    const r=await fetch('anim/'+a.name+'.svg');if(!r.ok)throw 0;
    const txt=await r.text();if(cur!==a)return;
    st.innerHTML=txt;svg=st.querySelector('svg');svg.removeAttribute('width');svg.removeAttribute('height');
  }catch(e){st.innerHTML='<img alt="'+a.title+'" src="anim/'+a.name+'.svg" style="width:100%">';}
  show(0);renderList($('#q').value);
  const on=list.querySelector('a.on');if(on)on.scrollIntoView({block:'nearest'});
}
$('#prev').onclick=()=>{pause();show(step-1)};
$('#next').onclick=()=>{pause();show(step+1)};
$('#play').onclick=play;
$('#slider').oninput=e=>{pause();show(+e.target.value)};
$('#speed').onchange=()=>{if(timer){pause();play();}};
addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'&&e.target.type==='search')return;
  if(e.key==='ArrowRight'){pause();show(step+1);e.preventDefault();}
  else if(e.key==='ArrowLeft'){pause();show(step-1);e.preventDefault();}
  else if(e.key===' '&&e.target.tagName!=='BUTTON'){play();e.preventDefault();}
});
addEventListener('hashchange',()=>open(location.hash.slice(1)));
renderList();open(location.hash.slice(1));
</script></body></html>
"""


def main():
    anims = g.build()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "anim").mkdir(parents=True)
    manifest = []
    for a in anims:
        (OUT / "anim" / f"{a['name']}.svg").write_text(a["svg"], encoding="utf-8")
        manifest.append(dict(
            name=a["name"], title=a["title"], subtitle=a["subtitle"], phase=a["phase"], marks=a["marks"],
            topics=[dict(path=t, title=g.title_of_topic(t)) for t in a["topics"]]))
    manifest.sort(key=lambda m: (m["phase"], m["topics"][0]["path"]))
    page = (PAGE.replace("__MANIFEST__", json.dumps(manifest, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__PHASES__", json.dumps(g.PHASES)).replace("__REPO__", REPO)
            .replace("__COUNT__", str(len(manifest))))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / ".nojekyll").write_text("")
    print(f"site: {len(manifest)} animations -> {OUT}  (title safe: {html.escape(manifest[0]['title'])})")
    import build_quiz  # after the wipe above: the quiz lives in _site/quiz/
    build_quiz.main()


if __name__ == "__main__":
    main()
