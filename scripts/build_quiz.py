#!/usr/bin/env python3
"""Build the interactive quiz page into _site/quiz/ (standard library only).

    python scripts/build_quiz.py

The questions come from the phase READMEs (see quiz_core.py). Answers are free text, so this is
self-assessment: think, reveal, then mark "I knew it" or "Not yet". Progress stays in the
browser (localStorage); nothing is sent anywhere.
"""
import html
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quiz_core as qc  # noqa: E402

OUT = qc.ROOT / "_site" / "quiz"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Engineer Roadmap: checkpoint quizzes</title>
<meta name="description" content="__COUNT__ self-check questions across 7 phases: RAG, LLMs, agents, MCP and production AI. Reveal the answer, mark yourself, track progress. Free and open source.">
<style>
:root{--bg:#0D1117;--card:#161B22;--line:#30363D;--text:#E6EDF3;--muted:#9DA7B3;--teal:#2DD4BF;--amber:#F5B042;--green:#3FB950;--red:#F85149}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header h1{font-size:18px;margin:0;flex:1;min-width:200px}a{color:var(--teal)}header a{font-size:14px;text-decoration:none}
main{max-width:760px;margin:0 auto;padding:22px 18px 60px}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
.tab{background:var(--card);border:1px solid var(--line);color:var(--text);border-radius:999px;padding:7px 14px;font-size:14px;cursor:pointer}
.tab[aria-pressed=true]{background:var(--teal);color:#0D1117;border-color:var(--teal);font-weight:700}
.tab small{color:inherit;opacity:.75;margin-left:4px}
.bar{height:8px;background:var(--card);border:1px solid var(--line);border-radius:99px;overflow:hidden;margin:6px 0 4px}
.bar>i{display:block;height:100%;background:var(--teal);width:0}
.meta{display:flex;justify-content:space-between;color:var(--muted);font-size:14px;margin-bottom:16px;gap:10px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px}
.card h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.q{font-size:21px;line-height:1.4;margin:0 0 18px;font-weight:600}
.ans{display:none;border-top:1px solid var(--line);padding-top:16px;margin-top:6px}.ans.on{display:block}
button.b{background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px 18px;font-size:15px;cursor:pointer}
button.b:hover:not(:disabled){border-color:var(--teal)}button.b:focus-visible,.tab:focus-visible,a:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
button.p{background:var(--teal);color:#0D1117;border-color:var(--teal);font-weight:700}
button.g{border-color:var(--green)}button.r{border-color:var(--red)}button:disabled{opacity:.4;cursor:default}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
code{background:#0D1117;border:1px solid var(--line);border-radius:6px;padding:1px 6px;font-size:.92em}
.done{text-align:center}.done .big{font-size:42px;font-weight:800;color:var(--teal)}
.hint{color:var(--muted);font-size:13px;margin-top:14px}.mark{font-size:13px;margin-left:8px}
.ok{color:var(--green)}.no{color:var(--red)}
</style></head><body>
<header><h1>AI Engineer Roadmap: checkpoint quizzes</h1>
<a href="__REPO__">GitHub repo</a><a href="../">Animated lessons</a></header>
<main>
<div class="tabs" id="tabs" role="group" aria-label="Phase"></div>
<div class="bar" aria-hidden="true"><i id="fill"></i></div>
<div class="meta"><span id="prog"></span><span id="score"></span></div>
<div class="card" id="card" aria-live="polite"></div>
<p class="hint">Self-check: think of your own answer first, then reveal. Your marks stay in this browser only. Questions come from the phase READMEs, so a wrong or unclear answer is a bug: <a href="__REPO__/issues">open an issue</a>.<br>Keys: Space reveals, 1 = I knew it, 2 = Not yet, left and right arrows move.</p>
</main>
<script>
const Q=__QUESTIONS__,PH=__PHASES__,KEY='aier-quiz-v1';
const $=s=>document.querySelector(s);
let st={};try{st=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
let phase='all',i=0,shown=false;
const list=()=>Q.map((q,n)=>({...q,id:n})).filter(q=>phase==='all'||q.phase===phase);
function tabs(){
  const t=$('#tabs');t.innerHTML='';
  const mk=(k,label,n)=>{const b=document.createElement('button');b.className='tab';b.type='button';b.setAttribute('aria-pressed',k===phase);
    b.innerHTML=label+'<small>'+n+'</small>';b.onclick=()=>{phase=k;i=0;shown=false;render()};t.appendChild(b)};
  mk('all','All',Q.length);for(const k of Object.keys(PH))mk(k,PH[k].replace(/^Phase (\d+):.*/,'Phase $1'),Q.filter(q=>q.phase===k).length);
}
function stats(L){
  const k=L.filter(q=>st[q.id]==='k').length,n=L.filter(q=>st[q.id]==='n').length;
  $('#fill').style.width=(L.length?100*(k+n)/L.length:0)+'%';
  $('#prog').textContent=(k+n)+' of '+L.length+' answered';
  $('#score').textContent=k+' known, '+n+' to review';
}
function render(){
  tabs();const L=list();stats(L);const c=$('#card');
  if(!L.length){c.textContent='No questions.';return}
  i=Math.max(0,Math.min(L.length-1,i));const q=L[i];
  const mark=st[q.id]==='k'?'<span class="mark ok">known</span>':st[q.id]==='n'?'<span class="mark no">review</span>':'';
  c.innerHTML='<h2>'+PH[q.phase]+' &middot; question '+(i+1)+' of '+L.length+mark+'</h2><p class="q" id="qt"></p>'+
    '<button class="b p" id="show" type="button">Show answer</button>'+
    '<div class="ans" id="ans"></div>'+
    '<div class="row"><button class="b" id="prev" type="button">&larr; Previous</button><button class="b" id="next" type="button">Next &rarr;</button></div>';
  $('#qt').textContent=q.q;$('#ans').innerHTML=q.a;
  $('#prev').disabled=i===0;
  const a=$('#ans');
  const reveal=()=>{shown=true;a.classList.add('on');$('#show').style.display='none';
    a.insertAdjacentHTML('beforeend','<div class="row"><button class="b g" id="k" type="button">I knew it</button><button class="b r" id="n" type="button">Not yet</button></div>');
    $('#k').onclick=()=>mk('k');$('#n').onclick=()=>mk('n')};
  const mk=v=>{st[q.id]=v;save();if(i<L.length-1){i++;shown=false;render()}else{finish(L)}};
  $('#show').onclick=reveal;$('#prev').onclick=()=>{i--;shown=false;render()};$('#next').onclick=()=>{if(i<L.length-1){i++;shown=false;render()}else finish(L)};
  render.reveal=reveal;render.mk=mk;
  if(shown)reveal();
}
function finish(L){
  stats(L);const k=L.filter(q=>st[q.id]==='k').length,n=L.filter(q=>st[q.id]==='n').length,un=L.length-k-n;
  $('#card').innerHTML='<div class="done"><div class="big">'+k+' / '+L.length+'</div><p>marked as known'+(n?', '+n+' to review':'')+(un?', '+un+' unanswered':'')+'.</p>'+
    '<div class="row" style="justify-content:center"><button class="b p" id="again" type="button">Review the ones I missed</button><button class="b" id="reset" type="button">Reset this set</button></div></div>';
  $('#again').onclick=()=>{const miss=L.filter(q=>st[q.id]!=='k');if(!miss.length){i=0;render();return}
    for(const q of miss)delete st[q.id];save();i=0;shown=false;render()};
  $('#reset').onclick=()=>{for(const q of L)delete st[q.id];save();i=0;shown=false;render()};
}
addEventListener('keydown',e=>{
  if(e.target.closest&&e.target.closest('button,a'))if(e.key===' '||e.key==='Enter')return;
  const ans=$('#ans');if(!ans)return;
  if(e.key===' '){e.preventDefault();if(!shown&&render.reveal)render.reveal()}
  else if(e.key==='1'&&shown){render.mk('k')}else if(e.key==='2'&&shown){render.mk('n')}
  else if(e.key==='ArrowRight'){const b=$('#next');if(b)b.click()}else if(e.key==='ArrowLeft'){const b=$('#prev');if(b&&!b.disabled)b.click()}
});
render();
</script></body></html>
"""


def main():
    items = qc.build()
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    questions = [{"phase": q["phase"], "q": q["q"], "a": q["a"]} for q in items]
    page = (PAGE.replace("__QUESTIONS__", json.dumps(questions, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__PHASES__", json.dumps(qc.PHASES)).replace("__REPO__", qc.REPO)
            .replace("__COUNT__", str(len(questions))))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"quiz: {len(questions)} questions -> {OUT}  ({html.escape(questions[0]['q'][:40])}...)")


if __name__ == "__main__":
    main()
