# Regenerates docs/learning-paths.md from the "Reading time" line on every topic page.
# Run from anywhere:   python scripts/gen_learning_paths.py
# CI runs it and fails if the committed file is out of date.
import glob, os, re
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

T = {}  # (phase_no, topic_no) -> (title, level, minutes, relpath)
for f in sorted(glob.glob('0[1-7]-*/topics/*.md')):
    t = open(f).read()
    m = re.search(r'\*\*Level:\*\* (\w+).*?\*\*Reading time:\*\* (\d+) min', t)
    assert m, f
    ph = int(f[:2]); n = int(os.path.basename(f)[:2])
    T[(ph, n)] = (t.split('\n')[0][2:], m.group(1), int(m.group(2)), f)
N = len(T)
PH = {}
for (ph, n) in T:
    PH[ph] = max(PH.get(ph, 0), n)
ICON = {'Beginner': '🟢', 'Intermediate': '🟡', 'Advanced': '🔴'}

def row(ph, n):
    title, lv, mn, f = T[(ph, n)]
    return f"[{title}](../{f})", f"{ICON[lv]}", mn

def day(label, items, do):
    cells, mins = [], 0
    for ph, n in items:
        link, ic, mn = row(ph, n)
        cells.append(f"{ic} {link} ({mn} min)")
        mins += mn
    if not items:
        return f"| {label} | | | {do} |", 0
    return f"| {label} | " + "<br/>".join(cells) + f" | {mins} min | {do} |", mins

def table(days):
    out = ["| Day | Read | Reading | Then do this |", "|---|---|---|---|"]
    tot = 0
    for label, items, do in days:
        r, m = day(label, items, do)
        out.append(r); tot += m
    return "\n".join(out), tot

A = [
 ("1", [(1,1),(1,2)], "Run `python 01-rag-foundations/code/mini_rag.py \"Why use multi-stage builds?\"`. Change the question and run it again."),
 ("2", [(1,3),(1,4),(1,5)], "Run `python 01-rag-foundations/code/rag_chunkers.py` and compare how the five chunkers split the same text."),
 ("3", [(1,9),(1,10),(1,11)], "Run `python 01-rag-foundations/code/rag_vector_math.py` and check the cosine number by hand."),
 ("4", [(1,16),(1,17)], "Run `python 01-rag-foundations/code/rag_bm25.py` and look at how BM25, dense and hybrid rank the same documents."),
 ("5", [(2,1),(2,2)], "Run `python 02-advanced-rag/code/adv_rerank.py` and compare the order before and after reranking."),
 ("6", [(2,5),(2,18)], "Run `python 02-advanced-rag/code/adv_query_transform.py` and `adv_incremental_index.py`."),
 ("7", [(2,20)], "Run `python 02-advanced-rag/code/adv_rag_metrics.py` and `eval_harness.py`. Write 10 questions about your own notes."),
 ("8", [(7,1),(7,2)], "Read how a RAG service is wrapped in an API and a container. Do not deploy anything yet."),
 ("9-10", [], "Build: point `mini_rag.py` at your own notes, add a 10-question eval set, report hit@3. See [09-projects](../09-projects/README.md)."),
]
B = [
 ("1", [(4,1),(4,2),(4,3)], "Run `python 04-agentic-ai/code/agent.py` and read each printed step."),
 ("2", [(4,4),(4,12)], "Run `python 04-agentic-ai/code/agent_react.py` and find where the loop stops."),
 ("3", [(4,7),(4,8)], "Run `python 04-agentic-ai/code/agent_memory.py` and compare short-term, long-term and episodic memory."),
 ("4", [(4,14),(4,16)], "Run `python 07-production-ai/code/prod_guardrails.py` and see what gets blocked and what gets masked."),
 ("5", [(6,1),(6,2)], "Run `python 06-mcp/code/mcp_mini_jsonrpc.py` and read the JSON-RPC messages it prints."),
 ("6", [(6,4),(6,7)], "Open `06-mcp/code/hello-mcp-server/src/index.ts`. Build it with `npm ci && npm run build` inside that folder."),
 ("7", [(7,11)], "Read the injection and tool-abuse section, then list which of your tools should need a human to approve."),
 ("8-10", [], "Build: a small MCP server for an API you use, with one tool that needs approval. See [mcp-safe-actions](https://github.com/techsaurabh-394/mcp-safe-actions)."),
]
CW = [
 ("Week 1", [(1,n) for n in range(1,PH[1]+1)], "Phase 1. Do the checkpoint quiz in the phase README."),
 ("Week 2", [(2,n) for n in range(1,PH[2]+1)], "Phase 2. Run every script in `02-advanced-rag/code/`, including `adv_llm_judge.py`."),
 ("Week 3", [(3,n) for n in range(1,PH[3]+1)], "Phase 3. Run `llm_attention.py`, `llm_decoding.py`, `sampling_demo.py`, `llm_finetune_data.py`, `llm_api_basics.py`, `llm_structured.py`, `llm_quantize.py`, `llm_lora_math.py`, `llm_serving_sim.py`, `llm_e2e_walkthrough.py`, `llm_qkv.py`, `llm_block.py`, `llm_training.py`, `llm_attn_variants.py` and `llm_moe_scaling.py`."),
 ("Week 4", [(4,n) for n in range(1,PH[4]+1)], "Phase 4. Run the four agent scripts."),
 ("Week 5", [(5,n) for n in range(1,PH[5]+1)], "Phase 5. Run `fw_mini_chain.py` and `fw_mini_graph.py`, then read the framework comparison."),
 ("Week 6", [(6,n) for n in range(1,PH[6]+1)], "Phase 6. Build the TypeScript server."),
 ("Week 7", [(7,n) for n in range(1,PH[7]+1)], "Phase 7. Run the five `prod_*.py` scripts."),
 ("Week 8", [], "Do [08-interview-prep](../08-interview-prep/README.md) and finish one project from [09-projects](../09-projects/README.md)."),
]
E = [
 ("1", [(3,16),(3,9)], "Run `python 03-llm-concepts/code/llm_api_basics.py`. Change the failure count and watch the retry waits double."),
 ("2", [(3,17)], "Run `python 03-llm-concepts/code/llm_structured.py`. Break the schema on purpose and read the error messages the repair loop sends back."),
 ("3", [(3,14),(3,13)], "Write a prompt for one task of yours, then list three ways the answer could be confidently wrong."),
 ("4", [(3,15)], "Run `python 03-llm-concepts/code/llm_finetune_data.py`. Decide: would prompting or RAG solve your task first?"),
 ("5", [(3,18)], "Run `python 03-llm-concepts/code/llm_quantize.py`. Compare 8-bit and 4-bit error, then find the outlier demo."),
 ("6", [(3,19)], "Run `python 03-llm-concepts/code/llm_lora_math.py`. Check the 4x4 example by hand."),
 ("7", [(3,20),(3,7)], "Run `python 03-llm-concepts/code/llm_serving_sim.py`. Change the request lengths until static and continuous batching tie."),
 ("8", [(2,20),(2,21)], "Run `python 02-advanced-rag/code/adv_llm_judge.py`. Swap the answer order and see which judge changes its mind."),
]
F = [
 ("1", [(3,1),(3,21)], "Run `python 03-llm-concepts/code/llm_e2e_walkthrough.py`. Follow the word 'the cat sat on the' from text to the next token."),
 ("2", [(3,22)], "Run `python 03-llm-concepts/code/llm_qkv.py`. Change one number in W_q and see which attention row moves."),
 ("3", [(3,2),(3,3),(3,4)], "Run `python 03-llm-concepts/code/llm_attention.py`. Remove the causal mask and compare the first row of weights."),
 ("4", [(3,23),(3,5)], "Run `python 03-llm-concepts/code/llm_block.py`. Watch the residual keep the signal alive through a deep stack."),
 ("5", [(3,24),(3,15)], "Run `python 03-llm-concepts/code/llm_training.py`. Watch the loss fall, then halve the learning rate."),
 ("6", [(3,7),(3,25)], "Run `python 03-llm-concepts/code/llm_attn_variants.py`. Compare the KV-cache sizes of MHA, GQA and MQA."),
 ("7", [(3,26),(3,20)], "Run `python 03-llm-concepts/code/llm_moe_scaling.py`. Change top-2 to top-1 and compare active parameters."),
]
sa, ta = table(A); sb, tb = table(B); sc, tc = table(CW); se, te = table(E); sf, tf = table(F)
total = sum(v[2] for v in T.values())
assert N == sum(PH_COUNT := [sum(1 for k in T if k[0] == p) for p in PH]), N
assert all(sum(1 for k in T if k[0] == p) == PH[p] for p in PH), 'topic numbers must be 1..N with no gaps'
assert tc == total, (tc, total)

doc = f"""# Learning paths

Pick the path that matches what you want to do. Every time below is the **reading time** printed on the topic page, added up by a script from the repo, plus time for running code. The number of topics is {N} and the total reading time is **{total} minutes (about {total/60:.0f} hours)**. Running code and building the project take longer than reading.

| I want to... | Path | Time |
|---|---|---|
| Build an app that answers questions from my documents | [Path A: RAG app](#path-a-build-a-rag-app-in-2-weeks) | 2 weeks, about 1 hour a day |
| Build agents and MCP tools | [Path B: agents and MCP](#path-b-build-agents-and-mcp-tools-in-2-weeks) | 2 weeks, about 1 hour a day |
| Learn everything | [Path C: full roadmap](#path-c-the-full-roadmap-in-8-weeks) | 8 weeks |
| Prepare for an interview | [Path D: interview week](#path-d-interview-in-one-week) | 1 week |
| Build with LLM APIs, from first call to serving | [Path E: LLM builder](#path-e-llm-builder-from-first-call-to-serving) | 8 days, about 1 hour a day |
| Understand how an LLM really works inside (Q, K, V, training, MoE) | [Path F: inside the LLM](#path-f-inside-the-llm-how-it-really-works) | 7 days, about 1 hour a day |

Not sure? Start with [what AI engineering is](what-is-ai-engineering.md) (5 minutes), then take Path A. Legend: 🟢 beginner · 🟡 intermediate · 🔴 advanced.

## Path A: build a RAG app in 2 weeks

**You will end with:** a small program that answers questions from your own notes, with citations and a score from a test set. **You need:** basic Python, [the setup](../00-start-here/README.md#setup-one-time). Reading in this path: {ta} minutes.

{sa}

## Path B: build agents and MCP tools in 2 weeks

**You will end with:** a small MCP server with a tool that waits for a human. **You need:** basic Python, and Node.js 18+ for day 6. Do Path A's first four days first if you have never heard of embeddings. Reading in this path: {tb} minutes.

{sb}

## Path C: the full roadmap in 8 weeks

All {N} topics, one phase a week, with the last week for interview prep and a project. Reading is about {total/60:.0f} hours in total; spread it so that you also run the code. Each week below lists every topic in the phase. Weekly reading time is in the right column.

{sc}

## Path D: interview in one week

Use [08-interview-prep](../08-interview-prep/README.md): one file a day, five days, answer out loud before opening the answers.

| Day | File |
|---|---|
| 1 | [LLM fundamentals](../08-interview-prep/01-llm-fundamentals.md) |
| 2 | [RAG](../08-interview-prep/02-rag.md) |
| 3 | [Agents and MCP](../08-interview-prep/03-agents-and-mcp.md) |
| 4 | [Evaluation and production](../08-interview-prep/04-evaluation-and-production.md) |
| 5 | [System design](../08-interview-prep/05-system-design.md) |
| 6 | Pick the weakest file and repeat it. Prepare one example from something you built for each answer. |
| 7 | Rest, or read the "Interview questions" of the topics you found hardest. |

If you have never studied these topics, one week is not enough for the topics themselves. Use Path A or B first.

## Path E: LLM builder, from first call to serving

**You will end with:** a working picture of the whole life of an LLM call: sending it, getting reliable output, making a model small and cheap to adapt, serving it fast, and grading answers fairly. **You need:** basic Python. Start with 🟢 days, and treat 🔴 days as optional. Reading in this path: {te} minutes.

{se}

## Path F: inside the LLM, how it really works

**You will end with:** a clear mental model of what happens inside a language model: tokens and embeddings, Q, K and V, the transformer block, how a model is trained, and the modern tricks (GQA, FlashAttention, RoPE, mixture of experts). **You need:** basic Python and high-school math. Do the days in order; day 2 is the one most people find hardest, so read it twice. Reading in this path: {tf} minutes.

{sf}

## If you fall behind
Skipping is fine. Do the topics marked 🟢 first, skip 🔴 ones until you need them, and use the [glossary](glossary.md) when a word is new. The checklist in the [README](../README.md#the-full-checklist) shows the level of every topic.

[Back to the README](../README.md)
"""
doc = doc.replace("\n| Week 8 | Interview prep", "\n| Week 8 | Interview prep")
open('docs/learning-paths.md', 'w').write(doc)
print(ta, tb, tc, total)
