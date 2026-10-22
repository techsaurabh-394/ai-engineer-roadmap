# 00 · Start here

**Goal:** get your computer ready and understand how this roadmap works. Time: 20 minutes.

## How the roadmap works
Each folder is one step. Do them in order.

| Step | Folder | You will learn | You will run |
|---|---|---|---|
| 1 | [01-llm-basics](../01-llm-basics) | tokens, context, temperature, hallucination | `sampling_demo.py` |
| 2 | [02-rag](../02-rag) | chunking, embeddings, retrieval, citations | `mini_rag.py` |
| 3 | [03-agents](../03-agents) | the agent loop, tool design, safety limits | `agent.py` |
| 4 | [04-mcp](../04-mcp) | the Model Context Protocol, building a server | `hello-mcp-server` |
| 5 | [05-evaluation](../05-evaluation) | measuring quality (hit@k, MRR) | `eval_harness.py` |
| 6 | [06-production](../06-production) | cost, latency, security, monitoring | checklist |
| 7 | [07-interview-prep](../07-interview-prep) | 47 questions with answers | out loud practice |
| 8 | [08-projects](../08-projects) | portfolio projects to build | your own repo |

Every lesson has the same shape: **Goal → Big idea → Run it → Try it → Checkpoint → Next.**

## Setup (one time)

### Python (steps 1, 2, 3, 5)
```bash
python --version                       # 3.10 or newer
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                 # all green = your setup works
```

### Node.js (step 4)
```bash
node --version                         # 18 or newer
```

### Optional: a Claude API key
Everything runs **without** an API key (offline demos use scripted or extractive answers). Add a key to see real model output:
```bash
export ANTHROPIC_API_KEY=your_key      # Windows PowerShell: $env:ANTHROPIC_API_KEY="your_key"
```

## How to learn from this
1. Read the lesson. Run the code. Change one thing. Run again.
2. Do the **Try it** exercises. They are the actual learning.
3. Answer the **Checkpoint** questions without looking.
4. Keep notes in your own words. Explaining is how you find gaps.

**Next:** [01 · LLM basics](../01-llm-basics/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
