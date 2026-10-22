<p align="center">
  <img src="assets/banner.svg" alt="AI Engineer Roadmap by Saurabh Pathak" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/techsaurabh-394/ai-engineer-roadmap/stargazers"><img src="https://img.shields.io/github/stars/techsaurabh-394/ai-engineer-roadmap?style=for-the-badge&logo=github&color=6ea8fe" alt="Stars"/></a>
  <a href="https://github.com/techsaurabh-394/ai-engineer-roadmap/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/techsaurabh-394/ai-engineer-roadmap/ci.yml?style=for-the-badge&label=CI" alt="CI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT"/></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs welcome"/></a>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>

# AI Engineer Roadmap: LLMs, RAG, AI Agents and MCP, step by step

A beginner-friendly, **hands-on roadmap** to become an AI engineer. Each step has a short lesson in plain English, an animated diagram, **runnable code with tests**, exercises, and checkpoint questions. It ends with **47 interview questions** and portfolio project ideas.

No API key needed to start: every demo runs offline. Add a [Claude](https://www.anthropic.com) key later to see real model output.

![Learning path](assets/roadmap.svg)

## What you will learn
| Step | Topic | You will run | Time |
|---|---|---|---|
| [00](00-start-here/README.md) | Setup and how to learn | `pytest` | 20 min |
| [01](01-llm-basics/README.md) | LLM basics: tokens, context, temperature, hallucination | `sampling_demo.py` | 1 hr |
| [02](02-rag/README.md) | RAG: chunking, embeddings, retrieval, citations | `mini_rag.py` | 2 hrs |
| [03](03-agents/README.md) | AI agents: the loop, tools, safety limits | `agent.py` | 2 hrs |
| [04](04-mcp/README.md) | MCP: build a server, connect Claude Desktop | `hello-mcp-server` | 2 hrs |
| [05](05-evaluation/README.md) | Evaluation: hit@k, MRR, golden sets | `eval_harness.py` | 1 hr |
| [06](06-production/README.md) | Production: cost, latency, security, monitoring | checklist | 1 hr |
| [07](07-interview-prep/README.md) | 47 interview questions with answers | out loud | 5 days |
| [08](08-projects/README.md) | Portfolio projects to build and share | your repo | your pace |

## Quick start
```bash
git clone https://github.com/techsaurabh-394/ai-engineer-roadmap.git
cd ai-engineer-roadmap
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                                  # all green = ready
python 02-rag/code/mini_rag.py "Why use multi-stage builds?"
```
Then open [00-start-here](00-start-here/README.md).

## The three ideas behind everything
<table>
<tr>
<td width="33%"><b>RAG</b><br/>Give the model your documents and make it cite them.<br/><img src="assets/rag-pipeline.svg" alt="RAG"/></td>
<td width="33%"><b>Agents</b><br/>Let the model choose tools in a loop, with limits.<br/><img src="assets/agent-loop.svg" alt="Agent loop"/></td>
<td width="33%"><b>MCP</b><br/>One standard way to connect AI apps to tools.<br/><img src="assets/mcp-architecture.svg" alt="MCP"/></td>
</tr>
</table>

## Repository map
```
ai-engineer-roadmap/
├── 00-start-here/      setup and how to learn
├── 01-llm-basics/      lesson + code/sampling_demo.py
├── 02-rag/             lesson + code/mini_rag.py
├── 03-agents/          lesson + code/agent.py
├── 04-mcp/             lesson + code/hello-mcp-server (TypeScript)
├── 05-evaluation/      lesson + code/eval_harness.py
├── 06-production/      production checklist
├── 07-interview-prep/  47 questions and answers
├── 08-projects/        projects to build and share
├── assets/             animated SVG diagrams
└── .github/workflows/  CI, link check, stale-issue cleanup
```

## Automated checks
Every push runs tests for all Python lessons, builds the MCP example, and validates the SVG diagrams. A weekly job checks that links in the docs still work.

## Related projects by the author
| Project | What it is |
|---|---|
| [rag-docs-chat](https://github.com/techsaurabh-394/rag-docs-chat) | PDF chat with cited answers and an evaluation script |
| [mcp-notes-server](https://github.com/techsaurabh-394/mcp-notes-server) | MCP server with 4 typed tools |
| [claude-skills](https://github.com/techsaurabh-394/claude-skills) | Reusable Claude Skills for SEO/GEO, blogging and MCP |
| [mernstackdev.com](https://mernstackdev.com) | Developer blog on React, Node.js and AI engineering |

## Contribute
Fix a typo, add a question, improve an explanation, or add your project. See [CONTRIBUTING](CONTRIBUTING.md). Look for issues labelled **good first issue**.

## Support this project
If this roadmap helped you, a ⭐ helps others find it, and you can support more free learning material:

<a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>

## Star history
[![Star History Chart](https://api.star-history.com/svg?repos=techsaurabh-394/ai-engineer-roadmap&type=Date)](https://star-history.com/#techsaurabh-394/ai-engineer-roadmap&Date)

## License
MIT. See [LICENSE](LICENSE).

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
