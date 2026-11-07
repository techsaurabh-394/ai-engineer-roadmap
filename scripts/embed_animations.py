#!/usr/bin/env python3
"""Insert (idempotently) an animated explainer into topic pages.

Run after gen_animations.py:  python scripts/embed_animations.py
Each topic gets a "Watch it" block placed just before "## How it works".
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
MARK = "<!-- anim:"

MAP = {
    "01-rag-foundations/topics/01-what-is-rag.md": ("rag-flow", "RAG step by step: documents, chunks, vectors, retrieval, answer"),
    "01-rag-foundations/topics/04-chunking.md": ("chunking", "Chunking with overlap: three overlapping pieces cut from one text"),
    "01-rag-foundations/topics/11-cosine-similarity.md": ("cosine-similarity", "Cosine similarity: a vector rotating against a query, with its score"),
    "01-rag-foundations/topics/16-hybrid-search.md": ("hybrid-search-flow", "Hybrid search: keyword and meaning results merged"),
    "02-advanced-rag/topics/20-rag-evaluation.md": ("eval-loop", "Evaluation loop: test set, run, score, improve"),
    "03-llm-concepts/topics/02-self-attention.md": ("attention", "Self-attention: one word weighting every other word"),
    "03-llm-concepts/topics/15-fine-tuning.md": ("fine-tuning-flow", "Fine-tuning: data, clean, train, evaluate, deploy"),
    "03-llm-concepts/topics/21-how-an-llm-works-end-to-end.md": ("llm-flow", "Inside an LLM: text, tokens, embeddings, attention, next token"),
    "04-agentic-ai/topics/01-what-is-an-ai-agent.md": ("agent-loop-anim", "The agent loop: goal, plan, act, observe"),
    "04-agentic-ai/topics/02-tool-calling.md": ("tool-calling-flow", "Tool calling: the model asks, your code runs the tool"),
    "04-agentic-ai/topics/04-react-pattern.md": ("react-loop", "ReAct loop: thought, action, observation, final answer"),
    "04-agentic-ai/topics/15-agent-evaluation.md": ("eval-loop", "Evaluation loop: test set, run, score, improve"),
    "06-mcp/topics/01-model-context-protocol.md": ("mcp-flow", "MCP: host, connect, list tools, call tool, result"),
}


def main():
    changed = 0
    for rel, (name, alt) in MAP.items():
        p = ROOT / rel
        text = p.read_text(encoding="utf-8")
        if MARK in text:
            continue
        depth = rel.count("/")
        img = "../" * depth + f"assets/anim/{name}.svg"
        block = (f"{MARK}{name} -->\n"
                 f"## Watch it\n"
                 f"![{alt}]({img})\n\n"
                 f"*The diagram plays on a loop. Read the caption in the box, one step at a time.*\n\n")
        new, n = re.subn(r"^## How it works", block + "## How it works", text, count=1, flags=re.M)
        if n == 0:
            print("no 'How it works' in", rel)
            continue
        p.write_text(new, encoding="utf-8")
        changed += 1
    print("embedded in", changed, "topic pages")


if __name__ == "__main__":
    main()
