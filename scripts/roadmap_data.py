#!/usr/bin/env python3
"""The words on the site's Roadmap page (#roadmap), one entry per phase.

Counts, reading times and levels are NOT here: build_site.py computes them from the real lessons.
What is here is the explanation a newcomer needs: why the phase exists, what you can do after it,
which ideas it covers and which runnable demos belong to it. test_roadmap.py checks that every
demo file and lab named here exists, so this file cannot point at something that is gone.

    python scripts/roadmap_data.py     # prints a one-line summary per phase
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

STEPS = [
    ("Read one lesson", "Each lesson is a few minutes: an analogy, a diagram, a worked example with real numbers."),
    ("Watch and play", "Step through the animation, then open the lab and change an input to see what moves."),
    ("Run the code", "Every phase ships small Python demos with tests. They need only the standard library."),
    ("Check yourself", "Answer the interview questions on each page, then take the phase quiz."),
]

PHASES = {
    "01": dict(
        tagline="Make an AI answer from your own documents",
        why="A model only knows what it was trained on. RAG (retrieval-augmented generation) fetches the right pieces of your "
            "documents first and hands them to the model, so answers are specific and can cite a source. Nearly every "
            "useful AI product starts here.",
        outcomes=["Split documents into chunks and explain why the cut points matter",
                  "Turn text into vectors and compare them with cosine similarity",
                  "Combine keyword search (BM25) and vector search into a hybrid"],
        ideas=["chunking", "embeddings", "cosine similarity", "BM25", "vector indexes (HNSW, IVF)", "hybrid search"],
        builds=["mini_rag.py: a complete small RAG pipeline", "rag_chunkers.py", "rag_bm25.py", "rag_hnsw_ivf.py"],
        demos=["01-rag-foundations/code/mini_rag.py", "01-rag-foundations/code/rag_chunkers.py",
               "01-rag-foundations/code/rag_bm25.py", "01-rag-foundations/code/rag_hnsw_ivf.py"],
        labs=["chunking", "similarity", "search"],
        needs="Basic Python: lists, dictionaries, functions, loops.",
    ),
    "02": dict(
        tagline="Fix RAG answers that are only okay",
        why="The basic pipeline works on easy questions and fails on hard ones. This phase teaches which lever to pull: "
            "reorder results, rewrite the question, use relationships between documents, keep the index fresh, and measure "
            "whether any change actually helped.",
        outcomes=["Rerank retrieved chunks and see the order improve",
                  "Rewrite or expand a query before searching (including HyDE)",
                  "Measure RAG quality with hit rate, MRR and an LLM judge"],
        ideas=["reranking", "query transformation", "GraphRAG", "incremental indexing", "evaluation metrics", "LLM-as-judge"],
        builds=["adv_rerank.py", "adv_query_transform.py", "adv_graph_rag.py", "eval_harness.py: a small evaluation harness"],
        demos=["02-advanced-rag/code/adv_rerank.py", "02-advanced-rag/code/adv_query_transform.py",
               "02-advanced-rag/code/adv_graph_rag.py", "02-advanced-rag/code/eval_harness.py"],
        labs=[],
        needs="Phase 1: you can chunk, embed and retrieve the top-k chunks.",
    ),
    "03": dict(
        tagline="Understand what happens inside the model",
        why="Once you know how a model reads and picks the next word, its behaviour stops being magic. You can predict "
            "why it forgets, why temperature changes the style, why long prompts cost more, and how to debug a bad answer.",
        outcomes=["Follow one sentence through a tiny transformer, step by step",
                  "Explain queries, keys and values with real numbers",
                  "Choose temperature, top-k and top-p on purpose, not by guessing"],
        ideas=["attention", "Q/K/V", "transformer block", "training", "sampling", "KV cache", "LoRA and quantization", "mixture of experts"],
        builds=["llm_e2e_walkthrough.py: a toy LLM end to end", "llm_qkv.py", "sampling_demo.py", "llm_lora_math.py"],
        demos=["03-llm-concepts/code/llm_e2e_walkthrough.py", "03-llm-concepts/code/llm_qkv.py",
               "03-llm-concepts/code/sampling_demo.py", "03-llm-concepts/code/llm_lora_math.py"],
        labs=["toyllm", "attention", "sampling"],
        needs="Basic Python and school maths. The idea of a probability is enough. No machine-learning background.",
    ),
    "04": dict(
        tagline="Give the model tools and a loop",
        why="A chat model only writes text. An agent also takes actions: it calls a tool, reads the result and decides what "
            "to do next. Building one from scratch shows what agent frameworks do for you, and where agents break.",
        outcomes=["Write a tool-calling loop with a step cap",
                  "Add short-term and long-term memory to an agent",
                  "Split work across several agents and add guardrails"],
        ideas=["tool calling", "ReAct", "memory", "planning", "multi-agent", "guardrails"],
        builds=["agent.py: a minimal agent", "agent_react.py", "agent_memory.py", "agent_multi.py"],
        demos=["04-agentic-ai/code/agent.py", "04-agentic-ai/code/agent_react.py",
               "04-agentic-ai/code/agent_memory.py", "04-agentic-ai/code/agent_multi.py"],
        labs=[],
        needs="Phase 3, especially context windows, hallucination and prompting. Python functions, dicts and JSON.",
    ),
    "05": dict(
        tagline="Know what the frameworks are for",
        why="LangChain, LangGraph, LlamaIndex, CrewAI and the rest package patterns you can already write by hand. "
            "Learning the pattern first lets you judge a framework, and choose none when none is needed.",
        outcomes=["Map each framework to the idea it wraps",
                  "Build a small chain and a small graph yourself",
                  "Decide when a framework helps and when it only adds weight"],
        ideas=["LangChain", "LangGraph", "LlamaIndex", "CrewAI and AutoGen", "DSPy", "Haystack"],
        builds=["fw_mini_chain.py", "fw_mini_graph.py"],
        demos=["05-frameworks/code/fw_mini_chain.py", "05-frameworks/code/fw_mini_graph.py"],
        labs=[],
        needs="Phases 1 to 4: RAG, tool calling and the agent loop.",
    ),
    "06": dict(
        tagline="Plug tools into any AI app with one protocol",
        why="The Model Context Protocol (MCP) is a standard way for an AI app to discover and call tools and read resources. "
            "Write a server once and many apps can use it.",
        outcomes=["Explain servers, clients, tools, resources and prompts",
                  "Read a JSON-RPC exchange line by line",
                  "Build a small custom MCP server"],
        ideas=["the protocol", "servers and clients", "tools", "resources", "prompts", "building a custom server"],
        builds=["mcp_mini_jsonrpc.py", "hello-mcp-server (TypeScript)"],
        demos=["06-mcp/code/mcp_mini_jsonrpc.py"],
        labs=[],
        needs="Phases 4 and 5. Basic JSON. Node.js 18 or newer only if you build the TypeScript server.",
    ),
    "07": dict(
        tagline="Turn a demo into a service people can rely on",
        why="A demo only has to work once. A service has to be fast, cheap, safe and observable for many users. "
            "This phase covers APIs, containers, caching, queues, monitoring, security and cost.",
        outcomes=["Cache and rate-limit model calls",
                  "Run slow work in a task queue",
                  "Trace, monitor and put guardrails around a live AI feature"],
        ideas=["FastAPI", "Docker and Kubernetes", "caching", "queues", "observability", "security", "cost control"],
        builds=["prod_cache_ratelimit.py", "prod_task_queue.py", "prod_observability.py", "prod_guardrails.py"],
        demos=["07-production-ai/code/prod_cache_ratelimit.py", "07-production-ai/code/prod_task_queue.py",
               "07-production-ai/code/prod_observability.py", "07-production-ai/code/prod_guardrails.py"],
        labs=[],
        needs="Basic Python and the earlier phases, especially MCP. No cloud account is needed for the code here.",
    ),
}

AFTER = [
    ("Interview prep", "08-interview-prep", "Questions with model answers, grouped by topic. Practise saying them out loud."),
    ("Projects", "09-projects", "Nine project ideas from beginner to advanced, and a checklist for what finished means."),
]

if __name__ == "__main__":
    for pid, p in sorted(PHASES.items()):
        print(pid, p["tagline"], "| labs:", ", ".join(p["labs"]) or "none", "| demos:", len(p["demos"]))
