# What is AI engineering?

**Read time: 5 minutes. No code.** This page answers three questions: what does an AI engineer actually do, what do you build with, and where should you start.

## The short answer
An AI engineer builds **software that uses an existing AI model**, such as Claude or GPT, to do something useful. You mostly do not train models. You connect a model to your data, your tools and your users, and you make sure it behaves.

Compare the roles:

| Role | Main job | Typical output |
|---|---|---|
| **AI engineer** (this roadmap) | Build apps around a ready-made model | A chat over company documents, an agent that files tickets, an API |
| Machine learning engineer | Train and tune models | A trained model and the pipeline that retrains it |
| Data scientist | Analyse data and run experiments | Reports, forecasts, prototypes |
| Software engineer | Build software | Any app; uses AI as one part |

The lines blur at real companies. This table is a simplification to help you choose what to study.

## The four problems you will keep solving

Every topic in this roadmap helps with one of these:

| Problem | In plain words | Where it is taught |
|---|---|---|
| **The model does not know my data** | It was not trained on your documents or on last week's news | RAG: [Phase 1](../01-rag-foundations/README.md) and [Phase 2](../02-advanced-rag/README.md) |
| **I do not understand why it behaves like that** | Answers change, it makes things up, it has a size limit | [Phase 3: LLM concepts](../03-llm-concepts/README.md) |
| **I want it to do things, not only talk** | Call an API, search, write a file, in several steps | Agents and MCP: [Phase 4](../04-agentic-ai/README.md), [Phase 6](../06-mcp/README.md) |
| **I need it to work for real users, safely and cheaply** | Speed, cost, logging, abuse | [Phase 7: production](../07-production-ai/README.md) |

[Phase 5](../05-frameworks/README.md) covers libraries such as LangChain that help you wire these together.

## Prompting, RAG or fine-tuning?
This is the first design question on almost every project. Try them in this order, because each one costs more than the last.

| If your problem is... | Try | Why |
|---|---|---|
| The answers have the wrong format or tone | **Better prompts** | Cheapest. Change a few lines of text |
| The model lacks facts about your data, or you need citations | **RAG** (retrieval-augmented generation) | Look up the right passages and hand them to the model |
| You need a consistent style or a small model that does one narrow task well | **Fine-tuning** | Train a little more on your examples. Needs good data |

Fine-tuning is not covered in this roadmap. Learn prompting and RAG first, since they solve most first projects.

## One request, step by step
Here is what happens when someone asks a company chatbot, "What is your refund policy?"

1. The question arrives at your **API** (Phase 7).
2. Your code turns the question into numbers called an **embedding** and searches a **vector database** for the closest passages ([Phase 1](../01-rag-foundations/topics/02-embeddings.md)).
3. A **reranker** puts the best passages first ([Phase 2](../02-advanced-rag/topics/01-reranking.md)).
4. Your code builds a **prompt**: the question plus those passages plus instructions such as "answer only from these passages and cite them" ([Phase 3](../03-llm-concepts/topics/14-prompt-engineering.md)).
5. The **model** writes the answer one token at a time ([Phase 3](../03-llm-concepts/README.md)).
6. If the user asks "cancel my order", an **agent** may call a tool to do it, with a person approving the risky step ([Phase 4](../04-agentic-ai/README.md)).
7. You **log** the whole request, check cost and watch for abuse ([Phase 7](../07-production-ai/README.md)).

You can run a tiny version of steps 2 to 5 now: `python 01-rag-foundations/code/mini_rag.py "Why use multi-stage builds?"`

## What you need to know first
- **Python basics:** variables, functions, lists, dictionaries, loops. That is enough to start.
- **Not needed to start:** maths beyond school level, machine learning theory, a GPU, or paid API keys. The demos run offline.
- **Helpful later:** HTTP and APIs, Docker basics, SQL, Git. Phase 7 teaches the AI-specific parts.

## Honest limits
- This roadmap teaches the ideas and gives you small, tested examples. It does not replace building a real project, which is where most of the learning happens.
- Tools and library APIs change fast. The ideas (retrieval, tool calling, evaluation, guardrails) change slowly. Learn the ideas first, then check the current docs for the library you pick.
- Nobody can promise you a job from reading anything. A finished, tested project that you can explain is the strongest evidence you can show.

## Where to go next
1. Open [learning paths](learning-paths.md) and pick the one that fits you.
2. Set up your computer: [00-start-here](../00-start-here/README.md).
3. Read [What is RAG?](../01-rag-foundations/topics/01-what-is-rag.md).
4. Keep the [glossary](glossary.md) open for new words.

[Back to the README](../README.md)
