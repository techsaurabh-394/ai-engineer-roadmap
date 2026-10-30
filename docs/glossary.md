# Glossary

Plain-English definitions of terms used across the roadmap. Each links to the topic that teaches it.

| Term | In plain English | Learn more |
|---|---|---|
| **LLM** (large language model) | A model that reads text and predicts the next piece of text, over and over, to write answers. | [Phase 3](../03-llm-concepts/README.md) |
| **Token** | The small piece of text a model reads and writes: a word, part of a word, or punctuation. Limits and prices are counted in tokens. | [Tokenization](../01-rag-foundations/topics/03-tokenization.md) |
| **Context window** | The most text (in tokens) the model can look at in one request, including your prompt and its answer. | [Context Window](../03-llm-concepts/topics/08-context-window.md) |
| **Prompt** | The instructions and text you send to the model. | [Prompt Engineering](../03-llm-concepts/topics/14-prompt-engineering.md) |
| **Hallucination** | When a model states something false in a confident way. | [Hallucination](../03-llm-concepts/topics/13-hallucination.md) |
| **RAG** (retrieval-augmented generation) | First find the most relevant pieces of your documents, then give them to the model so it answers from them and can cite them. | [What is RAG?](../01-rag-foundations/topics/01-what-is-rag.md) |
| **Embedding** | A list of numbers that represents the meaning of a piece of text, so similar meanings end up close together. | [Embeddings](../01-rag-foundations/topics/02-embeddings.md) |
| **Chunking** | Cutting long documents into smaller pieces so each one can be embedded, searched and shown to the model on its own. | [Chunking](../01-rag-foundations/topics/04-chunking.md) |
| **Vector database** | A store that keeps embeddings and quickly finds the ones closest to a query. | [Vector Databases](../01-rag-foundations/topics/10-vector-databases.md) |
| **Cosine similarity** | A score for how closely two embeddings point in the same direction. | [Cosine Similarity](../01-rag-foundations/topics/11-cosine-similarity.md) |
| **BM25** | A classic keyword-matching score used for search. | [BM25](../01-rag-foundations/topics/17-bm25.md) |
| **Hybrid search** | Running keyword search and embedding search together and merging the results. | [Hybrid Search](../01-rag-foundations/topics/16-hybrid-search.md) |
| **Reranking** | A second, more careful pass that re-orders the first search results so the best ones come first. | [Reranking](../02-advanced-rag/topics/01-reranking.md) |
| **Evals** (evaluation) | Tests that measure how good your system is, using a fixed set of questions with known good answers. | [RAG Evaluation](../02-advanced-rag/topics/20-rag-evaluation.md), [Agent Evaluation](../04-agentic-ai/topics/15-agent-evaluation.md) |
| **Agent** | A program where the model decides what to do next, usually by choosing tools, until the goal is met or a limit is hit. | [What is an AI Agent?](../04-agentic-ai/topics/01-what-is-an-ai-agent.md) |
| **Tool call** (function call) | The model asks your code to run a named function with arguments; your code runs it and sends back the result. The model never runs anything itself. | [Tool Calling](../04-agentic-ai/topics/02-tool-calling.md), [Function Calling](../04-agentic-ai/topics/03-function-calling.md) |
| **Agent loop** | The repeating cycle of think, act (call a tool), observe the result, and decide whether to stop. | [Agent Loops](../04-agentic-ai/topics/12-agent-loops.md) |
| **MCP** (Model Context Protocol) | An open standard for connecting AI apps to tools, data and prompts, so one server works with many clients. | [Model Context Protocol](../06-mcp/topics/01-model-context-protocol.md) |
| **Guardrails** | Checks around the model, on its inputs and outputs, that block or fix unsafe or invalid content. | [Agent Guardrails](../04-agentic-ai/topics/16-agent-guardrails.md), [Guardrails](../07-production-ai/topics/12-guardrails.md) |
| **Prompt injection** | Hostile instructions hidden in text the system reads (a web page, a document) that the model may follow. | [AI Security](../07-production-ai/topics/11-ai-security.md) |
| **Observability** | Recording what your system did (requests, steps, tokens, time, cost) so you can find out why it behaved a certain way. | [Observability](../07-production-ai/topics/13-observability.md) |
| **Temperature** | A setting that controls how random the model's word choices are. Low values give steadier answers, high values give more varied ones. | [Temperature](../03-llm-concepts/topics/09-temperature.md) |
| **Transformer** | The neural network design behind modern LLMs. It uses attention to weigh how much each word matters to every other word. | [Transformer Architecture](../03-llm-concepts/topics/01-transformer-architecture.md) |
| **Self-attention** | The step where each token looks at the other tokens in the text and decides which ones to focus on. | [Self-Attention](../03-llm-concepts/topics/02-self-attention.md) |
| **KV cache** | Saved intermediate results from earlier tokens, so the model does not recompute them for every new token it writes. | [KV Cache](../03-llm-concepts/topics/07-kv-cache.md) |
| **Query expansion** | Rewriting or adding to a search question so retrieval finds more of the relevant passages. | [Query Expansion](../02-advanced-rag/topics/05-query-expansion.md) |
| **HyDE** (hypothetical document embeddings) | Have the model write a made-up answer first, then search with that answer's embedding instead of the short question. | [HyDE](../02-advanced-rag/topics/07-hyde.md) |
| **Cost optimization** | Cutting token use and model spend, for example with shorter prompts, caching and smaller models, without hurting quality. | [Cost Optimization](../07-production-ai/topics/14-cost-optimization.md) |
