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
| **Fine-tuning** | Training a model a little more on your own examples so it learns a style or a narrow task. Try prompting and RAG first. | [Fine-Tuning](../03-llm-concepts/topics/15-fine-tuning.md) |
| **hit@k** | Out of your test questions, the share where the right chunk appeared in the top k results. | [RAG Evaluation](../02-advanced-rag/topics/20-rag-evaluation.md) |
| **MRR** (mean reciprocal rank) | Like hit@k, but it rewards finding the right chunk higher up: rank 1 scores 1, rank 2 scores 1/2, rank 3 scores 1/3. | [RAG Evaluation](../02-advanced-rag/topics/20-rag-evaluation.md) |
| **RRF** (reciprocal rank fusion) | A simple way to merge several ranked lists: each item scores the sum of 1 / (k + rank) over the lists, with k usually 60. | [Ensemble Retrieval](../02-advanced-rag/topics/12-ensemble-retrieval.md) |
| **ANN** (approximate nearest neighbour) | A search that may miss the very closest vector but is much faster. HNSW and IVF are two kinds. | [HNSW](../01-rag-foundations/topics/14-hnsw.md), [IVF Index](../01-rag-foundations/topics/15-ivf-index.md) |
| **Stride** | How far a sliding window moves each step. Overlap equals window minus stride. | [Sliding Window Chunking](../01-rag-foundations/topics/08-sliding-window-chunking.md) |
| **Rate limit** | A cap on how many requests are allowed in a time window, to stop overload and abuse. | [Redis](../07-production-ai/topics/04-redis.md) |
| **Backoff** | Waiting longer after each failed try (1 s, then 2 s, then 4 s) so you do not hammer a service that is already struggling. | [Calling an LLM API](../03-llm-concepts/topics/16-calling-an-llm-api.md) |
| **Structured output** | A model answer in a fixed shape (such as JSON with named fields) that your code can read without guessing. | [Structured Outputs](../03-llm-concepts/topics/17-structured-outputs.md) |
| **Quantization** | Storing a model's numbers with fewer bits (for example 4 instead of 16) so it uses less memory, at the cost of a small rounding error. | [Quantization](../03-llm-concepts/topics/18-quantization.md) |
| **LoRA** (low-rank adaptation) | Fine-tuning by freezing the model and training a small add-on, so it needs far less memory than changing every weight. | [LoRA and PEFT](../03-llm-concepts/topics/19-lora-and-peft.md) |
| **Continuous batching** | A serving trick: when one request finishes, a waiting request takes its place on the GPU straight away instead of waiting for the whole batch. | [Inference Serving and Batching](../03-llm-concepts/topics/20-inference-serving-and-batching.md) |
| **LLM-as-a-judge** | Using a language model to grade answers when there is no exact right answer to compare with. It has biases you must test for. | [LLM-as-a-Judge](../02-advanced-rag/topics/21-llm-as-a-judge.md) |
| **Cohen's kappa** | A score for how much two graders agree, after removing the agreement you would get by pure chance. 1 is perfect, 0 is chance. | [LLM-as-a-Judge](../02-advanced-rag/topics/21-llm-as-a-judge.md) |
| **Query, Key, Value (Q, K, V)** | Three lists of numbers made from every token. The query is what the token is looking for, the key is how it advertises itself, and the value is the content it hands over when matched. | [Q, K and V Explained From Scratch](../03-llm-concepts/topics/22-qkv-explained-from-scratch.md) |
| **Embedding vector** | The list of numbers that stands for one token. Tokens with similar meaning get similar numbers. | [How an LLM Works End to End](../03-llm-concepts/topics/21-how-an-llm-works-end-to-end.md) |
| **Logit** | A raw score the model gives each possible next token before it is turned into a probability. | [How an LLM Works End to End](../03-llm-concepts/topics/21-how-an-llm-works-end-to-end.md) |
| **Parameter (weight)** | One learned number inside the model. Training adjusts billions of them. | [How LLMs Are Trained](../03-llm-concepts/topics/24-how-llms-are-trained.md) |
| **Residual connection** | A shortcut that adds a layer's input back to its output, so information and learning signals can skip layers. | [Inside a Transformer Block](../03-llm-concepts/topics/23-inside-a-transformer-block.md) |
| **LayerNorm** | Rescales each token's numbers to a steady range so deep networks train stably. | [Inside a Transformer Block](../03-llm-concepts/topics/23-inside-a-transformer-block.md) |
| **Feed-forward network (FFN)** | A small network applied to each token on its own after attention. It holds much of what the model has memorised. | [Inside a Transformer Block](../03-llm-concepts/topics/23-inside-a-transformer-block.md) |
| **Cross-entropy loss** | The training score for next-token prediction: minus the log of the probability given to the right token. Lower is better. | [How LLMs Are Trained](../03-llm-concepts/topics/24-how-llms-are-trained.md) |
| **Gradient descent** | Repeatedly nudging every parameter a little in the direction that lowers the loss. | [How LLMs Are Trained](../03-llm-concepts/topics/24-how-llms-are-trained.md) |
| **RLHF / DPO** | Ways to tune a model toward answers people prefer, using ranked answer pairs. | [How LLMs Are Trained](../03-llm-concepts/topics/24-how-llms-are-trained.md) |
| **GQA / MQA** | Attention designs where many query heads share a few key/value heads, which shrinks the KV cache. | [Modern Attention Variants](../03-llm-concepts/topics/25-modern-attention-variants.md) |
| **FlashAttention** | An exact attention algorithm that works in small tiles so the huge score table never has to be stored in slow memory. | [Modern Attention Variants](../03-llm-concepts/topics/25-modern-attention-variants.md) |
| **RoPE** (rotary position embedding) | Encodes word position by rotating query and key vectors, so attention depends on how far apart two tokens are. | [Modern Attention Variants](../03-llm-concepts/topics/25-modern-attention-variants.md) |
| **Mixture of Experts (MoE)** | A model with many expert sub-networks where a router sends each token to only a few of them, so only some parameters are active per token. | [Mixture of Experts and Scaling Laws](../03-llm-concepts/topics/26-mixture-of-experts-and-scaling-laws.md) |
| **Scaling law** | The observed pattern that model loss falls smoothly as parameters, data and compute grow. | [Mixture of Experts and Scaling Laws](../03-llm-concepts/topics/26-mixture-of-experts-and-scaling-laws.md) |
| **Untrusted** | Text you did not write yourself (such as user input, web pages or retrieved files), so your system must not obey it as instructions. | [AI Security](../07-production-ai/topics/11-ai-security.md) |
| **Least privilege** | Giving each tool, service or user only the minimum permissions needed to do its job. | [AI Security](../07-production-ai/topics/11-ai-security.md) |
| **Escaping** | Converting special characters (such as `<` to `&lt;`) so model output is displayed as plain text instead of executed as code. | [AI Security](../07-production-ai/topics/11-ai-security.md) |
| **Defence in depth** | Stacking several independent protections (input checks, least privilege, output escaping, human approval) so one missed check does not cause a breach. | [AI Security](../07-production-ai/topics/11-ai-security.md) |
| **PII** (personally identifiable information) | Private data that can identify a person, such as names, email addresses, phone numbers or payment card numbers. | [AI Security](../07-production-ai/topics/11-ai-security.md), [Guardrails](../07-production-ai/topics/12-guardrails.md) |
