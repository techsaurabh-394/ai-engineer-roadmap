# 01 · LLM Fundamentals

### 1. What is a token, and why does it matter?
A token is the unit a model reads and writes: a word, part of a word, or punctuation, produced by a tokenizer. Context limits, latency and price are all measured in tokens, and the same text can cost different amounts on different models because tokenizers differ. Non-English text and code often use more tokens per word.

### 2. What is a context window, and why do long prompts still fail?
The context window is the maximum number of tokens (input plus output) the model can attend to in one request. Bigger windows don't guarantee good use of them: models can miss details buried in the middle of a long prompt, cost and latency grow with length, and irrelevant text adds noise. Put the most important instructions and evidence where the model attends best (start and end), and retrieve only what's relevant.

### 3. What do temperature and top-p do?
Both control randomness when sampling the next token. Lower temperature makes output more deterministic; higher makes it more varied. Top-p (nucleus sampling) restricts sampling to the smallest set of tokens whose probabilities add up to p. Use low values for extraction, classification and code, higher for brainstorming. Even at temperature 0 outputs are not guaranteed to be identical across runs.

### 4. Why do LLMs hallucinate, and how do you reduce it?
They generate plausible text from patterns, not from a verified fact store, so they can state wrong things confidently. Reduce it by grounding answers in retrieved sources, requiring citations, letting the model say "I don't know", constraining output with schemas, lowering temperature for factual tasks, and verifying outputs automatically where possible.

### 5. Prompting vs RAG vs fine-tuning: how do you choose?
Start with prompting (cheapest, fastest to iterate). Use RAG when the model needs knowledge it doesn't have or that changes often (private documents, recent data) and you need citations. Use fine-tuning to change behaviour, style or format, or to make a smaller model good at a narrow task. They combine well: fine-tune for behaviour, RAG for knowledge.

### 6. Explain attention in one minute.
Self-attention lets each token weigh every other token in the sequence when building its representation. The model computes query, key and value vectors; similarity between a token's query and the other tokens' keys decides how much of their values to mix in. Multiple heads learn different relationships in parallel. Compute grows with sequence length, which is why long contexts are expensive.

### 7. What are embeddings and what are they used for?
Embeddings are vectors that place text (or images, code) in a space where similar meaning lands close together. They power semantic search, clustering, deduplication, recommendations and the retrieval step of RAG. Similarity is usually measured with cosine similarity or dot product.

### 8. What is prompt injection?
An attack where untrusted text (a web page, document, email or tool result) contains instructions that the model follows as if they came from the developer or user. Mitigate with least-privilege tools, treating all retrieved or tool-returned text as data, confirmation for sensitive actions, output filtering, and isolating high-risk actions from untrusted content. No single prompt wording fully prevents it.

### 9. What is function calling / tool use?
The model returns a structured request to call a named function with arguments (validated against a schema); your code runs it and sends the result back. The model never executes anything itself. Good tool design means clear names and descriptions, narrow single-purpose tools, validated inputs and readable error messages.

### 10. How do you cut LLM cost and latency?
Use the smallest model that meets quality, cache repeated prompt prefixes and responses, shorten prompts and retrieved context, batch non-urgent work, stream output to improve perceived latency, route easy requests to cheaper models, and measure tokens per request so regressions are visible.
