# 04 · Evaluation & Production

### 1. How do you evaluate an LLM feature before launch?
Build a golden dataset of realistic inputs with expected outputs or grading criteria, define metrics tied to user value (correctness, groundedness, format validity, latency, cost), and run it automatically on every prompt, model or retrieval change. Include hard and adversarial cases, not only happy paths.

### 2. What are the pitfalls of LLM-as-judge?
Judges can prefer longer answers, favour their own style, be inconsistent, and miss subtle factual errors. Reduce risk with clear rubrics, pairwise comparisons, a different model as judge, spot-checking against human labels, and using deterministic checks (schema validation, exact match, unit tests) wherever possible.

### 3. How do you monitor an LLM app in production?
Trace every request (prompt, retrieved context, tool calls, output), record tokens, latency and cost, track user feedback and failure rates, sample outputs for human review, and alert on regressions. Without traces you can't debug a bad answer.

### 4. How do you version and test prompts?
Store prompts in version control, give them identifiers, run the evaluation set on every change, and roll out gradually (shadow or A/B) before full release. Treat a prompt change like a code change.

### 5. What guardrails are worth having?
Input checks (length, PII, abuse), output validation (schema, banned content, citation presence), permission checks outside the model, rate limits, and safe fallbacks when the model fails or refuses. Enforce security rules in code, not in the prompt alone.

### 6. How do you handle PII and privacy?
Minimise what you send to the model, redact or tokenise sensitive fields, avoid logging raw sensitive content, follow data-retention policies of your provider and your own, and enforce access control before retrieval.

### 7. How do you build resilience to provider failures?
Timeouts, retries with exponential backoff and jitter, circuit breakers, fallbacks to a secondary model or a cached/degraded answer, queueing for non-urgent work, and monitoring of provider error rates and latency.

### 8. How do you improve perceived latency?
Stream tokens as they're generated, show retrieved sources early, run retrieval and independent tool calls in parallel, use smaller models for simple steps, and cache common results.
