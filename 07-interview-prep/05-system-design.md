# 05 · System Design Prompts

Each prompt lists what a strong answer covers. Practise saying it aloud in 10 to 15 minutes.

## A. "Design a chat-with-company-documents system for 10,000 employees"
- **Requirements first:** document types and volume, freshness needs, latency target, permissions model, citation requirement.
- **Ingestion:** connectors, text extraction (OCR for scans), chunking with metadata (source, page, ACL), embeddings, idempotent updates and deletions.
- **Retrieval:** hybrid search, ACL filter at query time, reranker, query rewriting for follow-ups.
- **Generation:** grounded prompt, citations, "I don't know" behaviour, streaming.
- **Quality:** golden set, retrieval and faithfulness metrics, feedback loop from thumbs up/down.
- **Operations:** tracing, cost per query, caching, rate limits, audit logs, incremental re-indexing.

## B. "Design a customer-support agent that can look up orders and issue refunds"
- **Tools:** `get_order`, `search_help_articles`, `issue_refund`, `escalate_to_human`, each with strict schemas.
- **Safety:** refunds above a threshold need human approval; permission checks in code; idempotency keys to prevent double refunds.
- **Flow:** classify intent, retrieve policy, call tools, confirm with the customer, act, summarise.
- **Reliability:** step and cost limits, fallback to human, full conversation traces.
- **Evaluation:** scripted scenarios (happy path, angry customer, ambiguous order, prompt injection in the message), success rate and escalation rate.

## C. "Design an automated code-review bot"
- **Inputs:** diff, changed files, repo conventions, test results.
- **Approach:** retrieve related code and style rules, review per file then synthesise, post comments with line references, distinguish blocking issues from suggestions.
- **Quality:** low false-positive rate matters more than recall (noisy bots get ignored); measure acceptance rate of comments.
- **Safety:** never run untrusted code from the PR, limit repo permissions, handle secrets in diffs.

## D. "Design semantic search for an e-commerce catalogue"
- **Data:** titles, descriptions, attributes, images; embeddings per product.
- **Retrieval:** hybrid (keyword for SKUs and brands, dense for intent), attribute filters, business-rule boosts (stock, margin), reranking.
- **Evaluation:** query set with relevance labels, nDCG, click-through and conversion in A/B tests.
- **Operations:** incremental indexing as inventory changes, latency budget, caching popular queries.
