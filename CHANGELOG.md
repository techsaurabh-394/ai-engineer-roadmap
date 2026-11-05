# Changelog

All notable changes to this repository are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This repository has no version numbers or releases; entries are grouped by date.

## [Unreleased]

### Changed
- All 113 topic pages now follow one beginner-friendly shape: a **Before you start** line that links the pages to read first, a **Words to know** list that defines every term, a **Worked example** that follows one small case number by number (taken from the repo's own demos), and a three-point **Recap**. Reading times were raised to match and `docs/learning-paths.md` was regenerated (992 minutes in total).
- `00-start-here/README.md` explains the shape of a topic page and how to master a topic.
- Fixed one wrong number: the mean in the monitoring page's code comment is 517.5, not 507.5.

### Added
- Six deeper LLM topics in Phase 3 (the roadmap now has 113 topics and 339 topic interview questions), each with an offline standard-library demo and tests: [How an LLM Works End to End](03-llm-concepts/topics/21-how-an-llm-works-end-to-end.md), [Q, K and V Explained From Scratch](03-llm-concepts/topics/22-qkv-explained-from-scratch.md), [Inside a Transformer Block](03-llm-concepts/topics/23-inside-a-transformer-block.md), [How LLMs Are Trained](03-llm-concepts/topics/24-how-llms-are-trained.md), [Modern Attention Variants](03-llm-concepts/topics/25-modern-attention-variants.md) and [Mixture of Experts and Scaling Laws](03-llm-concepts/topics/26-mixture-of-experts-and-scaling-laws.md).
- Path F in `docs/learning-paths.md`: a 7-day "inside the LLM" route. Phase 3 README gains a beginner-first reading order, 4 new exercises and 5 new checkpoint quiz questions. Fifteen glossary terms added (Q/K/V, logit, residual, LayerNorm, cross-entropy, GQA, RoPE, MoE and more).
- 13 new practice questions (Part 9 in `docs/practice.md`, Q49 to Q61, easy to hard) for the six new topics. Every number was computed with the demo code. The page now has 61 questions; the score is out of 60.
- Six new topics, from basic to advanced, each with an offline standard-library demo and tests (the roadmap now has 107 topics and 321 topic interview questions):
  - Phase 3: [Calling an LLM API](03-llm-concepts/topics/16-calling-an-llm-api.md) (beginner), [Structured Outputs](03-llm-concepts/topics/17-structured-outputs.md) and [Quantization](03-llm-concepts/topics/18-quantization.md) (intermediate), [LoRA and PEFT](03-llm-concepts/topics/19-lora-and-peft.md) and [Inference Serving and Batching](03-llm-concepts/topics/20-inference-serving-and-batching.md) (advanced).
  - Phase 2: [LLM-as-a-Judge](02-advanced-rag/topics/21-llm-as-a-judge.md) (advanced).
- Path E in `docs/learning-paths.md`: an 8-day "LLM builder" route from the first API call to serving. `scripts/gen_learning_paths.py` now reads the topic counts and phase sizes from the files instead of hard-coding them.
- Seven glossary terms (backoff, structured output, quantization, LoRA, continuous batching, LLM-as-a-judge, Cohen's kappa).
- `docs/what-is-ai-engineering.md`: a 5-minute, no-code introduction for complete beginners.
- `docs/learning-paths.md`: four day-by-day plans (RAG app, agents and MCP, full 8 weeks, interview week). Reading times are summed from the topic pages (725 minutes for all 101 topics).
- `docs/practice.md`: 47 practice questions with answers (warm-up, RAG, LLM concepts, agents and MCP, production, calculations, run-the-demo, design scenarios) plus build challenges. Every number was computed with code.
- Seven new glossary terms (fine-tuning, hit@k, MRR, RRF, ANN, stride, rate limit).
- Topic 15 in Phase 3: [Fine-Tuning](03-llm-concepts/topics/15-fine-tuning.md), with a standard-library demo that checks a fine-tuning dataset (`llm_finetune_data.py`) and 8 tests. The roadmap now has 101 topics and 303 topic interview questions.
- `scripts/gen_learning_paths.py`, which regenerates `docs/learning-paths.md`, and a CI job that fails when the committed page is out of date.
- Six new SVG diagrams in `assets/`: learning path, big picture, RAG end to end, workflow vs agent, interview roadmap, production checklist.
- `docs/how-to-study.md` and `docs/glossary.md`.
- Issue templates for new topic ideas and broken links.
- A CI job that fails when a link to a local file in a Markdown page is broken.
- `CHANGELOG.md`.

### Changed
- README redesigned: badges, a "Start in 5 minutes" block with real output, a per-phase "What you'll need" table, a progress table, and an SVG overview replacing the Mermaid chart.
- Corrected claims that did not match the repo: demos use only the standard library (topic-page snippets and the optional `anthropic` package do not), 08-interview-prep has 43 questions plus 4 system-design prompts (not 47 questions), `mcp-notes-server` has 6 tools (not 4), and Phase 7 demos need no Docker, Redis or Kubernetes.
- The five existing SVGs gained a `<title>`, a `<desc>`, and a reduced-motion rule that turns their animation off for visitors who ask for less motion.
- Fixed a broken DeepEval link in the RAG Evaluation topic.
- `CONTRIBUTING.md`, the pull request template and `CITATION.cff` updated.

## [2026-10-22]

### Changed
- Fixed library snippets after testing them against current releases.
- Restructured the repository into 7 phases with 100 beginner-friendly topics, each with code, tests and interview questions.

### Added
- Phase 6 TypeScript MCP example, interview prep (08), projects (09), CI, link-check and stale-issue workflows.

## [Initial commit]

### Added
- Initial repository.

[Unreleased]: https://github.com/techsaurabh-394/ai-engineer-roadmap/compare/47b5212...HEAD
