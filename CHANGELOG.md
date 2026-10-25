# Changelog

All notable changes to this repository are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This repository has no version numbers or releases; entries are grouped by date.

## [Unreleased]

### Added
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
