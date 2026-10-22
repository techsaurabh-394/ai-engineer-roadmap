# 06 · Production

**Goal:** know what changes when an AI demo becomes a real service. Time: 1 hour. This is a checklist you can copy into your own project.

## Cost and latency
- [ ] Use the smallest model that meets your quality bar; route easy requests to cheaper models
- [ ] Cache repeated prompt prefixes and common answers
- [ ] Keep retrieved context short and relevant
- [ ] Stream responses so users see output immediately
- [ ] Track tokens and cost per request

## Reliability
- [ ] Timeouts on every external call
- [ ] Retries with exponential backoff and jitter
- [ ] A fallback (another model, a cached answer, or a graceful error)
- [ ] Step and cost caps on agents
- [ ] Idempotent actions so retries are safe

## Security and privacy
- [ ] Treat retrieved text and tool results as untrusted (prompt injection)
- [ ] Least-privilege credentials for every tool
- [ ] Human approval for destructive or high-impact actions
- [ ] Redact or avoid sending PII; avoid logging sensitive content
- [ ] Enforce access control **before** retrieval
- [ ] Keep secrets in environment variables, never in prompts or repos

## Quality
- [ ] A golden set that runs in CI (step 5)
- [ ] Prompts in version control, changed like code
- [ ] Gradual rollout (shadow or A/B) for prompt and model changes
- [ ] A feedback button and a way to add failures to the golden set

## Observability
- [ ] Trace every request: prompt, retrieved context, tool calls, output
- [ ] Dashboards for latency, error rate, cost and feedback
- [ ] Alerts on regressions

## Exercise
Pick your own AI project and tick this list honestly. Write the three unticked items that worry you most, and fix one this week.

**Next:** [07 · Interview prep](../07-interview-prep/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
