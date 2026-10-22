# 05 · Evaluation

**Goal:** measure whether your AI feature is actually good, so changes are decisions, not guesses. Time: 1 hour.

## Big idea
"It looks good" isn't evidence. Build a **golden set**: realistic questions with the expected answer or source. Run it on every change and watch the numbers.

For retrieval, two simple metrics:
- **hit@k:** for what fraction of questions does a correct chunk appear in the top k results?
- **MRR (mean reciprocal rank):** average of 1/rank of the first correct result. Rank 1 scores 1.0, rank 2 scores 0.5, rank 4 scores 0.25.

For generated answers, also check **faithfulness** (is every claim supported by the sources) and **citation correctness**.

## Run it
```bash
cd 05-evaluation/code
python eval_harness.py
pytest -q
```
The harness runs the golden questions through the RAG index from step 2 and prints hit@3 and MRR.

## Try it
1. Add 5 new questions about your own documents to `GOLDEN`.
2. Change the chunk size in `02-rag/code/mini_rag.py`. Re-run. Did the score move?
3. Add a question the system should fail on. Does the harness report the miss?
4. Write a rule: "no change merges if hit@3 drops". That is what a CI check does.

## Honest warning
A small golden set that you wrote yourself is easy to score 100% on. Treat it as a smoke test, then grow it with real user questions and past failures.

## LLM-as-judge
A model can grade answers too, but judges prefer long answers, can be inconsistent and miss subtle errors. Use clear rubrics, spot-check against human labels, and use deterministic checks (exact match, schema validation) wherever you can.

## Checkpoint
- What is the difference between hit@k and MRR?
- Why evaluate retrieval and generation separately?
- Name two pitfalls of LLM-as-judge.

(Answers: [interview prep, evaluation](../07-interview-prep/04-evaluation-and-production.md))

**Next:** [06 · Production](../06-production/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
