# Phase 10 · Frontier AI Engineering

**Goal:** understand the newer ideas that decide what an AI product costs, how fast it feels and whether you can trust its numbers: reasoning models, prompt caching, speculative decoding, preference tuning, distillation and honest evaluation.

**Who it's for:** learners who finished the LLM basics ([Phase 3](../03-llm-concepts/README.md)) and fine-tuning ([Phase 8](../10-fine-tuning/topics)) and want the levers that working AI teams actually pull.

**Prerequisites:**
- [Phase 3 · LLM Concepts](../03-llm-concepts/README.md), especially [KV Cache](../03-llm-concepts/topics/07-kv-cache.md) and [Context Window](../03-llm-concepts/topics/08-context-window.md).
- [Phase 8 · Fine-Tuning](../10-fine-tuning/topics) for supervised tuning and datasets.
- Basic Python and school maths. The code uses only the standard library.

**Time:** about 8-10 hours, including the code.

## Big idea
A model is not a fixed black box. You can spend more compute at answer time (thinking, voting), reuse work between requests (caching), guess ahead cheaply (speculative decoding), steer behaviour from comparisons (RLHF, DPO), shrink a big model into a small one (distillation) and measure all of it (evals). Each lever has a price and a failure mode, and each lesson here shows both with numbers you can reproduce in a few seconds.

## Topics

| # | Topic | Level | Link |
|---|---|---|---|
| 01 | Reasoning Models and Test-Time Compute | Advanced | [01-reasoning-models-and-test-time-compute.md](topics/01-reasoning-models-and-test-time-compute.md) |
| 02 | Context Engineering and Prompt Caching | Intermediate | [02-context-engineering-and-prompt-caching.md](topics/02-context-engineering-and-prompt-caching.md) |
| 03 | Speculative Decoding | Advanced | [03-speculative-decoding.md](topics/03-speculative-decoding.md) |
| 04 | Preference Tuning: RLHF and DPO | Advanced | [04-preference-tuning-rlhf-and-dpo.md](topics/04-preference-tuning-rlhf-and-dpo.md) |
| 05 | Distillation and Synthetic Data | Intermediate | [05-distillation-and-synthetic-data.md](topics/05-distillation-and-synthetic-data.md) |
| 06 | Evaluating Generative Models and Benchmarks | Advanced | [06-evaluating-generative-models-and-benchmarks.md](topics/06-evaluating-generative-models-and-benchmarks.md) |

## Learning order
1. **Cost and speed (01, 02, 03).** Thinking and voting, then caching, then speculative decoding.
2. **Behaviour (04, 05).** Preference tuning, then distillation and synthetic data.
3. **Trust (06).** Evaluating all of the above. If you read one lesson first, read 06.

If you are short on time, read 02, 01 and 06.

## Run the code
The demos in `code/` are offline, standard-library Python: no API key, no downloads, no GPU. Prices and probabilities in them are illustrative.

```bash
# from the repo root
python 12-frontier-ai/code/fa_reasoning.py     # majority voting, when it helps and hurts
python 12-frontier-ai/code/fa_context.py       # prefix cache hits and conversation cost
python 12-frontier-ai/code/fa_speculative.py   # acceptance rate and speed-up
python 12-frontier-ai/code/fa_preference.py    # reward-model loss and DPO loss
python 12-frontier-ai/code/fa_distill.py       # soft targets, temperature, data filter
python 12-frontier-ai/code/fa_evals.py         # pass@k, bootstrap interval, contamination

python -m pytest -q 12-frontier-ai/code        # run the tests (needs: pip install pytest)
```

| File | What it shows | Used in topics |
|---|---|---|
| [`fa_reasoning.py`](code/fa_reasoning.py) | exact and simulated majority vote, cost multiplier | 01 |
| [`fa_context.py`](code/fa_context.py) | prefix match, input cost with and without caching, break-even | 02 |
| [`fa_speculative.py`](code/fa_speculative.py) | expected tokens per pass, speed-up, best draft length | 03 |
| [`fa_preference.py`](code/fa_preference.py) | Bradley-Terry loss, DPO loss, beta effect | 04 |
| [`fa_distill.py`](code/fa_distill.py) | temperature softmax, KL loss, synthetic-row filter | 05 |
| [`fa_evals.py`](code/fa_evals.py) | pass@k, bootstrap interval, n-gram contamination | 06 |

## Try it
1. In `fa_reasoning.py`, find the smallest odd n where p = 0.55 gives a majority above 0.70.
2. In `fa_context.py`, change turns from 20 to 5. How much does caching save now?
3. In `fa_speculative.py`, find the acceptance rate below which a draft costing 0.2 never helps for k = 4.
4. In `fa_preference.py`, set beta to 1.0. What does the loss do for the same shift, and why can that be risky?
5. In `fa_distill.py`, add a banned phrase to the filter and count what is removed.
6. In `fa_evals.py`, change the 50-question run to 500 questions at 72 percent. How much narrower is the interval?

## Phase checkpoint quiz
Try to answer before opening each answer.

**1. Why can majority voting turn a 40 percent model into a 90 percent one, and when does it fail?**
<details><summary>Answer</summary>
When wrong answers are spread over many values, the right answer is the only one that repeats, so it wins the vote as n grows (0.40 becomes about 0.93 at n = 21 in the demo with five wrong values). It fails when mistakes are correlated: if the model gives one specific wrong answer more often than the right one, voting amplifies the mistake.
</details>

**2. How would you decide whether a task deserves extended thinking?**
<details><summary>Answer</summary>
Run a held-out set at several thinking budgets and plot accuracy against cost and latency. Tasks with many dependent steps or a cheap checker usually gain; lookups, formatting and classification usually do not. Choose the smallest budget where the curve flattens.
</details>

**3. What is the difference between voting and verifying?**
<details><summary>Answer</summary>
Voting picks the most common answer and needs no checker. Verifying generates candidates and uses a checker (tests, a calculator, a trained verifier) to pick one that passes. Verifying beats voting when checking is much cheaper than solving.
</details>

**4. Your cache-hit ratio fell from 90 percent to 2 percent after a release. What do you check first?**
<details><summary>Answer</summary>
Whether something that changes per request, such as a timestamp, user name or reordered tool list, moved into the early part of the prompt. A prefix cache matches only identical leading tokens, so any early change invalidates the rest. Diff the rendered prompts of two consecutive calls and find the first differing token.
</details>

**5. What does context engineering mean, beyond writing a good prompt?**
<details><summary>Answer</summary>
Deciding what information enters the window, in what order, and in what form: selecting retrieved facts, compressing history, isolating sub-tasks in separate contexts and persisting facts outside the window. It treats the window as a scarce, costly resource instead of a dumping ground.
</details>

**6. Why can too much context hurt accuracy?**
<details><summary>Answer</summary>
Long windows cost more and are slower, and models can miss or misuse facts buried in the middle or contradicted elsewhere in the prompt. Smaller, relevant context often beats a larger, noisier one, so measure with your own questions.
</details>

**7. Why is checking k tokens about as cheap as generating one?**
<details><summary>Answer</summary>
Decoding is limited by loading the weights from memory, not by arithmetic, so scoring k positions in one pass uses spare compute at almost the same time as scoring one position. The big model reads all the draft tokens in parallel, as it does with a prompt.
</details>

**8. Does speculative decoding reduce output quality?**
<details><summary>Answer</summary>
No, when implemented with the proper acceptance rule. Draft tokens are accepted or replaced so that the final text follows exactly the target model's distribution. Only latency changes.
</details>

**9. A team reports a 0.8x speed-up after enabling it. What do you suspect?**
<details><summary>Answer</summary>
A low acceptance rate or an expensive draft: perhaps a mismatched model family, a draft that is too large, or a batch size so high that no spare compute existed. Measure acceptance rate, draft cost and tokens per second, and shorten k or change the draft.
</details>

**10. Explain the DPO loss in plain words.**
<details><summary>Answer</summary>
It raises the probability of the chosen answer, relative to a frozen reference model, more than that of the rejected answer. The margin is beta times the difference of the two log-probability gains, and the loss is −log sigmoid of it. At the start the margin is 0 and the loss is ln 2 = 0.693.
</details>

**11. What is reward hacking and how does RLHF limit it?**
<details><summary>Answer</summary>
The model finds outputs that score high with the reward model but are not actually good, such as padded or oddly styled text. RLHF adds a penalty for drifting away from the reference model, and teams monitor with held-out human-labelled checks.
</details>

**12. When would you pick DPO over PPO-based RLHF?**
<details><summary>Answer</summary>
When you have a fixed preference dataset, a small team and want a stable, cheaper recipe with fewer moving parts. PPO-style RLHF fits when you need online learning from the model's own new outputs or when you have a programmatic reward such as tests passing.
</details>

**13. Why does a higher temperature help in distillation?**
<details><summary>Answer</summary>
It flattens the teacher's distribution so the relative probabilities of wrong answers become visible, for example 0.456 versus 0.142 instead of 0.823 versus 0.025. Those relationships are extra information the student can learn from.
</details>

**14. What is model collapse and how do you reduce the risk?**
<details><summary>Answer</summary>
Repeated training on model-generated text narrows diversity and compounds errors. Reduce it by mixing in real data, deduplicating, filtering with checks, tracking provenance and always evaluating on human-made held-out data.
</details>

**15. How do you check that synthetic data helped?**
<details><summary>Answer</summary>
Train on real only, synthetic only and a mix; evaluate all on the same held-out human-made set with confidence intervals. Keep synthetic examples out of the test set and spot-check random rows by hand.
</details>

**16. Explain pass@k and when pass@1 is more appropriate than pass@5.**
<details><summary>Answer</summary>
pass@k is the probability that at least one of k samples passes the tests, estimated as 1 − C(n−c, k)/C(n, k) from n samples with c passing. Use pass@1 when the user gets a single answer; pass@5 only if the product really shows or tests five candidates.
</details>

**17. Model A scores 72 percent and Model B 70 percent on 50 questions. What do you conclude?**
<details><summary>Answer</summary>
Nothing yet. A 95 percent bootstrap interval for 72 percent on 50 questions is roughly 58 to 84, which easily contains 70. You need more questions or a paired comparison on the same items before claiming a difference.
</details>

**18. How do you detect and handle benchmark contamination?**
<details><summary>Answer</summary>
Compare n-gram overlap between test items and training text, flag high-overlap items, report results with and without them, and prefer private or newly written test sets. Also keep your own evaluation data out of any synthetic data or tuning sets.
</details>

More practice: [interview prep · LLM fundamentals](../08-interview-prep/01-llm-fundamentals.md).

**Prev:** [Phase 9 · Multimodal AI](../11-multimodal-ai/README.md) · **Next:** [Back to the roadmap](../README.md)
