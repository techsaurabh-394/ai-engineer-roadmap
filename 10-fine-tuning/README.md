# Phase 8 · Fine-tuning

**Goal:** know when fine-tuning is the right tool, and how to do it well: prepare clean data, train a small LoRA or QLoRA adapter, read the training curves, and prove with a held-out test that the tuned model beats the base model.

**Who it's for:** readers who finished Phase 3 (LLM Concepts), or who already know what a token, a weight and a loss are. You do not need a GPU: the demos are plain Python with small numbers.

**Prerequisites:**
- [Phase 3 · LLM Concepts](../03-llm-concepts/README.md), especially [Fine-Tuning (the overview)](../03-llm-concepts/topics/15-fine-tuning.md), [Quantization](../03-llm-concepts/topics/18-quantization.md), [LoRA and PEFT](../03-llm-concepts/topics/19-lora-and-peft.md) and [How LLMs Are Trained](../03-llm-concepts/topics/24-how-llms-are-trained.md).
- Basic Python (lists, dictionaries, functions).
- Helpful: [Phase 1 · RAG Foundations](../01-rag-foundations/README.md), because RAG is the main alternative to fine-tuning.

**Time:** about 9-11 hours, including the code.

## Big idea
Fine-tuning changes **how a model behaves**, not **what it knows**. It is the last rung of a ladder (prompt, then RAG, then tune) and it only earns its cost when a test set shows the lower rungs failed. The work is mostly not the training: it is the dataset (clean, de-duplicated, split with no leakage), the loss mask (grade only the assistant's tokens), a cheap method (LoRA, or QLoRA on a 4-bit base), careful reading of the loss curves, and an honest evaluation against the base model.

## Topics

| # | Topic | Level | Link |
|---|---|---|---|
| 01 | Prompting vs RAG vs Fine-Tuning | Beginner | [01-prompting-vs-rag-vs-fine-tuning.md](topics/01-prompting-vs-rag-vs-fine-tuning.md) |
| 02 | Supervised and Instruction Tuning | Intermediate | [02-supervised-and-instruction-tuning.md](topics/02-supervised-and-instruction-tuning.md) |
| 03 | Dataset Preparation | Intermediate | [03-dataset-preparation.md](topics/03-dataset-preparation.md) |
| 04 | LoRA in Depth | Intermediate | [04-lora-in-depth.md](topics/04-lora-in-depth.md) |
| 05 | QLoRA and Quantized Training | Advanced | [05-qlora-and-quantized-training.md](topics/05-qlora-and-quantized-training.md) |
| 06 | Training Pipelines and Hyperparameters | Intermediate | [06-training-pipelines-and-hyperparameters.md](topics/06-training-pipelines-and-hyperparameters.md) |
| 07 | Evaluating Fine-Tuned Models | Advanced | [07-evaluating-fine-tuned-models.md](topics/07-evaluating-fine-tuned-models.md) |

## Learning order
1. **Decide (01).** Start here even if you are keen to train. It teaches you to diagnose the problem and to do the break-even sum.
2. **Teach and prepare (02-03).** How supervised tuning and the loss mask work, then how to build a dataset you can trust.
3. **Make it cheap (04-05).** LoRA in depth, then QLoRA. Read [LoRA and PEFT](../03-llm-concepts/topics/19-lora-and-peft.md) first if the formula is new.
4. **Run it and judge it (06-07).** Hyperparameters and curves, then the evaluation that decides whether to ship. Topic 07 also explains RLHF and DPO.

If you are short on time, read 01, 03 and 07: deciding, data, and proof.

## Run the code
The demos in `code/` are offline, deterministic, standard-library Python: no API key, no GPU, no downloads. All prices in `ft_decision.py` are made-up illustrative units, and the loss curves in `ft_lora_params.py` are a seeded simulation for practising how to read curves, not a real training run.

```bash
# from the repo root
python 10-fine-tuning/code/ft_decision.py       # break-even: months until fine-tuning pays for itself
python 10-fine-tuning/code/ft_dataset_prep.py   # validate, dedupe, scrub PII, group split, loss-mask labels
python 10-fine-tuning/code/ft_lora_params.py    # LoRA parameter counts, rank proof, merge check, QLoRA memory, schedules
python 10-fine-tuning/code/ft_eval.py           # exact match, F1, win rate, bootstrap interval, DPO loss

python -m pytest -q 10-fine-tuning/code         # run the tests (needs: pip install pytest)
```

| File | What it shows | Used in topics |
|---|---|---|
| [`ft_decision.py`](code/ft_decision.py) | tokens per month, prompt-size savings, break-even months, break-even volume, a rule table | 01 |
| [`ft_dataset_prep.py`](code/ft_dataset_prep.py) | schema validation, exact and near-duplicate removal, PII scrub, split by group, token-length stats, chat template and loss-mask labels | 02, 03 |
| [`ft_lora_params.py`](code/ft_lora_params.py) | LoRA parameter counts, rank of B A by Gaussian elimination, merge equivalence, QLoRA memory, LR schedule, loss curves | 04, 05, 06 |
| [`ft_eval.py`](code/ft_eval.py) | exact match, token F1, win rate, bootstrap and paired bootstrap, regression finder, DPO loss | 07 |

## Try it
1. In `ft_decision.py`, raise `requests_per_month` to 600,000. Roughly how much faster is break-even? Then find the volume at which a 12-month payback fails.
2. In `ft_decision.py`, set `tuned_price_multiplier` to 1.0. Does break-even at 200,000 requests improve, and by how much?
3. In `ft_dataset_prep.py`, lower the dedupe threshold from 0.8 to 0.5. Which extra records are dropped, and are they true duplicates?
4. Add a record with a name in the text. Does the PII scrub catch it? What would you need to add?
5. Change the split seed. Do the group overlaps stay empty? Do the split sizes change?
6. In `ft_lora_params.py`, add `down` to the target modules at r = 8. How many parameters is that, and what percent of the model?
7. Change `ToyTransformer` to 24 layers and `d_model` 3072. Recompute the QLoRA memory.
8. In `ft_eval.py`, make the tuned model wrong on three more questions. Does the paired interval still exclude zero?
9. In `ft_eval.py`, try beta 0.5 in `dpo_loss`. How does the loss for the same log-probabilities change?
10. In `simulate_curves`, reduce the penalty term (0.045) and see when early stopping stops firing.

## Prompting, RAG or fine-tuning?
| Need | Use |
|---|---|
| Clearer instructions or format | Prompting, with a few examples (start here) |
| Facts the model lacks, or facts that change, with citations | RAG ([Phase 1](../01-rag-foundations/README.md), [Phase 2](../02-advanced-rag/README.md)) |
| A consistent style or behaviour at high volume, or a small model good at one task | Fine-tuning (this phase), once a test set shows the other two fell short |

## Phase checkpoint quiz
Try to answer before opening each answer.

**1. A client wants to fine-tune the model on a price list that changes weekly. What do you recommend?**
<details><summary>Answer</summary>
Use RAG. Fine-tuning teaches behaviour rather than reliable facts, cannot cite a source, and goes stale as soon as the prices change. Retrieve the current list at question time, and keep fine-tuning for tone or format if needed.
</details>

**2. In this phase's illustrative scenario, a 1,200-token prompt shrinks to 150 tokens at 200,000 requests a month. How is break-even computed, and what is it?**
<details><summary>Answer</summary>
Divide the one-off cost (training plus data, 2,400 units) by the monthly saving (1,170 before minus 860 after, so 310). That is 7.74 months. The prices are made-up illustrative units, not real vendor prices.
</details>

**3. Why mask the prompt tokens in supervised fine-tuning?**
<details><summary>Answer</summary>
Prompt tokens are input, not the behaviour you want produced. Setting their labels to -100 makes the loss average only over the assistant's tokens, so capacity is not spent imitating users and long prompts do not drown out the answer.
</details>

**4. Why is the end-of-turn token included in the trained labels?**
<details><summary>Answer</summary>
It teaches the model when to stop. If it is masked out or missing, the tuned model may keep generating after the answer is finished.
</details>

**5. Why does supervised fine-tuning teach format and behaviour better than facts?**
<details><summary>Answer</summary>
The loss rewards matching the style and structure of replies. A format repeats in every example, so it is learned reliably, while each fact appears only once or twice, is stored unreliably, cannot be cited, and cannot be updated without retraining.
</details>

**6. Why split a fine-tuning dataset by group id instead of by row?**
<details><summary>Answer</summary>
Related rows (same customer, document, or paraphrases) share facts and wording. If they straddle train and test, the model can score well by remembering. In the demo a random row split leaks 4 groups between train and test, while the group split leaks none.
</details>

**7. What is the right order of the data-prep steps, and why?**
<details><summary>Answer</summary>
Validate, de-duplicate, scrub personal data, then split. If you split before de-duplicating, copies can already sit on both sides of the split and leak into the test set.
</details>

**8. How many trainable parameters does a LoRA adapter on one 4096 x 4096 matrix have at r = 8, and what share is that?**
<details><summary>Answer</summary>
r x (d_in + d_out) = 8 x 8192 = 65,536, against 16,777,216 for the full matrix: 0.3906%.
</details>

**9. Why does B start at zero in LoRA, and why is the update's rank at most r?**
<details><summary>Answer</summary>
With B = 0 the product B A is zero, so the model equals the base at step 0. B A is a (d_out x r) times (r x d_in) product, so all its columns are combinations of r columns, which limits it to r independent directions. The demo confirms it with a Gaussian-elimination rank.
</details>

**10. How do alpha and rank interact, and why does merging work?**
<details><summary>Answer</summary>
The update is scaled by alpha / r, so changing r with alpha fixed changes its strength. Merging works because the update is linear: W x + (alpha/r) B (A x) equals (W + (alpha/r) B A) x. The demo verifies this to about 1e-15.
</details>

**11. What is frozen and what is trained in QLoRA?**
<details><summary>Answer</summary>
The base weights are frozen in 4-bit NormalFloat (NF4) and dequantized on the fly. Only the LoRA adapters are trained, in higher precision, with gradients flowing through the dequantized weights.
</details>

**12. Name two memory tricks in the QLoRA paper besides NF4 and what each does.**
<details><summary>Answer</summary>
Double quantization quantizes the per-block quantization constants, cutting their overhead from 0.5 to about 0.127 bits per parameter. Paged optimizers let optimizer state move between GPU and CPU memory to survive memory spikes without running out.
</details>

**13. Micro-batch 4, accumulation 8, one device, 1,000 examples. What is the effective batch, and how many steps in 3 epochs?**
<details><summary>Answer</summary>
Effective batch = 4 x 8 x 1 = 32. Steps per epoch = ceil(1000 / 32) = 32, so 3 epochs are 96 steps.
</details>

**14. Training loss keeps falling but validation loss turns up. What is it called and what do you do?**
<details><summary>Answer</summary>
Overfitting. Stop, restore the checkpoint with the best validation loss, and then lower the epochs or learning rate, add or diversify data, or reduce adapter capacity.
</details>

**15. Why report a bootstrap interval, and what can an average hide?**
<details><summary>Answer</summary>
A score on a small held-out set is noisy, and a paired bootstrap interval shows whether the gain over the base model survives resampling; in the demo, 12 examples give a gain interval of [0.083, 0.833], but 4 examples include zero. An average can also hide regressions, such as the one answer that got worse while the mean rose.
</details>

**16. Write the DPO loss in words. What is its value when the policy equals the reference?**
<details><summary>Answer</summary>
Loss = -log sigmoid(beta x [(chosen log-prob minus its reference log-prob) minus (rejected log-prob minus its reference log-prob)]). When the policy equals the reference the margin is 0, so the loss is -log 0.5 = ln 2 = 0.6931, and it falls as the policy raises the chosen answer and lowers the rejected one.
</details>

More practice: [interview prep · LLM fundamentals](../08-interview-prep/01-llm-fundamentals.md).

**Prev:** [Phase 7 · Production AI](../07-production-ai/README.md) · **Next:** [Phase 9 · Multimodal AI](../11-multimodal-ai/README.md)
