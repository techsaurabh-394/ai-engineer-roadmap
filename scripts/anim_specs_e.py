"""Animation content for Phase 8 (fine-tuning)."""
from anim_core import P, L, B  # noqa: F401

# ---------------- Phase 8: fine-tuning ----------------
P("ft-decision", "Prompting, RAG or fine-tuning?", "Try the cheapest lever first, and do the sums before you train.", """
Prompt|clearer asks|1. Start with a better prompt and a few examples|It is the cheapest lever: minutes, no training.
Facts|use RAG|2. If the model lacks facts, look them up with RAG|Facts change and need sources; training cannot cite.
Behaviour?|style, format|3. Still wrong in style or format? Consider tuning|It must repeat on many requests.
Count cost|break-even|4. Compare monthly saving with the one-off cost|Shorter prompts at high volume pay back.
Decide|tune or not|5. Tune only if it pays back and you can test it|Otherwise keep prompting or RAG.
""", topic="01-prompting-vs-rag-vs-fine-tuning")

P("ft-sft-masking", "Supervised tuning with a loss mask", "Show the model finished chats, but grade only the assistant's words.", """
Pairs|prompt + reply|1. Collect examples of the exact replies you want|Quality beats quantity.
Template|chat format|2. Render each chat with the model's own template|Role markers and end tokens must match exactly.
Mask|labels -100|3. Hide prompt tokens from the loss with -100|Only the reply and its end token are graded.
Train|next token|4. Train the model to predict each reply token|It learns format and behaviour, not new facts.
Check|held out|5. Test on chats it never saw|Compare with the base model.
""", color="blue", topic="02-supervised-and-instruction-tuning")

P("ft-dataset-prep", "Dataset preparation", "Clean the data first: most fine-tuning failures start here.", """
Validate|schema|1. Reject records with bad roles or empty replies|Check every record against one schema.
Dedupe|exact + near|2. Drop exact and near-duplicate examples|Copies are over-counted and leak into tests.
Scrub|PII|3. Replace emails and phone numbers with placeholders|A regex is a first pass, not a guarantee.
Split|by group|4. Split by group id so no group crosses the line|Related rows must stay on one side.
Measure|token lengths|5. Check the length stats and the longest rows|Know what gets truncated before you train.
""", color="amber", topic="03-dataset-preparation")

L("ft-lora-in-depth", "LoRA in depth", "Freeze W, train a small B and A, then optionally merge them.", """
Freeze W|1. The big weight matrix W stays frozen|It receives no updates, so no gradients or optimizer state.
Add B and A|2. Train two thin matrices: B (d by r) and A (r by d)|Parameters: r times (d in plus d out).
Scale|3. Add the update as W + (alpha / r) times B A|Alpha and r together set how loud the update is.
Merge|4. Fold B A into W for serving with no extra delay|Or keep the adapter as a small file to swap.
""", color="pink", topic="04-lora-in-depth")

P("ft-qlora", "QLoRA", "A frozen 4-bit base with small adapters trained in higher precision.", """
Load base|4-bit NF4|1. Store the base weights in 4-bit NormalFloat|About 4.5 bits each with plain block constants.
Freeze|no updates|2. The 4-bit base is frozen and never trained|Only the adapters get gradients.
Adapters|LoRA in 16-bit|3. Add LoRA adapters kept in higher precision|Weights are dequantized on the fly for each matmul.
Save memory|DQ + paging|4. Double quantization and paged optimizers help|Named in the paper to trim memory and spikes.
Train|one GPU|5. Train on the loss as usual, then ship the adapter|Compare with the 16-bit model before you trust it.
""", color="teal", topic="05-qlora-and-quantized-training")

L("ft-training-loop", "Training pipeline", "Warm up, train, watch both loss curves, and keep a checkpoint.", """
Warm up|1. Raise the learning rate from near zero to its peak|Early steps are noisy; a gentle start avoids damage.
Accumulate|2. Add gradients over several small batches|Effective batch = micro batch x accumulation steps.
Decay|3. Lower the rate as training goes on|A cosine or linear decay gives a calm finish.
Watch|4. Compare training and validation loss|If validation rises while training falls, stop.
Checkpoint|5. Save the best checkpoint and the seed|So you can resume, roll back and reproduce.
""", color="amber", topic="06-training-pipelines-and-hyperparameters")

B("ft-evaluation", "Evaluating a tuned model", "Same held-out questions, base against tuned, with a margin of error.",
  ["exact match", "token F1", "win rate"],
  [("Base model", [0.42, 0.50, 0.50], None, "1. Score the base model on held-out questions", "It is the baseline: 0.42 exact match, 0.50 F1."),
   ("Tuned model", [0.92, 0.99, 0.75], None, "2. Score the tuned model on the same questions", "0.92 exact match, 0.99 F1, and it wins 7 of 12."),
   ("Check range", [0.92, 0.99, 0.75], [True, False, False], "3. Put a confidence interval around the gain", "12 questions is few: the range is wide, so grow the set."),
   ("Regressions", [0.92, 0.99, 0.75], [False, True, False], "4. Look for answers that got worse", "One question regressed: an average hides it.")],
  color="blue", topic="07-evaluating-fine-tuned-models")
