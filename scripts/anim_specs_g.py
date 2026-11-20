"""Animation content for Phase 10 (frontier AI engineering)."""
from anim_core import P, L, B  # noqa: F401

_T = "12-frontier-ai/topics/"

P("fa-reasoning-test-time-compute", "Test-time compute", "Spend more tokens at answer time to buy accuracy.", """
Ask|one question|1. The same question goes to the model|Easy ones need no extra thinking.
Think|longer chain|2. A reasoning model writes steps before answering|Thinking tokens are billed and fill the window.
Sample|n attempts|3. Or run n independent attempts|21 attempts of 800 tokens is 168x one short answer.
Vote|most common|4. Pick the most common final answer|p = 0.6 becomes 0.8256 at n = 21.
Check|correlated?|5. Beware repeated wrong answers|At p = 0.45 with one wrong value, n = 21 gives 0.321.
Budget|measure it|6. Set a thinking budget and test on your data|Pick the smallest budget where accuracy stops rising.
""", color="amber", topic=_T + "01-reasoning-models-and-test-time-compute")

P("fa-context-and-prompt-caching", "Prompt caching", "Only an identical prefix is reused, so order matters.", """
Layers|stable first|1. Build the prompt from stable to changing|System, tools, examples, history, then the new message.
Hash|prefix match|2. The server matches the leading tokens it has seen|Only an unbroken identical prefix counts.
Hit|2001 of 2002|3. Edit only the end and nearly all tokens are reused|These are billed at the cheaper cached rate.
Miss|0 of 2002|4. Put a timestamp first and nothing is reused|One early change voids every token after it.
Cost|$0.61 to $0.10|5. A 20-turn chat falls from $0.6120 to $0.1047|Demo prices are illustrative.
Measure|hit ratio|6. Log cached versus uncached tokens|A sudden drop after a deploy points to a moved timestamp.
""", color="teal", topic=_T + "02-context-engineering-and-prompt-caching")

P("fa-speculative-decoding", "Speculative decoding", "A cheap draft guesses, the big model checks in one pass.", """
Draft|small model|1. A small model proposes k tokens|It is quick, and often right on predictable text.
Verify|one big pass|2. The big model scores all k positions at once|Checking k tokens costs about as much as making one.
Keep|agreed prefix|3. Keep tokens up to the first disagreement|At a mismatch the big model's own token is used.
Bonus|+1 token|4. The big model adds one token of its own|So each round yields 1 to k + 1 tokens.
Maths|a = 0.7, k = 4|5. Expected 2.773 tokens per big-model pass|That is 1.98x faster when the draft costs 10 percent.
Risk|weak draft|6. A poor draft can be slower than none|a = 0.3 at 40 percent cost gives 0.55x.
""", color="blue", topic=_T + "03-speculative-decoding")

P("fa-preference-tuning", "Preference tuning", "Learn from which of two answers people choose.", """
Pairs|win vs lose|1. Collect prompts with a better and a worse answer|From raters, rubrics or tests passing and failing.
Reward|score answers|2. RLHF trains a reward model on the pairs|Loss is minus log sigmoid of the score gap.
RL|PPO loop|3. The model practises against that reward model|A penalty keeps it near the reference model.
DPO|no reward model|4. DPO uses the pairs directly with one loss|Raise chosen versus rejected, relative to the reference.
Start|loss = ln 2|5. At the start the loss is 0.6931|Moving log-probs by 5 each way gives 0.3133.
Watch|bias checks|6. Watch for reward hacking, length bias, sycophancy|A rising score is not proof of better answers.
""", color="pink", topic=_T + "04-preference-tuning-rlhf-and-dpo")

P("fa-distillation-synthetic-data", "Distillation and synthetic data", "A student copies a teacher; filter what it writes.", """
Teacher|big model|1. A big model answers or produces probabilities|Soft probabilities carry more than one label.
Soften|temperature T|2. Divide logits by T before softmax|T = 3 turns 0.823 into 0.456 for the top class.
Student|small model|3. Train the student to match the teacher|Loss is KL divergence times T squared.
Write|synthetic rows|4. Or have the teacher write training examples|Cheap, but errors can compound.
Filter|dedupe, check|5. Drop short, boilerplate and duplicate rows|The demo keeps 2 of 5 rows.
Test|held-out set|6. Compare on human-made data you never trained on|Real only, synthetic only and a mix.
""", color="teal", topic=_T + "05-distillation-and-synthetic-data")

P("fa-evaluating-generative-models", "Trustworthy evals", "Right metric, an error bar, a clean test set.", """
Metric|match the use|1. Pick the metric your product needs|Code uses pass@k with unit tests.
pass@k|n=10, c=3|2. Estimate pass@k from n samples, c correct|pass@1 is 0.30 and pass@3 is 0.7083.
Samples|50 questions|3. Small test sets are noisy|72 percent on 50 questions is really 58 to 84.
Bootstrap|resample|4. Resample the results to get an interval|A rival at 70 percent is inside the range.
Leak|n-gram overlap|5. Check test items against the training text|One demo item shares 0.8 of its 5-grams.
Lock|never tune|6. Keep a locked test set and refresh it|Add each production failure to the dev set.
""", color="amber", topic=_T + "06-evaluating-generative-models-and-benchmarks")
