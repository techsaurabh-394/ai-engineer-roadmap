"""Animation content for phases 3 and 4 (LLM concepts, agentic AI).

Pipeline step: "Label|sub label|caption line 1|caption line 2".
Loop step:     "Label|caption line 1|caption line 2".
Bars step:     see B() in anim_core.
"""
from anim_core import P, L, B

# ---------------- Phase 3: LLM concepts ----------------
P("01-transformer-architecture", "Transformer architecture", "The design behind modern LLMs, from tokens to a next-token guess.", """
Tokens|input|1. The text is split into tokens|Each token becomes an ID number.
Embed + pos|vectors|2. Each token becomes a vector, with its position added|So the model knows what and where.
Blocks|x N|3. Pass through a stack of identical blocks|Each block: attention, then a feed-forward network.
Scores|every word|4. The last layer scores every word in the vocabulary|One score per possible next token.
Next token|pick one|5. Pick the next token and repeat|That is how text is generated.
""")

P("attention", "Attention: who should this word look at?", "The word sat scores every word. Higher weight means more attention.", topic="02-self-attention", custom=True)

P("03-multi-head-attention", "Multi-head attention", "Several small attentions side by side, each tracking something different.", """
Input|token vectors|1. Token vectors come in|One vector per token.
Split|into heads|2. Split the work into several smaller heads|For example 8 heads.
Attend|in parallel|3. Each head runs its own attention|One may track grammar, another meaning.
Join|concatenate|4. Join the results of all heads|Back into one vector per token.
Mix|output|5. A final layer mixes them together|Richer than a single attention.
""")

P("04-positional-encoding", "Positional encoding", "Attention has no sense of order, so position is added to each token.", """
Tokens|no order yet|1. Attention alone sees a bag of tokens|Order is invisible to it.
Number|each position|2. Make a position signal for each place|Position 1, 2, 3 and so on.
Add|to the vectors|3. Add the position signal to each token vector|Same word, different place, different vector.
Order|now known|4. Dog bites man differs from man bites dog|Now the model can tell them apart.
Variants|sinusoidal, RoPE|5. Methods differ: fixed waves, learned, or rotation (RoPE)|Modern models often use RoPE.
""")

P("05-decoder-only-models", "Decoder-only models", "Read left to right and predict the next token. GPT, Claude and Llama work this way.", """
Prompt|tokens|1. The prompt is a list of tokens|The model starts from these.
Mask|no peeking|2. Each token sees only the tokens before it|Future tokens are hidden.
Predict|next token|3. Predict the next token|A probability for every word in the vocabulary.
Append|add it|4. Add the chosen token to the text|The text is now one token longer.
Repeat|until stop|5. Repeat until a stop token or the length limit|That is a whole reply.
""")

P("06-encoder-decoder-models", "Encoder-decoder models", "One part reads the input, another writes the output.", """
Input|source text|1. The input, such as a sentence to translate|All of it is available at once.
Encoder|reads all|2. The encoder reads the whole input|Every token can see every other token.
Memory|encoded input|3. It produces an encoded version of the input|Think of it as the model's notes.
Decoder|writes|4. The decoder writes the output token by token|At every step it looks back at the notes (cross-attention).
Output|translation|5. The output, such as the translation|Used by models like T5 and the original Transformer.
""")

P("07-kv-cache", "KV cache", "Remember each token's keys and values so they are never recomputed.", """
Token 1|compute K, V|1. For the first token, compute its key and value|Attention needs them.
Save|in the cache|2. Save them in the cache|Memory on the GPU.
Token 2|only new K, V|3. For a new token, compute only its own key and value|Not the whole prompt again.
Reuse|read the cache|4. Attention reads the old keys and values from the cache|No repeated work.
Result|faster|5. Much faster generation|Cost: cache memory grows with text length.
""")

P("08-context-window", "Context window", "The most tokens a model can handle in one request, input and output together.", """
Input|your prompt|1. Your prompt takes up tokens|System text, history, documents and the question.
Output|the reply|2. The reply takes up tokens too|It is generated after the prompt.
Sum|both count|3. Input plus output must fit the limit|For example 128,000 tokens in total.
Full|limit reached|4. When it is full, something must go|Older text is cut, or the request fails.
Plan|budget tokens|5. Budget your tokens|Retrieve less, summarise old turns, set a max output.
""")

B("temperature", "Temperature", "A dial for randomness: low is predictable, high is varied.",
  ["cat", "dog", "car", "sky", "pie"], [
    ("Temperature 0.2", [0.90, 0.07, 0.02, 0.01, 0.00], None, "1. Low temperature (0.2): the top choice wins almost every time", "Safe and repeatable. Good for facts and code."),
    ("Temperature 1.0", [0.50, 0.25, 0.12, 0.08, 0.05], None, "2. Normal temperature (1.0): the model's own probabilities", "The top choice wins often, but others get a chance."),
    ("Temperature 2.0", [0.28, 0.24, 0.20, 0.15, 0.13], None, "3. High temperature (2.0): the choices flatten out", "More surprising, and more likely to go wrong."),
  ], topic="09-temperature")

B("top-k", "Top-k sampling", "Keep only the k most likely tokens, then pick from those.",
  ["the", "a", "this", "that", "my", "our"], [
    ("All tokens", [0.40, 0.25, 0.15, 0.10, 0.06, 0.04], None, "1. The model gives a probability to every token", "Here, 6 tokens. Real vocabularies have tens of thousands."),
    ("Keep top 3", [0.40, 0.25, 0.15, 0.10, 0.06, 0.04], [1, 1, 1, 0, 0, 0], "2. With k = 3, keep only the 3 most likely tokens", "The rest are thrown away."),
    ("Renormalise", [0.50, 0.31, 0.19, 0.0, 0.0, 0.0], [1, 1, 1, 0, 0, 0], "3. Rescale the kept tokens so they add up to 1", "0.40, 0.25, 0.15 divided by 0.80. Sample from these."),
  ], topic="10-top-k")

B("top-p", "Top-p (nucleus) sampling", "Keep the smallest group of tokens whose probabilities reach p.",
  ["the", "a", "this", "that", "my", "our"], [
    ("All tokens", [0.40, 0.25, 0.15, 0.10, 0.06, 0.04], None, "1. Sort tokens from most to least likely", "Same probabilities as the top-k example."),
    ("Reach p = 0.90", [0.40, 0.25, 0.15, 0.10, 0.06, 0.04], [1, 1, 1, 1, 0, 0], "2. Add from the top until the total reaches 0.90", "0.40, 0.65, 0.80, 0.90. So keep 4 tokens."),
    ("Renormalise", [0.44, 0.28, 0.17, 0.11, 0.0, 0.0], [1, 1, 1, 1, 0, 0], "3. Rescale the kept tokens to add up to 1", "Sample from these 4. The group size adapts to each situation."),
  ], topic="11-top-p")

P("12-beam-search", "Beam search", "Keep the few best partial sentences instead of just the best next token.", """
Start|top 3|1. Keep the 3 best first tokens (beam width 3)|Each one starts a partial sentence.
Extend|each beam|2. Extend every beam by one token|Several candidates per beam.
Score|total odds|3. Score each partial sentence by its total probability|Multiply the token probabilities, or add their logs.
Prune|keep best 3|4. Keep only the best 3 again|Repeat until they end.
Pick|best sentence|5. Pick the highest-scoring full sentence|Often better than greedy, but less varied.
""")

P("13-hallucination", "Hallucination", "Fluent, confident text that is not true.", """
Question|unknown fact|1. A question the model cannot truly know|A rare fact, or something after its training.
Predict|likely words|2. The model predicts plausible words, not verified facts|It has no built-in fact checker.
Fluent|confident tone|3. The answer sounds fluent and sure|That makes it believable.
False|not supported|4. But it is wrong or unsupported|A made-up citation, number or name.
Fix|ground it|5. Fixes: RAG with citations, checks, allow I do not know|Make it show its sources.
""")

L("prompt-loop", "Prompt engineering loop", "Write, test on real cases, measure, refine.", """
Write|1. Write the prompt|Instructions, examples, output format.
Test|2. Run it on 20+ real cases|Not just one lucky example.
Measure|3. Score the outputs|Pass or fail against what you need.
Refine|4. Fix the biggest failure first|Then test again.
""", topic="14-prompt-engineering")

P("llm-flow", "From text to answer inside an LLM", "What happens to your prompt, in order.", """
Text|your prompt|1. You type a prompt|Just plain text.
Tokens|word pieces|2. Text is split into tokens|Tokens are word pieces, each with an ID number.
Embeddings|ID to vector|3. Each token becomes a vector|A list of numbers that carries meaning.
Attention|who matters|4. Attention mixes in context|Every token looks at the others to find what matters.
Next token|probabilities|5. The model scores every possible next token|It picks one, adds it, and repeats.
""", color="blue", topic="21-how-an-llm-works-end-to-end")

P("fine-tuning-flow", "Fine-tuning, step by step", "Teach a base model your style or task.", """
Data|examples|1. Collect example inputs and ideal outputs|Quality beats quantity.
Clean|check it|2. Clean and split into train and test|Remove duplicates and wrong labels.
Train|LoRA or full|3. Train: nudge the weights toward your examples|LoRA trains a tiny add-on, so it is cheap.
Evaluate|on test set|4. Test on examples the model never saw|Compare with the base model.
Deploy|serve it|5. Ship it, then keep monitoring|Retrain when your data changes.
""", color="blue", topic="15-fine-tuning")

P("16-calling-an-llm-api", "Calling an LLM API", "Messages go out over HTTPS, text and a token count come back.", """
Messages|with roles|1. Build a list of messages with roles|system, user and assistant.
Send|over HTTPS|2. Send them with your API key|Set a timeout.
Model|generates|3. The model generates a reply|Billed per input and output token.
Response|text + usage|4. You get text plus the tokens used|Log the usage to track cost.
Handle|retry, log|5. Handle errors with retries and backoff|Rate limits and timeouts do happen.
""", color="blue")

P("17-structured-outputs", "Structured outputs", "Make the model return JSON your code can use, and still check it.", """
Schema|define the JSON|1. Define the shape you need|Field names, types, allowed values.
Ask|include schema|2. Give the schema to the model|Many APIs can enforce it.
Reply|JSON text|3. The model replies in JSON|As text, not yet trusted data.
Validate|always|4. Validate against the schema|Even enforced outputs can have wrong values.
Use|or retry|5. Use it, or retry with the error message|Never pass unchecked data on.
""", color="blue")

P("18-quantization", "Quantization", "Store weights as small integers to save memory.", """
Weights|16-bit decimals|1. A model stores weights as 16-bit decimals|Billions of them.
Scale|find the range|2. Find the range of the values|A scale factor maps decimals to integers.
Round|8 or 4-bit ints|3. Round each weight to a small integer|8-bit or 4-bit.
Smaller|less memory|4. Memory drops about 2x (8-bit) or 4x (4-bit)|Versus 16-bit weights.
Check|quality|5. Test quality after shrinking|A small accuracy drop is common.
""", color="blue")

P("19-lora-and-peft", "LoRA and PEFT", "Fine-tune a tiny add-on instead of the whole model.", """
Freeze|base weights|1. Freeze the big model|Its weights never change.
Add|tiny add-on|2. Add two small matrices, A and B, beside a weight|Much smaller than the weight itself.
Train|only A and B|3. Train only A and B|Well under 1 percent of the parameters.
Merge|W + B x A|4. Their product is added to the weight|Or kept separate as an adapter.
Swap|one per task|5. Keep one small adapter per task|Swap them on the same base model.
""", color="blue")

P("20-inference-serving-and-batching", "Inference serving and batching", "Keep the GPU busy by working on many requests at once.", """
Requests|arrive|1. Requests arrive at different times|Different lengths too.
Queue|scheduler|2. A scheduler holds them in a queue|It decides who runs next.
Batch|many at once|3. Run many requests together on the GPU|One GPU pass serves them all.
Swap|finished out|4. Continuous batching swaps finished requests for new ones|Without waiting for the slowest one.
GPU|stays busy|5. The GPU stays busy|More requests per second, lower cost.
""", color="blue")

P("22-qkv-explained-from-scratch", "Q, K and V explained", "Every token asks a question, advertises itself, and carries content.", """
Token|a vector|1. Each token starts as a vector|Three learned matrices will transform it.
Query|what I seek|2. Query (Q): what am I looking for?|Made by multiplying by matrix Wq.
Key|how I advertise|3. Key (K): what do I offer?|Made by multiplying by matrix Wk.
Score|Q dot K|4. Score = the Q of one token dotted with the K of another|A high score means a good match.
Mix|weight the V|5. Softmax the scores, then average the V vectors|V (value) is the content that gets passed on.
""", color="blue")

P("23-inside-a-transformer-block", "Inside a transformer block", "Attention plus three supporting parts.", """
Input|x|1. A vector for each token comes in|Call it x.
Attention|mix tokens|2. Attention mixes information between tokens|Each token learns from the others.
Add + Norm|residual|3. Add the input back (residual), then LayerNorm|Keeps the signal and the numbers stable.
FFN|per token|4. A feed-forward network transforms each token|Where much of the learned knowledge sits.
Add + Norm|output|5. Add and normalise again, then pass it on|The next block does the same.
""", color="blue")

P("24-how-llms-are-trained", "How LLMs are trained", "From next-token guessing to following instructions.", """
Pretrain|next token|1. Pretraining: guess the next token over a huge pile of text|This teaches language and facts.
Loss|how wrong|2. Loss measures how wrong each guess was|One number to push down.
Backprop|find the blame|3. Backpropagation finds how to nudge every weight|Then the weights are updated a little.
SFT|follow orders|4. SFT: train on example questions with good answers|The model learns to follow instructions.
RLHF|human taste|5. RLHF: tune toward answers people prefer|More helpful, safer, better formatted.
""", color="blue")

P("25-modern-attention-variants", "Modern attention variants", "Faster and smaller attention for long contexts.", """
MQA|share K and V|1. MQA: all heads share one set of keys and values|A much smaller KV cache.
GQA|groups share|2. GQA: groups of heads share keys and values|A middle road between quality and memory.
Flash|tiled, exact|3. FlashAttention: the same exact result, computed in tiles|Less memory traffic, so it is faster.
Window|local only|4. Sliding window: each token sees only the last W tokens|Cost no longer grows with the full length.
RoPE|rotate Q and K|5. RoPE: encode position by rotating Q and K|Works well with long contexts.
""", color="blue")

P("26-mixture-of-experts-and-scaling-laws", "Mixture of experts and scaling laws", "Many specialists, but only a few run per token.", """
Token|arrives|1. A token arrives at the MoE layer|Many expert feed-forward blocks wait there.
Router|picks experts|2. A router picks the best 2 experts for this token|Different tokens use different experts.
Experts|run top 2|3. Only those 2 experts run|Most of the model stays idle, saving compute.
Combine|weighted sum|4. Combine their outputs with the router weights|Then continue to the next layer.
Scaling|laws|5. Scaling laws: loss falls predictably with more size, data and compute|Used to plan how big to build.
""", color="blue")

# ---------------- Phase 4: agentic AI ----------------
L("agent-loop-anim", "The agent loop", "An agent is a model plus tools plus memory, in a loop.", """
Goal|1. Receive the goal|For example: book the cheapest train.
Plan|2. Decide the next step|The model picks a tool or answers.
Act|3. Run the tool|Your code executes it safely.
Observe|4. Read the result|Add it to memory, check if the goal is met.
""", color="amber", topic=("01-what-is-an-ai-agent", "12-agent-loops"))

P("tool-calling-flow", "Tool calling, step by step", "The model asks, your code acts.", """
Question|from the user|1. User asks something the model cannot know|For example: what is the weather in Delhi?
Model|picks a tool|2. The model replies with a tool request|It names the tool and its inputs. It does not run anything.
Your code|runs the tool|3. YOUR code runs the function|This is where you check permissions and limits.
Result|sent back|4. The result goes back to the model|As a new message in the conversation.
Answer|in plain words|5. The model writes the final answer|Using the tool result as facts.
""", color="amber", topic=("02-tool-calling", "03-function-calling"))

L("react-loop", "ReAct loop: Think, Act, Observe", "Like a detective: each clue changes the next step.", """
Thought|1. Think: what do I need?|Example: I need France's population.
Action|2. Act: choose a tool and input|Example: lookup(topic = france).
Observation|3. Observe: your code runs the tool|Result: about 68 million.
Final Answer|4. Enough facts? Write the answer|If not, loop again, up to a step limit.
""", topic="04-react-pattern")

P("05-planning-agents", "Planning agents", "Write the whole plan first, then carry it out.", """
Goal|a big task|1. Receive a big task|Too large for one step.
Plan|list the steps|2. The model writes a list of steps|Before doing anything.
Execute|one by one|3. Carry out the steps in order|Using tools as needed.
Check|did it work?|4. Check each result against the plan|Is the step really done?
Re-plan|if stuck|5. If something fails, update the plan|Then continue.
""", color="amber")

L("reflection-loop", "Reflection loop", "Draft, critique, learn from it, try again.", """
Draft|1. Write a first answer|Do not expect it to be perfect.
Critique|2. Review it, or ask a critic|What is wrong or missing?
Note|3. Write down the lessons|Short and concrete.
Retry|4. Try again using the notes|Stop after a few rounds.
""", color="amber", topic="06-reflection")

P("07-memory", "Agent memory", "The model forgets between calls, so your code remembers for it.", """
New info|from the chat|1. New information appears in the conversation|The model will not keep it by itself.
Short-term|this chat|2. Recent turns stay in the prompt|Trimmed to fit the window.
Long-term|saved outside|3. Important facts are saved in a database|Kept across sessions.
Recall|search it|4. Search for what is relevant to the new question|Not everything, just the useful bits.
Prompt|add context|5. Add the recalled items to the prompt|The model now seems to remember.
""", color="amber")

P("08-short-term-memory", "Short-term memory", "The recent conversation, re-sent on every turn.", """
Turns|chat so far|1. The conversation so far is a list of messages|User, assistant and tool messages.
Re-send|every call|2. Send it all again with each new request|The model has no memory of its own.
Limit|window size|3. It must fit in the context window|Cost grows with every turn too.
Trim|drop old turns|4. Drop or shorten the oldest turns|Keep the system prompt and the latest messages.
Summarise|optional|5. Optionally summarise old turns into one message|Saves tokens, keeps the gist.
""", color="amber")

P("09-long-term-memory", "Long-term memory", "Facts saved outside the model and found again later.", """
Save|to a database|1. Save important information outside the model|A database or a file.
Embed|as a vector|2. Embed it so it can be searched by meaning|The same trick as RAG.
Later|new session|3. Days later, a new conversation starts|The old chat is gone.
Search|what is relevant|4. Search the memory using the new question|Return only the most relevant items.
Use|in the prompt|5. Put them in the prompt|The agent picks up where it left off.
""", color="amber")

P("10-semantic-memory", "Semantic memory", "Stable facts that stay true across conversations.", """
Learn|a fact|1. The agent learns a fact|For example: my timezone is IST.
Store|as a fact|2. Store it as a simple fact|timezone = IST.
Stable|stays true|3. It stays true across conversations|Unlike a one-off event.
Recall|any chat|4. Recall it in any later chat|No need to ask again.
Update|when changed|5. Update it when it changes|So old facts do not mislead the agent.
""", color="amber")

P("11-episodic-memory", "Episodic memory", "A log of what the agent tried and how it turned out.", """
Act|try a task|1. The agent tries a task|Using tools and making choices.
Log|what happened|2. Log the task, the actions and the outcome|Success or failure, and why.
Store|as episodes|3. Store each as an episode|Searchable later.
Recall|similar cases|4. On a new task, recall similar episodes|What worked before? What failed?
Improve|next time|5. Use them to do better|Learning from experience, without retraining.
""", color="amber")

P("13-multi-agent-systems", "Multi-agent systems", "Several agents with different roles pass work to each other.", """
Task|a big job|1. A big job comes in|Too broad for one agent.
Planner|splits it|2. A planner agent splits it into parts|One part per specialist.
Workers|each a role|3. Worker agents each handle one part|Each has its own prompt and tools.
Hand off|messages|4. They pass results to each other as messages|For example: researcher to writer to reviewer.
Result|combined|5. The results are combined into one answer|Set limits so the agents cannot loop forever.
""", color="amber")

P("14-human-in-the-loop", "Human in the loop", "The agent pauses so a person can approve risky steps.", """
Plan|agent proposes|1. The agent proposes its next action|For example: send this email.
Pause|ask a human|2. It pauses before anything risky|The state is saved.
Review|approve or edit|3. A person approves, edits or rejects|They see exactly what will happen.
Act|if approved|4. It continues only if approved|Otherwise it stops or re-plans.
Log|record it|5. Record who decided what|An audit trail.
""", color="amber")

P("16-agent-guardrails", "Agent guardrails", "Code-level limits on what an agent can see and do.", """
Input|check it|1. Check the input first|Block clearly unsafe requests.
Tools|allowlist|2. Only allow listed tools|Anything else is refused.
Arguments|validate|3. Validate every tool argument|Types, ranges, allowed values.
Limits|steps and cost|4. Cap the steps, time and money|So it cannot run away.
Output|filter|5. Filter what goes back to the user|Remove secrets and unsafe content.
""", color="amber")
