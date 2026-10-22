# 01 · LLM basics

**Goal:** understand how a language model produces text, so you can predict its behaviour and debug it. Time: 1 hour.

## Big idea
A language model reads **tokens** (pieces of words) and, one at a time, predicts the next token. It outputs a score for every possible token; those scores become probabilities; one token is picked. That's all generation is, repeated.

- **Tokens:** the units of input and output. Context limits and prices are counted in tokens.
- **Context window:** the most tokens the model can look at in one request (your prompt plus its reply).
- **Temperature:** how adventurous the pick is. Low = almost always the top choice. High = more variety.
- **Top-p:** only consider the smallest set of tokens that together cover probability p.
- **Hallucination:** the model produces fluent text that is not true. It predicts plausible words, it doesn't look facts up.

## Run it
```bash
cd 01-llm-basics/code
python sampling_demo.py
```
You will see how the same scores turn into different probabilities at temperature 0.2, 1.0 and 2.0. At 0.2 one token wins almost every time; at 2.0 the choice spreads out.

## Try it
1. Change `LOGITS` so "banana" has the highest score. What does the model now "believe"?
2. Add a temperature of 0.05 and 5.0. What happens to the probabilities?
3. Change `p` in `top_p_filter` from 0.9 to 0.5. Which tokens survive?
4. Write down when you'd choose low temperature (extraction, code) and when high (brainstorming).

## Prompting, RAG or fine-tuning?
| Need | Use |
|---|---|
| Better instructions or format | Prompting (start here, it's free) |
| Knowledge the model lacks (your documents, recent facts), with citations | RAG (step 2) |
| A consistent style or behaviour, or a small model good at a narrow task | Fine-tuning |

## Checkpoint
- What is a token? Why does it matter for cost?
- Why can a very long prompt still give worse answers?
- Why does the model hallucinate, and name two ways to reduce it.
- What is prompt injection, and why can't one clever prompt fully stop it?

(Answers: [interview prep, LLM fundamentals](../07-interview-prep/01-llm-fundamentals.md))

**Next:** [02 · RAG](../02-rag/README.md)

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a><br/>
  <a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>
</p>
