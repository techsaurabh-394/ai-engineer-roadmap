# How to study this repo

This guide turns the roadmap into a routine you can actually follow. Nothing here is required; it is one way that works.

## 1. Set up once (20 minutes)
Follow [00-start-here](../00-start-here/README.md): create a virtual environment, run `pip install -r requirements.txt`, and run `pytest -q`. Node.js is only needed for the TypeScript server in Phase 6.

## 2. Pick a plan and a pace
Not sure what to read first? [Learning paths](learning-paths.md) has day-by-day plans for four goals. New to the field? Read [What is AI engineering?](what-is-ai-engineering.md) first.

These are the paces suggested in [00-start-here](../00-start-here/README.md#suggested-pace):

| Pace | Time per day | Finish in |
|---|---|---|
| Relaxed | 30 min (1 topic) | about 4 months |
| Steady | 1 hour (2 topics) | about 2 months |
| Intensive | 3 hours | about 3 weeks |

A weekly rhythm that works well: 5 days of new topics, 1 day to review the interview questions of the week, 1 day off. Skip a phase if you already know it, but do its checkpoint quiz first.

## 3. What to do for each topic
Every topic page has the same shape, so use the same routine each time:

1. **Read** the one-line definition and the "Explain it like I'm new" analogy. If you can say the idea in your own words, move on.
2. **Look at the diagram** in "How it works" and follow the numbered steps.
3. **Run the code.** Run the demo script the topic points to, then change one number and run it again. Predict the change before you run.
4. **Answer the 3 interview questions out loud** before opening the hidden answers.
5. **Check the sources** in "Go deeper" for the one idea you found least clear.
6. **Tick the box** in the [checklist](../README.md#the-full-checklist) on your fork.

Some topics show code that needs extra packages (for example LangChain or FastAPI). Those snippets are marked with `pip install`. The scripts in `code/` folders never need them. The [what-you'll-need table](../README.md#what-youll-need) lists what each phase uses.

## 4. At the end of each phase
- Do the matching part of the [practice questions](practice.md) (Part 2 for Phases 1 and 2, Part 3 for Phase 3, Part 4 for Phases 4 to 6, Part 5 for Phase 7, Part 9 for the newer topics in Phases 2 and 3).
- Take the checkpoint quiz in the phase README.
- Look up any term you hesitated on in the [glossary](glossary.md).
- Update the progress table in the README on your fork.

## 5. Build something
After Phase 4, pick one project from [09-projects](../09-projects/README.md) and finish it. Extending [`mini_rag.py`](../01-rag-foundations/code/mini_rag.py) with your own notes is a good first one.

## 6. Interview week
Use [08-interview-prep](../08-interview-prep/README.md): one file per day for five days, answer out loud, and prepare one example from something you built for every answer.

## If you get stuck
Re-read the analogy, run the demo with a smaller input, then open an issue using the "bug" or "new topic idea" template.
