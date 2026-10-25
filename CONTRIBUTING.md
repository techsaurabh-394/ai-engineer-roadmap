# Contributing

Thanks for helping beginners learn AI engineering.

**Easy ways to help**
- Fix a typo or unclear explanation in any lesson
- Add or correct an interview question (use the existing format)
- Add an exercise or a checkpoint question to a lesson
- Fix or replace a broken link (use the "Broken link" issue template)
- Add your finished project to [09-projects](09-projects/README.md)

**Code and content changes**
1. Open an issue for anything bigger than a small fix. There are templates for bugs, new topic ideas and broken links.
2. Fork, branch from `main`, make the change.
3. Run `pytest -q` from the repo root (Python 3.10 or newer; CI uses 3.12). If you changed `06-mcp/code/hello-mcp-server`, also run `npm ci && npm run build` inside it (Node 18 or newer; CI uses 22).
4. Open a pull request using the template. Keep lessons simple, runnable and in plain English.

**Rules of thumb**
- Keep every number true. The README states topic counts per phase and question counts; if you add or remove a topic or question, update those numbers in the same pull request.
- Scripts in `code/` folders use only the standard library. If a lesson needs a package, show it in a snippet and mark it with a `pip install` line.
- Every topic page keeps the same sections (see [00-start-here](00-start-here/README.md#every-topic-page-has-the-same-shape)).
- Diagrams in `assets/` are hand-written SVG with a `<title>`, a `<desc>` and a dark background. CI checks that they are valid XML.
- Links in Markdown files that point to files in this repo must resolve. CI checks this on every pull request.

Maintained by Saurabh Pathak.
