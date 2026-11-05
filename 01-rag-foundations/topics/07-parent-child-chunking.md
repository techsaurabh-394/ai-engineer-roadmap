# Parent Child Chunking
> **Level:** Intermediate · **Reading time:** 8 min · **Phase:** 1 — RAG Foundations

**In one line:** Parent-child chunking searches over small "child" chunks for precise matches, then gives the model the larger "parent" chunk each child came from, for full context.

**Before you start:** [Chunking](04-chunking.md), [Recursive Chunking](05-recursive-chunking.md) and [Metadata](09-metadata.md). You should know what a chunk is and that a chunk can carry extra facts such as an ID.

## Explain it like I'm new
Think of a library index card. The card is tiny: "Egg boiling times — page 112". It is easy to find the right card. But you do not read the card to cook; you open the book to page 112 and read the whole section. The card is the **child**. The page is the **parent**.

Chunking has a built-in conflict (see [Chunking](04-chunking.md)): small chunks are best for *finding*, large chunks are best for *understanding*. Parent-child chunking stops choosing. It stores both sizes and links them.

- **Child chunk:** small (for example 100–200 tokens). It is embedded and searched.
- **Parent chunk:** large (for example 500–2,000 tokens, or a whole section). It is stored but usually not embedded. Each child keeps a `parent_id` in its [metadata](09-metadata.md).

At query time you search children, collect their parent IDs, remove duplicates, and send the parents to the LLM. In LangChain this is called the **ParentDocumentRetriever**. LlamaIndex has a related idea called auto-merging retrieval.

Words to know:
- **Child chunk:** a small chunk that is embedded and searched.
- **Parent chunk:** the larger chunk a child was cut from, sent to the LLM.
- **parent_id:** the label on each child that says which parent it belongs to.
- **Deduplicate:** keep only one copy when the same item appears several times.
- **Key-value store:** simple storage where you look an item up by its key (here, `parent_id`).
- **Retriever:** the component that takes a question and returns matching chunks.

## How it works
1. Split each document into parent chunks (by section or with a large [recursive](05-recursive-chunking.md) split).
2. Split each parent into child chunks.
3. Give each child a `parent_id`. Embed children and store them in the vector database.
4. Store parents in a simple key-value store (`parent_id → text`).
5. At query time: search children (top-k), map them to parent IDs, deduplicate, fetch parents.
6. Send parents (not children) to the LLM.

```mermaid
flowchart LR
  P1[Parent p0: full section] --> C1[child p0-c0]
  P1 --> C2[child p0-c1]
  P2[Parent p1] --> C3[child p1-c0]
  Q[Question] -->|vector search| C2
  C2 -->|parent_id = p0| P1
  P1 -->|send to LLM| L[LLM]
```

Worked example: 3 children are retrieved with parent IDs `p4, p4, p9`. After deduplication you send 2 parents, `p4` and `p9`. If parents are 800 tokens each, the prompt carries about 1,600 tokens of context.

### Worked example
The repo demo splits the sample text into three parents (`p0`, `p1`, `p2`), each cut into two children. Real output:
```text
p0-c0    -> parent p0: 'Docker images are built in layers. '
p0-c1    -> parent p0: 'Each instruction in a Dockerfile adds a layer.\n'
p1-c0    -> parent p1: 'Multi-stage builds use several FROM lines. '
p1-c1    -> parent p1: 'Only the final stage ends up in the shipped image, so the image stays small.\n'
p2-c0    -> parent p2: 'Refresh tokens should live in an httpOnly cookie. '
p2-c1    -> parent p2: 'Rotate refresh tokens on every use.'
```
Now ask "How do I keep the image small?". Trace it (we reason through the search by hand here, it is not demo output):
1. Search compares the question with the six children only.
2. Words like "image" and "small" point to `p1-c1`, "...the image stays small", as the best match.
3. Its `parent_id` is `p1`, so we fetch parent `p1`, the whole paragraph, including the sentence about multi-stage builds.
4. If a second child from `p1` also matched, we would still send `p1` once, thanks to deduplication.

The small child made the match precise. The parent gives the LLM the context around it.

## Code example
From [`code/rag_chunkers.py`](../code/rag_chunkers.py), building on `recursive_chunks`:

```python
def parent_child_chunks(text, parent_size=400, child_size=100):
    parents, children = {}, []
    for p_idx, parent in enumerate(recursive_chunks(text, parent_size)):
        pid = f"p{p_idx}"
        parents[pid] = parent
        for c_idx, child in enumerate(recursive_chunks(parent, child_size)):
            children.append({"id": f"{pid}-c{c_idx}", "parent_id": pid, "text": child})
    return parents, children

def retrieve_parents(question, children, parents, search, k=3):
    hits = search(question, [c["text"] for c in children], k)   # indexes of best children
    seen, out = set(), []
    for i in hits:
        pid = children[i]["parent_id"]
        if pid not in seen:
            seen.add(pid); out.append(parents[pid])
    return out
```

`search` can be any retriever, for example BM25 from [`code/rag_bm25.py`](../code/rag_bm25.py).

**In real projects** (LangChain, with any vector store and embeddings you already use):

```python
from langchain_classic.retrievers import ParentDocumentRetriever   # LangChain 1.x; older: langchain.retrievers
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,                      # holds child embeddings
    docstore=InMemoryStore(),                     # holds parents
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
)
retriever.add_documents(docs)
results = retriever.invoke("Where should refresh tokens be stored?")
```

Import paths move between LangChain versions; check the docs for your version.

## When to use it / when not to
- Good when answers need surrounding context: manuals, policies, legal or medical text, code docs.
- Good when questions are short and specific but the answer spans a section.
- Not needed when items are already short (FAQs, product records).
- Watch the prompt budget: k children can expand into k large parents.

## Common mistakes
- Forgetting to deduplicate parents, so the same section appears three times in the prompt.
- Parents so large that only one or two fit in the context window.
- Losing the link (no `parent_id`) when re-ingesting or updating documents.
- Evaluating retrieval on children while the LLM sees parents. Measure what the model actually receives.

## Recap
If you remember only three things:
- Search small child chunks for precision, then send the larger parent chunk to the LLM for context.
- Each child stores a `parent_id`, and only children are usually embedded.
- Deduplicate parents and cap their size, because several children can expand into very large prompts.

## Interview questions
**1. What problem does parent-child chunking solve?**
<details><summary>Answer</summary>
The trade-off between precise retrieval (small chunks) and enough context for generation (large chunks). It searches small children and returns their large parents.
</details>

**2. Which chunks are embedded?**
<details><summary>Answer</summary>
Usually only the children. Parents are stored by ID in a document store and fetched after the search.
</details>

**3. What is a risk of this approach?**
<details><summary>Answer</summary>
Prompt size can grow quickly, and duplicate or irrelevant parent text can add noise. You need deduplication and a cap on the number or size of parents.
</details>

## Go deeper
- LangChain retrievers (ParentDocumentRetriever) — https://python.langchain.com/
- LlamaIndex (auto-merging retrieval) — https://docs.llamaindex.ai/

**Prev:** [Semantic Chunking](06-semantic-chunking.md) · **Next:** [Sliding Window Chunking](08-sliding-window-chunking.md) · **Back to phase:** [Phase 1](../README.md)
