"""Animation content for phases 1 and 2 (RAG foundations, advanced RAG).

Format of a step in P()/L(): "Label|sub label|caption line 1|caption line 2".
"""
from anim_core import P, L

# ---------------- Phase 1: RAG foundations ----------------
P("rag-flow", "RAG, step by step", "Watch one question travel through the whole pipeline.", """
Documents|your files|1. Start with your own documents|PDFs, wiki pages, docs. The model has never seen these.
Chunk|small pieces|2. Cut them into small chunks|Small pieces are easier to match to a question.
Embed|text to numbers|3. Turn each chunk into a vector|Similar meaning gives similar numbers.
Store|vector database|4. Save the vectors in a vector database|Done once, ahead of time.
Retrieve|top matches|5. Embed the question and find the closest chunks|Cosine similarity picks the best 3 to 5.
Generate|answer + cites|6. Give the chunks to the LLM with the question|It answers from them and cites the source.
""", topic="01-what-is-rag")

P("02-embeddings", "Embeddings", "Text becomes a list of numbers that carries its meaning.", """
Text|a sentence|1. Start with a piece of text|For example: How do I reset my password?
Model|embedding model|2. An embedding model reads the text|It was trained so similar meanings land close together.
Vector|list of numbers|3. Out comes a vector, a list of numbers|Often 384 to 3072 numbers, like [0.12, -0.80, 0.33, ...].
Space|meaning map|4. Each vector is a point in a meaning space|Reset password and forgot login sit near each other.
Compare|near = alike|5. Close points mean similar meaning|Search finds the points nearest to your question.
""")

P("03-tokenization", "Tokenization", "Text is cut into tokens, and tokens become ID numbers.", """
Text|raw string|1. Start with raw text|For example: Unbelievable results
Split|into pieces|2. A tokenizer splits it into tokens|Words or word pieces. The exact split depends on the model.
IDs|token to number|3. Each token gets an ID number|The tokenizer has a fixed vocabulary table.
Model|reads IDs|4. The model reads only the ID numbers|It never sees letters directly.
Count|tokens = cost|5. Tokens decide cost and limits|You pay per token, and the context window counts tokens.
""")

P("chunking", "Chunking with overlap", "A long text is cut into overlapping pieces so no idea is split in half.", topic="04-chunking", custom=True)

P("05-recursive-chunking", "Recursive chunking", "Split on the biggest natural break first, then smaller ones only if needed.", """
Document|long text|1. Start with a long document|It has paragraphs, lines and sentences inside.
Paragraphs|try first|2. Split on paragraph breaks first|Keep each paragraph whole if it fits the size limit.
Too big?|check size|3. A piece is still too big?|Split only that piece on the next break: lines.
Sentences|then words|4. Keep going down: sentences, then words|Only as far as needed, never further.
Chunks|natural pieces|5. Result: chunks that follow natural breaks|Ideas stay together more often than with fixed cuts.
""")

P("06-semantic-chunking", "Semantic chunking", "Start a new chunk where the meaning of the text shifts.", """
Sentences|split text|1. Split the text into sentences|Then embed every sentence.
Embed|a vector each|2. Each sentence becomes a vector|So we can measure how close two sentences are.
Compare|neighbours|3. Compare each sentence with the next one|Use cosine similarity between neighbours.
Find drops|topic shifts|4. A sharp drop in similarity means a new topic|That is where a new chunk should start.
Chunks|by meaning|5. Cut at the drops|Each chunk is about one topic.
""")

P("07-parent-child-chunking", "Parent-child chunking", "Search small pieces, answer with the big piece around them.", """
Parents|big sections|1. Split documents into big parents|For example, whole sections.
Children|small pieces|2. Split each parent into small children|Small pieces match questions precisely.
Search|children only|3. Search over the children only|Small vectors give sharp matches.
Fetch|their parent|4. Look up the parent of each matching child|The metadata stores the parent ID.
Answer|rich context|5. Send the parents to the LLM|It gets the full surrounding context.
""")

P("08-sliding-window-chunking", "Sliding window chunking", "Fixed windows that move by less than their size, so borders overlap.", """
Settings|size and step|1. Choose a window size and a step|For example: window 200 tokens, step 150 tokens.
Window 1|tokens 1-200|2. Window 1 covers the first 200 tokens|Then the window moves by the step, not by its full size.
Window 2|tokens 151-350|3. Window 2 starts at token 151|Tokens 151 to 200 are in both windows.
Window 3|tokens 301-500|4. Window 3 starts at token 301|The overlap keeps border sentences intact.
Result|overlapping|5. Every border sits inside a full window|The price: more chunks to store and search.
""")

P("09-metadata", "Metadata", "Extra facts stored next to each chunk make search, citing and debugging easier.", """
Chunk|text piece|1. Each chunk has text and a vector|That alone does not say where it came from.
Attach|metadata|2. Store facts next to it|Source file, page, date, author, access level.
Filter|before search|3. Filter by metadata first|For example: year >= 2025 and team = security.
Cite|show sources|4. Show where each answer came from|The source and page come from the metadata.
Debug|trace errors|5. Debug with it|You can see which file a wrong answer came from.
""")

P("10-vector-databases", "Vector databases", "Store embeddings and find the nearest ones fast, even among millions.", """
Chunks|text + metadata|1. Chunks go in with text and metadata|One record per chunk.
Embed|a vector each|2. Each chunk is embedded|Using one embedding model for everything.
Index|build the index|3. The database builds an index like HNSW or IVF|So it need not compare every vector.
Query|embed question|4. At question time, embed the question the same way|Same model, same meaning space.
Top-k|nearest chunks|5. Return the k nearest chunks, fast|Even among millions of vectors.
""")

P("cosine-similarity", "Cosine similarity", "The angle between two embedding vectors tells you how similar their meaning is.", topic="11-cosine-similarity", custom=True)

P("12-euclidean-distance", "Euclidean distance", "The straight-line distance between two points. A worked example.", """
Points|A and B|1. Two points: A = (1, 2) and B = (4, 6)|Think of each as a tiny 2-number embedding.
Differences|3 and 4|2. Subtract each coordinate: 4-1 = 3 and 6-2 = 4|One difference per dimension.
Squares|9 and 16|3. Square each difference: 9 and 16|Squaring removes minus signs.
Add|sum = 25|4. Add them: 9 + 16 = 25|Now take the square root.
Root|distance = 5|5. Square root of 25 is 5|Smaller means more similar. 0 means identical.
""")

P("13-dot-product", "Dot product", "Multiply matching numbers, then add. A worked example.", """
Vectors|a and b|1. a = (1, 2, 3) and b = (4, 5, 6)|Same length, so we can pair the numbers up.
Multiply|matching pairs|2. Multiply each pair: 1x4, 2x5, 3x6 = 4, 10, 18|One product per position.
Add|total|3. Add the products: 4 + 10 + 18 = 32|32 is the dot product.
Read it|bigger = aligned|4. A bigger number means more aligned, and longer, vectors|Direction and length both count.
Cosine|ignores length|5. Cosine is the dot product divided by both lengths|So it compares direction only.
""")

P("14-hnsw", "HNSW index", "Hop from point to closer point, from a sparse top layer down to all the points.", """
Top layer|few points|1. Start at the top layer, which has few points|Long jumps cover big distances quickly.
Hop|get closer|2. Hop to the neighbour closest to the query|Keep going while you are getting closer.
Go down|next layer|3. When stuck, drop to the next layer|More points there, and shorter jumps.
Bottom|all points|4. At the bottom layer, search around carefully|Every vector lives on this layer.
Result|approx top-k|5. Return the best k found|Very fast, and almost always right (it is approximate).
""")

P("15-ivf-index", "IVF index", "Group vectors into clusters, then search only the nearest clusters.", """
Cluster|group vectors|1. Before searching, group vectors into clusters|Each cluster has a centre point.
Query|arrives|2. A query comes in|Embed it like everything else.
Nearest|few centres|3. Find the few nearest cluster centres|A setting called nprobe says how many.
Scan|only those|4. Compare against vectors in those clusters only|Skip all the other clusters.
Top-k|results|5. Return the best matches|Faster, but may miss a match in a skipped cluster.
""")

P("hybrid-search-flow", "Hybrid search", "Keyword search and meaning search, merged.", """
Query|user question|1. One question comes in|It is sent down two paths at once.
BM25|exact words|2. Keyword path: BM25 finds exact word matches|Great for names, codes and rare terms.
Dense|meaning|3. Meaning path: embeddings find similar ideas|Great for paraphrases.
Merge|RRF|4. Reciprocal Rank Fusion merges both rankings|A chunk ranked high in both rises to the top.
Top-k|to the LLM|5. The top chunks go to the model|You get the strengths of both.
""", topic="16-hybrid-search")

P("17-bm25", "BM25", "A classic keyword score: frequent in this document, rare overall.", """
Query|split to words|1. The query is split into words|Each word is scored separately.
Count|term frequency|2. Count how often each word appears in a document|More is better, with diminishing returns.
Rarity|IDF|3. Rare words count more than common ones|The is worth little. kubelet is worth a lot.
Length|normalise|4. Long documents are adjusted down|So length alone does not win.
Score|rank them|5. Add up the word scores and rank the documents|Highest score first.
""")

P("18-sparse-retrieval", "Sparse retrieval", "Very long vectors, mostly zeros, one slot per vocabulary word.", """
Vocabulary|a slot per word|1. Give every vocabulary word its own slot|Vectors can be 30,000 slots long.
Fill|mostly zeros|2. A text fills only the slots of its own words|Almost every slot stays zero.
Match|same words|3. A query matches documents that share its words|Fast with an inverted index.
Weigh|TF-IDF, BM25|4. Weights show how important each word is|Rare, frequent-here words weigh most.
Limit|no synonyms|5. Car and automobile do not match|That gap is what dense retrieval fills.
""")

P("19-dense-retrieval", "Dense retrieval", "Short meaning-based vectors find ideas, even with different words.", """
Query|a question|1. Embed the question into a short vector|Hundreds of numbers, all non-zero.
Documents|embedded earlier|2. Documents were embedded the same way beforehand|Stored in a vector index.
Compare|similarity|3. Compare vectors with cosine or dot product|Closer means more similar in meaning.
Rank|closest first|4. The closest vectors win|They match meaning, even without shared words.
Limit|rare terms|5. Weak on exact codes and names|Pair it with keyword search: hybrid search.
""")

P("20-metadata-filtering", "Metadata filtering", "Apply rules like team and year first, then search by meaning.", """
Question|has rules inside|1. Security policy from 2025|Part of it is meaning, part is a rule.
Rules|team, year|2. Turn the rules into filters|team = security and year >= 2025.
Filter|keep matches|3. Keep only chunks whose metadata matches|Thousands of chunks become hundreds.
Search|by meaning|4. Run the vector search on what is left|Fewer, better candidates.
Results|relevant|5. Results are on-topic, current and allowed|Also a good place for access control.
""")

# ---------------- Phase 2: advanced RAG ----------------
P("02-reranking", "Reranking", "A slower, more careful second pass that fixes the order.", """
Search|fast, top 50|1. A fast search returns the top 50 chunks|Good at finding candidates, rough at ordering them.
Pairs|query + chunk|2. Pair the question with each chunk|One pair per candidate.
Rerank|careful scores|3. A reranker scores each pair with more care|Slower, so it only sees the few candidates.
Sort|new order|4. Re-sort by the new score|The truly best chunk moves to the top.
Top 5|to the LLM|5. Keep the best 5 for the prompt|Less noise, better answers.
""", topic="01-reranking")

P("02-cross-encoder", "Cross-encoder", "Reads the question and the document together, and gives one score.", """
Pair|question + doc|1. Join the question and a document into one input|They are read together, not separately.
Read|one transformer|2. One transformer reads both at once|Words in each can attend to words in the other.
Score|one number|3. It outputs a single relevance score|Higher means a better match.
Repeat|per candidate|4. Repeat for every candidate|Cost grows with the number of candidates.
Tradeoff|accurate, slow|5. More accurate than bi-encoders, but slower|Use it to rerank, not to search everything.
""", topic="02-cross-encoder")

P("03-bi-encoder", "Bi-encoder", "Question and documents are encoded separately, then compared as vectors.", """
Documents|encoded once|1. Encode every document alone, ahead of time|Store the vectors in an index.
Query|encoded alone|2. Encode the question alone at query time|One quick model call.
Compare|vectors|3. Compare vectors with cosine similarity|No model call per document.
Fast|index|4. Works with a vector index over millions|This is why it is the first-stage search.
Tradeoff|less precise|5. Less precise, since the two never meet inside the model|Fix it with a cross-encoder reranker.
""")

P("04-context-compression", "Context compression", "Trim retrieved chunks to the parts that actually help.", """
Retrieve|chunks|1. Retrieve chunks as usual|They often contain extra, unrelated sentences.
Read|against question|2. Check each chunk against the question|A small model or filter does this.
Trim|keep relevant|3. Drop sentences that do not help|Keep only the useful parts.
Shorter|prompt|4. Build a shorter prompt|Fewer tokens go to the LLM.
Benefit|cost + focus|5. Lower cost and less noise|The model focuses on what matters.
""")

P("05-query-expansion", "Query expansion", "Add related words so the search finds documents that phrase it differently.", """
Question|short, few words|1. A short question: cheap car insurance|Documents might say affordable vehicle cover.
Expand|add terms|2. Add synonyms and related phrases|car also gets automobile and vehicle.
Search|wider net|3. Search with the expanded query|More ways to match.
Merge|results|4. Merge and de-duplicate the results|Keep the best ranking.
Result|more recall|5. Finds documents that use other words|The right chunk is less likely to be missed.
""")

P("06-multi-query-retrieval", "Multi-query retrieval", "One question becomes several, and the results are merged.", """
Question|one|1. One user question|One wording can miss relevant chunks.
LLM|rewrites|2. An LLM writes 3 to 5 different versions|Different angles on the same need.
Search|each version|3. Run a search for every version|Each finds slightly different chunks.
Merge|de-duplicate|4. Merge the results and remove duplicates|Chunks found by many versions rank higher.
Answer|better recall|5. The LLM answers from the merged chunks|Fewer misses than a single query.
""")

P("07-hyde", "HyDE", "Search with a made-up answer, because it looks like real documents.", """
Question|short|1. Start with a short question|Questions rarely look like the documents answering them.
LLM|fake answer|2. The LLM writes a made-up answer|It may contain mistakes. That is okay.
Embed|the fake one|3. Embed the fake answer, not the question|It sits closer to real answers in meaning space.
Search|near the fake|4. Find the real chunks nearest to it|Real documents, found more reliably.
Answer|real sources|5. Answer from the real chunks and drop the fake|The fake answer is only a search aid.
""")

P("08-step-back-prompting", "Step-back prompting", "Also ask the broader question behind the specific one.", """
Question|very specific|1. A specific question|For example: why did my pod die at 512Mi?
Step back|go general|2. Ask the broader question behind it|For example: how does Kubernetes limit memory?
Retrieve|for both|3. Retrieve chunks for both questions|General background plus the specific details.
Combine|context|4. Put both sets of chunks in the prompt|The model sees the principle and the case.
Answer|grounded|5. Answer the specific question using both|Better reasoning on hard questions.
""")

P("09-self-query-retriever", "Self-query retriever", "An LLM splits a question into search text and metadata filters.", """
Question|natural text|1. Security docs from 2025 about backups|It mixes meaning with rules.
LLM|splits it|2. An LLM splits it into two parts|Search text and structured filters.
Parts|text + filters|3. Text: backups. Filters: team = security, year = 2025|Both come from the same sentence.
Run|both|4. Run the filtered vector search|Rules first, meaning second.
Results|precise|5. Precise results, without the user writing filters|Check that the LLM wrote valid filters.
""")

P("10-multi-vector-retriever", "Multi-vector retriever", "Many vectors per document, but the full document comes back.", """
Document|full text|1. Start with a full document|One big text.
Make|many vectors|2. Make several vectors for it|A summary, questions it answers, small pieces.
Index|vectors only|3. Index those vectors|Each points back to the document ID.
Match|any vector|4. A question can match any one of them|More ways to find the right document.
Return|full document|5. Return the original document|Not the small text that matched.
""")

P("11-parent-document-retriever", "Parent document retriever", "Search small chunks, return the larger section they came from.", """
Small chunks|searched|1. Search over small chunks for accuracy|They match questions sharply.
Match|best chunks|2. Find the best matching small chunks|Each knows its parent ID.
Parent|look it up|3. Look up each chunk's parent section|From a simple document store.
Dedupe|same parent|4. If several chunks share a parent, return it once|No repeats in the prompt.
Return|parent|5. The LLM gets the full section|Complete context, not a fragment.
""")

P("12-ensemble-retrieval", "Ensemble retrieval", "Several retrievers run together and their rankings are fused.", """
Question|one|1. One question|Sent to several retrievers at once.
Keyword|BM25|2. A keyword retriever returns a ranked list|Strong on exact terms.
Vector|dense|3. A vector retriever returns another ranked list|Strong on meaning.
Fuse|RRF|4. Reciprocal Rank Fusion merges the lists|Each list adds 1/(60 + rank) to a document score.
Top-k|final list|5. One final ranked list|Documents ranked high in several lists win.
""")

P("13-knowledge-graph-rag", "Knowledge graph RAG", "Store facts as linked entities and retrieve by following the links.", """
Documents|text|1. Start with documents|Full of facts about people, places and things.
Extract|facts and links|2. Extract entities and relations|Asha works at Acme. Acme owns Zeta.
Graph|nodes + edges|3. Store them as a graph|Entities are nodes. Relations are edges.
Traverse|follow links|4. Find the entity in the question, then follow its links|Hop along edges to related facts.
Answer|with facts|5. Give the connected facts to the LLM|It can answer multi-step relationship questions.
""")

P("14-graphrag", "GraphRAG", "Summaries of communities answer questions about the whole collection.", """
Documents|text|1. Start with all your documents|Too many to read in one prompt.
Graph|entities|2. Build an entity graph|Who and what appears, and how they connect.
Communities|group them|3. Group the graph into communities of linked entities|Each community is one theme.
Summaries|pre-written|4. An LLM writes a summary for each community|Done ahead of time.
Answer|big picture|5. Big-picture questions use the summaries|For example: what are the main themes?
""")

P("15-multi-modal-rag", "Multi-modal RAG", "Retrieve images, tables and charts, not only text.", """
Inputs|many formats|1. Your data has more than text|Charts, scans, screenshots, tables.
Embed|shared space|2. Embed each type|Often into one shared meaning space.
Search|any type|3. A text question can find an image or a table|Because they share the space.
Fetch|the item|4. Fetch the matching image or table|With its source details.
Answer|multimodal LLM|5. A model that reads images answers from it|It can describe the chart it was given.
""")

P("16-sql-rag", "SQL RAG (text-to-SQL)", "The model writes a query, your database answers.", """
Question|plain words|1. How many orders last month?|The answer lives in a table, not in documents.
LLM|writes SQL|2. The LLM sees the table schema and writes a query|SELECT COUNT(*) FROM orders WHERE ...
Check|is it safe?|3. Validate the query before running it|Read-only, allowed tables, row limit.
Run|database|4. Run it on the database|Result rows come back.
Answer|from rows|5. The LLM turns the rows into a sentence|The numbers come from the database, not from guessing.
""")

P("17-api-rag", "API RAG", "Fetch live data at question time and hand it to the model.", """
Question|a live fact|1. Where is order 42?|Documents cannot know. It changes by the minute.
Choose|pick the API|2. The model picks an API and arguments|For example: orders API with id 42.
Call|live API|3. Your code calls the API|Add timeouts and allowed hosts.
Data|fresh JSON|4. Fresh data comes back|Status: shipped, arrives Friday.
Answer|from response|5. The model answers from the response|Always current.
""")

P("18-incremental-indexing", "Incremental indexing", "Re-embed only what changed, delete what disappeared.", """
Scan|find changes|1. Hash every file and compare with the last run|A hash changes when the content changes.
Changed|re-embed|2. New or changed files are re-embedded|Replace their old chunks.
Deleted|remove|3. Files that disappeared lose their chunks|So answers never cite deleted files.
Unchanged|skip|4. Unchanged files are skipped|This is where the time is saved.
Done|index is fresh|5. The index is up to date in seconds|Instead of rebuilding everything.
""")

P("19-streaming-retrieval", "Streaming retrieval", "Send results as soon as each piece is ready.", """
Question|arrives|1. A question arrives|Start working at once.
Retrieve|first hits|2. Start as soon as the first chunks arrive|No need to wait for every result.
Generate|token by token|3. The model writes the answer token by token|Each token is sent as soon as it is ready.
Stream|to the user|4. The user sees words appear immediately|It feels much faster.
Update|index stays live|5. New documents are indexed continuously|So fresh data is searchable quickly.
""")

L("eval-loop", "The evaluation loop", "Measure, change one thing, measure again.", """
Test set|1. Keep a fixed set of questions|With known good answers.
Run|2. Run your system on all of them|Log every answer and every source.
Score|3. Score: accuracy, faithfulness, hit@k|Numbers, not feelings.
Improve|4. Change ONE thing, then re-run|Keep it only if the score goes up.
""", topic=("02-advanced-rag/topics/20-rag-evaluation", "04-agentic-ai/topics/15-agent-evaluation"))

P("21-llm-as-a-judge", "LLM-as-a-judge", "A model grades answers, but only with a good rubric and checks.", """
Answers|to grade|1. You have answers with no exact match to compare|Opinions, summaries, explanations.
Rubric|clear criteria|2. Write a rubric with clear criteria|For example: correct, grounded, concise, each 1 to 5.
Judge|LLM scores|3. The judge model scores each answer|Ask for a short reason too.
Check|vs humans|4. Compare with human labels on a sample|Measure agreement, for example Cohen's kappa.
Fix bias|swap order|5. Fight bias: swap answer order, hide who wrote it|Judges favour long answers and their own style.
""")
