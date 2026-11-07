"""Animation content for phases 5, 6 and 7 (frameworks, MCP, production AI)."""
from anim_core import P, L, B

# ---------------- Phase 5: frameworks ----------------
P("05-langchain-flow", "LangChain: a chain of building blocks", "Pieces joined with the pipe operator.", """
Prompt|template|1. A prompt template fills in your variables|Question and context go into the text.
Model|chat model|2. The model receives the filled prompt|Swap models without changing the rest.
Parser|clean output|3. A parser turns the reply into text or JSON|So your code can use it.
Pipe|join with a pipe|4. Join them with the pipe symbol: prompt, model, parser|The output of each step feeds the next.
Run|invoke()|5. Call invoke() once and the chain runs|Add a retriever or tools the same way.
""", topic="01-langchain")

P("02-langgraph", "LangGraph: a graph of steps", "Nodes do the work, edges decide what comes next, state is shared.", """
State|shared data|1. Define the shared state|Messages, notes, counters.
Nodes|do work|2. Each node is a function that reads and updates state|For example: plan, search, write.
Edges|what is next|3. Edges connect the nodes|Some are conditional: if the check fails, go back.
Loop|cycles allowed|4. Cycles let an agent retry or keep working|With a step limit for safety.
Run|invoke|5. Run the graph from START to END|You can save state, pause and resume.
""")

P("03-llamaindex", "LlamaIndex", "Connect your documents to an LLM: load, split, index, retrieve, answer.", """
Load|readers|1. Load your files with readers|PDFs, web pages, databases.
Split|nodes|2. Split them into nodes (chunks)|With metadata attached.
Index|vector index|3. Build an index of embeddings|Stored in memory or a vector database.
Retrieve|top nodes|4. A retriever finds the best nodes for a question|Optionally rerank them.
Answer|query engine|5. The query engine asks the LLM to answer|Sources included.
""")

L("04-openai-agents-sdk", "OpenAI Agents SDK: the Runner loop", "Agent plus Runner: the Runner repeats until there is a final output.", """
Agent|1. Define an agent: model, instructions, tools|This is its job description.
Runner|2. Runner calls the model with the conversation|The model chooses a tool, a handoff, or a final answer.
Tool|3. Tool calls are run and the results added|Guardrails can check inputs and outputs.
Output|4. Repeat until there is a final output|Or hand off to another agent.
""", topic="04-openai-agents-sdk")

P("05-crewai", "CrewAI", "A crew of role-playing agents working through tasks.", """
Agents|role + goal|1. Define agents with a role, a goal and a backstory|For example: researcher, writer.
Tasks|what to do|2. Define tasks and who does each|With the expected output.
Crew|team + process|3. Put them in a crew with a process|Sequential, or a manager decides.
Kickoff|start|4. kickoff() runs the tasks in order|Each result feeds the next task.
Result|final output|5. The last task's output is the result|Watch cost: many agents means many calls.
""")

P("06-autogen", "AutoGen", "Agents solve problems by messaging each other.", """
Agents|assistants|1. Create agents, each with its own instructions|For example: coder and reviewer.
Message|they talk|2. One sends a message, another replies|The conversation is the program.
Tools|code and APIs|3. Agents can run code and call tools|Ideally inside a sandbox.
Rules|when to stop|4. A termination rule ends the chat|A keyword, a message limit, or a human.
Result|chat history|5. The last messages hold the result|Always set a message limit.
""")

P("07-dspy", "DSPy", "Declare what a step does, and let an optimizer write the prompt.", """
Signature|inputs, outputs|1. Declare the task: question -> answer|No prompt writing yet.
Module|how it runs|2. Choose a module that runs it|DSPy builds the prompt for you.
Metric|how to score|3. Define a metric and a few examples|What counts as a good answer.
Optimise|compile|4. The optimizer tries prompts and examples|It keeps what scores best.
Program|better prompt|5. You get an optimised program|Re-optimise when you change the model.
""")

P("08-haystack", "Haystack", "Pipelines of components with connected inputs and outputs.", """
Components|the parts|1. Pick components|Each has named inputs and outputs.
Add|to a pipeline|2. Add them to a Pipeline|Give each one a name.
Connect|output to input|3. Connect them explicitly|retriever.documents to prompt.documents.
Run|pipeline.run()|4. Run it with your inputs|Data flows along the connections.
Inspect|each step|5. You can see each component's output|Easy to debug and swap parts.
""")

# ---------------- Phase 6: MCP ----------------
P("mcp-flow", "MCP: how an AI app uses a tool server", "One standard way to plug tools into any AI app.", """
Host app|Claude, IDE|1. The host app starts an MCP client|The host is the AI app you use.
Connect|handshake|2. Client and server agree on features|They exchange name, version and capabilities.
List|tools/list|3. The client asks: what can you do?|The server returns tools with descriptions and schemas.
Call|tools/call|4. The model picks a tool and the client calls it|Inputs follow the tool's JSON schema.
Result|content back|5. The server returns the result|The model reads it and continues.
""", color="pink", topic="01-model-context-protocol")

P("02-mcp-server", "MCP server", "A small program that offers tools, resources and prompts.", """
Start|stdio|1. The host launches your server as a process|Messages travel over stdin and stdout.
Declare|capabilities|2. It says what it supports|Tools, resources, prompts.
List|what is offered|3. It answers list requests|Names, descriptions, input schemas.
Handle|requests|4. It runs a tool or reads a resource when asked|Validate every input.
Reply|result or error|5. It returns a result or a clear error|Never print stray text to stdout.
""", color="pink")

P("03-mcp-client", "MCP client", "The part inside an AI app that talks to one server.", """
Spawn|start the server|1. Start the server process, or connect to it|One client per server.
Initialise|negotiate|2. Send initialize and agree on versions and features|Then confirm it is ready.
Discover|list tools|3. Ask for tools, resources and prompts|Pass the tools to the model.
Forward|model's choice|4. When the model picks a tool, send tools/call|The host may ask the user first.
Return|result|5. Give the result back to the model|It continues its reply.
""", color="pink")

P("04-mcp-tools", "MCP tools", "Actions the model can choose to call.", """
Define|name + schema|1. Give each tool a name, a description and an input schema|The description is how the model decides.
Offer|tools/list|2. The server lists the tools|The client passes them to the model.
Choose|the model|3. The model picks a tool and fills the arguments|Matching the schema.
Run|tools/call|4. The server validates and runs it|Narrow tools are safer than broad ones.
Return|content|5. It returns text or data to the model|With isError set if something failed.
""", color="pink")

P("05-mcp-resources", "MCP resources", "Read-only data identified by a URI.", """
Expose|URIs|1. The server lists resources, each with a URI|For example: notes://welcome.
Choose|the host app|2. The host decides what to load|Not the model by itself.
Read|resources/read|3. The client reads the URI|The server returns the content.
Context|in the prompt|4. The content is added to the model's context|As background material.
Update|optional|5. Servers can notify when a resource changes|So the host can reload it.
""", color="pink")

P("06-mcp-prompts", "MCP prompts", "Reusable message templates the user picks, like slash commands.", """
Define|template|1. The server defines a prompt with arguments|For example: summarise-note(note_id).
List|prompts/list|2. The client lists the available prompts|They appear as commands in the app.
Pick|the user|3. The user picks one and fills the arguments|The user chooses, not the model.
Get|prompts/get|4. The server returns the ready-made messages|With the arguments filled in.
Send|to the model|5. The host sends them to the model|A repeatable workflow in one click.
""", color="pink")

P("07-build-a-custom-mcp-server", "Build a custom MCP server", "A notes server, from idea to a connected host.", """
Design|narrow tools|1. Design a few narrow tools|add_note, search_notes, list_notes.
Validate|inputs|2. Validate every input|Types, length, allowed values.
Resource|notes as URIs|3. Expose notes as resources and add a prompt|Read-only and reusable.
Test|inspector|4. Test with the MCP Inspector|Before connecting any AI app.
Connect|host config|5. Add it to a host's config and try it|Check errors and logs.
""", color="pink")

# ---------------- Phase 7: production AI ----------------
P("01-fastapi-for-ai", "FastAPI for AI", "Wrap your AI function in a validated HTTP API.", """
Function|your AI code|1. Start with a Python function|For example: answer(question).
Model|request schema|2. Describe the request with a Pydantic model|Bad input is rejected automatically.
Route|POST /ask|3. Add a route that calls your function|FastAPI turns it into an endpoint.
Docs|auto page|4. Open /docs for an interactive API page|Generated for free.
Serve|uvicorn|5. Run it with uvicorn|Add timeouts and logging.
""", color="teal")

P("02-docker", "Docker", "Pack the app and everything it needs into an image.", """
Dockerfile|recipe|1. Write a Dockerfile: base image, install, copy, run|The recipe for the image.
Build|image|2. docker build makes an image|A frozen copy of your app and its tools.
Run|container|3. docker run starts a container from it|Isolated, and the same everywhere.
Small|multi-stage|4. Multi-stage builds keep the image small|Fewer files, smaller attack surface.
Ship|registry|5. Push the image to a registry|Servers pull and run it.
""", color="teal")

P("03-kubernetes", "Kubernetes", "Run many containers across many machines and keep them healthy.", """
Deploy|desired state|1. Declare what you want: 3 copies of this image|In a YAML file.
Schedule|on nodes|2. Kubernetes places pods on machines with room|It picks the nodes.
Heal|restart|3. If a pod dies, it starts a new one|Health checks detect problems.
Scale|more or fewer|4. Add copies when traffic grows|Autoscaling can do it by itself.
Expose|service|5. A Service gives one stable address|Traffic is shared across the pods.
""", color="teal")

P("04-redis", "Redis", "A very fast in-memory store for caching and rate limits.", """
Request|arrives|1. A question arrives|Same questions come again and again.
Check|cache|2. Look in Redis for a saved answer|Very fast, because it lives in memory.
Hit|return it|3. Found: return it in milliseconds|No model call, no cost.
Miss|ask the LLM|4. Not found: call the model, then save the answer|With an expiry time (TTL).
Limit|rate limiting|5. Redis counters also cap requests per user|Protects cost and the service.
""", color="teal")

P("05-celery", "Celery", "Run slow jobs in the background so the API answers fast.", """
API|receives job|1. A user uploads a document|Processing it takes minutes.
Queue|send task|2. The API puts a task on a queue and replies at once|Here is your job ID.
Worker|picks it up|3. A background worker takes the task|Embeds and indexes the document.
Status|poll or notify|4. The user checks the job ID for progress|Pending, running, done.
Retry|on failure|5. Failed tasks are retried with backoff|Make tasks safe to run twice.
""", color="teal")

P("06-rabbitmq", "RabbitMQ", "A broker that routes messages from producers to queues to consumers.", """
Producer|sends|1. A producer publishes a message|It does not know who will read it.
Exchange|routes|2. An exchange routes it using rules|By topic, key or fan-out.
Queue|holds|3. The message waits in a queue|Safely stored until someone is ready.
Consumer|processes|4. A consumer takes it and does the work|Then acknowledges it.
Ack|or redeliver|5. No ack: the message goes back for another try|So work is not lost.
""", color="teal")

L("07-monitoring", "Monitoring", "Watch the numbers and alert a human when they go wrong.", """
Measure|1. Measure: latency, errors, cost, tokens|Per request and per minute.
Chart|2. Put them on a dashboard|Look for trends and spikes.
Alert|3. Alert when a limit is crossed|For example: error rate above 2 percent.
Fix|4. A human investigates and fixes it|Then add a check so it is caught sooner.
""", color="teal", topic="07-monitoring")

P("08-langsmith", "LangSmith", "Record every step of an LLM run so you can debug and evaluate.", """
Trace|records a run|1. Turn on tracing for your app|Each run is recorded.
Steps|every call|2. See each step: prompt, tool, retrieval, output|With time and token counts.
Debug|find the bug|3. Open a failed run and find where it went wrong|The retrieval, the prompt or the tool.
Dataset|save examples|4. Save good and bad runs as a test dataset|Real cases, not guesses.
Evaluate|compare|5. Re-run tests after every change|See if the score went up.
""", color="teal")

P("09-mlflow", "MLflow", "Record experiments so the best run can be compared and reproduced.", """
Run|an experiment|1. Start a run for each experiment|Different settings each time.
Log|params + scores|2. Log parameters, metrics and files|Temperature, chunk size, accuracy.
Compare|in the UI|3. Compare runs side by side|Sort by the metric you care about.
Pick|the best|4. Choose the best run|Its settings are all saved.
Reproduce|later|5. Re-run it any time with the same settings|And register the winning model.
""", color="teal")

P("10-prompt-versioning", "Prompt versioning", "Treat prompts like code: version, compare, roll out slowly, roll back.", """
Change|new prompt|1. Edit a prompt and give it a new version|v7 becomes v8.
Test|on examples|2. Test v8 against v7 on the same set|Same questions, same scoring.
Roll out|5 percent|3. Roll out to a small share of traffic|Watch quality and cost.
Compare|real data|4. Compare results with the old version|Is v8 really better?
Roll back|if worse|5. If worse, switch back at once|Because every version is saved.
""", color="teal")

P("11-ai-security", "AI security", "Assume anything the model reads or writes can be hostile or wrong.", """
Threat|prompt injection|1. Documents or web pages may hide instructions|The model may follow them.
Limit|least privilege|2. Give the agent only the tools it needs|And read-only where possible.
Check|inputs|3. Validate everything the model asks for|Arguments, URLs, file paths.
Approve|risky steps|4. Ask a human before irreversible actions|Sending, deleting, paying.
Protect|secrets|5. Keep keys out of prompts and logs|Filter outputs before showing them.
""", color="teal")

P("12-guardrails", "Guardrails", "Checks before and after the model call.", """
Input|check|1. Check what the user sends|Block unsafe or off-topic requests.
Model|call|2. Call the model|With clear rules in the system prompt.
Output|check|3. Check what comes back|Format, safety, no secrets.
Fix|or reject|4. Clean it, retry, or reject it|Never show a bad output.
Log|record it|5. Log every block|To improve the rules.
""", color="teal")

P("13-observability", "Observability", "Record enough per request to explain why the system did what it did.", """
Request|with an ID|1. Give each request an ID|It follows the request everywhere.
Record|steps and cost|2. Record each step with time, tokens and cost|Plus any errors.
Join|logs + traces|3. Connect logs, metrics and traces by that ID|One story per request.
Ask|why?|4. When something is wrong, open that request|See exactly what happened.
Learn|improve|5. Fix the cause, then add a test|So it does not come back.
""", color="teal")

P("14-cost-optimization", "Cost optimisation", "Fewer tokens, cheaper models, fewer calls, same quality.", """
Measure|cost per request|1. Measure cost per request and per user|You cannot cut what you cannot see.
Trim|fewer tokens|2. Shorten prompts and retrieve fewer chunks|Tokens are the bill.
Cache|reuse answers|3. Cache repeated questions|A cache hit costs almost nothing.
Route|cheap first|4. Send easy questions to a cheaper model|Keep the strong model for hard ones.
Guard|quality bar|5. Re-run your tests after each change|Never drop below your quality bar.
""", color="teal")

P("15-deploy-a-production-ai-application", "Deploy a production AI app", "From a working demo to a reliable service.", """
Package|Docker image|1. Put the API in a Docker image|The same everywhere.
Wire|cache, queue, DB|2. Connect Redis, a queue and a vector database|Each runs as its own service.
Secure|keys + limits|3. Secure it: secrets, auth, rate limits|And guardrails around the model.
Watch|monitor|4. Add monitoring, tracing and alerts|Before real users arrive.
Release|slowly|5. Release gradually, with a way to roll back|Then keep measuring and improving.
""", color="teal")
