# Bounded Graph Runner and Trace Extractor

A small Python utility for running a LangGraph with a caller-provided recursion
limit and returning a compact, ordered execution report. It supports graphs
that answer directly or make tool calls, and converts expected input and graph
failures into declared result statuses.

This repository exposes reusable Python functions; `main.py` is not currently
a command-line interface. The supplied research graph uses a Groq-compatible
client and a deterministic local evidence-search tool.

## Requirements

- Python 3.10 or newer
- A Groq API key only when running the supplied research graph with the real
  client

Install the pinned dependencies in a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the Supplied Graph

Set the API key in the current PowerShell session, then build the graph and
pass it to `run_graph`:

```powershell
$env:GROQ_API_KEY = "your-api-key"
python -c "from main import run_graph; from utils.research_graph import build_research_graph; result = run_graph(build_research_graph(), 'Why might Wi-Fi be intermittent?', 10); print(result)"
```

The graph factory is `utils.research_graph.build_research_graph()`. It accepts
an optional LLM client for deterministic or custom integrations. The local
`search_evidence` tool searches the records in `utils/local_tools.py`; the
Groq client is configured in `utils/llm_client.py` and reads `GROQ_API_KEY`.

The runner can also be used from application code:

```python
from main import run_graph
from utils.research_graph import build_research_graph

graph = build_research_graph()
result = run_graph(graph, "How can I reduce battery drain?", recursion_limit=10)

print(result.status.value)
print(result.final_answer)
print(f"Recorded {result.step_count} graph updates")
for step in result.trace:
    print(step.node_name, step.message_type, step.tool_names, step.output_snippet)
```

The recursion limit must be a positive integer; booleans are rejected even
though Python treats them as integers. User text must be a non-empty string and
is stripped before it is sent to the graph.

## Result Contract

`run_graph(graph, user_text, recursion_limit)` returns a `RunResult` with:

- `status`: one of the statuses below
- `final_answer`: the latest non-empty assistant message without tool calls,
  or `None` when no answer is available
- `trace`: an immutable tuple of ordered `TraceStep` values
- `step_count`: the number of records in `trace`
- `message`: optional status detail; raw graph exception text is not returned

Statuses:

| Status | Meaning |
| --- | --- |
| `COMPLETED` | The graph produced a final assistant answer. |
| `INVALID_INPUT` | User text or recursion limit failed validation; the graph was not called. |
| `EMPTY_OUTPUT` | The graph finished without a final assistant answer. |
| `LIMIT_REACHED` | LangGraph raised `GraphRecursionError`. |
| `GRAPH_ERROR` | Another exception occurred while streaming the graph. |

Each trace step records the node name, latest message type, tool names, tool
call IDs, and a bounded output snippet. Updates are recorded in the order
received from LangGraph, including events emitted before a graph failure.
Normalization and snippet bounds are implemented by
`utils.trace_helpers.trace_step_from_update`.

## Run Tests

The tests use fake graph streams and do not call Groq or require an API key:

```powershell
python -m pytest -q
```

Coverage includes input validation, trace ordering and normalization, each run
status, exact stream configuration, partial-trace preservation, and ensuring
graph exceptions do not leak their raw text.