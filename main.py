"""Bounded Graph Runner and Trace Extractor.

Scenario
--------
A command-line research assistant needs a compact execution report for every
graph run. A supplied LangGraph may answer directly or call a local evidence
tool before producing its final answer.

Your task
---------
Complete only these three TODO functions:

1. build_run_input
2. extract_trace
3. run_graph

The result types, trace-normalization helpers, compiled research graph, Groq
client, evidence tool, and pytest fixtures are supplied. The application can
use GROQ_API_KEY during normal execution. Pytest injects a mocked LLM client,
so evaluation makes no real API call and requires no API key or network access.

Required behavior
-----------------
build_run_input:
- Reject blank or non-string user_text with ValueError.
- Reject recursion_limit values that are booleans, non-integers, or below 1.
- Strip surrounding whitespace from valid user text.
- Return GraphRunInput containing one HumanMessage and a config dictionary
  with the exact key "recursion_limit".

extract_trace:
- Accept LangGraph update events in their original order.
- Create one TraceStep for every node update, preserving event and node order.
- Use the supplied trace_step_from_update helper so node names, message types,
  tool names, tool-call IDs, and bounded output snippets are normalized.

run_graph:
- Build the graph input and config with build_run_input.
- Invalid input returns RunStatus.INVALID_INPUT without invoking the graph.
- Stream the supplied graph with stream_mode="updates".
- On success, return the final assistant answer, ordered trace, and a step_count
  equal to the number of TraceStep records.
- A completed run without a final assistant answer returns EMPTY_OUTPUT.
- GraphRecursionError returns LIMIT_REACHED instead of escaping.
- Any other graph exception returns GRAPH_ERROR without exposing raw exception
  text. Preserve trace events produced before either failure.

The evaluator checks only observable behavior. It does not require a particular
private implementation style.

Not required
------------
You do not need to build graph nodes, implement tools, write prompts, call a
real model during evaluation, configure LangSmith, build a CLI, or deploy an
application.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from utils.models import GraphRunInput, RunResult, RunStatus, TraceStep
from utils.trace_helpers import final_answer_from_events, trace_step_from_update


def build_run_input(user_text: str, recursion_limit: int) -> GraphRunInput:
    """Validate user input and build the graph input plus execution config."""
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string")
    if isinstance(recursion_limit, bool) or not isinstance(recursion_limit, int):
        raise ValueError("recursion_limit must be a positive integer")
    if recursion_limit < 1:
        raise ValueError("recursion_limit must be a positive integer")

    return GraphRunInput(
        graph_input={"messages": [HumanMessage(content=user_text.strip())]},
        config={"recursion_limit": recursion_limit},
    )


def extract_trace(
    events: Iterable[Mapping[str, Any]],
) -> list[TraceStep]:
    """Convert ordered LangGraph update events into normalized trace steps."""
    trace: list[TraceStep] = []
    for event in events:
        for node_name, update in event.items():
            trace.append(trace_step_from_update(node_name, update))
    return trace


def run_graph(graph: Any, user_text: str, recursion_limit: int) -> RunResult:
    """Run a graph safely and return its final answer plus bounded trace."""
    try:
        run_input = build_run_input(user_text, recursion_limit)
    except ValueError:
        return RunResult(
            status=RunStatus.INVALID_INPUT,
            final_answer=None,
            trace=(),
            step_count=0,
        )

    events: list[Mapping[str, Any]] = []
    try:
        for event in graph.stream(
            run_input.graph_input,
            config=run_input.config,
            stream_mode="updates",
        ):
            events.append(event)
    except GraphRecursionError:
        trace = tuple(extract_trace(events))
        return RunResult(
            status=RunStatus.LIMIT_REACHED,
            final_answer=None,
            trace=trace,
            step_count=len(trace),
            message="Graph stopped at the recursion limit.",
        )
    except Exception:
        trace = tuple(extract_trace(events))
        return RunResult(
            status=RunStatus.GRAPH_ERROR,
            final_answer=None,
            trace=trace,
            step_count=len(trace),
            message="Graph execution failed.",
        )

    trace = tuple(extract_trace(events))
    final_answer = final_answer_from_events(events)
    status = RunStatus.COMPLETED if final_answer is not None else RunStatus.EMPTY_OUTPUT
    return RunResult(
        status=status,
        final_answer=final_answer,
        trace=trace,
        step_count=len(trace),
    )
