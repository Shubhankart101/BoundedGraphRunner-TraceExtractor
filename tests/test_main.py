from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from main import build_run_input, extract_trace, run_graph
from utils.models import OUTPUT_SNIPPET_LIMIT, RunStatus


class FakeGraph:
    def __init__(self, events=(), error: Exception | None = None):
        self.events = list(events)
        self.error = error
        self.calls = []

    def stream(self, graph_input, *, config, stream_mode):
        self.calls.append((graph_input, config, stream_mode))
        yield from self.events
        if self.error is not None:
            raise self.error


def test_build_run_input_strips_text_and_uses_expected_config():
    result = build_run_input("  research question  ", 5)

    assert len(result.graph_input["messages"]) == 1
    assert isinstance(result.graph_input["messages"][0], HumanMessage)
    assert result.graph_input["messages"][0].content == "research question"
    assert result.config == {"recursion_limit": 5}


@pytest.mark.parametrize("user_text", [None, 1, "", " \t\n"])
def test_build_run_input_rejects_invalid_text(user_text):
    with pytest.raises(ValueError):
        build_run_input(user_text, 5)


@pytest.mark.parametrize("recursion_limit", [True, False, 0, -1, 1.5, "5", None])
def test_build_run_input_rejects_invalid_recursion_limit(recursion_limit):
    with pytest.raises(ValueError):
        build_run_input("valid question", recursion_limit)


def test_extract_trace_preserves_event_and_node_order_and_normalizes_updates():
    events = [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_evidence",
                                "args": {"query": "wifi"},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            "tools": {
                "messages": [
                    ToolMessage(
                        content="evidence result",
                        name="search_evidence",
                        tool_call_id="call-1",
                    )
                ]
            },
        },
        {"agent": {"messages": [AIMessage(content="x" * 100)]}},
    ]

    trace = extract_trace(events)

    assert [step.node_name for step in trace] == ["agent", "tools", "agent"]
    assert trace[0].message_type == "ai"
    assert trace[0].tool_names == ("search_evidence",)
    assert trace[0].tool_call_ids == ("call-1",)
    assert trace[1].message_type == "tool"
    assert trace[1].tool_names == ("search_evidence",)
    assert trace[1].tool_call_ids == ("call-1",)
    assert len(trace[2].output_snippet) == OUTPUT_SNIPPET_LIMIT


def test_run_graph_returns_final_answer_and_uses_updates_stream():
    graph = FakeGraph(
        [
            {"agent": {"messages": [AIMessage(content="  final answer  ")]}}
        ]
    )

    result = run_graph(graph, " question ", 7)

    assert result.status is RunStatus.COMPLETED
    assert result.final_answer == "final answer"
    assert result.step_count == len(result.trace) == 1
    graph_input, config, stream_mode = graph.calls[0]
    assert graph_input["messages"][0].content == "question"
    assert config == {"recursion_limit": 7}
    assert stream_mode == "updates"


def test_run_graph_returns_empty_output_when_no_final_assistant_answer_exists():
    graph = FakeGraph(
        [{"agent": {"messages": [HumanMessage(content="not an answer")]}}]
    )

    result = run_graph(graph, "question", 5)

    assert result.status is RunStatus.EMPTY_OUTPUT
    assert result.final_answer is None
    assert result.step_count == len(result.trace) == 1


def test_run_graph_invalid_input_does_not_invoke_graph():
    graph = FakeGraph()

    result = run_graph(graph, "  ", 5)

    assert result.status is RunStatus.INVALID_INPUT
    assert result.final_answer is None
    assert result.trace == ()
    assert result.step_count == 0
    assert graph.calls == []


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (GraphRecursionError("recursion details"), RunStatus.LIMIT_REACHED),
        (RuntimeError("private failure details"), RunStatus.GRAPH_ERROR),
    ],
)
def test_run_graph_maps_failures_and_preserves_partial_trace(error, expected_status):
    graph = FakeGraph(
        [{"agent": {"messages": [AIMessage(content="partial output")]}}],
        error=error,
    )

    result = run_graph(graph, "question", 5)

    assert result.status is expected_status
    assert result.final_answer is None
    assert [step.node_name for step in result.trace] == ["agent"]
    assert result.step_count == len(result.trace) == 1
    assert str(error) not in repr(result)