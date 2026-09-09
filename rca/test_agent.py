"""
Self-test for agent.py's control-flow logic that doesn't need a live LLM
call: the forced-conclusion retry added after observing the free-tier model
flake on C001 (succeeded once, failed once, same complaint, same code), and
route_after_agent's routing decisions. Uses mocking to simulate model
responses -- same "test what doesn't need the API" split as tools.py and
extraction/test_pipeline.py.

Run with:
    cd rca && python test_agent.py
"""
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END

import agent


def test_invoke_forced_conclusion_retries_once_on_empty_response():
    empty_response = AIMessage(content="", tool_calls=[])
    good_response = AIMessage(content="", tool_calls=[
        {"name": "submit_conclusion", "args": {"root_cause_category": "Equipment"}, "id": "call_1"}])

    with patch.object(agent, "_model_concluding") as mock_model:
        mock_model.invoke.side_effect = [empty_response, good_response]
        result = agent._invoke_forced_conclusion([])

    assert mock_model.invoke.call_count == 2, "should retry exactly once after an empty first response"
    assert result is good_response
    print("_invoke_forced_conclusion: retried once after an empty response, returned the successful retry")


def test_invoke_forced_conclusion_does_not_retry_twice():
    """Only ONE retry -- if the retry also comes back empty, accept that
    (agent.investigate() then correctly reports conclusion_failed) rather
    than retrying forever."""
    empty_response = AIMessage(content="", tool_calls=[])

    with patch.object(agent, "_model_concluding") as mock_model:
        mock_model.invoke.return_value = empty_response
        result = agent._invoke_forced_conclusion([])

    assert mock_model.invoke.call_count == 2, \
        f"expected exactly 2 attempts (1 + 1 retry), got {mock_model.invoke.call_count}"
    assert result is empty_response
    print("_invoke_forced_conclusion: gives up after exactly 1 retry (2 attempts total), doesn't loop forever")


def test_invoke_forced_conclusion_skips_retry_on_success():
    good_response = AIMessage(content="", tool_calls=[{"name": "submit_conclusion", "args": {}, "id": "call_1"}])

    with patch.object(agent, "_model_concluding") as mock_model:
        mock_model.invoke.return_value = good_response
        result = agent._invoke_forced_conclusion([])

    assert mock_model.invoke.call_count == 1, "should not retry when the first call already succeeds"
    print("_invoke_forced_conclusion: no wasted retry call when the first attempt already succeeds")


def test_route_after_agent_routing():
    no_tools_state = {"messages": [AIMessage(content="done", tool_calls=[])]}
    assert agent.route_after_agent(no_tools_state) == END

    investigating_state = {"messages": [AIMessage(
        content="", tool_calls=[{"name": "query_downtime", "args": {}, "id": "c1"}])]}
    assert agent.route_after_agent(investigating_state) == "tools"

    concluding_state = {"messages": [AIMessage(
        content="", tool_calls=[{"name": "submit_conclusion", "args": {}, "id": "c1"}])]}
    assert agent.route_after_agent(concluding_state) == END

    print("route_after_agent: no-tool-calls -> END, investigation tool -> 'tools', submit_conclusion -> END")


def test_call_agent_forces_conclusion_when_investigating_model_narrates_instead_of_calling():
    """Reproduces exactly what was observed live on gpt-4o-mini: the
    investigating model reasons through the right answer and writes "I will
    now submit the conclusion" as plain text, without an actual tool call.
    Before this fix, route_after_agent would treat that as terminal and the
    whole investigation would be thrown away as conclusion_failed, despite
    the model having already reasoned to the correct answer."""
    narrated = AIMessage(content="...analysis... I will now submit the conclusion.", tool_calls=[])
    real_conclusion = AIMessage(content="", tool_calls=[
        {"name": "submit_conclusion", "args": {"root_cause_category": "Equipment"}, "id": "call_1"}])

    with patch.object(agent, "_model_investigating") as mock_investigating, \
         patch.object(agent, "_model_concluding") as mock_concluding:
        mock_investigating.invoke.return_value = narrated
        mock_concluding.invoke.return_value = real_conclusion
        result = agent.call_agent({"messages": [], "steps": 0})

    assert mock_concluding.invoke.call_count == 1, \
        "a narrated-but-not-called conclusion should force a real submit_conclusion call"
    assert result["messages"][0] is real_conclusion
    print("call_agent: investigating model narrating a conclusion instead of calling the tool "
          "correctly forces a real submit_conclusion call rather than ending the investigation")


if __name__ == "__main__":
    test_invoke_forced_conclusion_retries_once_on_empty_response()
    test_invoke_forced_conclusion_does_not_retry_twice()
    test_invoke_forced_conclusion_skips_retry_on_success()
    test_route_after_agent_routing()
    test_call_agent_forces_conclusion_when_investigating_model_narrates_instead_of_calling()
    print("\nAll agent self-tests passed (no LLM/API calls made).")
