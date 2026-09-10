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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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


def test_verify_material_citation_ignores_non_material_categories():
    conclusion = {"root_cause_category": "Equipment", "root_cause_hypothesis": "whatever, not checked here"}
    is_valid, reason = agent._verify_material_citation(
        conclusion, {"batch_code": "M02-2026W13", "defect_description": "x"})
    assert is_valid and reason == ""
    print("_verify_material_citation: non-Material categories pass through unchecked")


def test_verify_material_citation_accepts_genuine_cross_machine_match():
    """C008 (M01) and C010 (M03) are the real Material storyline -- different
    machines, byte-identical defect_description. This is what a CORRECT
    Material citation looks like, and it must still pass."""
    investigated = agent._complaints_by_id["C008"]
    conclusion = {"root_cause_category": "Material",
                  "root_cause_hypothesis": "Cross-machine pattern: C010 is on a different machine and shares "
                                            "the exact same symptom."}
    is_valid, reason = agent._verify_material_citation(conclusion, investigated)
    assert is_valid, f"expected a genuine cross-machine citation to pass, got rejected: {reason}"
    print("_verify_material_citation: accepts a real cross-machine, matching-symptom citation (C008 citing C010)")


def test_verify_material_citation_rejects_same_machine_recurrence():
    """The exact failure mode from eval_results.json: C001 is on M02;
    C002/C003/C004 are ALSO on M02 with the same symptom -- that's
    recurrence, not a cross-machine pattern, even though the model kept
    citing it as Material after three rounds of prompt fixes."""
    investigated = agent._complaints_by_id["C001"]
    conclusion = {"root_cause_category": "Material",
                  "root_cause_hypothesis": "Multiple complaints C002, C003, C004 on the same batch confirm "
                                            "a material issue."}
    is_valid, reason = agent._verify_material_citation(conclusion, investigated)
    assert not is_valid
    assert "SAME" in reason and "M02" in reason
    print("_verify_material_citation: rejects same-machine recurrence cited as Material -- the real eval failure")


def test_verify_material_citation_rejects_no_cited_ids():
    conclusion = {"root_cause_category": "Material", "root_cause_hypothesis": "Multiple customers reported this."}
    is_valid, reason = agent._verify_material_citation(
        conclusion, {"batch_code": "M02-2026W13", "defect_description": "x"})
    assert not is_valid and "doesn't cite" in reason
    print("_verify_material_citation: rejects a Material conclusion with no complaint IDs to check at all")


def _submit_conclusion_message(call_id: str, category: str, hypothesis: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "submit_conclusion", "id": call_id, "args": {
        "root_cause_category": category, "root_cause_hypothesis": hypothesis,
        "confidence": "medium", "recommended_corrective_action": "x"}}])


def test_investigate_retries_on_invalid_material_citation_then_accepts_correction():
    """End-to-end (mocked, no real graph/API involved): the initial
    submit_conclusion wrongly claims Material citing a same-machine
    complaint; the retry loop should send exactly one correction and accept
    a fixed Equipment conclusion on the second try."""
    bad = _submit_conclusion_message("call_1", "Material", "C002, C003, C004 same batch confirm this.")
    fixed = _submit_conclusion_message("call_2", "Equipment", "Calibration drift per D0131.")
    fake_final_state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="investigate"), bad],
                         "steps": 3}

    with patch.object(agent.graph, "invoke", return_value=fake_final_state), \
         patch.object(agent, "_invoke_forced_conclusion", return_value=fixed) as mock_correct:
        result = agent.investigate("C001")

    assert mock_correct.call_count == 1, "should correct exactly once when the fix is accepted on the first retry"
    assert result["conclusion"]["root_cause_category"] == "Equipment"
    assert result["material_verification_failed"] is False
    assert len(result["verification_log"]) == 1
    assert "SAME" in result["verification_log"][0]["reason"]
    print("investigate(): caught a bad same-machine Material citation, corrected in one retry, "
          "returned the fixed Equipment conclusion")


def test_investigate_gives_up_after_max_verification_retries():
    """If the model stubbornly re-asserts the same invalid Material citation
    every time, the retry loop must stop at MAX_VERIFICATION_RETRIES and
    flag material_verification_failed rather than looping forever."""
    initial = _submit_conclusion_message("c0", "Material", "C002, C003 same batch.")
    fake_final_state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="investigate"), initial],
                         "steps": 3}
    stubborn = _submit_conclusion_message("cN", "Material", "Still C002, C003, C004 same batch.")

    with patch.object(agent.graph, "invoke", return_value=fake_final_state), \
         patch.object(agent, "_invoke_forced_conclusion", return_value=stubborn) as mock_correct:
        result = agent.investigate("C001")

    assert mock_correct.call_count == agent.MAX_VERIFICATION_RETRIES
    assert result["material_verification_failed"] is True
    assert len(result["verification_log"]) == agent.MAX_VERIFICATION_RETRIES
    print(f"investigate(): stops after exactly {agent.MAX_VERIFICATION_RETRIES} correction attempts, "
          f"flags material_verification_failed instead of looping forever")


def test_investigate_accepts_an_adhoc_complaint_not_in_the_dataset():
    """investigate() must also accept a fresh complaint dict -- not just a
    known complaint_id -- for the "type your own complaint" path (the demo
    console, and main.py's POST /investigate). No complaint_id given, so
    one should be generated."""
    clean = _submit_conclusion_message("call_1", "Equipment", "Calibration drift per D0131.")
    fake_final_state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="investigate"), clean],
                         "steps": 2}
    adhoc = {"batch_code": "M02-2026W13", "defect_description": "Blistering on inner shoulder."}

    with patch.object(agent.graph, "invoke", return_value=fake_final_state):
        result = agent.investigate(adhoc)

    assert result["complaint_id"].startswith("ADHOC-")
    assert result["complaint"]["batch_code"] == "M02-2026W13"
    assert result["conclusion"]["root_cause_category"] == "Equipment"
    print("investigate(): accepts an ad-hoc complaint dict not present in complaints.json, "
          f"generating {result['complaint_id']!r} since none was supplied")


def test_investigate_preserves_a_caller_supplied_adhoc_complaint_id():
    clean = _submit_conclusion_message("call_1", "Equipment", "Calibration drift per D0131.")
    fake_final_state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="investigate"), clean],
                         "steps": 2}
    adhoc = {"complaint_id": "WALKIN-001", "batch_code": "M02-2026W13", "defect_description": "Blistering."}

    with patch.object(agent.graph, "invoke", return_value=fake_final_state):
        result = agent.investigate(adhoc)

    assert result["complaint_id"] == "WALKIN-001"
    print("investigate(): a caller-supplied complaint_id on an ad-hoc complaint is kept as-is, not overwritten")


def test_investigate_rejects_adhoc_complaint_missing_required_fields():
    try:
        agent.investigate({"defect_description": "no batch code given"})
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "batch_code" in str(e)
    print("investigate(): an ad-hoc complaint missing batch_code is rejected before touching the graph")


def test_investigate_rejects_adhoc_complaint_with_unparseable_batch_code():
    try:
        agent.investigate({"batch_code": "not-a-real-code", "defect_description": "x"})
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "batch_code" in str(e).lower()
    print("investigate(): an ad-hoc complaint with a malformed batch_code is rejected with a clear reason, "
          "instead of failing deep inside a tool call")


def test_confidence_score_clean_equipment_run_scores_perfectly():
    conclusion = {"root_cause_category": "Equipment", "root_cause_hypothesis": "Explained by D0131."}
    scoring = agent._confidence_score(conclusion, material_verification_failed=False, retries_used=0)
    assert scoring["score"] == 1.0, scoring
    print(f"_confidence_score: clean Equipment run with a cited ID scores {scoring['score']} (perfect)")


def test_confidence_score_clean_material_run_still_scores_below_threshold():
    """The round-3 finding, now expressed as a score instead of a hardcoded
    rule: Material measured 50% on unseen data vs 100% elsewhere, so even a
    PERFECTLY clean Material run (no retries, real citation) still can't
    clear the threshold, because the category signal alone caps it."""
    conclusion = {"root_cause_category": "Material", "root_cause_hypothesis": "Cross-machine match: C010."}
    scoring = agent._confidence_score(conclusion, material_verification_failed=False, retries_used=0)
    assert scoring["score"] < agent.CONFIDENCE_THRESHOLD, scoring
    print(f"_confidence_score: a CLEAN Material run still scores {scoring['score']} "
          f"(below the {agent.CONFIDENCE_THRESHOLD} threshold) -- the category prior caps it")


def test_confidence_score_penalizes_retries_and_missing_citation():
    conclusion_no_cite = {"root_cause_category": "Equipment", "root_cause_hypothesis": "no ID cited here"}
    scoring = agent._confidence_score(conclusion_no_cite, material_verification_failed=False, retries_used=0)
    assert scoring["citation_signal"] == 0.0 and scoring["score"] < 1.0

    conclusion_retried = {"root_cause_category": "Equipment", "root_cause_hypothesis": "D0131 explains it."}
    scoring2 = agent._confidence_score(conclusion_retried, material_verification_failed=False, retries_used=2)
    assert scoring2["verification_signal"] < 1.0 and scoring2["score"] < 1.0
    print("_confidence_score: missing citation and extra retries both pull the score down independently")


def test_human_review_reasons_clean_equipment_run_needs_no_review():
    conclusion = {"root_cause_category": "Equipment", "root_cause_hypothesis": "Explained by D0131."}
    reasons, scoring = agent._human_review_reasons(conclusion, conclusion_failed=False,
                                                     material_verification_failed=False, retries_used=0)
    assert reasons == [] and scoring["score"] == 1.0
    print("_human_review_reasons: a clean, cited Equipment conclusion needs no review")


def test_human_review_reasons_flags_conclusion_failed():
    conclusion = {"root_cause_category": "Insufficient evidence", "root_cause_hypothesis": "fallback"}
    reasons, scoring = agent._human_review_reasons(conclusion, conclusion_failed=True,
                                                     material_verification_failed=False, retries_used=0)
    assert len(reasons) == 1 and "conclusion_failed" in reasons[0] and scoring["score"] == 0.0
    print("_human_review_reasons: conclusion_failed always flags for review, regardless of category")


def test_human_review_reasons_flags_material_verification_failed():
    conclusion = {"root_cause_category": "Material", "root_cause_hypothesis": "C010."}
    reasons, scoring = agent._human_review_reasons(conclusion, conclusion_failed=False,
                                                     material_verification_failed=True, retries_used=2)
    assert any("material_verification_failed: True" in r for r in reasons)
    assert scoring["verification_signal"] == 0.0
    print(f"_human_review_reasons: material_verification_failed drives the score to {scoring['score']} and is flagged")


def test_human_review_reasons_flags_material_category_by_default():
    """Same round-3 finding as the confidence_score test above, checked at
    the review-gate level this time: a clean-looking Material run still
    gets flagged, unlike every other category, because the system's own
    measured track record says to double-check it regardless of whether
    anything else looks wrong."""
    conclusion = {"root_cause_category": "Material", "root_cause_hypothesis": "Cross-machine match: C010."}
    reasons, scoring = agent._human_review_reasons(conclusion, conclusion_failed=False,
                                                     material_verification_failed=False, retries_used=0)
    assert any("reliability prior" in r for r in reasons)
    print(f"_human_review_reasons: a CLEAN Material conclusion (score {scoring['score']}) is still flagged")


def test_human_review_reasons_flags_insufficient_evidence():
    conclusion = {"root_cause_category": "Insufficient evidence", "root_cause_hypothesis": "genuine, not a fallback"}
    reasons, scoring = agent._human_review_reasons(conclusion, conclusion_failed=False,
                                                     material_verification_failed=False, retries_used=0)
    assert any("could not reach a conclusion" in r for r in reasons)
    assert scoring["score"] == 1.0  # the trigger fires regardless of score -- it's not a low-confidence signal
    print("_human_review_reasons: a genuine (non-fallback) Insufficient evidence conclusion is flagged "
          f"even though its own score is {scoring['score']} -- this trigger is separate from the score")


if __name__ == "__main__":
    test_invoke_forced_conclusion_retries_once_on_empty_response()
    test_invoke_forced_conclusion_does_not_retry_twice()
    test_invoke_forced_conclusion_skips_retry_on_success()
    test_route_after_agent_routing()
    test_call_agent_forces_conclusion_when_investigating_model_narrates_instead_of_calling()
    test_verify_material_citation_ignores_non_material_categories()
    test_verify_material_citation_accepts_genuine_cross_machine_match()
    test_verify_material_citation_rejects_same_machine_recurrence()
    test_verify_material_citation_rejects_no_cited_ids()
    test_investigate_retries_on_invalid_material_citation_then_accepts_correction()
    test_investigate_gives_up_after_max_verification_retries()
    test_investigate_accepts_an_adhoc_complaint_not_in_the_dataset()
    test_investigate_preserves_a_caller_supplied_adhoc_complaint_id()
    test_investigate_rejects_adhoc_complaint_missing_required_fields()
    test_investigate_rejects_adhoc_complaint_with_unparseable_batch_code()
    test_confidence_score_clean_equipment_run_scores_perfectly()
    test_confidence_score_clean_material_run_still_scores_below_threshold()
    test_confidence_score_penalizes_retries_and_missing_citation()
    test_human_review_reasons_clean_equipment_run_needs_no_review()
    test_human_review_reasons_flags_conclusion_failed()
    test_human_review_reasons_flags_material_verification_failed()
    test_human_review_reasons_flags_material_category_by_default()
    test_human_review_reasons_flags_insufficient_evidence()
    print("\nAll agent self-tests passed (no LLM/API calls made).")