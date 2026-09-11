"""
Experiment: ONE reusable investigator builder, instantiated once per
root-cause category with its own narrow prompt and tool subset -- instead
of agent.py's single shared prompt juggling all categories at once.

Why this exists: DAY3_STATUS.md documents a real whack-a-mole problem --
strengthening the shared prompt's Material-detection rules broke
Equipment-detection, and vice versa, across three rounds of edits. That
happens because all categories' detection logic lives in one prompt,
fought over by one model call. Giving each category its own short,
non-competing prompt makes that structurally impossible: there's no
shared prompt left to fight over.

Design, deliberately minimal:
  - 3 category investigators (Human, Equipment, Material), each built by
    the SAME function (_build_investigator_graph), just parameterized by
    CATEGORY_CONFIGS -- one reusable agent, not three hand-written ones.
  - Dispatched in PARALLEL (they're independent -- none needs another's
    result), via a plain ThreadPoolExecutor. No LLM call decides this;
    with only 3 fixed categories there's nothing to route, unlike the
    "which of 50 tools is relevant" problem in notes/scaling-architecture.md.
  - Each investigator reports a STRUCTURED finding (submit_finding), not
    free text -- {supports_category, confidence, supporting_record_ids,
    reasoning}. That structure is what lets verification be mechanical
    and per-category-specific, same idea as agent.py's
    _verify_material_citation, just one small check per category instead
    of one bespoke check for the one category that happened to break.
  - Synthesis (picking the winning category) is plain code, not another
    LLM call -- with structured, pre-verified findings, "which one wins"
    is a priority lookup, not a judgment call.

Same return contract as agent.investigate() (conclusion, conclusion_failed,
evidence_chain, etc.) so eval.py can point at either implementation --
see AGENT_IMPL in eval.py.
"""
import concurrent.futures
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from tools import (_decode_batch_code, _downtime, _query_shifts,
                    decode_batch_code, query_downtime, query_shifts,
                    query_complaints, query_complaints_by_machine)

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
_complaints_by_id = {c["complaint_id"]: c for c in json.loads((DATA_DIR / "complaints.json").read_text())}

MAX_STEPS = 6  # each investigator only has one narrow job -- needs far fewer
               # turns than agent.py's MAX_STEPS=8 juggling all categories
MAX_VERIFICATION_RETRIES = 2  # same bounded-retry discipline as agent.py's
                               # verification loop -- this is what was MISSING the
                               # first time this experiment ran (see
                               # notes/per-category-experiment-results.md): a
                               # rejected finding was just discarded with no
                               # chance to self-correct


def _build_model() -> ChatOpenAI:
    """Same provider toggle as agent.py, duplicated on purpose -- this
    module is a self-contained experiment being A/B tested against
    agent.py, not a shared dependency of it."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "openrouter":
        return ChatOpenAI(model=os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
                           temperature=0, api_key=os.environ.get("OPENROUTER_API_KEY"),
                           base_url="https://openrouter.ai/api/v1")
    return ChatOpenAI(model=os.environ.get("RCA_MODEL", "gpt-4o-mini"), temperature=0)


@tool
def submit_finding(supports_category: bool, confidence: str, supporting_record_ids: list[str], reasoning: str) -> dict:
    """Call this once you've finished checking your ONE assigned signal.
    supports_category must be True only if you found concrete evidence in the records YOU looked up --
    never guess.
    supporting_record_ids: the exact record IDs (e.g. downtime_id, shift_id, complaint_id) that support
    your finding -- empty list if supports_category is False.
    confidence: high, medium, or low.
    reasoning: one or two sentences, must reference the record IDs you cited."""
    return {"supports_category": supports_category, "confidence": confidence,
            "supporting_record_ids": supporting_record_ids, "reasoning": reasoning}


@tool
def submit_finding_equipment(supports_category: bool, confidence: str, supporting_record_ids: list[str],
                              reasoning: str, is_anomaly_not_routine: bool) -> dict:
    """Call this once you've finished checking your ONE assigned signal.
    supports_category must be True only if you found concrete evidence in the records YOU looked up --
    never guess.
    supporting_record_ids: the exact record IDs (downtime_id or complaint_id) that support your finding --
    empty list if supports_category is False.
    confidence: high, medium, or low.
    reasoning: one or two sentences, must reference the record IDs you cited.
    is_anomaly_not_routine: ONLY relevant if you're citing a downtime_id. Must be True only if that record
    describes catching/fixing an actual problem (e.g. a calibration drift, a breakdown) -- NOT a routine
    scheduled event (e.g. "Scheduled PM", ordinary changeover). A routine event existing is not evidence of
    anything; do not set supports_category=True on a downtime record alone unless this is also True. If
    you're citing complaint IDs for a recurrence pattern instead of a downtime record, set this to True
    (not applicable to that evidence type)."""
    return {"supports_category": supports_category, "confidence": confidence,
            "supporting_record_ids": supporting_record_ids, "reasoning": reasoning,
            "is_anomaly_not_routine": is_anomaly_not_routine}


CATEGORY_CONFIGS = {
    "Human": {
        "tools": [decode_batch_code, query_shifts],
        "submit_tool": submit_finding,
        "prompt": """You are a narrow specialist investigator at a tyre/cycle plant, checking ONLY ONE thing: \
whether a trainee-staffed shift explains a manufacturing defect. You are NOT responsible for any other \
root-cause category -- do not consider material, equipment, or process causes at all.

1. Decode the complaint's batch code to get the machine and its production date window.
2. Query shift records for that machine in that window.
3. If any shift has operator_experience == "trainee", that is your evidence: supports_category=True, cite \
that shift_id in supporting_record_ids.
4. If no trainee shift exists in that window, supports_category=False, empty supporting_record_ids.
5. Call submit_finding. Never guess -- only report True if you found an actual trainee shift record."""
    },
    "Equipment": {
        "tools": [decode_batch_code, query_downtime, query_complaints_by_machine],
        "submit_tool": submit_finding_equipment,
        "prompt": """You are a narrow specialist investigator at a tyre/cycle plant, checking ONLY ONE thing: \
whether an equipment issue -- a discrete downtime/maintenance event, OR gradual wear shown by recurrence -- \
explains a manufacturing defect. You are NOT responsible for any other root-cause category -- do not \
consider human, material, or process causes at all.

1. Decode the complaint's batch code to get the machine and its production date window.
2. Query downtime for that machine, extending the window about 2 weeks PAST its end date: a calibration or \
maintenance fix is sometimes only logged after the defective batch already shipped, and can still explain \
it -- check whether the record describes catching/fixing something that would already have been affecting \
production before that date.
3. If you find such a record, that is your evidence -- but ONLY if it describes an actual anomaly being \
caught/fixed (e.g. "recalibrated after drift found"), not a routine scheduled event (e.g. "Scheduled PM"). \
A routine event existing proves nothing; do not treat one as evidence just because it's the only downtime \
record around. If it's genuinely an anomaly: supports_category=True, cite that downtime_id,
is_anomaly_not_routine=True.
4. If not, check this machine's full complaint history (query_complaints_by_machine) for recurrence across \
several non-adjacent weeks with no single explaining event -- that recurrence pattern itself is evidence of \
gradual equipment wear. If found, supports_category=True, cite the recurring complaint_ids,
is_anomaly_not_routine=True (not applicable to this evidence type, so leave it True).
5. If neither applies, supports_category=False, empty supporting_record_ids.
6. Call submit_finding_equipment. Never guess -- only report True if you found a genuine anomaly record or \
a real multi-week recurrence pattern (list specific complaint IDs, not just "multiple complaints")."""
    },
    "Material": {
        "tools": [decode_batch_code, query_complaints],
        "submit_tool": submit_finding,
        "prompt": """You are a narrow specialist investigator at a tyre/cycle plant, checking ONLY ONE thing: \
whether a shared material batch explains a manufacturing defect. You are NOT responsible for any other \
root-cause category -- do not consider human, equipment, or process causes at all.

1. Decode the complaint's batch code to get the machine and its production date window.
2. Query other complaints in that same date window.
3. Look for a complaint on a genuinely DIFFERENT machine_id (not this one) with a matching or clearly \
similar defect_description (the same symptom). That is your ONLY valid evidence.
4. A complaint on the SAME machine is NOT this signal, no matter how many there are -- that's just multiple \
customers reporting one batch, which proves nothing about the cause. A different-machine complaint with an \
UNRELATED symptom is also not evidence -- with many complaints spread across a year, unrelated storylines \
sometimes land in the same week by pure coincidence. Do not report supports_category=True for either.
5. Call submit_finding. Never guess -- only report True if you found a genuine different-machine, \
matching-symptom complaint, and cite its complaint_id."""
    },
}

PRIORITY = ["Human", "Equipment", "Material"]  # first verified-True finding in this order wins


def _build_investigator_graph(category: str) -> dict:
    """The one reusable builder -- identical graph shape every call, only
    the prompt, tool subset, and submit tool differ per category. Returns
    a bundle (not just the compiled graph) so investigate() can reuse
    model_concluding directly for the verification-retry loop, the same
    way agent.py's _invoke_forced_conclusion gets reused for its retries."""
    config = CATEGORY_CONFIGS[category]
    submit_tool = config["submit_tool"]
    submit_name = submit_tool.name
    model = _build_model()
    model_with_tools = model.bind_tools(config["tools"] + [submit_tool])
    model_concluding = model.bind_tools([submit_tool], tool_choice=submit_name)
    tool_node = ToolNode(config["tools"])

    class State(MessagesState):
        steps: int

    def call_model(state: State) -> dict:
        steps = state.get("steps", 0)
        messages = state["messages"]
        if steps >= MAX_STEPS:
            forced = HumanMessage(content=f"Step limit reached. Call {submit_name} now with your best "
                                           f"answer based on what you've found so far.")
            response = model_concluding.invoke(messages + [forced])
        else:
            response = model_with_tools.invoke(messages)
            if not getattr(response, "tool_calls", None):
                forced = HumanMessage(content=f"Call {submit_name} now with your finding -- the summary "
                                               f"you just wrote is not a structured answer.")
                response = model_concluding.invoke(messages + [response, forced])
        return {"messages": [response], "steps": steps + 1}

    def route(state: State):
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if not tool_calls:
            return END
        if any(tc["name"] == submit_name for tc in tool_calls):
            return END
        return "tools"

    builder = StateGraph(State)
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return {"graph": builder.compile(), "model_concluding": model_concluding, "submit_name": submit_name}


_INVESTIGATORS = {category: _build_investigator_graph(category) for category in CATEGORY_CONFIGS}


def _invoke_forced_finding(model_concluding, messages):
    """Same discipline as agent.py's _invoke_forced_conclusion -- retry
    once if the model returns no tool call at all, then accept whatever
    comes back rather than looping forever."""
    response = model_concluding.invoke(messages)
    if not getattr(response, "tool_calls", None):
        response = model_concluding.invoke(messages)
    return response


def _build_evidence_chain(messages, submit_name: str = "submit_finding") -> list[dict]:
    """Same idea as agent.py's build_evidence_chain -- walk the messages,
    pair each investigation tool call with its result. submit_finding is
    the termination signal here, same role submit_conclusion plays there."""
    by_call_id = {}
    for m in messages:
        if getattr(m, "type", None) == "tool":
            by_call_id[m.tool_call_id] = m.content
    chain = []
    step = 1
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"] == submit_name:
                continue
            chain.append({"step": step, "tool": tc["name"], "args": tc["args"],
                          "result": by_call_id.get(tc["id"])})
            step += 1
    return chain


def _extract_finding(messages, submit_name: str = "submit_finding") -> dict | None:
    last = messages[-1]
    for tc in getattr(last, "tool_calls", None) or []:
        if tc["name"] == submit_name:
            return tc["args"]
    return None


def _verify_human_finding(finding: dict, complaint: dict) -> bool:
    if not finding.get("supports_category"):
        return True  # nothing to verify for a negative finding
    ids = finding.get("supporting_record_ids") or []
    if not ids:
        return False
    window = _decode_batch_code(complaint["batch_code"])
    shifts = _query_shifts(window["machine_id"], window["start_date"], window["end_date"])
    trainee_ids = {s["shift_id"] for s in shifts if s["operator_experience"] == "trainee"}
    return any(sid in trainee_ids for sid in ids)


def _verify_material_finding(finding: dict, complaint: dict) -> bool:
    if not finding.get("supports_category"):
        return True
    ids = finding.get("supporting_record_ids") or []
    if not ids:
        return False
    own_machine = _decode_batch_code(complaint["batch_code"])["machine_id"]
    own_symptom = complaint["defect_description"]
    for cid in ids:
        cited = _complaints_by_id.get(cid)
        if cited is None:
            continue
        cited_machine = _decode_batch_code(cited["batch_code"])["machine_id"]
        if cited_machine != own_machine and cited["defect_description"] == own_symptom:
            return True
    return False


def _verify_equipment_finding(finding: dict, complaint: dict) -> bool:
    if not finding.get("supports_category"):
        return True
    ids = finding.get("supporting_record_ids") or []
    if not ids:
        return False
    machine = _decode_batch_code(complaint["batch_code"])["machine_id"]
    own_week = _decode_batch_code(complaint["batch_code"])["week"]

    real_downtime_ids = {d["downtime_id"] for d in _downtime if d["machine_id"] == machine}
    if any(cid in real_downtime_ids for cid in ids):
        # A real downtime record alone isn't enough -- it must also be self-reported
        # as an anomaly, not routine (e.g. "Scheduled PM"). This is the model's OWN
        # structured claim, not free text the verifier has to parse or guess at --
        # diagnosed directly on C008, where a routine maintenance record got waved
        # through as "could have addressed issues" before this check existed.
        return bool(finding.get("is_anomaly_not_routine", False))

    # Otherwise this must be a genuine recurrence claim: same-machine complaints
    # spread across a DIFFERENT week than the one under investigation, not just
    # any same-machine complaint -- a same-week co-occurrence is coincidence, the
    # same failure mode agent.py's old shared prompt had for Material, just here
    # for Equipment's recurrence branch instead. At least 2 citations required too,
    # so one lucky nearby complaint alone doesn't count as "a pattern."
    cited_other_weeks = set()
    for cid in ids:
        cited = _complaints_by_id.get(cid)
        if cited and _decode_batch_code(cited["batch_code"])["machine_id"] == machine:
            week = _decode_batch_code(cited["batch_code"])["week"]
            if week != own_week:
                cited_other_weeks.add(week)
    return len(ids) >= 2 and len(cited_other_weeks) >= 1


_VERIFIERS = {"Human": _verify_human_finding, "Equipment": _verify_equipment_finding,
              "Material": _verify_material_finding}

# Same review-gate logic as agent.py's, duplicated rather than imported --
# this module is a self-contained experiment being A/B tested against
# agent.py, not a shared dependency of it (see the module docstring's
# provider-toggle comment for the same rationale). Category priors are the
# real measured per-category hit rates from round 3's held-out eval (see
# notes/per-category-experiment-results.md), for THIS module specifically.
_CATEGORY_RELIABILITY = {
    "equipment": 1.0, "human": 1.0, "insufficient evidence": 1.0,
    "material": 0.5, "process": 0.7, "scheduling": 0.7,
}
_RECORD_ID_RE = re.compile(r"\b[DSC]\d{3,4}\b")
CONFIDENCE_THRESHOLD = 0.80


def _confidence_score(conclusion: dict, material_verification_failed: bool, retries_used: int) -> dict:
    if material_verification_failed:
        verification_signal = 0.0
    elif retries_used == 0:
        verification_signal = 1.0
    elif retries_used == 1:
        verification_signal = 0.6
    else:
        verification_signal = 0.3

    category = (conclusion.get("root_cause_category") or "").strip().lower()
    category_signal = _CATEGORY_RELIABILITY.get(category, 0.7)

    hypothesis = conclusion.get("root_cause_hypothesis", "") or ""
    if category == "insufficient evidence":
        citation_signal = 1.0
    else:
        citation_signal = 1.0 if _RECORD_ID_RE.search(hypothesis) else 0.0

    score = round(0.30 * verification_signal + 0.50 * category_signal + 0.20 * citation_signal, 3)
    return {"score": score, "verification_signal": verification_signal,
            "category_signal": category_signal, "citation_signal": citation_signal}


def _human_review_reasons(conclusion: dict, conclusion_failed: bool, material_verification_failed: bool,
                           retries_used: int) -> tuple[list[str], dict]:
    reasons = []
    if conclusion_failed:
        reasons.append("no investigator produced a real structured finding (conclusion_failed) -- "
                        "whatever category is shown is a fallback, not a genuine answer")
        return reasons, {"score": 0.0, "verification_signal": 0.0, "category_signal": 0.0, "citation_signal": 0.0}

    scoring = _confidence_score(conclusion, material_verification_failed, retries_used)
    category = (conclusion.get("root_cause_category") or "").strip().lower()

    if category == "insufficient evidence":
        reasons.append("no specialist investigator found verified supporting evidence -- recommend "
                        "manual investigation rather than accepting this as final")
    if scoring["score"] < CONFIDENCE_THRESHOLD:
        weak = []
        if scoring["verification_signal"] < 1.0:
            weak.append(f"verification signal {scoring['verification_signal']} (retries used: {retries_used}, "
                        f"material_verification_failed: {material_verification_failed})")
        if scoring["category_signal"] < 1.0:
            weak.append(f"category '{conclusion.get('root_cause_category')}' has a measured reliability prior "
                        f"of only {scoring['category_signal']} from held-out testing")
        if scoring["citation_signal"] < 1.0:
            weak.append("hypothesis does not cite any specific record ID")
        reasons.append(f"confidence score {scoring['score']} is below the {CONFIDENCE_THRESHOLD} threshold "
                        f"-- " + "; ".join(weak))
    return reasons, scoring


def investigate(complaint_id: str) -> dict:
    complaint = _complaints_by_id.get(complaint_id)
    if complaint is None:
        raise ValueError(f"Unknown complaint_id: {complaint_id}")

    def run_one(category: str):
        investigator = _INVESTIGATORS[category]
        submit_name = investigator["submit_name"]
        messages = [
            SystemMessage(content=CATEGORY_CONFIGS[category]["prompt"]),
            HumanMessage(content=f"Investigate this complaint:\n{json.dumps(complaint, indent=2)}"),
        ]
        final_state = investigator["graph"].invoke({"messages": messages, "steps": 0},
                                                     config={"recursion_limit": MAX_STEPS * 2 + 4})
        messages = final_state["messages"]
        steps_used = final_state.get("steps", 0)

        finding = _extract_finding(messages, submit_name)
        verification_log = []

        # The retry loop that was MISSING the first time this experiment ran:
        # a rejected finding used to just get discarded. Now it gets the same
        # bounded, feedback-driven retry that worked so well for agent.py's
        # Material check.
        if finding is not None:
            is_valid = _VERIFIERS[category](finding, complaint)
            while not is_valid and len(verification_log) < MAX_VERIFICATION_RETRIES:
                verification_log.append({"attempt": len(verification_log) + 1, "category": category})
                last_ai_message = messages[-1]
                submit_call = next((tc for tc in getattr(last_ai_message, "tool_calls", None) or []
                                     if tc["name"] == submit_name), None)
                if submit_call is not None:
                    # Same OpenAI API requirement as agent.py: a pending tool_call
                    # must be answered before anything else can follow it.
                    messages = messages + [ToolMessage(
                        content="Rejected by verification -- see correction note.",
                        tool_call_id=submit_call["id"])]
                correction = HumanMessage(content=(
                    "Your finding needs correction: the evidence you cited doesn't hold up "
                    "(wrong machine, a routine record mistaken for an anomaly, or similar). "
                    f"Reconsider what you already looked up and call {submit_name} again."))
                messages = messages + [correction]
                response = _invoke_forced_finding(investigator["model_concluding"], messages)
                messages = messages + [response]
                finding = _extract_finding(messages, submit_name)
                if finding is None:
                    break
                is_valid = _VERIFIERS[category](finding, complaint)
            if finding is not None:
                finding["_verified"] = is_valid

        return category, messages, steps_used, finding, verification_log

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CATEGORY_CONFIGS)) as pool:
        results = list(pool.map(run_one, CATEGORY_CONFIGS))

    findings = {}
    sub_agent_failed = {}
    all_evidence_chain = []
    step = 1
    total_steps_used = 0
    combined_verification_log = []

    for category, messages, steps_used, finding, verification_log in results:
        total_steps_used += steps_used
        combined_verification_log.extend(verification_log)
        submit_name = _INVESTIGATORS[category]["submit_name"]
        for item in _build_evidence_chain(messages, submit_name):
            item = dict(item)
            item["step"] = step
            item["investigator"] = category
            all_evidence_chain.append(item)
            step += 1

        sub_agent_failed[category] = finding is None
        findings[category] = finding

    winner = None
    for category in PRIORITY:
        f = findings.get(category)
        if f and f.get("supports_category") and f.get("_verified"):
            winner = category
            break

    conclusion_failed = all(sub_agent_failed.values())  # only a hard failure if EVERY investigator
                                                          # produced no structured finding at all

    if winner is not None:
        f = findings[winner]
        conclusion = {
            "root_cause_category": winner,
            "root_cause_hypothesis": f["reasoning"],
            "confidence": f["confidence"],
            "recommended_corrective_action": f"Address the {winner.lower()} issue identified: {f['reasoning']}",
        }
    elif conclusion_failed:
        conclusion = {"root_cause_category": "Insufficient evidence",
                      "root_cause_hypothesis": "No investigator produced a structured finding.",
                      "confidence": "low", "recommended_corrective_action": "Re-run or investigate manually."}
    else:
        conclusion = {
            "root_cause_category": "Insufficient evidence",
            "root_cause_hypothesis": "No specialist investigator found verified supporting evidence: "
                                      + "; ".join(f"{c}={findings[c]['reasoning']}" for c in PRIORITY if findings.get(c)),
            "confidence": "low",
            "recommended_corrective_action": "No single category's evidence held up to verification -- "
                                              "escalate for manual review.",
        }

    material_verification_failed = bool(findings.get("Material") and findings["Material"].get("supports_category")
                                         and not findings["Material"].get("_verified"))
    review_reasons, confidence_scoring = _human_review_reasons(
        conclusion, conclusion_failed, material_verification_failed, len(combined_verification_log))

    return {
        "complaint_id": complaint_id,
        "complaint": complaint,
        "evidence_chain": all_evidence_chain,
        "conclusion": conclusion,
        "conclusion_failed": conclusion_failed,
        "material_verification_failed": material_verification_failed,
        "verification_log": combined_verification_log,
        "sub_agent_failed": sub_agent_failed,
        "findings_by_category": findings,
        "confidence_score": confidence_scoring,
        "needs_human_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "steps_used": total_steps_used,
    }
