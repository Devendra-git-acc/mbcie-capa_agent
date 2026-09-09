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
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
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


CATEGORY_CONFIGS = {
    "Human": {
        "tools": [decode_batch_code, query_shifts],
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
        "prompt": """You are a narrow specialist investigator at a tyre/cycle plant, checking ONLY ONE thing: \
whether an equipment issue -- a discrete downtime/maintenance event, OR gradual wear shown by recurrence -- \
explains a manufacturing defect. You are NOT responsible for any other root-cause category -- do not \
consider human, material, or process causes at all.

1. Decode the complaint's batch code to get the machine and its production date window.
2. Query downtime for that machine, extending the window about 2 weeks PAST its end date: a calibration or \
maintenance fix is sometimes only logged after the defective batch already shipped, and can still explain \
it -- check whether the record describes catching/fixing something that would already have been affecting \
production before that date.
3. If you find such a record, that is your evidence: supports_category=True, cite that downtime_id.
4. If not, check this machine's full complaint history (query_complaints_by_machine) for recurrence across \
several non-adjacent weeks with no single explaining event -- that recurrence pattern itself is evidence of \
gradual equipment wear. If found, supports_category=True, cite the recurring complaint_ids.
5. If neither applies, supports_category=False, empty supporting_record_ids.
6. Call submit_finding. Never guess -- only report True if you found an actual record or a genuine \
multi-week recurrence pattern (list specific complaint IDs, not just "multiple complaints")."""
    },
    "Material": {
        "tools": [decode_batch_code, query_complaints],
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


def _build_investigator_graph(category: str):
    """The one reusable builder -- identical graph shape every call, only
    the prompt and tool subset differ per category."""
    config = CATEGORY_CONFIGS[category]
    model = _build_model()
    model_with_tools = model.bind_tools(config["tools"] + [submit_finding])
    model_concluding = model.bind_tools([submit_finding], tool_choice="submit_finding")
    tool_node = ToolNode(config["tools"])

    class State(MessagesState):
        steps: int

    def call_model(state: State) -> dict:
        steps = state.get("steps", 0)
        messages = state["messages"]
        if steps >= MAX_STEPS:
            forced = HumanMessage(content="Step limit reached. Call submit_finding now with your best "
                                           "answer based on what you've found so far.")
            response = model_concluding.invoke(messages + [forced])
        else:
            response = model_with_tools.invoke(messages)
            if not getattr(response, "tool_calls", None):
                forced = HumanMessage(content="Call submit_finding now with your finding -- the summary "
                                               "you just wrote is not a structured answer.")
                response = model_concluding.invoke(messages + [response, forced])
        return {"messages": [response], "steps": steps + 1}

    def route(state: State):
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if not tool_calls:
            return END
        if any(tc["name"] == "submit_finding" for tc in tool_calls):
            return END
        return "tools"

    builder = StateGraph(State)
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


_GRAPHS = {category: _build_investigator_graph(category) for category in CATEGORY_CONFIGS}


def _build_evidence_chain(messages) -> list[dict]:
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
            if tc["name"] == "submit_finding":
                continue
            chain.append({"step": step, "tool": tc["name"], "args": tc["args"],
                          "result": by_call_id.get(tc["id"])})
            step += 1
    return chain


def _extract_finding(messages) -> dict | None:
    last = messages[-1]
    for tc in getattr(last, "tool_calls", None) or []:
        if tc["name"] == "submit_finding":
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
        return True  # a real discrete downtime record is sufficient on its own

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


def investigate(complaint_id: str) -> dict:
    complaint = _complaints_by_id.get(complaint_id)
    if complaint is None:
        raise ValueError(f"Unknown complaint_id: {complaint_id}")

    def run_one(category: str):
        graph = _GRAPHS[category]
        messages = [
            SystemMessage(content=CATEGORY_CONFIGS[category]["prompt"]),
            HumanMessage(content=f"Investigate this complaint:\n{json.dumps(complaint, indent=2)}"),
        ]
        final_state = graph.invoke({"messages": messages, "steps": 0},
                                    config={"recursion_limit": MAX_STEPS * 2 + 4})
        return category, final_state["messages"], final_state.get("steps", 0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CATEGORY_CONFIGS)) as pool:
        results = list(pool.map(run_one, CATEGORY_CONFIGS))

    findings = {}
    sub_agent_failed = {}
    all_evidence_chain = []
    step = 1
    total_steps_used = 0

    for category, messages, steps_used in results:
        total_steps_used += steps_used
        for item in _build_evidence_chain(messages):
            item = dict(item)
            item["step"] = step
            item["investigator"] = category
            all_evidence_chain.append(item)
            step += 1

        finding = _extract_finding(messages)
        sub_agent_failed[category] = finding is None
        if finding is not None:
            finding["_verified"] = _VERIFIERS[category](finding, complaint)
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

    return {
        "complaint_id": complaint_id,
        "complaint": complaint,
        "evidence_chain": all_evidence_chain,
        "conclusion": conclusion,
        "conclusion_failed": conclusion_failed,
        "material_verification_failed": bool(findings.get("Material") and findings["Material"].get("supports_category")
                                              and not findings["Material"].get("_verified")),
        "verification_log": [],
        "sub_agent_failed": sub_agent_failed,
        "findings_by_category": findings,
        "steps_used": total_steps_used,
    }
