"""
LangGraph ReAct agent for CAPA-style root-cause investigation.

Graph shape: two nodes, "agent" and "tools", looping until the model calls
the dedicated submit_conclusion tool (or the step cap is hit). Using a
tool call as the termination signal -- rather than parsing free text --
keeps the final answer structured, matching how the rest of this project
is built (task 2 leans on structured output for the same reason).

Nothing about the evidence chain is hidden in framework state: every tool
call and its result lives in `messages`, and build_evidence_chain() below
reads that list straight out. That's what run_single.py serializes.
"""
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS

load_dotenv()
# NOTE: reads provider/key/model env vars directly rather than from
# shared/config.py so this stays runnable standalone (`cd rca && python
# run_single.py`). Once main.py (Day 2) puts the repo root on sys.path,
# switch this to `from shared.config import ...`.

DATA_DIR = Path(__file__).parent / "data"
_complaints_by_id = {c["complaint_id"]: c for c in json.loads((DATA_DIR / "complaints.json").read_text())}

MAX_STEPS = 8

SYSTEM_PROMPT = """You are a CAPA (Corrective and Preventive Action) investigator at a combined tyre \
and cycle (bicycle) manufacturing plant. You are given one customer/dealer defect complaint. Your job \
is to determine the root cause by autonomously consulting three internal record systems: downtime logs, \
shift/operator assignment logs, and other complaints (both same-week and this machine's full history).

Plant context:
- Product batches are identified by a batch code like "M02-2026W13": machine M02, ISO week 13 of 2026.
- Root causes here fall into standard CAPA categories: Process, Material, Equipment, Human, Scheduling \
-- or "Insufficient evidence" if the three available sources genuinely don't support a conclusion.

Investigation method:
1. Always decode the complaint's batch code first, to get the machine and its production date window.
2. Gather all four signals before concluding anything, in this order:
   a. Downtime records for the machine -- query a window that extends about 2 weeks PAST the production \
window's end date too, not just the exact week: calibration drift and similar equipment issues are often \
only caught and logged as a fix some time after the defective batch already shipped, so a relevant record \
can post-date the batch itself. Check whether a later record describes catching/fixing something that would \
already have been affecting production before that date.
   b. Shift records for the machine in the production window -- specifically look for operator_experience \
== "trainee".
   c. Other complaints in the SAME date window on a DIFFERENT machine_id (not this one).
   d. This machine's full complaint history across all time, for recurrence over several non-adjacent weeks.
3. Weigh the signals in this priority order -- use the FIRST one that applies, don't let a later, weaker \
signal override an earlier, specific one:
   a. A trainee-staffed shift (2b) on the exact machine/window is a specific, attributable cause -> "Human". \
Conclude this confidently even if other complaints also exist for the same batch -- a trainee-staffing issue \
and multiple customers receiving the resulting defective units are not mutually exclusive; the trainee signal \
is more specific and should win.
   b. A discrete downtime/maintenance event (2a) that explains the defect (including a later fix that reveals \
an earlier-present problem, per 2a) -> "Equipment" or "Process" as fits.
   c. Complaints on a genuinely DIFFERENT machine_id in the same window (2c), with a matching or clearly \
similar defect_description to the complaint you're investigating (the same symptom, e.g. both describing \
the same kind of defect) -> "Material" (a shared input feeding multiple lines). Multiple complaints in the \
same date range is not enough by itself -- with 18 complaints spread across a year, unrelated storylines' \
windows sometimes overlap by pure calendar coincidence; a different-machine complaint describing a clearly \
DIFFERENT symptom is a coincidence to ignore, not evidence. Other complaints that reference the exact same \
batch code on the SAME machine are also NOT this signal -- that's just multiple customers reporting one \
defective batch, which is consistent with every category equally and proves nothing about which one. Never \
use same-machine complaint volume, same-week-but-different-symptom complaints, or a downtime record's \
category alone, as Material evidence -- only a different machine_id WITH a matching symptom counts.
   d. Recurrence on the SAME machine across several non-adjacent weeks (2d), with no discrete cause in 2a/2b \
and no cross-machine pattern in 2c -> "Equipment" (gradual wear, e.g. a degrading mold/fixture). There is no \
single record to cite here -- list the recurring complaint IDs and weeks as the evidence.
   e. If none of a-d apply -- downtime clean, shifts clean, no cross-machine pattern, no recurrence -- \
conclude "Insufficient evidence", even if multiple people complained about the same batch. Complaint volume \
by itself is never a root-cause category; a confident wrong answer is worse than an honest "can't determine \
this from available records" here.
4. When ready, call submit_conclusion. Cite the actual record IDs (downtime_id / shift_id / complaint_id) \
your conclusion rests on wherever you have them, or the recurring complaint IDs if that's the evidence."""


@tool
def submit_conclusion(root_cause_category: str, root_cause_hypothesis: str,
                       confidence: str, recommended_corrective_action: str) -> dict:
    """Call this once your investigation is complete, to submit your final root-cause conclusion.
    root_cause_category must be one of: Process, Material, Equipment, Human, Scheduling, Insufficient evidence.
    confidence must be one of: high, medium, low.
    root_cause_hypothesis should cite specific record IDs wherever your conclusion is based on one."""
    return {
        "root_cause_category": root_cause_category,
        "root_cause_hypothesis": root_cause_hypothesis,
        "confidence": confidence,
        "recommended_corrective_action": recommended_corrective_action,
    }


def _build_model() -> ChatOpenAI:
    """Provider toggle, switched purely by env var (LLM_PROVIDER) so the same
    code runs against paid OpenAI or a free OpenRouter model with no code
    change -- just a different .env. OpenRouter speaks the OpenAI-compatible
    API, so this only needs a base_url + a different key, no new dependency.

    Caveat: forced tool-calling (tool_choice="submit_conclusion" below) is
    reliable on OpenAI's function-calling models but not guaranteed on every
    free OpenRouter model -- if submit_conclusion never gets forced-called at
    the step cap, that's the first thing to check when testing a new
    OPENROUTER_MODEL."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        return ChatOpenAI(
            model=model,
            temperature=0,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://github.com"),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "mbcie-rca-agent"),
            },
        )
    elif provider == "openai":
        return ChatOpenAI(model=os.environ.get("RCA_MODEL", "gpt-4o-mini"), temperature=0)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}: expected 'openai' or 'openrouter'")


_model = _build_model()
_model_investigating = _model.bind_tools(ALL_TOOLS + [submit_conclusion])
_model_concluding = _model.bind_tools([submit_conclusion], tool_choice="submit_conclusion")

_tool_node = ToolNode(ALL_TOOLS)  # submit_conclusion deliberately excluded -- see route_after_agent


class InvestigationState(MessagesState):
    steps: int


def _invoke_forced_conclusion(messages):
    """Call the forced submit_conclusion tool_choice, retrying once if the
    model returns no tool call at all. Observed directly during testing: the
    identical free-tier model succeeded on one run of a complaint and failed
    (empty response, no tool_calls) on another run of the SAME complaint --
    stochastic flakiness, not a hard incompatibility, so a retry is the right
    fix rather than giving up after one attempt."""
    response = _model_concluding.invoke(messages)
    if not getattr(response, "tool_calls", None):
        response = _model_concluding.invoke(messages)
    return response


def call_agent(state: InvestigationState) -> dict:
    steps = state.get("steps", 0)
    messages = state["messages"]
    if steps >= MAX_STEPS:
        forced = HumanMessage(content="You've reached the investigation step limit. Call submit_conclusion "
                                       "now with your best hypothesis, or 'Insufficient evidence' if you "
                                       "can't support one.")
        response = _invoke_forced_conclusion(messages + [forced])
    else:
        response = _model_investigating.invoke(messages)
        if not getattr(response, "tool_calls", None):
            # Observed directly, even on gpt-4o-mini (not just flaky free
            # models): the model sometimes narrates its conclusion as plain
            # text -- reasoning through the right answer and literally
            # writing "I will now submit the conclusion" -- without actually
            # making the tool call. route_after_agent treats any no-tool-call
            # response as terminal, so without this, a well-reasoned answer
            # would be silently thrown away as conclusion_failed well before
            # MAX_STEPS. Force a real submit_conclusion call from the same
            # context instead of accepting narrated intent as a dead end.
            forced = HumanMessage(content="Call submit_conclusion now with that conclusion -- the summary "
                                           "you just wrote is not a structured answer.")
            response = _invoke_forced_conclusion(messages + [response, forced])
    return {"messages": [response], "steps": steps + 1}


def route_after_agent(state: InvestigationState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls:
        return END
    if any(tc["name"] == "submit_conclusion" for tc in tool_calls):
        return END
    return "tools"


_graph_builder = StateGraph(InvestigationState)
_graph_builder.add_node("agent", call_agent)
_graph_builder.add_node("tools", _tool_node)
_graph_builder.add_edge(START, "agent")
_graph_builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
_graph_builder.add_edge("tools", "agent")
graph = _graph_builder.compile()


def build_evidence_chain(messages) -> list[dict]:
    """Walk the message history and pair each investigation tool call with
    its result, in order. This -- not anything inside LangGraph's internal
    state -- is the evidence chain returned to the caller."""
    by_call_id = {}
    for m in messages:
        if getattr(m, "type", None) == "tool":
            by_call_id[m.tool_call_id] = m.content

    chain = []
    step = 1
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"] == "submit_conclusion":
                continue
            chain.append({
                "step": step,
                "tool": tc["name"],
                "args": tc["args"],
                "result": by_call_id.get(tc["id"]),
            })
            step += 1
    return chain


def investigate(complaint_id: str) -> dict:
    complaint = _complaints_by_id.get(complaint_id)
    if complaint is None:
        raise ValueError(f"Unknown complaint_id: {complaint_id}")

    initial_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Investigate this complaint:\n{json.dumps(complaint, indent=2)}"),
    ]
    final_state = graph.invoke({"messages": initial_messages, "steps": 0},
                                config={"recursion_limit": MAX_STEPS * 2 + 4})

    messages = final_state["messages"]
    evidence_chain = build_evidence_chain(messages)

    conclusion = None
    last = messages[-1]
    for tc in getattr(last, "tool_calls", None) or []:
        if tc["name"] == "submit_conclusion":
            conclusion = tc["args"]
    conclusion_failed = conclusion is None
    if conclusion is None:
        # The model failed to produce the forced submit_conclusion tool call
        # (seen on some free-tier OpenRouter models under tool_choice). This
        # can happen to land on "Insufficient evidence" by coincidence, but
        # it is NOT a real conclusion -- callers must check conclusion_failed
        # before treating root_cause_category as a genuine answer, or a hit-rate
        # score will silently count a broken run as a correct one.
        conclusion = {"root_cause_category": "Insufficient evidence",
                      "root_cause_hypothesis": "Agent did not reach a structured conclusion.",
                      "confidence": "low",
                      "recommended_corrective_action": "Re-run or investigate manually.",
                      "raw_final_message": getattr(last, "content", None)}

    return {
        "complaint_id": complaint_id,
        "complaint": complaint,
        "evidence_chain": evidence_chain,
        "conclusion": conclusion,
        "conclusion_failed": conclusion_failed,
        "steps_used": final_state.get("steps", 0),
    }
