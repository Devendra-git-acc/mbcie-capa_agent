"""
Tool functions for the CAPA-style RCA agent.

Five tools, each independently testable with no LLM involved:
  - decode_batch_code        : parses "M02-2026W13" into machine_id + a
                                calendar date range. Deterministic arithmetic
                                like ISO week boundaries belongs in a tool,
                                not left to the model.
  - query_downtime            : downtime events for a machine in a date range
  - query_shifts               : shift/operator records for a machine in a
                                date range
  - query_complaints           : other complaints in a given week, for
                                spotting cross-machine patterns (storyline 3
                                needs this)
  - query_complaints_by_machine : ALL complaints for one machine across all
                                time, for spotting recurrence over weeks that
                                query_complaints' date window can't see
                                (storyline 4, gradual mold wear, needs this --
                                there's no single discrete record, only a
                                pattern spread across non-adjacent weeks)

These are plain Python functions wrapped with @tool at the bottom so
agent.py can bind them straight to the model. Kept framework-light on
purpose: everything above the @tool line is pure functions you can unit
test without LangGraph or an API key at all.
"""
import json
import re
from datetime import date, timedelta
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).parent / "data"

_complaints = json.loads((DATA_DIR / "complaints.json").read_text())
_downtime = json.loads((DATA_DIR / "downtime.json").read_text())
_shifts = json.loads((DATA_DIR / "shifts.json").read_text())

BATCH_RE = re.compile(r"^(M\d{2})-(\d{4})W(\d{2})$")


def _decode_batch_code(batch_code: str) -> dict:
    m = BATCH_RE.match(batch_code.strip())
    if not m:
        raise ValueError(f"Unrecognized batch code format: {batch_code!r}")
    machine_id, year, week = m.group(1), int(m.group(2)), int(m.group(3))
    monday = date.fromisocalendar(year, week, 1)
    sunday = monday + timedelta(days=6)
    return {
        "machine_id": machine_id,
        "year": year,
        "week": week,
        "start_date": monday.isoformat(),
        "end_date": sunday.isoformat(),
    }


def _query_downtime(machine_id: str, start_date: str, end_date: str) -> list[dict]:
    return [r for r in _downtime
            if r["machine_id"] == machine_id and start_date <= r["date"] <= end_date]


def _query_shifts(machine_id: str, start_date: str, end_date: str) -> list[dict]:
    return [r for r in _shifts
            if r["machine_id"] == machine_id and start_date <= r["date"] <= end_date]


def _query_complaints(start_date: str, end_date: str, exclude_complaint_id: str | None = None) -> list[dict]:
    return [c for c in _complaints
            if start_date <= c["date_reported"] <= end_date
            and c["complaint_id"] != exclude_complaint_id]


def _query_complaints_by_machine(machine_id: str, exclude_complaint_id: str | None = None) -> list[dict]:
    result = []
    for c in _complaints:
        if c["complaint_id"] == exclude_complaint_id:
            continue
        if _decode_batch_code(c["batch_code"])["machine_id"] == machine_id:
            result.append(c)
    return result


# --- LangChain tool wrappers -------------------------------------------
@tool
def decode_batch_code(batch_code: str) -> dict:
    """Decode a product batch/lot code (e.g. 'M02-2026W13') into the
    machine ID and the calendar date range (Mon-Sun) it was produced in.
    Always call this first for a new complaint -- you need the date range
    before downtime/shift lookups mean anything."""
    return _decode_batch_code(batch_code)


@tool
def query_downtime(machine_id: str, start_date: str, end_date: str) -> list[dict]:
    """Look up downtime events for one machine within a date range
    (YYYY-MM-DD, inclusive). Returns reason category, detail, shift, and
    duration for each stop in that window."""
    return _query_downtime(machine_id, start_date, end_date)


@tool
def query_shifts(machine_id: str, start_date: str, end_date: str) -> list[dict]:
    """Look up shift/operator assignments for one machine within a date
    range (YYYY-MM-DD, inclusive). Returns operator ID and experience level
    for each shift in that window."""
    return _query_shifts(machine_id, start_date, end_date)


@tool
def query_complaints(start_date: str, end_date: str, exclude_complaint_id: str = "") -> list[dict]:
    """Look up OTHER complaints reported within a date range, optionally
    excluding the one currently under investigation. Use this to check
    whether a defect is isolated to one machine or shows up across several
    -- a cross-machine pattern in the same window usually points to a
    shared input (material) rather than a single machine or operator."""
    return _query_complaints(start_date, end_date, exclude_complaint_id or None)


@tool
def query_complaints_by_machine(machine_id: str, exclude_complaint_id: str = "") -> list[dict]:
    """Look up EVERY complaint ever recorded for one machine, across all time
    -- not limited to a single production week. Use this when downtime and
    shift records for the current production window are clean (no discrete
    event, no staffing anomaly) and there's no cross-machine pattern either:
    a machine with multiple complaints spread across several non-adjacent
    weeks, with different operators and shifts each time, points to gradual
    equipment wear (e.g. a wearing mold or fixture) rather than a one-off
    triggering event -- there is no single record to cite for that, the
    recurrence itself is the evidence."""
    return _query_complaints_by_machine(machine_id, exclude_complaint_id or None)


ALL_TOOLS = [decode_batch_code, query_downtime, query_shifts, query_complaints, query_complaints_by_machine]


# --- Self-test, no LLM required -----------------------------------------
if __name__ == "__main__":
    decoded = _decode_batch_code("M02-2026W13")
    assert decoded["machine_id"] == "M02" and decoded["week"] == 13, decoded
    print("decode_batch_code:", decoded)

    dt = _query_downtime("M02", "2026-01-01", "2026-12-31")
    assert len(dt) > 0, "expected some M02 downtime across the year"
    print(f"query_downtime(M02, full year): {len(dt)} records")

    sh = _query_shifts("M05", *[decoded["start_date"], decoded["end_date"]])
    print(f"query_shifts(M05, W13): {len(sh)} records")

    # storyline check: the M02 calibration fix should show up in week 15
    fix_window = _decode_batch_code("M02-2026W15")
    fix_dt = _query_downtime("M02", fix_window["start_date"], fix_window["end_date"])
    cal_events = [r for r in fix_dt if "Calibration" in r["reason_detail"]]
    assert len(cal_events) == 1, f"expected exactly 1 calibration event in W15, got {len(cal_events)}"
    print("Storyline check (M02 W15 calibration event):", cal_events[0])

    # storyline check: material batch complaints should co-occur across M01 + M03 in week 17
    cross = _query_complaints("2026-04-19", "2026-04-26")
    machines_hit = {_decode_batch_code(c["batch_code"])["machine_id"] for c in cross}
    print("Machines with complaints in W17 window:", machines_hit)
    assert {"M01", "M03"}.issubset(machines_hit), machines_hit

    # storyline check: mold-wear (M04) should show recurrence across several
    # non-adjacent weeks when queried across all time, not just one window
    m04_history = _query_complaints_by_machine("M04")
    m04_weeks = {_decode_batch_code(c["batch_code"])["week"] for c in m04_history}
    print(f"query_complaints_by_machine(M04): {len(m04_history)} complaints across weeks {sorted(m04_weeks)}")
    assert len(m04_history) >= 3 and len(m04_weeks) >= 3, \
        f"expected mold-wear recurrence across several weeks, got {m04_weeks}"

    print("\nAll tool self-tests passed.")
