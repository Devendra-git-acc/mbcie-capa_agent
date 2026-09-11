"""
Integrity check for the Task 1 synthetic dataset, independent of any LLM.

Not "does the agent get the right answer" (untestable here, no API access)
-- this checks something more basic and more important: is the ground
truth actually clean? Specifically, could the RANDOM background noise
have accidentally planted a false signal in a window that's supposed to
be clean, which would corrupt the test regardless of how good the agent
is?
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).parent
complaints = json.loads((BASE / "data" / "complaints.json").read_text())
downtime = json.loads((BASE / "data" / "downtime.json").read_text())
shifts = json.loads((BASE / "data" / "shifts.json").read_text())
# ANSWER_KEY=answer_key_unseen.json validates the held-out generalization set
# (rca/generate_data_unseen.json) instead of the original tuned-against one --
# same env-var-switch pattern as eval.py's AGENT_IMPL.
answer_key = json.loads((BASE / os.environ.get("ANSWER_KEY", "answer_key.json")).read_text())

import re
from datetime import date, timedelta
BATCH_RE = re.compile(r"^(M\d{2})-(\d{4})W(\d{2})$")


def decode(batch_code):
    m = BATCH_RE.match(batch_code)
    machine_id, year, week = m.group(1), int(m.group(2)), int(m.group(3))
    monday = date.fromisocalendar(year, week, 1)
    return machine_id, monday.isoformat(), (monday + timedelta(days=6)).isoformat()


SUSPICIOUS_CATEGORIES = {"Mechanical"}  # a real discrete-failure signal
SUSPICIOUS_DETAILS = {"Calibration check"}  # would look like a competing calibration story
SEARCH_PADDING_DAYS = 14  # must match agent.py's system-prompt instruction

issues = []
checked = 0

by_complaint = {c["complaint_id"]: c for c in complaints}

for cid, truth in answer_key.items():
    complaint = by_complaint[cid]
    machine_id, start, _own_week_end = decode(complaint["batch_code"])
    end = (date.fromisoformat(_own_week_end) + timedelta(days=SEARCH_PADDING_DAYS)).isoformat()
    checked += 1

    window_downtime = [d for d in downtime if d["machine_id"] == machine_id and start <= d["date"] <= end]
    window_shifts = [s for s in shifts if s["machine_id"] == machine_id and start <= s["date"] <= end]
    trainee_shifts = [s for s in window_shifts if s["operator_experience"] == "trainee"]
    suspicious_downtime = [d for d in window_downtime
                            if d["reason_category"] in SUSPICIOUS_CATEGORIES
                            or d["reason_detail"] in SUSPICIOUS_DETAILS]

    category = truth["root_cause_category"]
    expected_downtime_ids = set(truth.get("supporting_downtime_ids", []))
    expected_shift_ids = set(truth.get("supporting_shift_ids", []))

    # 1. Every ID the answer key claims as supporting evidence must actually
    #    exist, in the window, on the right machine.
    for did in expected_downtime_ids:
        if did not in {d["downtime_id"] for d in window_downtime}:
            issues.append(f"{cid}: claimed supporting downtime {did} not found in {machine_id} window {start}..{end}")
    for sid in expected_shift_ids:
        if sid not in {s["shift_id"] for s in window_shifts}:
            issues.append(f"{cid}: claimed supporting shift {sid} not found in {machine_id} window {start}..{end}")

    # 2. Storylines that are supposed to be clean on downtime must actually
    #    be clean -- any suspicious event not already accounted for is a
    #    contamination bug.
    if category in {"Human", "Material", "Equipment"} and not expected_downtime_ids:
        unexplained = [d for d in suspicious_downtime if d["downtime_id"] not in expected_downtime_ids]
        if unexplained:
            issues.append(f"{cid} ({category}, meant to have no downtime signal): found unexplained "
                           f"suspicious downtime {[d['downtime_id'] for d in unexplained]} "
                           f"({[d['reason_detail'] for d in unexplained]})")

    # 3. Storylines that are supposed to be clean on shifts (i.e. not the
    #    trainee-operator story) must not have a trainee show up by chance.
    if category != "Human" and not expected_shift_ids:
        if trainee_shifts:
            issues.append(f"{cid} ({category}, meant to have no shift signal): found unexplained trainee "
                           f"shift(s) {[s['shift_id'] for s in trainee_shifts]}")

    # 4. Red herring must be clean on EVERYTHING -- extra strict check.
    if category == "Insufficient evidence":
        if suspicious_downtime:
            issues.append(f"{cid} (red herring): suspicious downtime present: "
                           f"{[d['reason_detail'] for d in suspicious_downtime]}")
        if trainee_shifts:
            issues.append(f"{cid} (red herring): trainee shift present: {[s['shift_id'] for s in trainee_shifts]}")

# 5. Storyline-3 specific: verify the cross-machine pattern actually shows
#    up in a single complaints query (this is what the agent's
#    query_complaints tool needs to find).
material_cids = [cid for cid, t in answer_key.items() if t["root_cause_category"] == "Material"]
material_dates = [by_complaint[cid]["date_reported"] for cid in material_cids]
expected_material_machines = {decode(by_complaint[cid]["batch_code"])[0] for cid in material_cids}
w_start, w_end = min(material_dates), max(material_dates)
cross_machines = set()
for c in complaints:
    if w_start <= c["date_reported"] <= w_end:
        cross_machines.add(decode(c["batch_code"])[0])
if not expected_material_machines.issubset(cross_machines):
    issues.append(f"Material storyline: {expected_material_machines} not all visible in one date-range query "
                   f"({w_start}..{w_end}) -- found machines {cross_machines}")

# 6. Storyline-1 specific: the fix event must chronologically post-date
#    every complaint it's meant to explain.
calib_cids = [cid for cid, t in answer_key.items() if "calibration" in t["root_cause"].lower()]
for cid in calib_cids:
    complaint_date = by_complaint[cid]["date_reported"]
    fix_id = list(answer_key[cid]["supporting_downtime_ids"])[0]
    fix_date = next(d["date"] for d in downtime if d["downtime_id"] == fix_id)
    if fix_date <= complaint_date:
        issues.append(f"{cid}: calibration fix ({fix_date}) does not post-date the complaint ({complaint_date})")

print(f"Checked {checked} complaints against {len(downtime)} downtime records and {len(shifts)} shift records.\n")
if issues:
    print(f"FOUND {len(issues)} ISSUE(S):\n")
    for i in issues:
        print(" -", i)
else:
    print("No contamination or consistency issues found. All supporting-evidence IDs resolve correctly, "
          "'clean' storylines have no accidental suspicious downtime or trainee shifts, the material "
          "cross-machine pattern is queryable in one window, and the calibration fix event post-dates "
          "the complaints it explains.")
