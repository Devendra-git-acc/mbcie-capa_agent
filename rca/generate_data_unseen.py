"""
Generates a SECOND, held-out batch of synthetic complaints/downtime/shifts
-- same category distribution and storyline shapes as the original
18-complaint dataset (generate_data.py), on entirely new machines and
weeks, with deliberately different symptom wording -- specifically to
test whether the agent's design (prompt + mechanical verification)
generalizes, or whether it was only ever tuned to fit the exact 18
complaints every fix in this project was diagnosed against.

Design constraints, on purpose:
  - Appends to (does not replace) the existing complaints.json/
    downtime.json/shifts.json -- the original 18 complaints and every ID
    referenced throughout DAY1-3_STATUS.md, the journal, and every prior
    eval_results.json stay byte-identical. This script only adds new
    records with fresh, non-colliding IDs (continuing from the existing
    max: D0131 -> D0132+, S1260 -> S1261+, C018 -> C019+).
  - The new ground truth goes to a SEPARATE file, answer_key_unseen.json,
    not merged into answer_key.json -- so it's unambiguous which
    complaints were part of the tuned-against set and which are the
    held-out generalization test. eval.py reads whichever file the
    ANSWER_KEY env var points at (defaults to answer_key.json).
  - Same category proportions as the original 18 (9 Equipment [4
    calibration-drift + 5 mold-wear], 3 Human, 4 Material, 2 red herring)
    for a directly comparable percentage, not just "some more data."
  - Deliberately different defect_description wording per storyline (not
    the same string with the machine ID swapped) -- Material's mechanical
    verifier checks exact string equality of defect_description across
    machines, so reusing the *exact* original wording would make this a
    weaker test of generalization than genuinely new phrasing that still
    has to independently satisfy the same check.

Run with:
    cd rca && python generate_data_unseen.py
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(999)  # deliberately different from generate_data.py's seed(42) -- an
                   # independent draw, not a continuation of the same stream

DATA_DIR = Path(__file__).parent / "data"
YEAR = 2026
WEEKS = list(range(20, 32))  # 12 weeks: W20-W31, past the original W10-W19 range
SHIFTS = ["morning", "evening", "night"]
DEALERS = ["Dealer-Pune-02", "Dealer-Nashik-04", "Dealer-Surat-01", "Dealer-Indore-07",
           "Dealer-Ahmedabad-03", "Dealer-Nagpur-05", "Dealer-Rajkot-02", "Dealer-Vadodara-01"]

MACHINES = {
    "M07": {"type": "Tyre Curing Press",     "line": "tyre"},
    "M08": {"type": "Frame Welding Station", "line": "cycle"},
    "M09": {"type": "Tyre Building Machine", "line": "tyre"},
    "M10": {"type": "Tyre Curing Press",     "line": "tyre"},
    "M11": {"type": "Tyre Building Machine", "line": "tyre"},
    "M12": {"type": "Wheel Assembly Station","line": "cycle"},
}

DOWNTIME_REASONS = {
    "Setup":        ["Batch changeover", "Tooling change", "First-article inspection"],
    "Mechanical":   ["Belt drive fault", "Sensor fault", "Hydraulic leak"],
    "Maintenance":  ["Scheduled PM", "Calibration check", "Part replacement"],
    "Material":     ["Waiting on compound delivery", "Staging delay"],
    "Quality Hold": ["Scrap review", "In-process inspection hold"],
    "Staffing":     ["Shift handover delay", "Break coverage gap"],
}

# New operator pool, IDs continuing past the original's OP001-OP015
OPERATORS = [
    {"operator_id": f"OP{i:03d}", "experience": "experienced", "experience_months": random.randint(24, 96)}
    for i in range(16, 28)
] + [
    {"operator_id": f"OP{i:03d}", "experience": "trainee", "experience_months": random.randint(1, 5)}
    for i in range(28, 31)
]


def batch_code(machine_id: str, week: int) -> str:
    return f"{machine_id}-{YEAR}W{week:02d}"


def week_dates(week: int) -> list[date]:
    monday = date.fromisocalendar(YEAR, week, 1)
    return [monday + timedelta(days=i) for i in range(7)]


def pick_operator(experience: str | None = None) -> dict:
    pool = [o for o in OPERATORS if experience is None or o["experience"] == experience]
    return random.choice(pool)


# --- Load existing data to continue ID sequences without collision --------
existing_downtime = json.loads((DATA_DIR / "downtime.json").read_text())
existing_shifts = json.loads((DATA_DIR / "shifts.json").read_text())
existing_complaints = json.loads((DATA_DIR / "complaints.json").read_text())

_downtime_seq = max(int(d["downtime_id"][1:]) for d in existing_downtime) + 1
_shift_seq = max(int(s["shift_id"][1:]) for s in existing_shifts) + 1
_complaint_seq = max(int(c["complaint_id"][1:]) for c in existing_complaints) + 1

downtime_records = []
shift_records = []
complaints = []
answer_key_unseen = {}


# ---------------------------------------------------------------------------
# 1. Background noise for the new machines
# ---------------------------------------------------------------------------
for machine_id in MACHINES:
    for week in WEEKS:
        days = week_dates(week)
        for _ in range(random.randint(1, 3)):
            d = random.choice(days)
            category = random.choice(["Setup", "Maintenance", "Material", "Staffing"])
            downtime_records.append({
                "downtime_id": f"D{_downtime_seq:04d}",
                "machine_id": machine_id,
                "date": d.isoformat(),
                "shift": random.choice(SHIFTS),
                "duration_minutes": random.choice([15, 20, 30, 45, 60]),
                "reason_category": category,
                "reason_detail": random.choice(DOWNTIME_REASONS[category]),
            })
            _downtime_seq += 1
        for d in days:
            for shift in SHIFTS:
                op = pick_operator("experienced")
                shift_records.append({
                    "shift_id": f"S{_shift_seq:04d}",
                    "machine_id": machine_id,
                    "date": d.isoformat(),
                    "shift": shift,
                    "operator_id": op["operator_id"],
                    "operator_experience": op["experience"],
                    "operator_experience_months": op["experience_months"],
                })
                _shift_seq += 1


def add_downtime(machine_id, d, shift, category, detail, duration=45):
    global _downtime_seq
    rec = {"downtime_id": f"D{_downtime_seq:04d}", "machine_id": machine_id, "date": d.isoformat(),
           "shift": shift, "duration_minutes": duration, "reason_category": category, "reason_detail": detail}
    downtime_records.append(rec)
    _downtime_seq += 1
    return rec["downtime_id"]


def overwrite_shift(machine_id, d, shift, operator_id, experience, months):
    for rec in shift_records:
        if rec["machine_id"] == machine_id and rec["date"] == d.isoformat() and rec["shift"] == shift:
            rec["operator_id"] = operator_id
            rec["operator_experience"] = experience
            rec["operator_experience_months"] = months
            return rec["shift_id"]
    raise ValueError("no matching background shift to overwrite")


def add_complaint(machine_id, week, defect_desc, product_line, severity="medium"):
    global _complaint_seq
    d = random.choice(week_dates(week)[2:])
    cid = f"C{_complaint_seq:03d}"
    complaints.append({
        "complaint_id": cid, "date_reported": d.isoformat(), "batch_code": batch_code(machine_id, week),
        "product_line": product_line, "defect_description": defect_desc,
        "customer_or_dealer": random.choice(DEALERS), "severity": severity,
    })
    _complaint_seq += 1
    return cid


# ---------------------------------------------------------------------------
# 2. Five new storylines, same shape and proportions as the original,
#    deliberately different symptom wording and machines
# ---------------------------------------------------------------------------

# --- Storyline 6: Equipment / calibration drift, analog of storyline 1 ----
fix_date = week_dates(23)[1]  # must land within 14 days of the EARLIEST complaint week's (21) end date,
                               # not just the latest -- week 24 was too late, caught by validate_dataset.py
fix_id_6 = add_downtime("M07", fix_date, "evening", "Maintenance",
                         "Calibration check - curing press pressure sensor recalibrated after drift found",
                         duration=90)
s6_ids = []
for wk, n in [(21, 2), (22, 2)]:
    for _ in range(n):
        cid = add_complaint("M07", wk,
                             "Sidewall blistering with air trapped beneath the rubber surface, discovered "
                             "during routine post-cure inspection; consistent with an under-vulcanized batch.",
                             "tyre", severity="high")
        s6_ids.append(cid)
        answer_key_unseen[cid] = {
            "root_cause_category": "Equipment",
            "root_cause": "Curing press (M07) pressure sensor calibration drifted low, producing an "
                           "under-vulcanized batch and trapped-air blistering. Drift was undetected until the "
                           "week 23 calibration check.",
            "supporting_downtime_ids": [fix_id_6],
            "supporting_shift_ids": [],
            "notes": "Analog of the original storyline 1 (M02, calibration drift) -- new machine, new week, "
                     "different wording (pressure vs temperature sensor, blistering vs tread separation).",
        }

# --- Storyline 7: Human / trainee operator, analog of storyline 2 --------
trainee_7 = pick_operator("trainee")
s7_shift_ids = []
for d in week_dates(23)[3:5]:
    sid = overwrite_shift("M08", d, "night", trainee_7["operator_id"], "trainee", trainee_7["experience_months"])
    s7_shift_ids.append(sid)
s7_ids = []
for _ in range(3):
    cid = add_complaint("M08", 23,
                         "Weld porosity visible at chainstay/bottom-bracket joint during final QC; "
                         "inconsistent penetration noted, bead appears cold.",
                         "cycle", severity="high")
    s7_ids.append(cid)
    answer_key_unseen[cid] = {
        "root_cause_category": "Human",
        "root_cause": "Frame welding (M08) night shift in week 23 was staffed by a trainee operator "
                      f"({trainee_7['operator_id']}, {trainee_7['experience_months']} months experience) with "
                      "no senior sign-off logged. No mechanical or maintenance event on M08 that week.",
        "supporting_downtime_ids": [],
        "supporting_shift_ids": s7_shift_ids,
        "notes": "Analog of the original storyline 2 (M05, trainee welder) -- new machine, new week, "
                 "different weld-defect wording (porosity/cold bead vs crack/under-penetration).",
    }

# --- Storyline 8: Material / shared batch, cross-machine, analog of storyline 3 ---
s8_ids = []
for machine_id in ["M09", "M10"]:
    for _ in range(2):
        cid = add_complaint(machine_id, 25,
                             "Bead area cracking observed near the rim seat shortly after mounting; no "
                             "installation damage reported by fitter.",
                             "tyre", severity="high")
        s8_ids.append(cid)
        answer_key_unseen[cid] = {
            "root_cause_category": "Material",
            "root_cause": "Both M09 and M10 (different machines, different operators, no downtime events) "
                           "produced affected batches in week 25, pointing to a shared input rather than a "
                           "machine or operator issue -- consistent with a defective incoming bead-wire lot "
                           "feeding both lines that week.",
            "supporting_downtime_ids": [],
            "supporting_shift_ids": [],
            "notes": "Analog of the original storyline 3 (M01+M03, compound batch) -- new machines, new week, "
                     "different symptom wording (bead-area cracking vs sidewall cracking).",
        }

# --- Storyline 9: Equipment / gradual wear, analog of storyline 4 ---------
s9_ids = []
for wk in [21, 23, 26, 28, 30]:
    cid = add_complaint("M11", wk,
                         "Slight tread thickness variation noted across circumference during QC scan; "
                         "localized thinning on one shoulder, otherwise within tolerance.",
                         "tyre", severity="low")
    s9_ids.append(cid)
    answer_key_unseen[cid] = {
        "root_cause_category": "Equipment",
        "root_cause": "Recurring low-severity complaints on M11 spread across 5 non-adjacent weeks, multiple "
                       "different shifts and operators, with no discrete downtime or maintenance event and no "
                       "shared material batch -- pattern is consistent with gradual mold wear rather than a "
                       "single triggering event.",
        "supporting_downtime_ids": [],
        "supporting_shift_ids": [],
        "notes": "Analog of the original storyline 4 (M04, mold wear) -- new machine, new weeks, different "
                 "wording (thickness variation vs uneven tread wear).",
    }

# --- Storyline 10: Red herring, analog of storyline 5 ---------------------
s10_ids = []
for _ in range(2):
    cid = add_complaint("M12", 27,
                         "Front hub bearing grinding noise reported by customer after roughly two weeks of "
                         "use; no visible damage found on inspection.",
                         "cycle", severity="medium")
    s10_ids.append(cid)
    answer_key_unseen[cid] = {
        "root_cause_category": "Insufficient evidence",
        "root_cause": "No anomaly in downtime or shift records for M12 in week 27: experienced operator, no "
                       "mechanical or maintenance events, no cross-machine pattern. The true cause (if any) "
                       "sits outside these three sources -- most likely an incoming bearing lot defect, which "
                       "this plant's systems don't track.",
        "supporting_downtime_ids": [],
        "supporting_shift_ids": [],
        "notes": "Analog of the original storyline 5 (M06, red herring) -- new machine, new week, different "
                 "wording (front hub, two weeks of use vs wheel bearing, first week of use).",
    }

# ---------------------------------------------------------------------------
# 3. Sanitize accidental contamination, same discipline as generate_data.py
# ---------------------------------------------------------------------------
SEARCH_PADDING_DAYS = 14
SUSPICIOUS_CATEGORIES = {"Mechanical"}
SUSPICIOUS_DETAILS = {"Calibration check"}


def _protected_window(machine_id, week):
    days = week_dates(week)
    return days[0].isoformat(), (days[-1] + timedelta(days=SEARCH_PADDING_DAYS)).isoformat()


def sanitize_clean_windows():
    protected_downtime_ids = {rid for t in answer_key_unseen.values() for rid in t.get("supporting_downtime_ids", [])}
    protected_shift_ids = {rid for t in answer_key_unseen.values() for rid in t.get("supporting_shift_ids", [])}
    fixed = []
    for cid, truth in answer_key_unseen.items():
        complaint = next(c for c in complaints if c["complaint_id"] == cid)
        machine_id = complaint["batch_code"].split("-")[0]
        week = int(complaint["batch_code"].split("W")[1])
        start, end = _protected_window(machine_id, week)

        if not truth.get("supporting_downtime_ids"):
            for d in downtime_records:
                if d["downtime_id"] in protected_downtime_ids:
                    continue
                if d["machine_id"] == machine_id and start <= d["date"] <= end and \
                        (d["reason_category"] in SUSPICIOUS_CATEGORIES or d["reason_detail"] in SUSPICIOUS_DETAILS):
                    fixed.append(f"{cid}: sanitized {d['downtime_id']}")
                    d["reason_category"] = "Staffing"
                    d["reason_detail"] = "Shift handover delay"

        if not truth.get("supporting_shift_ids"):
            for s in shift_records:
                if s["shift_id"] in protected_shift_ids:
                    continue
                if s["machine_id"] == machine_id and start <= s["date"] <= end and \
                        s["operator_experience"] == "trainee":
                    replacement = pick_operator("experienced")
                    fixed.append(f"{cid}: sanitized {s['shift_id']}")
                    s["operator_id"] = replacement["operator_id"]
                    s["operator_experience"] = "experienced"
                    s["operator_experience_months"] = replacement["experience_months"]
    return fixed


_sanitize_log = sanitize_clean_windows()

# ---------------------------------------------------------------------------
# 4. Merge into the existing data files (so tools.py can query the new
#    machines/complaints with zero code changes) and write the held-out
#    answer key SEPARATELY
# ---------------------------------------------------------------------------
merged_downtime = existing_downtime + downtime_records
merged_shifts = existing_shifts + shift_records
merged_complaints = existing_complaints + complaints

merged_downtime.sort(key=lambda r: (r["date"], r["machine_id"]))
merged_shifts.sort(key=lambda r: (r["date"], r["machine_id"], r["shift"]))
merged_complaints.sort(key=lambda r: r["complaint_id"])

(DATA_DIR / "downtime.json").write_text(json.dumps(merged_downtime, indent=2))
(DATA_DIR / "shifts.json").write_text(json.dumps(merged_shifts, indent=2))
(DATA_DIR / "complaints.json").write_text(json.dumps(merged_complaints, indent=2))
(Path(__file__).parent / "answer_key_unseen.json").write_text(json.dumps(answer_key_unseen, indent=2))

if _sanitize_log:
    print(f"Sanitized {len(_sanitize_log)} accidental noise collision(s):")
    for line in _sanitize_log:
        print("  -", line)
    print()
else:
    print("No accidental noise collisions found in the new storylines' clean windows.\n")

print(f"New complaints added:      {len(complaints)}  (total now: {len(merged_complaints)})")
print(f"New downtime records:      {len(downtime_records)}  (total now: {len(merged_downtime)})")
print(f"New shift records:         {len(shift_records)}  (total now: {len(merged_shifts)})")
print(f"Held-out answer key entries: {len(answer_key_unseen)}  -> rca/answer_key_unseen.json")
print()
print("New storylines: calibration-drift x%d, trainee-operator x%d, material-batch x%d, "
      "mold-wear x%d, red-herring x%d" % (len(s6_ids), len(s7_ids), len(s8_ids), len(s9_ids), len(s10_ids)))
