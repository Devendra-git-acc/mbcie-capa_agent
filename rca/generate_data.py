"""
Synthetic dataset generator for the MBCIE Task 1 CAPA-style RCA agent.

Plant: a combined tyre + cycle (bicycle) manufacturing facility.
Produces three linked sources (complaints, downtime, shifts) joined by a
batch/lot code convention: "{machine_id}-{year}W{week:02d}".

Root-cause categories follow the standard manufacturing CAPA taxonomy:
Process / Material / Equipment / Human / Scheduling.
Downtime reason categories follow standard shop-floor practice:
Setup, Mechanical, Maintenance, Material, Quality Hold, Staffing.

Five storylines are planted (4 causal + 1 red herring). A hidden
answer_key.json records ground truth; it is NOT given to the agent at
runtime -- eval.py (Day 3) is the only consumer of it.

Deterministic (seeded) so it's reproducible and regenerable.
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

YEAR = 2026
WEEKS = list(range(10, 20))  # 10 weeks: W10-W19

MACHINES = {
    "M01": {"type": "Tyre Curing Press",     "line": "tyre"},
    "M02": {"type": "Tyre Curing Press",     "line": "tyre"},
    "M03": {"type": "Tyre Building Machine", "line": "tyre"},
    "M04": {"type": "Tyre Building Machine", "line": "tyre"},
    "M05": {"type": "Frame Welding Station", "line": "cycle"},
    "M06": {"type": "Wheel Assembly Station","line": "cycle"},
}

SHIFTS = ["morning", "evening", "night"]

# Mostly experienced operators, a handful of trainees -- realistic pool
OPERATORS = [
    {"operator_id": f"OP{i:03d}", "experience": "experienced", "experience_months": random.randint(24, 96)}
    for i in range(1, 13)
] + [
    {"operator_id": f"OP{i:03d}", "experience": "trainee", "experience_months": random.randint(1, 5)}
    for i in range(13, 16)
]

DEALERS = ["Dealer-Pune-02", "Dealer-Nashik-04", "Dealer-Surat-01", "Dealer-Indore-07",
           "Dealer-Ahmedabad-03", "Dealer-Nagpur-05", "Dealer-Rajkot-02"]

DOWNTIME_REASONS = {
    "Setup":        ["Batch changeover", "Tooling change", "First-article inspection"],
    "Mechanical":   ["Belt drive fault", "Sensor fault", "Hydraulic leak"],
    "Maintenance":  ["Scheduled PM", "Calibration check", "Part replacement"],
    "Material":     ["Waiting on compound delivery", "Staging delay"],
    "Quality Hold": ["Scrap review", "In-process inspection hold"],
    "Staffing":     ["Shift handover delay", "Break coverage gap"],
}


def batch_code(machine_id: str, week: int) -> str:
    return f"{machine_id}-{YEAR}W{week:02d}"


def week_dates(week: int) -> list[date]:
    monday = date.fromisocalendar(YEAR, week, 1)
    return [monday + timedelta(days=i) for i in range(7)]


def pick_operator(experience: str | None = None) -> dict:
    pool = [o for o in OPERATORS if experience is None or o["experience"] == experience]
    return random.choice(pool)


# ---------------------------------------------------------------------------
# 1. Background noise: routine downtime + routine shifts across all machines
# ---------------------------------------------------------------------------
downtime_records = []
shift_records = []
_downtime_seq = 1
_shift_seq = 1

for machine_id in MACHINES:
    for week in WEEKS:
        days = week_dates(week)
        # 1-3 routine downtime events per machine per week
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
        # every machine staffed on every shift, every day, by an experienced operator by default
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
    rec = {
        "downtime_id": f"D{_downtime_seq:04d}",
        "machine_id": machine_id,
        "date": d.isoformat(),
        "shift": shift,
        "duration_minutes": duration,
        "reason_category": category,
        "reason_detail": detail,
    }
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


# ---------------------------------------------------------------------------
# 2. Planted storylines
# ---------------------------------------------------------------------------
complaints = []
answer_key = {}
_complaint_seq = 1


def add_complaint(machine_id, week, defect_desc, product_line, severity="medium"):
    global _complaint_seq
    d = random.choice(week_dates(week)[2:])  # complaint lands a couple days into the week
    cid = f"C{_complaint_seq:03d}"
    complaints.append({
        "complaint_id": cid,
        "date_reported": d.isoformat(),
        "batch_code": batch_code(machine_id, week),
        "product_line": product_line,
        "defect_description": defect_desc,
        "customer_or_dealer": random.choice(DEALERS),
        "severity": severity,
    })
    _complaint_seq += 1
    return cid


# --- Storyline 1: Equipment / calibration drift on M02 (weeks 13-14) -------
# Curing press temperature sensor drifts low -> under-cured tread -> trapped
# air pockets -> tread separation. Caught and fixed in week 15.
fix_date = week_dates(15)[1]
fix_downtime_id = add_downtime("M02", fix_date, "evening", "Maintenance",
                                "Calibration check - curing press temperature sensor recalibrated after drift found",
                                duration=90)
s1_ids = []
for wk, n in [(13, 2), (14, 2)]:
    for _ in range(n):
        cid = add_complaint("M02", wk,
                             "Tread separation reported after short mileage; internal inspection shows air "
                             "pocket / blister beneath tread on inner shoulder, consistent with under-cure.",
                             "tyre", severity="high")
        s1_ids.append(cid)
        answer_key[cid] = {
            "root_cause_category": "Equipment",
            "root_cause": "Curing press (M02) temperature sensor calibration drifted low, producing "
                           "under-cured tread and trapped-air blistering. Drift was undetected until the "
                           "week 15 calibration check.",
            "supporting_downtime_ids": [fix_downtime_id],
            "supporting_shift_ids": [],
            "notes": "Fix event happens AFTER the complaints -- agent should notice the calibration downtime "
                     "record post-dates the batch and infer the drift was already present, not caused by the fix.",
        }

# --- Storyline 2: Human / trainee operator on M05 night shift (week 16) ---
trainee = pick_operator("trainee")
s2_shift_ids = []
for d in week_dates(16)[3:5]:
    sid = overwrite_shift("M05", d, "night", trainee["operator_id"], "trainee", trainee["experience_months"])
    s2_shift_ids.append(sid)
s2_ids = []
for _ in range(3):
    cid = add_complaint("M05", 16,
                         "Weld crack at frame down-tube/seat-tube joint discovered during final QC; "
                         "weld bead appears inconsistent / under-penetrated.",
                         "cycle", severity="high")
    s2_ids.append(cid)
    answer_key[cid] = {
        "root_cause_category": "Human",
        "root_cause": "Frame welding (M05) night shift in week 16 was staffed by a trainee operator "
                      f"({trainee['operator_id']}, {trainee['experience_months']} months experience) with no "
                      "senior sign-off logged. No mechanical or maintenance event on M05 that week.",
        "supporting_downtime_ids": [],
        "supporting_shift_ids": s2_shift_ids,
        "notes": "No downtime anomaly at all -- the only signal is in shift data. Tests whether the agent "
                 "keeps investigating after downtime comes back clean instead of stopping there.",
    }

# --- Storyline 3: Material / bad compound batch, cross-machine (week 17) --
# Same incoming rubber compound batch feeds M01 and M03 -- different
# machines, different operators, no downtime anomaly on either.
s3_ids = []
for machine_id in ["M01", "M03"]:
    for _ in range(2):
        cid = add_complaint(machine_id, 17,
                             "Sidewall cracking visible near mid-sidewall region shortly after entering "
                             "service; no impact damage reported by customer.",
                             "tyre", severity="high")
        s3_ids.append(cid)
        answer_key[cid] = {
            "root_cause_category": "Material",
            "root_cause": "Both M01 and M03 (different machines, different operators, no downtime events) "
                           "produced affected batches in week 17, pointing to a shared input rather than a "
                           "machine or operator issue -- consistent with a defective incoming rubber compound "
                           "lot feeding both lines that week.",
            "supporting_downtime_ids": [],
            "supporting_shift_ids": [],
            "notes": "Single-machine investigation looks clean on both M01 and M03 individually. The signal "
                     "only appears when comparing complaints ACROSS machines in the same week -- this is what "
                     "the query_complaints tool is for.",
        }

# --- Storyline 4: Equipment / gradual mold wear on M04 (spread out) -------
s4_ids = []
for wk in [11, 13, 15, 17, 19]:
    cid = add_complaint("M04", wk,
                         "Uneven tread wear pattern noted on inspection; localized shallow tread depth on "
                         "one shoulder, otherwise within spec.",
                         "tyre", severity="low")
    s4_ids.append(cid)
    answer_key[cid] = {
        "root_cause_category": "Equipment",
        "root_cause": "Recurring low-severity complaints on M04 spread across 5 non-adjacent weeks, multiple "
                       "different shifts and operators, with no discrete downtime or maintenance event and no "
                       "shared material batch -- pattern is consistent with gradual mold wear rather than a "
                       "single triggering event.",
        "supporting_downtime_ids": [],
        "supporting_shift_ids": [],
        "notes": "No single 'smoking gun' record -- root cause is inferred from the ABSENCE of a discrete "
                 "cause plus a recurring pattern over time. Hardest of the four to get right.",
    }

# --- Storyline 5: Red herring on M06 (week 18) -----------------------------
s5_ids = []
for _ in range(2):
    cid = add_complaint("M06", 18,
                         "Wheel bearing noise / roughness reported by customer within first week of use.",
                         "cycle", severity="medium")
    s5_ids.append(cid)
    answer_key[cid] = {
        "root_cause_category": "Insufficient evidence",
        "root_cause": "No anomaly in downtime or shift records for M06 in week 18: experienced operator, no "
                       "mechanical or maintenance events, no cross-machine pattern. The true cause (if any) "
                       "sits outside these three sources -- most likely an incoming bearing lot defect, which "
                       "this plant's systems don't track. Correct behaviour is to say so, not to force a "
                       "plausible-sounding answer from complaints/downtime/shifts alone.",
        "supporting_downtime_ids": [],
        "supporting_shift_ids": [],
        "notes": "This is the deliberate red herring. Every source comes back clean by design.",
    }

# ---------------------------------------------------------------------------
# 3. Sanitize accidental contamination in windows meant to stay clean
# ---------------------------------------------------------------------------
# A complaint's "search window" is its own production week PLUS this many
# days forward -- fixes/calibration checks are often logged after a defect
# surfaces downstream, not within the exact production week. agent.py's
# system prompt tells the agent to search this same padded window, so the
# data has to actually be clean (or signal-bearing) across that full span,
# not just the bare ISO week.
SEARCH_PADDING_DAYS = 14
SUSPICIOUS_CATEGORIES = {"Mechanical"}
SUSPICIOUS_DETAILS = {"Calibration check"}


def _protected_window(machine_id, week):
    days = week_dates(week)
    start = days[0]
    end = days[-1] + timedelta(days=SEARCH_PADDING_DAYS)
    return start.isoformat(), end.isoformat()


def sanitize_clean_windows():
    protected_downtime_ids = {rid for t in answer_key.values() for rid in t.get("supporting_downtime_ids", [])}
    protected_shift_ids = {rid for t in answer_key.values() for rid in t.get("supporting_shift_ids", [])}
    fixed = []

    for cid, truth in answer_key.items():
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
                    fixed.append(f"{cid}: sanitized {d['downtime_id']} ({d['reason_detail']} -> Staffing/Shift "
                                  f"handover delay) -- accidental noise collision in a window meant to stay clean")
                    d["reason_category"] = "Staffing"
                    d["reason_detail"] = "Shift handover delay"

        if not truth.get("supporting_shift_ids"):
            for s in shift_records:
                if s["shift_id"] in protected_shift_ids:
                    continue
                if s["machine_id"] == machine_id and start <= s["date"] <= end and \
                        s["operator_experience"] == "trainee":
                    replacement = pick_operator("experienced")
                    fixed.append(f"{cid}: sanitized {s['shift_id']} (trainee -> experienced) -- accidental "
                                  f"trainee placement in a window meant to stay clean")
                    s["operator_id"] = replacement["operator_id"]
                    s["operator_experience"] = "experienced"
                    s["operator_experience_months"] = replacement["experience_months"]
    return fixed


_sanitize_log = sanitize_clean_windows()

# ---------------------------------------------------------------------------
# 4. Write outputs
# ---------------------------------------------------------------------------
downtime_records.sort(key=lambda r: (r["date"], r["machine_id"]))
shift_records.sort(key=lambda r: (r["date"], r["machine_id"], r["shift"]))
complaints.sort(key=lambda r: r["complaint_id"])

(OUT_DIR / "complaints.json").write_text(json.dumps(complaints, indent=2))
(OUT_DIR / "downtime.json").write_text(json.dumps(downtime_records, indent=2))
(OUT_DIR / "shifts.json").write_text(json.dumps(shift_records, indent=2))
(Path(__file__).parent / "answer_key.json").write_text(json.dumps(answer_key, indent=2))

if _sanitize_log:
    print(f"Sanitized {len(_sanitize_log)} accidental noise collision(s):")
    for line in _sanitize_log:
        print("  -", line)
    print()
else:
    print("No accidental noise collisions found in clean windows.\n")

print(f"complaints:        {len(complaints)}")
print(f"downtime records:  {len(downtime_records)}")
print(f"shift records:     {len(shift_records)}")
print(f"answer key entries:{len(answer_key)}")
print()
print("Storylines: calibration-drift x%d, trainee-operator x%d, material-batch x%d, "
      "mold-wear x%d, red-herring x%d" % (len(s1_ids), len(s2_ids), len(s3_ids), len(s4_ids), len(s5_ids)))
