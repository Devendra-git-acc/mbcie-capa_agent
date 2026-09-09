# Data grounding notes (Task 1)

Quick record of what shaped the synthetic dataset, so this doesn't have to
be re-derived when writing the README on Day 3.

## Root cause taxonomy
Manufacturing CAPA practice generally groups root causes into five
categories: **Process, Material, Equipment, Human, Scheduling** (plus
"insufficient evidence" as a valid non-answer when the investigation
genuinely can't support a conclusion). This dataset's five storylines map
onto four of those five, deliberately skipping a forced fit into all five:

| Storyline | Machine(s) | Category | Signal lives in |
|---|---|---|---|
| Calibration drift | M02 | Equipment | downtime (fix event, post-dates the batch) |
| Trainee operator | M05 | Human | shifts only (downtime is clean) |
| Bad material batch | M01 + M03 | Material | cross-machine complaint pattern only |
| Gradual mold wear | M04 | Equipment | absence of a discrete event + recurrence over time |
| Red herring | M06 | Insufficient evidence | nothing -- all three sources clean by design |

## Defect vocabulary
Real tire-defect writeups consistently point to a small set of failure
modes: trapped air/blisters from the curing (vulcanization) step, tread
separation, sidewall cracking, contamination, and uneven wear from
alignment-type issues. The curing step specifically is where
temperature/pressure control problems do the most damage, which is why
the calibration-drift storyline routes through a curing press (M02) and
manifests as blistering/tread separation rather than a generic "defect."

## Downtime taxonomy
Shop-floor downtime tracking converges on a small top-level bucket list
(commonly 6-8 codes): Setup/Changeover, Mechanical, Maintenance
(including calibration), Material, Quality Hold, Staffing. This dataset
uses that same shape for `reason_category`, so the downtime log reads
like a real CMMS export rather than free text invented per-record.

## Why this matters for the "data judgment" evaluation criterion
None of this is decorative -- it's what makes the planted storylines
resolve to a specific, defensible field (`reason_category`,
`root_cause_category`) instead of a prose description the agent has to
interpret loosely. A grader skimming the JSON should recognize the
categories as the real thing, not an invented-for-the-assessment schema.
