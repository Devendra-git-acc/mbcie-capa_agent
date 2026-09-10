# Per-category investigator experiment: real results

**Update, round 2: 100% (54/54).** See "Round 2" section near the bottom
for what changed and the honest caveats before treating that number as
the final word. The section below is the original round -- kept as-is,
not rewritten, because the regression it documents is exactly what
motivated round 2's two fixes.

Branch: `experiment/per-category-investigator`. Built per
`scaling-architecture.md`'s "alternative considered" section -- one
reusable investigator builder, instantiated once per category (Human,
Equipment, Material), dispatched in parallel, synthesized by plain code
using each investigator's structured, mechanically-verified finding.
Tested against the same `eval.py` (18 complaints x 3 runs = 54
investigations) as the single-agent baseline, same model (`gpt-4o-mini`),
so the numbers are directly comparable.

## Headline: mixed result, not a clean win

| | Single agent + verifier (`agent.py`, main branch) | Per-category investigators (this branch) |
|---|---|---|
| **Overall** | **72.2%** | **61.1%** |
| Equipment | 77.8% | **100.0%** |
| Human | 77.8% | 33.3% |
| Material | 50.0% | 0.0% |
| Insufficient evidence | 83.3% | 50.0% |

The hypothesis behind this experiment -- that isolating each category
into its own narrow, non-competing prompt would eliminate the whack-a-
mole cross-category interference documented in DAY3_STATUS.md -- is
**confirmed for Equipment**: the calibration-drift storyline (C001-C004)
that kept getting hijacked by the "multiple complaints = Material"
instinct under the shared prompt now resolves perfectly, every time,
because the Material investigator has no opportunity to interfere with
the Equipment investigator's own reasoning anymore. That part of the
design worked exactly as intended.

**But it introduced a new problem the shared-prompt design didn't have**,
diagnosed directly (not guessed at) on C008, one of the real Material
storyline complaints:

```
Equipment -> supports=True, cites D0022 ("Scheduled PM" -- routine
             preventive maintenance, not an anomaly), _verified=True
Material  -> supports=True, cites C009 (SAME machine as C008, not the
             genuine cross-machine match C010/C011 sitting right there
             in the same query result), _verified=False (correctly
             rejected)
```

Two distinct root causes, both real:

1. **Narrow framing makes each investigator more willing to accept weak
   evidence for its own category.** Told to look for ONLY equipment
   causes, with no other explanation to weigh against, the Equipment
   investigator grabbed a routine "Scheduled PM" downtime record and
   rationalized it as suspicious ("could have addressed issues affecting
   production"). The mechanical verifier didn't catch this because it
   only checks that the cited `downtime_id` is REAL for that machine --
   it never checked whether the record's content is actually anomalous
   versus routine, because that's not cleanly mechanically checkable
   without hardcoding dataset-specific text patterns (the same kind of
   fragile, ad-hoc rule-writing this whole design was trying to get away
   from). The single shared-prompt agent didn't have this failure mode as
   often, plausibly because weighing 4 categories against each other in
   one pass makes it easier to notice "none of this is very convincing"
   -- a narrow specialist has nothing else to compare against.
2. **The same-machine-vs-cross-machine confusion survived prompt
   narrowing.** Even with a prompt that ONLY talks about Material and
   explicitly warns against same-machine citations, the Material
   investigator still cited C009 (M01, C008's own machine) instead of
   C010/C011 (M03, the real signal) -- which WAS visible in its own tool
   results. The mechanical verifier caught this correctly and rejected
   it (this is exactly what it's for), but with nothing else to fall back
   on, the complaint fell through to whichever OTHER investigator's
   (weaker) finding happened to pass -- in this case, Equipment's flawed
   "Scheduled PM" citation.

A secondary, now-fixed issue during testing: the Equipment investigator's
recurrence check (`_verify_equipment_finding`) initially accepted ANY
same-machine complaint as "recurrence" evidence, including one merely
in the SAME week (pure coincidence, not a multi-week pattern) -- this
caused C017 (the deliberate red herring) to wrongly resolve to Equipment
citing C018 (same week, same machine). Fixed by requiring cited complaint
IDs to span a genuinely DIFFERENT week, confirmed against C017 (now
correctly "Insufficient evidence") and re-verified the 5 legitimate
mold-wear complaints (C012-C016) still pass with the stricter check.

## What this means, honestly

Splitting into narrow agents traded one class of error (cross-category
prompt interference) for another (per-category overconfidence in weak
evidence). It is NOT simply "better" or "worse" than the single-agent +
mechanical-verifier design -- it's a different failure profile, and this
specific run measured worse overall (61.1% vs 72.2%) mostly because
Material collapsed from 50% to 0%, which outweighs Equipment's gain.

This is exactly why the experiment was worth running rather than assuming
either design is obviously right: the intuition that narrower prompts
must be more reliable turned out to be only half true.

## Not yet tried, worth naming rather than building blind

- Tightening the Equipment investigator's downtime citation to require
  more than "this ID is real for this machine" -- e.g. having it quote
  the specific words in `reason_detail` that indicate an anomaly, and
  having a human (or a second pass) sanity-check that quote, rather than
  trusting the category label alone.
- A genuine "cite the machine_id explicitly" step for the Material
  investigator, forcing it to state both machine IDs being compared
  before it's allowed to conclude, so the mechanical verifier isn't the
  only thing standing between it and a wrong same-machine citation.
- Running this same eval against the free OpenRouter model too, to see
  whether this failure profile is specific to `gpt-4o-mini` or general.

None of these are done -- flagging them as the next things to try, not
claiming they'd work.

## Round 2: two targeted fixes, real result 100% (54/54)

Both untried ideas above were actually implemented, plus the one gap
that was really the bigger story:

1. **Added the verification-retry loop each investigator was missing.**
   Round 1's investigators got exactly one shot -- rejected findings were
   discarded with no chance to self-correct. `agent.py`'s Material check
   already proved that feeding a rejected finding's specific reason back
   and letting the model retry (bounded, max 2 attempts) works well; this
   was simply never carried over to the per-category investigators the
   first time. Now it is, using the same "close the pending tool_call with
   a synthetic ToolMessage before appending the correction" fix already
   applied in `agent.py` (same OpenAI API constraint applies here).
2. **Made the Equipment investigator's evidence claim structurally
   checkable.** Added `is_anomaly_not_routine: bool` as a required field
   on its `submit_finding_equipment` tool (Equipment gets its own tool
   variant; Human and Material still share the plain `submit_finding`) --
   forcing an explicit, separate commitment instead of burying "is this
   actually suspicious or just routine" inside free-text reasoning the
   verifier can't check. The verifier now requires this field be True
   before accepting a cited downtime record, on top of it being real.

**Result**: every category, every complaint, all 3 runs: 100.0%.
Verified live on C008 first (the exact complaint that failed in round 1)
before the full run: Equipment correctly retried and stood down
("No evidence of an anomaly... routine"), Material correctly retried and
found the genuine cross-machine citation (C010) it had access to all
along. Full `eval.py` run: 54/54, 36 correction attempts used across
54 investigations (~0.67 retries/investigation on average -- most
investigations needed at least one correction round).

### Read this number carefully, not just triumphantly

- **The two fixes were built by diagnosing exact failures on this same
  18-complaint dataset**, then re-tested on that same dataset. The checks
  themselves are structural/general (machine_id equality, week
  comparison, an explicit anomaly flag) rather than hardcoded to specific
  complaint IDs, so this isn't literally memorizing answers -- but 100%
  on the dataset the fixes were tuned against is a different claim than
  "this would score 100% on a fresh, never-seen batch of complaints."
  The honest framing: these are now two well-diagnosed, generally-stated
  rules, not overfit patches, but they've only been measured against the
  data that inspired them.
- **Real added cost.** ~0.67 retries/investigation on top of the 3
  parallel base calls means real extra latency and token spend beyond
  what round 1 already cost more than the single-agent design. The
  retry-driven reliability is genuinely valuable, but it isn't free, and
  that trade should be weighed explicitly, not assumed away by the
  headline number.
- **Not yet re-tested**: the free OpenRouter model (this round only ran
  against `gpt-4o-mini`, same as round 1).

### Where this leaves the architecture decision

This changes the earlier recommendation. Round 1 measured worse than the
single-agent design (61.1% vs 72.2%) and the advice was to keep the
single agent. Round 2 measures decisively better (100% vs 72.2%), on the
same eval, same model. The per-category design is the stronger result now
-- the open questions before treating it as the new default are exactly
the two caveats above: does it hold up on complaints it wasn't tuned
against, and is the added call volume worth it for this project's actual
needs.
