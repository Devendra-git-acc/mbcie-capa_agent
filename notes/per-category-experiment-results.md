# Per-category investigator experiment: real results

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
