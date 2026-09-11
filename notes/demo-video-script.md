# Demo video script

A full shot-by-shot script for the submission video, under the brief's
5-minute cap. Every line is written to be read close to verbatim; every
action is numbered so there's no ambiguity about what's on screen when
each line is spoken. Total budget below adds to **5:00 exactly** -- treat
that as tight, not loose, and use the [cutting guide](#if-youre-running-long)
if a take runs over.

## Before you hit record

1. `uvicorn main:app --reload`, wait for `Application startup complete.`
   **Don't record the cold start** -- the LLM provider connection and
   first import already happened by the time recording begins.
2. Open `http://127.0.0.1:8000/` in a clean browser window/tab. Zoom the
   page to whatever reads clearly on your recording resolution (Ctrl/Cmd
   `+` a couple of times if recording at 1080p on a high-DPI screen).
3. Log in once with `reviewer` / your real `BASIC_AUTH_PASSWORD` -- stay
   connected for the whole recording, don't log in on camera.
4. Have a real PDF or a phone photo of any receipt/invoice sitting on the
   desktop, ready to drag in during the Task 2 upload beat.
5. Do one silent practice run of the whole path below first. LLM output
   isn't identical every time -- know roughly what each step returns
   *this* run before you're narrating it live, so nothing on screen
   surprises you mid-sentence. If a result differs from what's written
   below, say what's actually on screen, not the scripted line -- the
   contingency notes under each beat cover the realistic alternatives.
6. Close anything else that might pop a notification during recording.

Screen recorder running, browser at the landing page, both tabs visible.
Begin.

---

## 0:00 -- 0:15 -- Intro

**On screen:** the demo console landing page, Task 1 tab active, nothing
clicked yet.

**Say, word for word:**
> "This is my submission for the MBCIE AI Engineer assessment. It's two
> systems behind one FastAPI service. Task 1 is a LangGraph agent that
> autonomously investigates manufacturing defect complaints. Task 2 is a
> document extraction pipeline with cost control. Everything I'm about to
> show is running live against the real API -- this isn't a slide deck or
> a recording of a mock."

**Action:** none yet -- let the page sit still while you say this so the
opening frame isn't a blur of cursor movement.

---

## 0:15 -- 1:05 -- Task 1: an existing complaint from the dataset

**Say:**
> "Task 1 starts with a synthetic factory dataset I generated myself --
> no real dataset was supplied for this assessment. It's a tyre and cycle
> plant: a complaint log, machine downtime records, and operator shift
> data, with root causes deliberately planted in the data rather than
> random. This one" --

**Action 1:** click the complaint dropdown, select **C002**.

**Say (continuing):**
> -- "is a real complaint from that dataset."

**Action 2:** click **Run investigation**.

**Say, while the request is in flight (this typically takes 5-40 seconds --
keep talking, don't sit in silence):**
> "This is a real LangGraph ReAct loop running right now -- the agent is
> deciding for itself which tools it needs. It might decode the batch
> code into a machine and date range, pull downtime records for that
> window, check who was staffing the shift, or look for the same defect
> on other machines. It's not a fixed pipeline -- the order and the choice
> of which tools to call is the model's own decision each time."

**Action 3:** once the result lands, point the cursor at, in order:
- the **root cause category** and hypothesis text
- the **confidence score** tile row
- one or two expanded rows of the **evidence chain**

**Say:**
> "The confidence score here isn't the model grading its own answer --
> that's something I deliberately avoided, because a model's self-reported
> confidence doesn't actually correlate well with whether it's right. This
> is computed from three things I can independently check: whether the
> citation survived a mechanical verification step, how reliable this
> category has historically been, and whether it actually cited a real
> record. And this evidence chain is exactly what it says -- every tool
> call the agent made, in order, with the real result it got back."

**Contingency:** if C002 happens to land on a different category than
Equipment this run (non-determinism is real and documented, not a bug),
just say what's on screen -- e.g. "this run it landed on X" -- and move
on. Don't stop the recording over it.

---

## 1:05 -- 1:50 -- Task 1: a complaint that isn't in the dataset at all

**Say:**
> "It doesn't have to be a complaint that already exists in the dataset,
> though. I can type a completely new one."

**Action 1:** click into the "type your own complaint" form.

**Action 2:** type into **Batch code**:
```
M02-2026W13
```

**Action 3:** type into **Defect description**:
```
Customer reports the tyre came apart at the tread after very little use -- looks like it wasn't cured properly.
```

**Action 4:** click **Investigate this complaint**.

**Say, while it runs:**
> "This exact complaint was never in complaints.json -- I just typed it.
> What makes this work is that the evidence the agent needs -- the
> downtime records, the shift records -- already exists independently of
> any specific complaint. A machine's operating history doesn't wait for
> someone to file a complaint about it. So the agent is investigating this
> against the real history for machine M02 that week, the same as it
> would for any real new complaint in production."

**Action 5:** once it resolves, point at the category and note it's
reasoning from the same underlying evidence as the dataset complaint on
the same machine -- proving it's not just keyword-matching the wording.

**Say:**
> "Same machine, same evidence, completely different wording from the
> fixture complaint -- and it reasons its way to the same conclusion."

---

## 1:50 -- 2:20 -- Task 1: the honest "I don't know" case

**Say:**
> "And if I give it a machine that genuinely isn't in the data at all --"

**Action 1:** clear the form, type into **Batch code**:
```
M99-2026W40
```

**Action 2:** type into **Defect description**:
```
Unexplained surface discoloration.
```

**Action 3:** click **Investigate this complaint**.

**Say, while it runs:**
> "-- it doesn't invent a plausible-sounding answer just because I asked
> it a question."

**Action 4:** once it resolves to "Insufficient evidence," scroll to the
**review queue** panel and point at the new entry that just appeared.

**Say:**
> "It honestly says there's insufficient evidence, and that automatically
> routes it into a persisted human-review queue instead of shipping a
> guess. That queue updates after every single investigation, and clears
> an entry again if a later re-run comes back clean."

---

## 2:20 -- 3:10 -- Task 2: extracting a fixture document

**Action 1:** switch to the **Task 2** tab.

**Say:**
> "Task 2 is document extraction. There are ten invoices and purchase
> orders here in genuinely inconsistent formats -- some are digital PDFs,
> some are scanned images, different vendors, different layouts, none
> templated the same way."

**Action 2:** select **D05** from the sample documents dropdown, click
**Extract this document**.

**Say, while it runs:**
> "This runs OCR if the document's scanned, or pulls text directly if
> it's a digital PDF, then structures it into a validated schema through
> one LLM call."

**Action 3:** once it resolves, point at:
- the extracted fields and the line-item table
- the confidence breakdown, specifically `needs_review: true`
- the review reason text

**Say:**
> "This one's flagged. The subtotal, tax, and total don't actually
> reconcile -- and that's not something I asked the model to judge, it's
> arithmetic checked in code, the same way I check the agent's evidence in
> Task 1 mechanically rather than trusting its own self-assessment. So
> instead of silently accepting a possibly-wrong extraction, it's routed
> to review with the specific reason why."

---

## 3:10 -- 3:35 -- Task 2: uploading a real file

**Say:**
> "And this isn't limited to the ten fixtures."

**Action 1:** drag the real PDF or invoice photo you prepared earlier into
the upload panel.

**Action 2:** click **Extract upload**.

**Say, while it runs:**
> "I can drop in any real file and it runs the exact same pipeline, live,
> against something it's never seen before."

**Action 3:** briefly show the result once it lands -- no need to dwell,
this beat is proving it works on arbitrary input, not re-explaining the
confidence breakdown again.

---

## 3:35 -- 3:55 -- Both review queues, side by side

**Action 1:** show the Task 2 review queue panel with D05's entry.

**Action 2:** switch back to Task 1, show its review queue with the
M99 entry from earlier still there.

**Say:**
> "Both tasks share the same underlying idea: route anything low-confidence
> to a persisted queue instead of guessing, rather than it being two
> unrelated features that happen to look similar."

---

## 3:55 -- 4:40 -- The numbers

This beat doesn't need live interaction -- cut to the terminal, the
README, or just stay on the console while you say it as voiceover.

**Say:**
> "On evaluation: I ran each of the eighteen complaints in the dataset
> three times each, not once, because I found directly during testing
> that this agent isn't fully deterministic even at temperature zero --
> the identical complaint produced two different categories across two
> consecutive runs. Across fifty-four total runs, the last full evaluation
> came back at 100% accuracy, with zero failed conclusions.
>
> More importantly, I also built a second, completely held-out dataset --
> eighteen new synthetic complaints on different machines, generated after
> the agent's design was already finalized, that it was never tuned
> against. On that set it held 88.9%. Three of the four categories stayed
> at 100%; one category, Material, dropped to 50%, with a specific,
> reproducible pattern I wasn't able to fully explain. I documented that
> openly rather than leaving it out.
>
> For Task 2, measured cost comes out to thirty-six cents per thousand
> documents at gpt-4o-mini pricing, and I've got one concrete idea for
> cutting that further: stop paying the model to compute line-item totals,
> since that's pure arithmetic the pipeline can already compute itself for
> free."

---

## 4:40 -- 5:00 -- Close

**Say:**
> "All of this -- the architecture decisions and why I made them, the full
> evaluation methodology, and an honest limitations section covering what
> I'd still fix and what I'd do differently with more time -- is written
> up in the README. Thanks for watching."

**Action:** end on either the demo console or the README open in an
editor, whichever you cut to.

---

## If you're running long

Cut in this order -- each one costs the least proof value per second saved:

1. The M99 "Insufficient evidence" beat (1:50-2:20) -- the ad-hoc complaint
   beat right before it already proves the "type your own" feature works;
   this one only adds the honesty angle, which you can still say in one
   sentence during the close instead.
2. The Task 2 real-file upload (3:10-3:35) -- the sample-document
   extraction already proves the pipeline works end-to-end.
3. Trim the numbers section (3:55-4:40) down to just the four headline
   figures, no surrounding explanation: *"100% on the tuned set, 88.9% on
   a genuinely held-out set, thirty-six cents per thousand documents,
   with one concrete cost-reduction idea."*

**Never cut:** one full Task 1 investigation shown end-to-end (any of the
three Task 1 beats), one full Task 2 extraction shown end-to-end, and at
least one review-queue entry actually visible on screen. Those three are
the things a reviewer cannot verify from the README text alone -- everything
else in this script is explanation a reviewer could also get by reading.
