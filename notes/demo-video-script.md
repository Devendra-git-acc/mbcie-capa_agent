# Demo video script

Target: **under 5 minutes**, per the brief. Written to be read almost
verbatim while screen-recording the demo console at `http://127.0.0.1:8000/`
(after `uvicorn main:app --reload`, with `.env` filled in and the server
already warm -- don't record the server cold-starting). Times are a
budget, not a stopwatch -- talk at a natural pace and trim the wrap-up if
you're running long rather than rushing the middle.

Log into the console once at the start (`reviewer` / your `BASIC_AUTH_PASSWORD`)
and leave it connected for the whole recording.

---

## 0:00 -- 0:15 -- Intro

**Say:**
> "This is my submission for the MBCIE AI Engineer assessment -- two
> systems behind one FastAPI service. Task 1 is a LangGraph agent that
> investigates manufacturing defect complaints on its own. Task 2 is a
> document extraction pipeline with cost control. I'll run both live
> against the real API, not a slide deck."

**Show:** the demo console landing page, both tabs visible.

---

## 0:15 -- 1:15 -- Task 1, dataset complaint

**Say:**
> "First, an existing complaint from the synthetic dataset I generated --
> a tyre plant with complaint logs, machine downtime, and shift records,
> with root causes deliberately planted in the data."

**Do:**
1. Task 1 tab, pick **C002** (or any Equipment complaint) from the dropdown.
2. Click **Run investigation**. While it's running (5-40s):

**Say (while it runs):**
> "This is a real LangGraph ReAct loop -- the agent decides for itself
> which tools to call. It might decode the batch code, pull downtime
> records, check shift staffing, and look for the same defect on other
> machines, in whatever order it decides it needs."

**Do:** once it finishes, point at:
- the conclusion + category
- the **confidence score breakdown** -- explain it's a computed score
  (verification outcome + category reliability + citation), not the
  model's self-reported confidence
- expand one or two steps of the **evidence chain**

**Say:**
> "The confidence score isn't the model grading itself -- it's built from
> signals I can actually check: did the citation survive a mechanical
> verification, how reliable has this category historically been, and
> did it cite a real record. And every tool call it made is right here,
> in order."

---

## 1:15 -- 2:15 -- Task 1, ad-hoc complaint

**Say:**
> "It doesn't have to be a complaint that already exists in the dataset --
> I can type a brand new one."

**Do:** in the "type your own complaint" form:
- Batch code: `M02-2026W13`
- Defect description: `Customer reports the tyre came apart at the tread after very little use -- looks like it wasn't cured properly.`
- Click **Investigate this complaint**.

**Say (while it runs):**
> "This complaint was never in complaints.json. The agent is investigating
> it against the real downtime and shift history for that machine and
> week -- the same records that already exist independent of any specific
> complaint."

**Do:** once done, briefly show the result (should land on Equipment,
same underlying evidence as C001/C002 -- reinforces it's reasoning over
retrieved data, not keyword-matching wording).

**Optional, if time allows -- the honest "I don't know" case:**
- Batch code: `M99-2026W40`
- Defect description: `Unexplained surface discoloration.`

**Say:**
> "And if I give it a machine that's genuinely not in the data, it
> doesn't invent an answer -- it says 'Insufficient evidence,' and gets
> automatically flagged for human review instead."

**Do:** point at the **review queue** panel updating with the new entry.

---

## 2:15 -- 3:15 -- Task 2, sample document + upload

**Say:**
> "Task 2 is document extraction. Ten invoices and purchase orders in
> genuinely inconsistent formats -- some digital PDFs, some scanned
> images, different vendors and layouts."

**Do:**
1. Switch to the Task 2 tab.
2. Pick **D05** from the sample documents, click **Extract this document**.
3. Once done, show the extracted fields, the line-item table, and the
   **confidence breakdown** -- point out `needs_review: true` with its
   specific reason (arithmetic doesn't reconcile).

**Say:**
> "This one's flagged -- the subtotal, tax, and total don't add up, so
> instead of silently accepting a possibly-wrong extraction, it's routed
> to review with the exact reason why."

**Do:** drag a real PDF (or a photo of a receipt/invoice) into the upload
panel, click **Extract upload**.

**Say:**
> "And this isn't limited to the ten fixtures -- I can drop in any real
> file and it runs the same pipeline live."

---

## 3:15 -- 3:45 -- Review queues

**Do:** show both `GET /review-queue` panels (Task 1 and Task 2) with at
least one real entry each from what you just ran.

**Say:**
> "Both tasks route low-confidence results to a persisted review queue
> instead of guessing -- this updates after every investigation and
> extraction, and clears again once something's re-run clean."

---

## 3:45 -- 4:30 -- Numbers (evaluation + cost)

**Say (can be voiceover over the README or terminal, no need to re-run anything live):**
> "On evaluation: I ran each of the 18 fixture complaints 3 times, not
> once, because I found during testing that this agent isn't fully
> deterministic even at temperature zero -- the same complaint produced
> different categories on two consecutive runs. Across 54 runs, the last
> full evaluation hit 100%. On a completely held-out set of 18 new
> synthetic complaints the agent never saw while I was building it, it
> held 88.9% -- three of four categories at 100%, with one specific,
> honestly-documented gap in the Material category I wasn't able to fully
> explain.
>
> For Task 2, measured cost is 36 cents per thousand documents at
> gpt-4o-mini pricing, and I've got one concrete idea for cutting that
> further -- stop paying the model to compute line totals, since that's
> just arithmetic the pipeline can do itself."

---

## 4:30 -- 5:00 -- Close

**Say:**
> "Everything here is in the README, including an honest limitations
> section -- what's genuinely solid, what I'd still fix, and what I'd do
> differently with more time. Thanks for watching."

**Do:** end on the demo console or the README in an editor -- whichever
reads better on screen.

---

## If you're cutting for time

Drop, in this order, before cutting anything above:
1. The optional M99 "Insufficient evidence" example (2:15 section)
2. The Task 2 upload demo -- the sample-document extraction alone still
   proves the pipeline works
3. Shorten the numbers section to just the two headline figures (100%
   tuned / 88.9% held-out; $0.36/1,000 docs) without the surrounding
   explanation

Never cut: one real Task 1 investigation end-to-end, one real Task 2
extraction end-to-end, and at least one review-queue flag actually
appearing on screen -- those three are the parts a reviewer can't verify
from the README alone.
