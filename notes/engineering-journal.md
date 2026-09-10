# Engineering journal: how this system actually got built

This is the full story, in order, of building the CAPA root-cause-analysis
agent (Task 1) and the document extraction pipeline (Task 2) for this
assessment -- every real bug found, why it happened, how it was diagnosed,
what was tried, what worked, what didn't, and why. Written so it can be
read start to finish and explained out loud, e.g. in an interview, not
just skimmed as a changelog.

The short version, if you need it before the long one: this project's
real engineering story isn't "I built an agent." It's "I built an agent,
measured it honestly, found it was wrong in specific and often surprising
ways, and used that evidence -- not intuition -- to decide what to fix and
how." Almost every interesting decision below was made *because* a test
or a real run exposed something, not because it seemed like a good idea
in the abstract.

---

## Part 0: what this system actually is

**Task 1** is a LangGraph ReAct agent that investigates a manufacturing
defect complaint (from a synthetic tyre/cycle plant) by autonomously
querying three data sources -- downtime logs, shift/operator records, and
other complaints -- and concluding a root-cause category (Equipment,
Material, Human, Process, Scheduling, or "Insufficient evidence") with a
cited evidence chain. It's exposed as a FastAPI service with basic auth.

**Task 2** is a document-extraction pipeline: given an invoice or purchase
order (a real uploaded file, digital PDF or scanned image, single or
multi-page), it extracts structured JSON (vendor, line items, totals),
scores its own confidence from three independent signals, and routes
low-confidence extractions to a human-review queue instead of guessing.

Both were built, then stress-tested against real data and real model
calls, and both round of testing turned up real, sometimes serious bugs
that are documented below alongside the fixes.

---

## Part 1: the foundation (Day 1)

Day 1's work (not something I built from scratch in this conversation,
but verified carefully before building on top of it) produced:

- A synthetic dataset -- complaints, downtime logs, shift records --
  generated with five deliberately planted "storylines," each mapping to
  a real CAPA root-cause category: a calibration-drift Equipment case, a
  trainee-operator Human case, a cross-machine Material case, a
  gradual-wear Equipment case (mold wear), and one deliberate red herring
  designed to resolve to "Insufficient evidence." The categories and
  vocabulary were grounded in real CAPA/tyre-defect/CMMS conventions
  (documented in `notes/data-research-notes.md`), not invented for the
  assessment.
- Four tools for the agent to use (`decode_batch_code`, `query_downtime`,
  `query_shifts`, `query_complaints`), each independently unit-tested
  with no LLM involved.
- An independent dataset-validator (`validate_dataset.py`) that checks
  the synthetic data for contamination -- e.g. that a "clean" storyline
  window doesn't accidentally contain a suspicious downtime record that
  would give the agent a false lead. Running it actually caught and led
  to fixing two real bugs in the data generation before any of this
  session's work started.
- A basic agent skeleton, not yet tested against a live model (the
  original build environment couldn't reach the OpenAI API at all).

The important thing about Day 1's practice, which the rest of this
project kept following: **every piece of logic that could be tested
without an LLM call was tested that way first.** Tool functions, data
generation, contamination checks -- all free, all fast, all run before a
single dollar was spent on a real model call. This habit is why so many
of the bugs below were caught cheaply instead of expensively.

---

## Part 2: Day 2 -- provider flexibility and the first three real bugs

### Making the system provider-agnostic

Before any live testing could happen, there needed to be a way to run
against a real model. OpenAI wasn't available yet (no key), but a free
OpenRouter model was. Rather than hardcode either, `agent.py` got a
`_build_model()` function switched purely by an env var
(`LLM_PROVIDER=openai|openrouter`) -- since OpenRouter speaks the
OpenAI-compatible API, this only needed a different `base_url` and key,
no new dependency, no code branching anywhere else in the system. This
paid off repeatedly: the whole system was later flipped to real OpenAI
with a one-line `.env` change and zero code changes.

### Bug #1: the .env file that wasn't actually being loaded

The very first live test failed with "OPENAI_API_KEY is not set" --
despite `LLM_PROVIDER=openrouter` being set and a real OpenRouter key
being in `.env`. The cause: `run_single.py` checked `os.environ` for the
right key *before* importing `agent.py` -- and `load_dotenv()` only
lived inside that import. So `.env` was never actually read before the
check ran; the script only ever worked before if the key had been
exported directly in the shell, which is not what the documented usage
told people to do. Fixed by moving `load_dotenv()` to the very top of
`run_single.py`, before any environment check. Small fix, but a good
example of a bug that's invisible until you actually run the thing the
way the docs say to run it.

### Bug #2: an entire storyline the agent had no way to detect

Testing complaint C012 (the mold-wear/gradual-equipment-wear storyline)
produced the wrong category on the very first live run. Investigating
why: this storyline's actual signal is recurring complaints on the same
machine spread across several non-adjacent weeks, with no single
"smoking gun" record -- the *absence* of a discrete cause plus a pattern
over time is itself the evidence. But the existing toolset only offered
`query_complaints(date_range)`, which can't see across time; it can only
see one window at a time. The agent, unable to find the real pattern,
grabbed an unrelated background-noise downtime record in the window and
built a plausible-sounding but wrong story around it.

This wasn't a prompting problem -- the agent literally didn't have the
tool it needed. Added `query_complaints_by_machine(machine_id)` (all
complaints for one machine across all time, no date filter) plus a
system-prompt step telling the agent when to reach for it. Retested:
correct.

**The lesson that recurs throughout this whole project**: before
assuming a wrong answer is a reasoning failure, check whether the agent
actually had access to the right evidence at all. Several bugs below
follow this same shape.

### Bug #3: a "correct" answer that was actually a coincidence, and would have poisoned every future measurement

Testing C017 (the deliberate red herring, meant to resolve to
"Insufficient evidence") appeared to work: the agent's answer matched
ground truth. Looking closer at the actual returned object revealed the
`root_cause_hypothesis` field said, verbatim, *"Agent did not reach a
structured conclusion."* -- this wasn't a real answer. `agent.py`'s
fallback path, used when the model fails to produce the forced
`submit_conclusion` tool call at all, defaults to `"Insufficient
evidence"` -- which is also the correct answer for this specific
complaint. A genuinely broken run (the model returned nothing usable)
had, by pure coincidence, landed on the same label as the ground truth.

This is a dangerous bug specifically because it's invisible at the level
of "did the category match" -- a naive hit-rate script comparing
`root_cause_category` strings would count this as a hit. Fixed by adding
a `conclusion_failed: bool` field to `investigate()`'s return value, and
making sure every downstream consumer (first `run_single.py`, later
`eval.py`) checks that flag before ever counting a run as a match. This
fix mattered a lot later -- see Part 4, where the eval harness's careful
scoring is what let later regressions and improvements actually be
trusted as real numbers rather than noise.

### Building the FastAPI layer

Task 1's brief explicitly required a FastAPI service with basic auth;
Task 2's did not, but was added anyway for demo/deployment purposes once
asked. Implementation notes worth keeping:

- Basic auth compares credentials with `secrets.compare_digest`, not
  `==`, specifically to avoid a timing side-channel that could let an
  attacker infer the password one byte at a time from response latency.
- If `BASIC_AUTH_PASSWORD` is left blank, the service does NOT silently
  accept an empty password (which `secrets.compare_digest("", "")` would
  otherwise do, since two empty strings match) -- it generates a random
  one at startup and prints it, so a demo service is never trivially
  open by default.
- `GET /health` is deliberately unauthenticated, so a monitor or reviewer
  can check liveness without credentials, while every real endpoint
  requires them.
- A later addition, `POST /extract`, is a genuine file-upload endpoint --
  not limited to replaying the 10 fixture documents -- which is what
  makes Task 2's API a real service rather than a canned demo. Building
  it surfaced a real dependency gotcha: FastAPI's `UploadFile` silently
  requires the `python-multipart` package, which is never imported
  directly anywhere in the code -- it would have failed at the first
  real upload request in a fresh environment, not at import time, not at
  `pip install` time for anything already in `requirements.txt`. Caught
  and added before it could surprise a reviewer.

### The senior-engineer review pass: four more real bugs, found by re-reading, not by running

At one point the work paused specifically to re-read every file touched
so far with a critical eye, rather than just moving forward. This paid
off -- it found:

1. **`extraction/pipeline.py`'s batch runner had zero fault
   tolerance.** The LLM call inside `extract_document()` had no
   try/except. This wasn't hypothetical: it's exactly what crashed a
   real test run later that same day, when OpenRouter's rate limit hit
   mid-batch (see below) and the whole script died, losing every
   document after the failed one and never writing the summary file.
   Fixed with a per-document try/except that records a failure and
   moves on, plus the batch runner itself wrapping each document call
   for non-LLM failures too (a corrupt file, etc).
2. **Fixing bug #1 exposed a second bug.** The batch summary's
   `review_queue` builder did `r["confidence"]["review_reasons"]`
   unconditionally -- but a failed document (from bug #1's new handling)
   only has a `failure_reason` key, not `confidence`. Any batch
   containing a genuinely failed document would crash exactly at the
   point where the fault-tolerance fix was supposed to kick in. This
   wasn't found by inspection -- it was found by a test specifically
   written to simulate a broken document and confirm the batch survives
   it, which is a good example of testing the failure path, not just
   the happy path.
3. **No cost guard for near-empty extracted text.** A document where
   OCR or PDF extraction produced almost nothing was still being sent to
   the LLM, spending a real API call on something the model couldn't
   possibly do anything useful with. Added a length threshold that
   short-circuits before the LLM call entirely -- directly relevant
   since Task 2's brief explicitly asks for cost awareness.
4. **`main.py` leaked raw exceptions to API callers.** An upstream
   provider failure (rate limit, bad key, network error) would surface
   as an unhandled 500 with a full Python stack trace in the HTTP
   response -- a real information-leakage and unprofessional-API smell.
   Wrapped in try/except, mapped to a clean `502` with a structured
   message. This was later verified against a *real* failure, not a
   simulated one -- see Part 3.
5. **Task 2 had no equivalent of Task 1's ground-truth check.**
   `run_single.py` could compare an agent's answer to `answer_key.json`;
   nothing did the equivalent for extraction results against
   `ground_truth.json`. Built `check_accuracy.py` -- normalizes string
   fields, applies numeric tolerance to money fields, and reports both
   overall and per-field accuracy. Running it against the only two real
   samples available at the time (see below) immediately surfaced the
   exact currency-symbol bug described next, which is a good example of
   a review tool doing its job the very first time it ran.

### The OpenRouter rate-limit wall

Testing Task 2's pipeline against the full 10-document batch hit
OpenRouter's free-tier cap (50 requests/day) mid-batch -- only 2 of 10
documents (one clean digital doc, one deliberately-degraded scan) had
gone through the real LLM step before the quota ran out. Both behaved
exactly as designed (the clean one extracted at 0.99 confidence and was
accepted; the degraded one correctly refused to guess numbers from
garbled OCR text and landed at 0.45 confidence, routed to review) -- a
good sign, but 2/10 isn't a validated batch, and the honest status
write-up at the time said so plainly rather than papering over it.

Along the way, two real prompt-quality bugs were also found from those 2
live extractions: `currency` came back as the literal symbol `"Rs."`
instead of the ISO code `"INR"`, and `doc_type` came back as `"INVOICE"`
instead of the schema's lowercase literal. Both fixed by tightening the
`InvoiceExtraction` schema's field descriptions -- not chased further
with more live testing at the time, since the rate limit made further
verification impossible until either the quota reset or a different
provider became available.

---

## Part 3: switching to real OpenAI, and a bug that a paid model didn't fix

Once a real OpenAI key was added, `LLM_PROVIDER` flipped to `openai` in
one line, no code changes -- proof the provider-toggle design from Part
2 actually paid for itself.

### Bug: a correct answer, thrown away, because it was never actually submitted

Re-testing C001 (calibration drift) against `gpt-4o-mini` produced
`conclusion_failed: true` -- surprising, since this is normally the
"easy" storyline. Reading the raw final message revealed the model had
reasoned through the *entire correct answer* in prose -- Equipment,
citing the right downtime record, correctly reasoning about the
post-dated fix -- and ended with the sentence "I will now submit the
conclusion." As text. Not as an actual tool call.

This was a different bug from anything fixed in Part 2. The earlier fix
only covered the forced-conclusion path *at* the step cap (`MAX_STEPS`);
this failure happened at turn 4 of a typical 8-turn budget, nowhere near
the cap, in the *normal* investigating loop, where the graph's routing
logic treats any response without a tool call as terminal -- full stop,
whatever the text said about intentions. A well-reasoned, correct answer
was being silently discarded.

Fixed by having `call_agent` check, on every ordinary turn (not just at
the step cap): if the investigating model responds with no tool call at
all, force a real `submit_conclusion` call from that same context before
giving up. Verified with a mocked test reproducing the exact scenario,
then verified live -- the same complaint that had just failed now
resolved correctly.

**Why this matters as a lesson**: this bug survived through a *paid,
generally very tool-compliant model*, not just a flaky free one. It's a
reminder that "the free model is unreliable, the paid one won't have
this problem" is not a safe assumption -- test the paid model too,
don't just assume better infrastructure buys away every failure mode.

### The full Task 2 batch, finally validated

OpenAI has no daily rate cap, so the blocker from Part 2 was gone. Ran
the full 10-document batch for real: **7/10 fully correct, 3/10
correctly flagged for review** (via `check_accuracy.py` against
`ground_truth.json`, not eyeballing). The earlier currency/doc_type
prompt fixes were confirmed working live on 9 of the 10 documents. One
residual, minor case: the hardest document (35 line items across 2
scanned pages) reverted to the raw currency symbol on that one field
while still getting all 35 line items and the arithmetic exactly right
-- a small, understood trade-off, not chased with more prompt tuning
given it was a single field on the single hardest document.

### The cost-estimator that was quietly lying

`estimate_cost.py` had originally been written while testing against
the free OpenRouter model, which does a lot of hidden chain-of-thought
reasoning even for simple extraction tasks -- so its hardcoded advice
said "switch to a non-reasoning model" as the cost-reduction idea. Once
the system had actually switched to `gpt-4o-mini` (already a
non-reasoning model), that advice became not just outdated but
self-contradictory -- the script's own numbers showed 0% reasoning
overhead right next to advice about reasoning overhead. Fixed by making
both the diagnosis and the recommendation *derived from the actual
sample data* rather than hardcoded to whatever scenario the script was
last written against, so switching providers again can't produce stale,
contradictory output a second time.

Final real number, from the full 10-document batch: **$0.36 per 1,000
documents**, with a cost-reduction idea also derived from real data
(stop asking the model to compute `line_total`, which is just
`quantity * unit_price` and cheaper for code to derive than for the
model to output).

---

## Part 4: Day 3 -- building a real evaluation harness, and the whack-a-mole saga

### Why one test run per complaint isn't enough

Before building `eval.py`, Day 2's testing had already directly observed
`gpt-4o-mini` at `temperature=0` behaving non-deterministically -- the
identical C012 complaint produced a different category on two
consecutive, otherwise-identical runs. A single pass/fail check per
complaint would misrepresent reliability either way: a lucky pass looks
like success, an unlucky fail looks like a regression that isn't real.
`eval.py` was built to run every complaint **3 times**, reporting a real
hit rate, not a coin flip's worth of confidence.

It also tracks a second, more diagnostic metric alongside the category
match: **evidence surfaced** -- for storylines with concrete supporting
record IDs, did the tool calls made during that run ever actually
*return* those IDs, regardless of what the agent concluded? This exists
specifically to separate two very different failure modes that a single
hit-rate number can't tell apart: "the right evidence was right there
and the agent still got it wrong" (a reasoning problem) versus "the
agent never even saw the right evidence" (a tool/investigation problem,
the same shape as bug #2 in Part 2). This distinction turned out to
matter a great deal in what follows.

### First real run: 44.4%, and a category that had simply never been tested

Running the full 18-complaint x 3-run evaluation for the first time
produced an overall hit rate of 44.4% -- and one number stood out
immediately: **the Human/trainee-operator storyline scored 0% (0 out of
9 runs).** This was the first time in the whole project that storyline
had been tested at all; every spot-check up to this point had happened
to land on Equipment, Material, or the red herring.

Diagnosing it: every single run showed `evidence_surfaced: true` -- the
agent had genuinely retrieved the correct trainee shift records
(`S0978`, `S0981`) every time. It just didn't conclude "Human" from
them. Instead it defaulted to "Material," citing "multiple complaints on
this batch" -- the exact same bias already seen once before in the
red-herring case, now confirmed to generalize well beyond that one
complaint.

The same run also showed Material at only 16.7%, including a regression
on C008 -- a complaint that had correctly resolved to Material in an
earlier spot-check, the day before. The cause: a Day-2 prompt addition
meant to stop the same-batch bias had overcorrected, making the model
default to "Insufficient evidence" even when genuine cross-machine
evidence was sitting right there in its own tool results.

### The decision to rewrite instead of patch again

At this point the system prompt had already been patched three separate
times across Day 2 and the start of Day 3, each patch aimed at a
specific observed failure. Rather than add a fourth patch, the whole
"Investigation method" section of the system prompt was rewritten from
scratch as one coherent, priority-ordered decision tree -- gather all
signal types up front, then apply the single most specific one that
actually fits, instead of accumulated if-you-see-X-but-not-Y caveats
that had started contradicting each other.

Re-running the full eval: **61.1% overall.** Human and Material both
jumped to 100%. But a brand new regression appeared: **the
calibration-drift storyline (C001-C004), which had been essentially
perfect since Day 2, dropped to near 0%.**

Diagnosing this one didn't need another LLM call at all -- a plain
Python query against the dataset (`query_complaints` for C001's actual
window) showed the real cause directly: C002, C003, and C004 are the
*same* calibration-drift storyline as C001, on the *same* machine, with
the *same* symptom -- and the newly-strengthened Material-detection
instinct was now firing on that same-machine recurrence, misreading it
as a cross-machine pattern because it shared a symptom. The fix that had
just solved Human and Material had, in solving them, made the model
*more* willing to call anything "Material" whenever multiple
same-symptom complaints existed -- including cases where they were
correctly explained by something else entirely.

One more targeted fix was added: cross-machine Material evidence now
explicitly requires the cited complaint to be on a genuinely
*different* `machine_id`, not just a matching symptom. This also
surfaced a real, previously-unnoticed property of the dataset itself,
confirmed by direct query rather than assumed: with 18 complaints spread
across a year, unrelated storylines sometimes land in the same calendar
window by pure coincidence (an unrelated mold-wear complaint, C013,
happened to fall inside C001's own date window) -- and the real Material
storyline (C008-C011) is distinguishable from that coincidence because
its complaints have byte-identical `defect_description` strings across
machines, which the coincidental overlap does not.

Re-running the eval a third time: **61.1% again** -- stable across two
consecutive full runs, which matters, because it rules out "maybe that
regression was just noise" as an explanation. Equipment stayed down
specifically on the calibration-drift sub-case (the mold-wear sub-case,
a different signal type within the same category, stayed mostly
correct), while Human, Material, and (for the first time all project)
partial credit on the red herring case all held.

### Recognizing when to stop patching a prompt

By this point: three full 54-investigation evaluations (162 total live
investigations) plus many additional spot-checks had been run, and every
recent fix had *individually, verifiably* traded one storyline's
accuracy for another's -- not randomly, but as a direct, diagnosed
consequence of strengthening one category's detection logic inside a
prompt that all four categories still had to share. That's a real
structural signal, not bad luck: as long as every category's rules live
in one prompt fought over by one model call, strengthening any one of
them risks bleeding into the others. Continuing to patch the same prompt
a fifth, sixth, seventh time was judged very unlikely to converge
cleanly, and every additional round costs real API money to verify.
**Work stopped here and the honest number -- 61.1%, with a precise,
evidenced explanation of exactly which two storylines pulled it down and
why -- was written up rather than chased further with more prompting.**

---

## Part 5: the verification-loop breakthrough -- checking facts in code instead of asking the model to try harder

Independently of continued prompt tuning, a different kind of fix was
introduced: instead of relying on the model to reason its way past its
own bias, add a small piece of **deterministic, non-LLM code** that
mechanically checks whether a "Material" conclusion's cited evidence
actually holds up -- does the cited complaint reference a genuinely
different `machine_id`, with a matching symptom, compared to the
complaint under investigation? If not, the conclusion is rejected with a
specific, concrete reason, and the model gets one bounded retry
(`MAX_VERIFICATION_RETRIES = 2`) with that exact reason fed back, rather
than being asked to "try harder" in the abstract.

This is a different *kind* of fix from everything in Part 4, not just a
fourth iteration of the same kind. Prompting asks the model to reason
better; mechanical verification doesn't trust the model's reasoning at
all for the one specific claim being checked -- it checks a fact
(same/different `machine_id`, string equality of `defect_description`)
in plain Python, which cannot be talked out of being right by a
persuasive-sounding wrong argument the way a second LLM call reviewing
its own work potentially could be.

### A real crash, found by actually running it, not by inspection

The very first live test of this mechanism (running the full evaluation
against it) crashed with an OpenAI `400 Bad Request`:

> *"An assistant message with 'tool_calls' must be followed by tool
> messages responding to each 'tool_call_id'."*

The cause: the retry loop appended a plain correction message directly
after an AI message that still contained an **unanswered**
`submit_conclusion` tool call. In the normal graph flow, that call is
never answered by a tool response, because it's the deliberate
termination signal -- the graph just ends right after it, so nothing
ever needs to respond to it. But the retry loop was reusing that same
message history and appending *more* conversation after it, which
OpenAI's API flatly refuses: every tool call needs a matching response
before anything else can follow it, no exceptions.

Fixed by inserting a synthetic `ToolMessage` -- "Rejected by
verification, see correction note" -- addressed to that exact
`tool_call_id`, closing it out properly before the correction message
gets appended. This is the kind of bug that's essentially undiscoverable
by reading code alone; it only shows up when the exact message sequence
actually gets sent to a real API that enforces the constraint.

### Result: 72.2%, and the red herring solved for the first time all project

With the crash fixed and retested live -- confirmed on three consecutive
real runs of C001 that the mechanism reliably caught the model's
same-machine "Material" miscitation, fed back the specific reason, and
watched the model self-correct to the right answer on the very next
attempt, every time -- the full evaluation was run again:

**72.2% overall** (up from 61.1%). Equipment (calibration drift) jumped
to 77.8%, fixed without the regression this fix's predecessors kept
causing elsewhere. **"Insufficient evidence" -- the red herring, never
solved correctly across an entire project's worth of prompt tuning --
jumped to 83.3%.** Material dropped somewhat (to 50%), diagnosed
precisely rather than left as a mystery: on C008/C009, the model's first
citation named only same-machine complaints (correctly rejected), but on
the retry it abandoned "Material" as a category entirely instead of
going back to find the genuine cross-machine evidence (`C010`/`C011`)
that was sitting in its own already-retrieved tool results -- a
correction-*message*-wording gap (it wasn't being told clearly enough to
look harder for the right citation, just to "reconsider"), not a flaw in
the verification logic itself.

---

## Part 6: thinking through how this would need to change at real scale

A natural question followed the verification-loop success: does this
approach -- hand-written rules and hand-written verifiers -- hold up if
the number of root-cause categories, or the number of underlying data
sources, grew a lot? This section is design discussion, documented in
`notes/scaling-architecture.md`, not something built into the running
system -- but it's real engineering reasoning worth being able to
explain, and it directly shaped the per-category experiment in Part 7.

### Why "just let the agent reason more" probably doesn't scale either

The instinctive alternative to hand-written rules is: don't write rules
at all, just have the agent (or a second LLM call reviewing the first
one's work) reason more carefully. This was directly tested, in effect,
across Part 4's three prompt rewrites -- "reason more carefully about
Material vs Equipment" is exactly what each of those prompt changes was
asking for, phrased differently each time. It didn't converge; the same
underlying bias kept resurfacing in new disguises. Self-critique using
the same model doesn't have an independent way to know it's wrong,
because to that model, its own reasoning feels sound. What actually
worked in Part 5 was checking a fact in code that the model *can't*
reason its way around -- a categorically different intervention, not a
better-worded version of the same one.

### Two distinct scaling problems, worth naming separately

- **More rows per data source** (hundreds of records becoming millions):
  a backend problem, not an agent problem. The current tool functions do
  a plain in-memory list scan on every call, which is fine at the
  project's real scale (1,260 shift records) and would fall over at
  millions. The fix is boring and well-understood -- move the data into
  an indexed database and have the tool function run a real query
  instead of a Python loop. The agent's *interface* to that tool never
  changes; it's a backend swap the agent doesn't even know happened.
- **More distinct data sources** (3 sources becoming dozens): a genuinely
  harder problem, because handing a model dozens of tool schemas in one
  call both bloats the prompt and measurably degrades its ability to
  pick the right one. The standard fix -- the same mechanism behind
  `ToolSearch` in the tool-calling environment this whole project was
  built inside of -- is a lightweight routing step: search a short
  catalog (names and one-line descriptions only) to narrow down to the
  handful of genuinely relevant tools for a given case, and only load
  *those* tools' full schemas, leaving the rest invisible. This keeps
  the model's actual decision space small and constant regardless of
  how large the total tool catalog grows.

### Agent vs. tool: a distinction worth being precise about

A first draft of the scaling document used the word "Specialist" loosely
for every domain in a proposed design, which reads as "one agent per
domain" -- and that's not accurate to what's actually needed. An agent
is something that makes its own LLM call(s) to reason across multiple
steps; a tool is a plain function with no LLM inside it at all. For this
project's actual domains (downtime, shifts, complaints), there's nothing
to reason about *inside* any one of them -- "look up shift records for
this machine and window" is one query, not a multi-step investigation.
The corrected design has exactly **one** agent and a growing set of
**tools**, with a rare "agent-as-tool" exception reserved only for a
domain complex enough to genuinely need its own reasoning loop --
invisible to the main agent either way, which just sees inputs and
outputs like any other tool call.

### True multi-agent orchestration, as a distinct alternative

A genuinely different design was also worked through and diagrammed:
give each root-cause category its own agent (its own LLM call, its own
short, non-competing prompt), coordinated by an orchestrator that
dispatches to them and synthesizes their structured findings. The
motivating case for this is precise, not hand-wavy: it directly targets
the whack-a-mole problem from Part 4, where every category's detection
logic sharing one prompt is *why* strengthening one broke another. Split
into separate agents with separate prompts, that kind of cross-category
interference becomes structurally impossible -- there's no shared prompt
left to fight over.

The honest costs were named alongside it, not glossed over: more LLM
calls and real latency (mitigated by dispatching independent
investigators in parallel, not serially); a new failure surface at
synthesis, where an orchestrator now has to weigh several reports
instead of reading raw data directly; and each investigator needing its
own step-cap and retry discipline, not just one. This alternative was
documented as a real, viable design -- and then actually built and
tested, rather than left as a diagram. See Part 7.

---

## Part 7: building and testing the alternative, twice

### Round 1: a hypothesis half-confirmed, net negative

Built on a separate branch (`experiment/per-category-investigator`), the
design from Part 6 became real code: `rca/agent_per_category.py`, with
**one reusable investigator-building function**, instantiated three
times (Human, Equipment, Material) with each one's own short prompt and
own tool subset -- deliberately not three separately hand-written agent
modules, since the whole point was to avoid duplicated, divergent code.
The three investigators are dispatched **in parallel** (via a plain
thread pool, since they're independent of each other and none needs
another's result), and their structured findings are synthesized by
plain priority-ordered code -- no separate "orchestrator" LLM call was
needed, since with only 3 fixed categories there's no ambiguity about
which ones to check; that only becomes a real decision at the "dozens of
sources" scale discussed in Part 6.

Tested against the same `eval.py` harness, same model, for a directly
comparable number: **61.1% overall -- worse than the single-agent
design's 72.2%,** despite Equipment jumping to a clean 100%.

Diagnosing the net loss, precisely, on C008 (a real Material case):

1. **Equipment cited a routine "Scheduled PM" maintenance record as if
   it were suspicious.** Told to look for equipment causes and nothing
   else, with no other explanation to weigh it against, the narrow
   Equipment investigator was more willing to accept weak evidence for
   its own category than the single shared-prompt agent had been --
   which had the advantage of comparing all four possible explanations
   against each other in one pass, making it easier to notice "none of
   this is very convincing." Narrowing the frame removed exactly the
   comparison that had been catching this.
2. **Material still confused same-machine for cross-machine, even with
   an explicit prompt warning against it, and with no way to recover.**
   It cited `C009` (its own machine) instead of `C010` (the genuine
   cross-machine match sitting in the very same tool result) --
   correctly rejected by the mechanical verifier, but round 1 had no
   retry mechanism at all for the per-category investigators. A rejected
   finding was simply discarded, so the complaint fell through to
   whichever *other* investigator's weaker finding happened to be
   available -- in this case, Equipment's flawed one.

A separate, smaller bug was also found and fixed during this testing:
Equipment's recurrence check (the gradual-wear signal) was accepting any
same-machine complaint as "recurrence" evidence, including one merely in
the *same week* -- a coincidence, not a multi-week pattern -- which was
about to wrongly break the red-herring case (C017) the exact same way
Part 4's Equipment regression had happened. Fixed by requiring the cited
complaints to span a genuinely different week, and reverified against
all five legitimate mold-wear complaints to make sure the stricter check
didn't also break the real pattern.

All of this -- the real numbers, both root causes, the fix that didn't
fully land -- was written up honestly in
`notes/per-category-experiment-results.md` and the recommendation at the
time was to keep the single-agent design, since it measured better.

### Round 2: closing the actual gaps, 100% (54/54)

Two targeted fixes, both aimed precisely at the two diagnosed causes
above, not speculative:

1. **Gave each investigator the retry-with-feedback loop that Part 5 had
   already proven works.** This was an actual gap in round 1, not a new
   idea -- the single agent's Material check already used exactly this
   pattern (reject, feed back the specific reason, bounded retry) with
   great results; it simply hadn't been carried over to the per-category
   investigators the first time around. Reused the same
   pending-tool-call `ToolMessage` fix from Part 5, since the identical
   OpenAI API constraint applies here too.
2. **Made Equipment's evidence claim structurally checkable instead of
   buried in free text.** Equipment now has its own tool variant,
   `submit_finding_equipment`, with a required `is_anomaly_not_routine`
   boolean field -- forcing the model to commit explicitly and
   separately to "this is a genuine anomaly, not routine maintenance,"
   rather than letting that judgment hide inside a paragraph of
   reasoning the verifier had no way to check. The verifier now requires
   this flag be true before accepting a cited downtime record, on top
   of the record simply being real.

Verified live on C008 -- round 1's exact failure -- before spending
anything on a full run: Equipment retried once and correctly stood down
("no evidence of an anomaly... routine"), Material retried once and
correctly found the genuine cross-machine citation it had access to all
along. Full evaluation: **100.0%, 54 out of 54 investigations, every
category perfect**, using roughly 0.67 correction retries per
investigation on average (a real, non-trivial added cost, not free).

Two honest caveats went into the write-up alongside the celebration,
deliberately, rather than stopping at the headline number:

- **These two fixes were diagnosed against, and then re-tested on, the
  exact same 18-complaint dataset.** The checks themselves are
  structural and general -- machine-ID equality, week comparison, an
  explicit yes/no flag -- not hardcoded to specific complaint IDs, so
  this isn't literally memorizing the answer key. But "100% on the data
  that inspired the fix" is a meaningfully different, weaker claim than
  "100% on complaints it has never seen," and that distinction matters
  for how confidently this number should be reported.
- **Real added latency and cost.** Close to one extra correction round
  per investigation on average, stacked on top of three parallel base
  calls per complaint (versus one agent's single thread of calls in the
  original design). The reliability this buys is real and measured, not
  assumed -- but it isn't free, and a fair comparison has to weigh both
  sides, not just the accuracy number.

This result reversed the recommendation from round 1: the per-category
design, once it had the same retry discipline the single-agent design
already relied on, now beats it decisively (100% vs 72.2%) on the same
measurement. Whether it becomes the system's actual default going
forward is a decision that should weigh those two caveats explicitly,
not just the headline number -- which is exactly the kind of judgment
call this whole project tried to make with real evidence at every step,
rather than by assumption.

---

## Key principles, distilled (for explaining this out loud)

1. **Test everything that can be tested without a model call, first, and
   keep doing it as the system grows.** Every tool function, every
   dataset property, every verification rule in this project got a
   plain Python test with no LLM involved before it was trusted. This
   is what made bugs cheap to catch instead of expensive.
2. **A hit-rate number from one run of one complaint is not evidence of
   anything, if the underlying model isn't deterministic.** Multiple
   runs per case, a real evaluation harness, and metrics that separate
   "wrong evidence" from "wrong reasoning" (`evidence_surfaced`) are
   what made every later decision in this project trustworthy instead
   of anecdotal.
3. **A metric that happens to match ground truth isn't automatically a
   real success.** The `conclusion_failed` bug (Part 2) and the round-1
   per-category discard-on-rejection bug (Part 7) both show the same
   shape: a broken run can coincidentally look correct, and only
   checking *why* an answer was reached (not just whether it matched)
   catches that.
4. **When a fix trades one failure for another, that's a signal about
   the architecture, not bad luck.** Three consecutive prompt patches in
   Part 4 each individually, diagnosably broke something the previous
   patch had fixed -- that's what motivated stopping prompt-only
   iteration and trying a structurally different kind of fix
   (mechanical verification) instead of a fourth version of the same
   kind of fix.
5. **Mechanical, deterministic verification of a specific factual claim
   is a different (and often stronger) tool than asking a model to
   reason more carefully about the same claim.** This was the single
   highest-leverage idea in the whole project -- not a smarter prompt,
   a fact-check in plain code that the model's own persuasive reasoning
   can't talk its way around.
6. **Retry with the *specific* reason for rejection, not a generic "try
   again."** Verification without a feedback-driven retry loop (round 1
   of the per-category experiment) just throws away a wrong answer with
   nothing to replace it; verification *with* that retry (Part 5, and
   round 2 of Part 7) turns a rejection into a self-correction most of
   the time.
7. **Narrower isn't automatically more reliable.** Splitting one prompt
   into several narrow ones fixed exactly the problem it was built to
   fix (cross-category interference) and simultaneously introduced a
   new one (less internal skepticism, since there's nothing else to
   compare a weak piece of evidence against) -- the round-1 result is a
   genuinely useful, real counter-example to the intuition that
   "smaller, more focused prompts must be safer."
8. **A perfect score on the data you tuned against is a narrower claim
   than a perfect score in general**, and saying so explicitly, in
   writing, at the moment of the result, is worth more than letting the
   headline number speak for itself.
