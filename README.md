# MBCIE AI Engineer Assessment -- Devendra Sunil Khandale

Two systems, one FastAPI service:

- **Task 1** -- a LangGraph ReAct agent that autonomously investigates
  manufacturing defect complaints across three synthetic data sources
  (complaint log, machine downtime, operator shifts) and returns a
  root-cause hypothesis with a full evidence chain.
- **Task 2** -- a multi-format document extraction pipeline (digital PDFs
  + scanned images) with a per-document confidence score, a human-review
  queue for low-confidence results, and a measured cost-per-1,000-docs
  figure.

> **Reviewing this repo:** the branch to look at is
> `experiment/per-category-investigator`, not `main`. Everything below --
> the confidence scoring, the review queues, the ad-hoc complaint input,
> the demo console -- was built after `main` was last updated, and lives
> only on that branch. See [Limitations](#limitations) for why it wasn't
> merged during the assessment window.

## Quick start (aim: under 10 minutes)

```bash
git clone https://github.com/Devendra-git-acc/mbcie-capa_agent.git
cd mbcie-capa_agent
git checkout experiment/per-category-investigator

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Task 2 also needs the **Tesseract OCR binary** (not pip-installable --
`pytesseract` just calls out to it):
- Windows: [UB-Mannheim's installer](https://github.com/UB-Mannheim/tesseract/wiki), then add it to `PATH`
- macOS: `brew install tesseract`
- Linux: `apt-get install tesseract-ocr`

Then:

```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY (or switch LLM_PROVIDER=openrouter and set
# OPENROUTER_API_KEY instead), and set BASIC_AUTH_PASSWORD to something
# real rather than the placeholder

uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/** -- it redirects to a demo console covering
both tasks (see [Demo console](#demo-console-ui) below). Everything else
lives behind Basic Auth (username/password from `.env`) except `/health`.
Swagger docs at `/docs`.

If you'd rather not use a browser, `rca/run_single.py` and
`extraction/pipeline.py` (`python pipeline.py`) both run standalone without
the API layer at all.

**Docker**: a `Dockerfile`/`docker-compose.yml` exist (bundling Tesseract,
running as non-root, with a healthcheck) but were never verified with a
real `docker build` in this environment -- no Docker was available on the
development machine. Treat it as unverified, not broken; the
`pip install` path above is the tested route. See [Limitations](#limitations).

---

## Task 1 -- CAPA-style root-cause investigation

### The data

No dataset was supplied by the brief -- `rca/generate_data.py` builds a
synthetic tyre/cycle factory: 18 complaints, 276 downtime records, 2,772
shift records across 6 machines and ~20 production weeks, with root
causes **deliberately planted**, not random:

| Category | Storyline planted in the data |
|---|---|
| Equipment | A sensor calibration drift event on one machine explains a cluster of complaints |
| Human | A trainee-staffed shift correlates with a defect on another machine |
| Material | The *same* defect appears on two *different* machines in the same week -- a cross-machine pattern that only makes sense as a bad material batch, not equipment or human error on either machine alone |
| Insufficient evidence | A deliberate red herring -- a complaint with no explanatory downtime, no trainee shift, no cross-machine pattern, where the honest answer is "we don't know," not a forced guess |

`rca/validate_dataset.py` checks the generated data has no accidental
contamination (e.g. a stray trainee shift that would leak the wrong answer
into a storyline meant to test something else) before any agent ever sees
it. `rca/generate_data_unseen.py` later generated a second, disjoint batch
(18 more complaints, different machines/weeks) as a genuinely held-out
generalization test -- see [Evaluation](#evaluation-methodology).

### The agent

`rca/agent.py` -- a two-node LangGraph graph (`agent` <-> `tools`,
looping until a dedicated `submit_conclusion` tool call ends it). Tools:
`decode_batch_code`, `query_downtime`, `query_shifts`,
`query_complaints_by_machine`.

### Architecture decisions -- the why

**Why a ReAct agent instead of a single prompt with all the data stuffed
in.** No two complaints need the same evidence. One needs downtime
records, another needs shift records, a third needs a cross-machine
complaint lookup no other complaint needs. Deciding *which* sources are
relevant, and in what order, per complaint, is exactly the kind of
multi-step judgment call an agentic tool loop is for -- a single prompt
would either need every source stuffed in every time (wasteful, and
confusing when most of it is irrelevant) or a hand-written router deciding
relevance in advance (which is just reimplementing what the agent already
does, with none of its flexibility).

**Why a dedicated `submit_conclusion` tool instead of parsing free text.**
Forcing the final answer through a tool call with a fixed schema
(`root_cause_category`, `root_cause_hypothesis`, `confidence`,
`recommended_corrective_action`) means "is this a real, structured
conclusion" is a type check, not a regex over prose. It also gives a
clean termination signal for the graph's conditional edge -- the loop
doesn't guess when the model is "done," it waits for the one tool call
that means done.

**Why mechanical, code-level verification instead of only prompting
harder.** During testing, the agent repeatedly mistook same-machine
*recurrence* for a genuine cross-machine *Material* pattern -- exactly the
distinction that storyline exists to test -- and survived three separate
prompt corrections aimed at fixing it. `_verify_material_citation()`
checks this in code instead: it pulls the complaint IDs the model actually
cited, looks up their real machine and defect data, and rejects the
conclusion if the citation doesn't hold up, with one bounded retry
(`MAX_VERIFICATION_RETRIES = 2`) that tells the model exactly what was
wrong. This is the single result I'd point to if asked "what did you learn
building this": mechanical verification of a specific, well-understood
failure mode reliably closes it; a fourth prompt tweak was not going to.

**Why a weighted composite confidence score instead of asking the model
"how confident are you."** An LLM's self-reported confidence is
notoriously uncorrelated with whether it's actually right. `_confidence_score()`
instead combines three independently-checkable signals -- whether the
conclusion survived mechanical verification cleanly, a category-level
reliability prior (Material and Process/Scheduling conclusions are
capped lower even when clean, because they depend on comparisons across
records rather than a single directly-cited event), and whether the
hypothesis actually cites a real record ID -- into one score, gated at
0.80 for `needs_human_review`. A clean Material conclusion still scores
0.75 and gets flagged, by design: the category prior alone caps it below
threshold, independent of anything else going right.

**Why a single agent, not one investigator per category.** I built and
measured an alternative: three parallel, narrowly-prompted investigators
(Human/Equipment/Material) synthesized by plain code, on the theory that a
narrower prompt reasons more reliably than one prompt covering every
category. First pass actually regressed (61.1% vs. this design's 100% on
the same 18 complaints) for two concrete reasons -- Equipment's narrower
framing lost the cross-category comparison that catches routine
maintenance being mistaken for equipment failure, and Material had no
retry loop, so a rejected finding was just discarded instead of corrected.
Both fixed, and round 2 reached parity (100%, 54/54 runs). But on a
genuinely held-out dataset the per-category design fell to 88.9%, with a
specific, still-unexplained asymmetry -- the same storyline resolved
correctly on one machine and incorrectly on another. That's honestly
documented in `notes/per-category-experiment-results.md` as a real,
unresolved finding, not swept under the rug. Given that, and that the
single-agent design is materially simpler to reason about, verify, and
extend, it's what's exposed through the API. The multi-agent variant is
kept as a working, evaluated experiment (`rca/agent_per_category.py`),
not the shipped path -- see `notes/scaling-architecture.md` for when a
real multi-agent split would earn its complexity (many more categories,
enough that one prompt genuinely can't hold them all reliably).

**Why complaints don't have to already exist in the dataset.**
`POST /investigate` accepts a fresh complaint -- just a `batch_code` and
`defect_description` -- alongside `POST /investigate/{complaint_id}` for
the fixture set. This isn't a shortcut; it reflects how the system would
actually be used: `complaints.json` is only ever the *entry point*, the
real evidence (`downtime.json`, `shifts.json`, other complaints on the
same machine) already exists independent of any specific complaint, the
same way a real machine's operational history doesn't wait for someone to
complain about it. A batch code referencing a machine/week the data has no
records for isn't an error -- the agent correctly and honestly reaches
"Insufficient evidence," proving the honesty mechanism generalizes past
the one red-herring complaint it was built for.

### API surface

| Endpoint | What it does |
|---|---|
| `GET /complaints` | List fixture complaint IDs |
| `POST /investigate/{complaint_id}` | Investigate one fixture complaint |
| `POST /investigate` | Investigate a fresh, typed-in complaint (`batch_code`, `defect_description`, +optional fields) |
| `GET /investigate/review-queue` | Complaints whose most recent investigation needs a human -- persisted, updated on every investigation, removed again if a re-run comes back clean |

---

## Task 2 -- multi-format document extraction with cost control

### The data

`extraction/data/` -- 10 invoices/purchase orders in genuinely
inconsistent formats: 6 digital PDFs, 4 scanned-image sets (one,
`D09`, spanning two pages), different vendors, different layouts, some
with line-item tables, none templated identically. `extraction/generate_docs.py`
produced them; `extraction/data/ground_truth.json` is the answer key
`extraction/check_accuracy.py` scores against.

### The pipeline

`extraction/pipeline.py`: auto-detects digital-text vs. scanned (`is_pdf_actually_digital`),
extracts text directly via PyMuPDF or via Tesseract OCR, then a single LLM
call structures it into a Pydantic-validated `InvoiceExtraction` schema
(doc type/number/date, vendor, buyer, currency, line items, subtotal,
tax, total).

### Architecture decisions -- the why

**Why the confidence score is computed, not asked for.** Same principle
as Task 1: `score_document()` combines OCR/extraction-source confidence,
field completeness (how many required fields actually came back
non-null), and an **arithmetic consistency check computed in Python**
(does `subtotal + tax == total`, do the line items sum to the subtotal) --
not something the LLM is asked to self-assess. A document that "looks"
extracted fine but doesn't arithmetically reconcile is flagged
(`needs_review`) with the specific reason, rather than silently accepted.

**Why low-confidence documents go to a queue instead of being guessed.**
The brief requires this explicitly, and it's the same design as Task 1's
review gate: `extraction/output/review_queue.json`, exposed at
`GET /review-queue`, updated per-document (not just once per batch run --
see the [note below](#a-bug-found-while-re-verifying-this-project) on a
real gap this caught).

**Why the cost-reduction idea targets `line_total`, not the model or the
prompt.** Measured cost is **$0.36 per 1,000 documents** (gpt-4o-mini
pricing, ~900 input / ~380 output tokens/doc average, 10 fixture docs).
The concrete reduction idea: stop asking the model to emit `line_total`
per line item -- it's `quantity * unit_price`, pure arithmetic the
pipeline can compute itself post-hoc, so paying output tokens for the
model to produce it buys no accuracy (an LLM's multiplication isn't more
trustworthy than code doing it directly), and the cost scales with line-item
count (this run's worst case had 35 items on one document).

### A bug found while re-verifying this project

Worth stating plainly rather than hiding it: while re-verifying both
tasks end-to-end after adding features, I found `GET /review-queue`'s own
docstring claimed single-document extraction (`POST /extract/sample/{doc_id}`)
updated the same queue file `run_batch()` writes. It didn't --
`extraction/output/review_queue.json` had never actually been created by
anything reachable through the API. A document flagged `needs_review=true`
was silently invisible to the one endpoint whose job is surfacing it.
Fixed by sharing the review-reason logic between `run_batch()` and the API
endpoint and updating the queue incrementally per document, then verified
live (a real low-confidence document appearing in the queue, a clean one
not, a re-extraction removing an entry that's no longer flagged). Left in
here because catching this *is* the verification discipline this project
is built around, and a README that only reported clean results after the
fact wouldn't be honest about how it actually got that way.

### API surface

| Endpoint | What it does |
|---|---|
| `GET /documents` | List the 10 fixture document IDs |
| `POST /extract/sample/{doc_id}` | Extract one fixture document, persisted to `extraction/output/` |
| `POST /extract` | Extract a real uploaded file (PDF, or multiple page images for one scan) -- not persisted, genuinely ad hoc |
| `GET /review-queue` | Documents currently flagged for human review |

---

## Demo console (`/ui`)

A single dependency-free static page (`static/index.html`, no build step,
no CDN, served by FastAPI itself at `GET /ui`, with `/` redirecting there)
covering both tasks: a complaint picker *and* a "type your own complaint"
form for Task 1 with the full evidence chain, confidence breakdown, and
review queue rendered live; a sample-document picker *and* a real
drag-and-drop upload for Task 2 with the same. Everything on the page
calls the real API directly with Basic Auth entered in the page itself --
nothing on it is mocked.

---

## Evaluation methodology

`rca/eval.py` runs each of the 18 fixture complaints **3 times**, not
once -- this was not an arbitrary choice. Directly observed during
testing: the identical complaint produced a *different* category on two
consecutive runs at `temperature=0`. A single pass/fail per complaint
would misrepresent reliability either way (a lucky pass, or an unlucky
fail), so the eval reports a hit rate across repeated runs, plus separate
tracking for `conclusion_failed` (a broken run must never count as a
coincidental hit) and `evidence_surfaced` (did the right evidence get
retrieved at all, independent of whether the final reasoning used it
correctly -- separates a reasoning failure from a retrieval failure).

**Last full run** (`rca/eval_results.json`, 18 complaints x 3 runs = 54
investigations): **100% hit rate**, 0% `conclusion_failed`, 0%
`material_verification_failed`, 100% `evidence_surfaced`. This run
predates the confidence-scoring feature (no `human_review_rate` in that
file) -- it wasn't re-run afterward given real API cost/time during the
assessment window; see [Limitations](#limitations).

**Held-out generalization test** (`rca/answer_key_unseen.json` --
18 *new* synthetic complaints, different machines, different weeks,
generated after the agent's design was finalized, never used to tune
anything): **88.9%** (16/18). Equipment, Human, and Insufficient evidence
all held at 100%; Material fell to 50%, with a specific, reproducible,
still-unexplained asymmetry -- the same cross-machine storyline resolved
correctly on one machine pair and incorrectly on another, ruled out as
either a data-generation bug or a retrieval failure (both checked
directly) before being accepted as a genuine reasoning gap. Documented in
full in `notes/per-category-experiment-results.md`.

---

## Repository structure

```
rca/                        Task 1
  agent.py                    the shipped LangGraph agent
  agent_per_category.py       the evaluated multi-agent experiment (not shipped -- see above)
  tools.py                    decode_batch_code / query_downtime / query_shifts / query_complaints_by_machine
  data/                       generated synthetic complaints / downtime / shifts
  generate_data.py            builds the tuned dataset, with planted root causes
  generate_data_unseen.py     builds the held-out generalization dataset
  answer_key.json / answer_key_unseen.json    ground truth (only eval.py/validate_dataset.py read these)
  eval.py                     multi-run hit-rate evaluation harness
  validate_dataset.py         checks generated data for contamination before any agent sees it
  test_agent.py                mock-based unit tests, no API calls
  output/review_queue.json    Task 1's persisted human-review queue

extraction/                 Task 2
  pipeline.py                  extraction + confidence scoring + review-queue logic
  data/                        10 fixture invoices/POs, mixed digital + scanned
  generate_docs.py             built the fixture documents
  estimate_cost.py             derives the cost-per-1000-docs figure from real measured usage
  test_pipeline.py             mock-based unit tests, no API calls
  output/                      per-document JSON, accuracy report, cost estimate, review queue

main.py                     FastAPI app -- both tasks, Basic Auth, the demo console mount
static/index.html           the demo console
shared/config.py            shared auth config

notes/                       design discussion and process documentation, not required by the brief
  engineering-journal.md          full narrative of every issue found and fixed, in order
  per-category-experiment-results.md   the multi-agent experiment's results in detail
  scaling-architecture.md         what this would look like with far more tools/data sources
  production-readiness-agentic.md What's left for real production, agent-specific and general
  demo-video-script.md            script used to record the submission demo video
```

---

## Limitations

Honest account of what's incomplete and what I'd do differently with more
time, rather than presenting this as more finished than it is.

**Non-determinism is handled offline, not in production.** `eval.py`
proved this agent isn't fully deterministic even at `temperature=0` (the
same complaint produced different categories across runs). The eval
harness accounts for that by sampling 3 times; the live API doesn't --
`POST /investigate` returns whatever one sample happens to produce. With
more time, I'd add self-consistency voting (run N samples, take the
majority, treat disagreement itself as a review-trigger) at least for
borderline-confidence cases, rather than only measuring the problem
offline and not mitigating it live.

**Grounding verification is uniform in *format*, not in *rigor*.**
`_verify_material_citation()` is a genuine mechanical check -- it looks up
the cited complaint IDs and confirms they're real, cross-machine,
matching-symptom evidence. The `citation_signal` used for every *other*
category only checks that *something shaped like* a record ID appears in
the hypothesis text -- it doesn't confirm that ID was actually returned by
a tool call this run. A fabricated-but-plausible ID would currently score
the same as a real one outside Material. Generalizing the Material
verifier's actual grounding check to every category is the single highest-value
fix I'd make next.

**The held-out Material asymmetry is open, not resolved.** 88.9% on
genuinely unseen data, with one specific, reproducible failure I could not
fully explain -- see above. I'd want more held-out storylines specifically
targeting that asymmetry before trusting Material's real-world reliability
at the same level as the other three categories.

**No memory across investigations, by design, but worth being explicit
about the tradeoff.** Cross-complaint pattern-finding (recurrence,
cross-machine defects) is handled by deterministic tool queries over the
full dataset, not by the agent recalling past investigations -- this is
more reliable at the current scale (36 complaints) than an LLM-memory
layer would be. It stops being the right call once volume outgrows what
structured filtering handles well; at that point a semantic/vector memory
over resolved CAPAs becomes worth its cost. Not needed yet, but not free
forever either -- discussed further in `notes/production-readiness-agentic.md`.

**The review queues are read, not resolved.** Both `GET /investigate/review-queue`
and `GET /review-queue` tell you what's flagged; there's no endpoint yet
for a human to record a decision (approve/override + why). For a CAPA
process specifically, that decision trail -- who reviewed what, and why --
is the actual artifact an audit would ask for, not just the AI's own
output. Designing that properly means pausing the LangGraph run itself
before finalizing a flagged conclusion (via LangGraph's checkpointer/
`interrupt_before`) rather than flagging after the fact, which I didn't
have time to build.

**No structured logging or persisted reasoning trace.** Right now,
diagnosing what happened during a specific past investigation means
either reading whatever `print()` output was on screen when it ran, or
re-running it -- and a re-run isn't guaranteed to reproduce the same
result, given the non-determinism already discussed above.
`evidence_chain` and `verification_log` are returned in each response,
but nothing persists them centrally or makes them queryable afterward
(e.g. "show me every investigation last week that called
`query_complaints_by_machine` more than twice," or "what did the agent
actually see before concluding X on this complaint two days ago"). With
more time I'd replace the `print()` calls with structured logging
carrying a correlation ID per request, and either wire in LangSmith
(close to free here, since this is already LangGraph -- one env var) or
at minimum persist each run's full message history to a queryable store,
rather than treating the response body as the only record of what
happened. Covered in more depth, with concrete options, in
`notes/production-readiness-agentic.md`.

**Docker is unverified.** No Docker was available in the development
environment, so the `Dockerfile`/`docker-compose.yml` were written
carefully but never actually run. I'd verify this before calling it a
real deployment option rather than a best-effort one.

**`main` is stale relative to this branch.** All of the work described in
this README -- confidence scoring, both review queues, ad-hoc complaint
input, the demo console -- happened after `main` was last updated and
lives only on `experiment/per-category-investigator`. I'd normally merge
before calling something done; I kept them separate here so the original,
simpler baseline stays easy to diff against everything that came after it,
but a reviewer who only looks at `main` will see a materially earlier
version of this project.

**No production hardening beyond what the brief asked for.** No rate
limiting, no TLS termination (this app shouldn't hold certs itself
anyway), no per-user accounts (Basic Auth is one shared credential), no
CI pipeline running the test suite or `eval.py` automatically.
None of this was in scope for a 3-day prototype, but it's the honest gap
between "working prototype" and "something I'd put in front of real
plant data" -- covered in much more depth, with concrete mitigation
approaches for each, in `notes/production-readiness-agentic.md`.
