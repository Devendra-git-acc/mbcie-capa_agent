# Production readiness: gaps and resolutions

Not implemented -- a design/interview-prep document in the same spirit as
`scaling-architecture.md` and `per-category-experiment-results.md`: what
this system would still need before it runs unattended against real
manufacturing data, organized the way a senior AI engineer would actually
triage it rather than as a generic checklist.

Two parts, deliberately separated:

- **Part 1 -- agentic-AI-specific gaps.** Failure modes that exist
  *because this is an autonomous, tool-using, multi-step LLM system*, not
  because it's a web service. A perfectly hardened REST API around a bad
  agent design still ships bad conclusions.
- **Part 2 -- general production gaps.** The rest of what any backend
  service needs regardless of whether an LLM is involved.

A standalone **hallucination-reduction toolkit** sits at the end of Part 1,
since it's the single most-asked interview question about this kind of
system and the answer is scattered across several of the gaps below --
worth having in one place.

---

## Part 1: Agentic-AI-specific production gaps

### 1. Grounding verification is inconsistent across categories

**Problem.** `_verify_material_citation()` in `agent.py` mechanically
checks Material conclusions against real machine/complaint data. Equipment,
Human, and Scheduling conclusions get no equivalent check -- the
`citation_signal` in the confidence score just checks that *something
shaped like* a record ID (`_RECORD_ID_RE`) appears in the hypothesis text.
That is a format check, not a grounding check: a fabricated `D0999` scores
identically to a real, tool-returned one, because nothing confirms the
cited ID was actually returned by a tool call this run.

**Resolutions.**
- Generalize the Material verifier into a category-agnostic one. Build an
  "evidence index" while the graph runs -- every tool result's IDs get
  collected into a set as each node executes. At conclusion time, extract
  every cited ID from the hypothesis and check `cited_ids ⊆
  evidence_index`. Any category with an ungrounded ID gets the same
  retry-with-correction loop Material already has (the synthetic-
  `ToolMessage`-insertion pattern), not just Material.
- Force structured citation instead of free-text citation. Add an explicit
  `supporting_ids: list[str]` field to `submit_conclusion`'s schema,
  separate from `root_cause_hypothesis`. Checking a structured field
  against the evidence index is far more robust than regex-scraping prose
  -- the model can't bury a fabricated ID inside a sentence that happens
  to parse.
- Turn `citation_signal` into a fraction, not a binary: `grounded_ids /
  cited_ids`. A conclusion citing three IDs where one is fabricated should
  score worse than one citing a single real ID -- currently they score the
  same.
- Add a cheap LLM-judge faithfulness check as a second opinion, gated only
  to borderline confidence scores (roughly 0.6-0.85) to control cost: a
  second, smaller model call asks "does this hypothesis logically follow
  from these exact facts?" This catches *logical* hallucination (real IDs,
  wrong inference) that ID-matching alone cannot.
- Extend the existing retry discipline uniformly -- `MAX_VERIFICATION_RETRIES`
  and the correction-message pattern already proven for Material just needs
  to stop being gated on `category == "Material"`.

### 2. No defense against prompt injection via tool or complaint content

**Problem.** Complaint descriptions are free text that flows straight into
the LLM's context with no structural signal that it is *data*, not
*instructions*. OWASP's #1 LLM risk, and this system has no mitigation for
it today -- fine on a synthetic, single-author dataset; not fine the
moment real operators start typing complaint text.

**Resolutions.**
- Delimit and label untrusted content explicitly. Wrap every tool-returned
  record and every complaint description in explicit tags (e.g.
  `<untrusted_data>...</untrusted_data>`) and add a system-prompt clause
  that content inside those tags is never to be treated as instructions,
  regardless of what it says. Cheap, doesn't eliminate the risk, but
  raises the bar against naive injection.
- Dual-LLM / quarantine pattern (the strongest mitigation available).
  Route all free-text fields through a separate, privilege-less
  "extraction" LLM call first, whose only job is to pull structured facts
  into a fixed schema (e.g. `{defect_type, symptom}`). The orchestrator
  agent that makes the final decision never sees the raw text directly --
  only the sanitized structured extraction. This removes the attack
  surface instead of arguing with it.
- Heuristic pre-filter: cheap regex/keyword scan on incoming complaint
  text for instruction-like patterns (`ignore previous`, `system:`,
  role-play framing) before it reaches the agent. Weak alone (easily
  evaded), useful as one layer among several.
- Output-side grounding as the real backstop -- the same mechanism as gap
  #1. If every claim in the final conclusion must trace back to a real,
  tool-returned record ID, an injection attack can only succeed by making
  the model draw a *grounded but wrong* conclusion, which is a much
  narrower target than "say anything the attacker wants."
- Keep tools least-privilege (already true here). Every current tool is a
  read-only query -- no destructive/action tool exists for an injected
  instruction to steer toward. If a write/action tool is ever added (e.g.
  "auto-file this CAPA"), it needs a human-approval gate before execution,
  never a direct agent-triggered side effect.
- Add adversarial cases to `eval.py`'s golden set: complaints whose text
  contains injection attempts, with an expected outcome of "the
  conclusion is unaffected by the injected instruction." Run in CI, not
  just correctness cases.

### 3. Runaway control is per-mechanism, not a unified circuit breaker

**Problem.** The ReAct step cap and `MAX_VERIFICATION_RETRIES` are two
separate, hand-rolled limits. Nothing stops a future third retry
mechanism from being added without its own cap, and there is no single
wall-clock or total-cost ceiling across a run.

**Resolutions.**
- Use LangGraph's built-in `recursion_limit` (`graph.invoke(..., config=
  {"recursion_limit": N})`) as the framework-level backstop, rather than
  relying only on custom step-counting to be correct.
- Introduce one `RunBudget` object threaded through graph state: tracks
  tokens consumed, tool calls made, and elapsed wall-clock time. Every
  node checks the budget before proceeding and short-circuits to a
  "budget exceeded -> needs_human_review" terminal state if exceeded --
  one object, one source of truth, instead of scattered counters.
- Wrap the LLM client with a token-accounting decorator that sums usage
  against the per-run cap in real time, so a run can be killed mid-flight
  rather than only reported on afterward.
- Explicit wall-clock timeout around the whole `investigate()` call, as
  defense-in-depth against a hung provider call no step/token counter
  would catch.
- Centralize every limit in one settings module (`MAX_TOOL_CALLS`,
  `MAX_TOKENS_PER_RUN`, `MAX_WALL_CLOCK_SECONDS`, `MAX_VERIFICATION_RETRIES`)
  so a new retry mechanism reuses the existing budget check instead of
  inventing a new cap.
- Fail safe, not silent: any budget breach always resolves to
  `needs_human_review=True` with a distinct reason
  (`"budget_exceeded"`), never a silently truncated or best-effort answer.

### 4. Self-consistency exists only in offline eval, not production

**Problem.** `eval.py`'s own methodology exists *because* gpt-4o-mini at
temperature=0 was proven non-deterministic here during Day 2 testing --
the identical C012 complaint produced a different category on two
consecutive runs. That insight justified running 3 samples per complaint
offline. Production still runs exactly one sample and ships whatever it
gets, gambling on the exact non-determinism already proven to exist.

**Resolutions.**
- Adaptive multi-sample voting in production: wrap `investigate()` in
  `investigate_with_consensus(complaint_id, n=3)`, running N samples in
  parallel (reuse the `ThreadPoolExecutor` pattern already built for
  `agent_per_category.py`), then take the majority category vote.
- If there's no strict majority, treat that disagreement as a stronger,
  more principled review-trigger than any single run's self-reported
  confidence score.
- Cost-aware escalation, not always-3x: run one sample first, escalate to
  full N-sample voting only when that first run's confidence score lands
  in a borderline band (roughly 0.6-0.85). Cheap runs stay cheap; only
  ambiguous ones pay the 3x cost.
- Log agreement rate as a first-class production metric -- how often do
  repeated runs on the same or similar complaints agree. This becomes a
  live drift signal feeding the eval-gate discipline in gap #8.
- When flagged for review due to disagreement, surface *all* the
  conflicting conclusions and their confidence breakdowns to the reviewer,
  not just one opaque flag.

### 5. No reasoning trace, only a final artifact

**Problem.** `evidence_chain` / `verification_log` are returned inside
each response but live nowhere central, queryable, or replayable. There is
no way to ask "show me every investigation last week that called
`query_complaints_by_machine` more than twice" without re-running
everything -- and re-running it may give a different answer (gap #4).

**Resolutions.**
- Wire in LangSmith. Since this is already LangGraph, this is close to
  free -- set `LANGCHAIN_TRACING_V2=true` plus an API key and every
  node/tool call is automatically traced with a UI for inspecting,
  filtering, and diffing runs. Lowest effort, highest leverage.
- Self-hosted alternative: Langfuse -- same idea, open-source, OTel-based,
  useful when traces can't leave your own infrastructure.
- Vendor-neutral: OpenTelemetry GenAI semantic conventions -- emit spans
  per node/tool call, exportable to any existing OTel backend (Grafana
  Tempo, Honeycomb, etc.) rather than a dedicated LLM-ops tool.
- Minimum viable, no new dependency: persist the full LangGraph message
  history and tool-call sequence as JSON to a `traces` store keyed by
  `investigation_id`. Answers "what happened during run X" without
  re-running it, even with zero tooling investment.
- Thread a correlation ID through every log line and tool call within one
  investigation so even plain structured logs can be filtered to one
  run's complete story.

### 6. No reproducibility guarantee tied to a conclusion

**Problem.** Given a stored conclusion, nothing pins down exactly what
produced it -- model snapshot, prompt version, and the facts it was shown
can all have silently drifted by the time anyone asks "why did the agent
say that."

**Resolutions.**
- Pin and log the exact model snapshot, not the family alias --
  `gpt-4o-mini-2024-07-18`, not `gpt-4o-mini`. Providers roll aliases
  forward silently; the alias alone doesn't reproduce anything.
- Hash the system prompt and store the hash alongside every conclusion,
  with a lookup table mapping hash to full prompt text -- lets you
  reconstruct exactly which prompt version produced any historical
  decision even after later edits.
- Persist the raw tool outputs used in that run centrally (the data
  already exists in `evidence_chain`; make sure it's stored, not only
  returned and discarded).
- Log LLM call parameters including `seed` -- OpenAI's API supports a
  best-effort `seed`; it doesn't guarantee bit-identical output but
  meaningfully improves reproducibility over leaving it unset.
- Version the tools, not only the prompt. A tool bug (this project hit two
  real ones -- the Equipment recurrence-window check and the held-out
  dataset's fix-date bug) affects determinism as much as a prompt change;
  tag tool code with a version/hash too.
- Build a replay capability: given a stored trace, replay the same
  conversation by feeding back the *stored* tool outputs instead of
  issuing live tool calls. Isolates "did the facts change" from "did the
  model's reasoning change" when debugging a surprising conclusion.

### 7. Human-in-the-loop is bolted on after the fact

**Problem.** The review queue (`rca/output/review_queue.json` +
`GET /investigate/review-queue`) classifies a conclusion *after* the agent
has already fully run and committed it. Nothing stops a bad conclusion
from being written as "final" before a human ever sees it.

**Resolutions.**
- Use LangGraph's `interrupt_before` / checkpointer support. With a
  persistent checkpointer (`SqliteSaver` / `PostgresSaver`), the graph can
  genuinely pause before the `submit_conclusion` node commits -- e.g.
  whenever category is Material, verification failed, or pre-check
  confidence is low -- and wait for a human decision before continuing,
  rather than always running to a self-declared final answer.
- Split into a provisional vs. final conclusion node. The graph always
  computes a provisional conclusion; a conditional edge routes either to
  auto-finalize (clean, high-confidence) or to a paused
  "awaiting-human-approval" state (flagged cases) -- finalization becomes
  a distinct, gated step rather than an implicit side effect of the agent
  finishing.
- Change the API shape for flagged cases: `POST /investigate/{id}` returns
  `"status": "pending_review"` immediately instead of a completed
  conclusion; a separate `POST /investigate/{id}/approve` (or
  `/override`) resumes the paused graph via the checkpointer with the
  human's input injected as a message.
- This is also what actually closes the review-queue gap noted in the
  earlier general assessment: the resolve endpoint should *resume the
  graph*, not just record a decision next to an already-finalized JSON
  blob.

### 8. No agent-version regression gate

**Problem.** `eval.py` exists and works but nothing runs it automatically
on change, and there is no history of scores over time -- "did accuracy
improve after last week's prompt edit" isn't answerable without manually
re-running and remembering the previous number.

**Resolutions.**
- Wire `eval.py` into CI (GitHub Actions): trigger on any PR touching
  `agent.py`, `tools.py`, or the system prompt; compare against a stored
  baseline `eval_results.json`; fail the PR if hit rate drops or
  `human_review_rate` / `conclusion_failed_rate` rises past a threshold.
- Store eval results with the git commit SHA and timestamp in a small
  experiment log (even an append-only JSON list), or adopt MLflow /
  Weights & Biases for proper tracking with charts over time.
- Shadow deployment before promotion: run a new agent version in parallel
  with the current production version on live traffic without acting on
  its output, and diff the conclusions before cutting over.
- Canary rollout for prompt/model changes: route a small percentage of
  real traffic to the new version, watch `human_review_rate` and
  `conclusion_failed_rate` in production, then ramp -- the same discipline
  used for any ML model rollout, applied to prompts.
- Grow the golden set from real failures. Every production near-miss or
  reviewer override gets folded back into the eval set -- exactly the
  discipline already used building `answer_key_unseen.json` for held-out
  testing. Eval should compound over time, not stay frozen at 18 fixtures.

### 9. Multi-agent design has no failure contract

**Problem.** `agent_per_category.py`'s three parallel investigators
(Human/Equipment/Material) have no defined behavior for one of them timing
out, erroring, or returning malformed output.

**Resolutions.**
- Enforce a schema contract at the synthesis boundary. The investigator's
  `submit_finding` / `submit_finding_equipment` tool schemas already
  constrain output shape -- validate the synthesizer's *input* against
  that same schema explicitly, and reject/flag anything malformed rather
  than pass it through silently.
- Per-investigator timeout with graceful degradation:
  `ThreadPoolExecutor.submit` + `as_completed(timeout=...)`; if the
  Equipment investigator doesn't return in time, the synthesizer proceeds
  with the 2-of-3 findings it has, and the confidence score / review
  trigger reflects the gap (lower `category_signal`, or an explicit
  `"partial_investigation"` review reason) instead of either blocking the
  whole run or silently ignoring the missing input.
- Distinguish "investigator found nothing" from "investigator crashed" --
  semantically different (a valid negative vs. an infrastructure failure)
  and must never be conflated into the same downstream signal. A crash
  should always route to review; a genuine negative finding is normal
  operation.
- Circuit breaker per category: if one investigator fails repeatedly
  across runs, temporarily exclude it and flag all affected runs for
  review rather than let every subsequent investigation silently degrade
  the same way.

### 10. No retrieval strategy for scale

**Problem.** `query_downtime`, `query_complaints_by_machine`, etc. are
hand-written deterministic filters over a fixed 18-complaint dataset --
this is a fixture, not an architecture. It works today; it's not what
handles real data volume (see `scaling-architecture.md` for the
tool-count side of this same question).

**Resolutions.**
- Hybrid retrieval, not pure RAG. Keep the deterministic structured
  queries (machine_id, date range, category filters) for the precision
  they already give -- that's a strength, not a weakness. Add semantic
  search (a vector store: pgvector, FAISS) only for the fuzzy-matching
  case current tools can't do: "find complaints with a similar defect
  description across machines" when exact keyword/category filtering
  misses the match.
- Pre-aggregation tools for high-volume queries: when a query would return
  hundreds of records, add a tool that returns aggregates first (counts
  by category/machine/week) so the LLM reasons over a summary and can
  drill down, instead of being handed a flood of raw records that blows
  the context window.
- Pagination-aware tools -- let the agent explicitly request more results
  if the first page wasn't enough, instead of a tool that either dumps
  everything or truncates silently.
- Move the underlying store off flat JSON to something indexed (see Part
  2, gap 1) so tool query latency doesn't degrade linearly as the dataset
  grows from 18 complaints to 18,000.
- Two-stage retrieval as the general pattern: a cheap structured/keyword
  filter narrows the candidate set first, semantic search or LLM
  reasoning only runs over that narrowed set -- keeps cost and latency
  bounded regardless of corpus size.

---

### Hallucination-reduction toolkit

The single most interview-relevant question about any agentic system:
"how do you stop it from making things up." This project's own Day 2-3
experience already answered the core version of this empirically --
**mechanical, deterministic verification of LLM claims consistently beat
iterative prompt engineering** for well-defined failure modes (this is
exactly how `_verify_material_citation()` and its retry loop came about).
That finding generalizes into the fuller toolkit below.

1. **Mechanical / deterministic verification of claims (proven here).**
   Don't ask the model to be more careful -- check its output in code
   against ground truth it had access to. `_verify_material_citation()`
   checking cited complaint IDs against real machine/date/symptom data is
   the concrete example; gap #1 above is exactly "apply this uniformly,
   not just to Material."
2. **Force structured, checkable output over free-text prose.** A
   `supporting_ids: list[str]` field is machine-checkable; a claim buried
   in a sentence is not. Wherever a model asserts a fact that should be
   traceable to a source, make the source citation a structured field,
   not prose.
3. **Retrieval-grounding (RAG).** Constrain what the model can talk about
   to what was actually retrieved for this call. If the model is only
   ever shown the records a tool returned, hallucinated *entities* become
   far less likely than in an open-ended, memory-only response --
   although this alone does not stop hallucinated *inferences* over real
   data (hence #1 and #2 still matter on top of it).
4. **Self-consistency / ensemble voting.** Run the same prompt N times and
   take a majority vote (gap #4). Hallucinations are frequently
   inconsistent across samples even when the model is confident on any
   single one -- disagreement across samples is itself a signal, often a
   better one than the model's own stated confidence.
5. **A second LLM as verifier/judge (chain-of-verification).** Have a
   separate call check "does this conclusion follow from these exact
   facts," ideally with a different, cheaper model to control cost and
   reduce the chance both calls share the same blind spot. Best reserved
   for borderline-confidence cases rather than run on every request.
6. **Confidence calibration, not self-reported confidence.** This
   project's weighted `_confidence_score()` (verification signal +
   category-reliability prior + citation signal) is a deliberate rejection
   of asking the model "how confident are you" -- LLM self-reported
   confidence is notoriously uncorrelated with actual correctness. A
   composite score built from independently checkable signals is a much
   more honest number.
7. **Decoding controls.** Temperature=0 reduces but does not eliminate
   hallucination or non-determinism -- proven directly in this project
   (C012 flipped category at temperature=0 across two runs). Don't treat
   temperature=0 as a correctness guarantee; it only narrows variance.
8. **Explicit permission to say "I don't know."** The `"Insufficient
   evidence"` category and `_human_review_reasons()`'s always-on trigger
   for it exist specifically so the model has a legitimate, rewarded exit
   ramp instead of being implicitly pressured to always produce a
   confident-sounding category. A system prompt that only offers
   "confidently pick one" options will get confidently wrong answers more
   often.
9. **Escalate low-confidence or disagreeing outputs to a human** rather
   than auto-accepting them -- the review-queue mechanism (gap #7) is the
   final backstop when every upstream mitigation still leaves residual
   uncertainty. Hallucination mitigation is defense-in-depth, not a single
   solved problem; the review gate is what catches whatever gets through.
10. **Keep context focused and relevant (avoid "lost in the middle").**
    Retrieving too much irrelevant data alongside the relevant facts
    measurably degrades a model's ability to use the relevant facts
    correctly, even when they're technically present in context. The
    pre-aggregation and two-stage retrieval ideas in gap #10 aren't only a
    scale/cost concern -- a smaller, more relevant context is also a
    hallucination mitigation in its own right.
11. **Fine-tuning / domain adaptation** (heavier-weight, usually a later
    lever). If a domain's failure modes are well-understood and stable
    (e.g. consistently confusing two categories), fine-tuning on
    corrected examples can reduce the base hallucination rate more
    durably than prompt patches -- but it's slower to iterate and should
    follow, not replace, the cheaper mitigations above.

---

## Part 2: General production gaps

These apply to this system as a backend service, independent of the LLM
inside it.

### 1. State/data layer

**Problem.** Everything is flat JSON files (`complaints.json`,
`review_queue.json`, `eval_results.json`). Two concrete issues:
- `_update_rca_review_queue()` in `main.py` does read-modify-write on a
  file with no locking -- two concurrent `POST /investigate` calls will
  lose one of the writes. The same pattern exists in Task 2's
  `review_queue.json`.
- No multi-instance story: the moment a second replica runs for
  availability, each has its own local file and the queue silently forks.

**Resolutions.**
- Move complaints, investigations, review-queue, and audit trail into a
  real relational database (Postgres) with proper transactions --
  eliminates the lost-update race by construction.
- If staying single-instance for now, at minimum add file locking
  (`filelock` / OS-level advisory locks) around every read-modify-write,
  as a stopgap that doesn't require a new infra dependency.
- Design the schema so investigations are append-only rows, not
  overwritten JSON blobs -- this also directly serves the audit-trail need
  in gap 2 below.
- Use a managed/replicated Postgres (RDS, Cloud SQL, etc.) rather than a
  self-hosted single instance if uptime matters, so the database itself
  isn't a new single point of failure replacing the JSON-file one.

### 2. Human review workflow lacks an audit trail

**Problem.** The review queue is currently read-only from the API's
perspective -- there's no way to record who reviewed an item, what they
decided, or why. For a CAPA/quality process specifically, that decision
trail *is* the regulatory artifact an auditor asks for -- "the AI flagged
it, human X approved/overrode it on date Y" -- not the AI's conclusion
alone.

**Resolutions.**
- Add `POST /investigate/review-queue/{complaint_id}/resolve` capturing
  reviewer identity, decision (approve / override / escalate), a required
  free-text reason, and a timestamp.
- Make the audit log append-only -- never overwrite a prior resolution,
  even if the item is re-reviewed; a correction is a new row referencing
  the old one, not a silent edit.
- Tie this into gap 7 in Part 1: ideally the resolve endpoint resumes a
  paused LangGraph checkpoint rather than just annotating an
  already-finalized JSON blob.
- Add per-user accounts and basic RBAC (who can view the queue vs. who can
  resolve items) rather than the current single shared Basic Auth
  credential, so the audit trail can actually attribute a decision to a
  specific person.

### 3. Observability (request/infra level, distinct from agent tracing)

**Problem.** Failure diagnosis currently means reading print statements or
replaying a call by hand. This is separate from Part 1's agent-reasoning
tracing gap -- this is ordinary service-level observability.

**Resolutions.**
- Structured logging (JSON logs) with a request/correlation ID per API
  call, replacing the current `print()` statements.
- Metrics exported to Prometheus/Grafana (or equivalent): request latency,
  error rate by endpoint, token usage and cost per request, and the
  agent-level rates (`conclusion_failed_rate`, `human_review_rate`)
  surfaced as live dashboards, not only inside individual JSON responses.
- Alerting rules on top of those metrics: page or notify when the live
  `human_review_rate` drifts meaningfully from what `eval.py` measured
  offline, or when error rate/latency crosses a threshold.
- Centralized log aggregation (even a simple ELK/Loki stack) so logs
  survive past a single instance's local disk and are searchable across
  instances once there's more than one.

### 4. Security

**Problem.** Single shared Basic Auth credential, no TLS handled by the
app itself, secrets in a `.env` file, no rate limiting, and upload
validation that only checks file extension, not actual content.

**Resolutions.**
- Terminate TLS at a reverse proxy / load balancer in front of the app
  (nginx, an API gateway, or the cloud provider's own LB) -- this service
  shouldn't hold certificates itself.
- Move secrets out of `.env` into a secrets manager (AWS Secrets
  Manager, HashiCorp Vault, or the cloud provider's native equivalent) for
  any real deployment.
- Replace shared Basic Auth with per-user accounts and RBAC (ties to Part
  2 gap 2), or at minimum rotate the shared credential and stop treating
  it as a long-lived secret.
- Add rate limiting (e.g. `slowapi` for FastAPI, or gateway-level
  throttling) -- both an abuse and a cost-control measure given every
  request triggers a billed LLM call.
- Validate uploaded file content by magic bytes / actual MIME sniffing,
  not just the filename extension, before passing it into the extraction
  pipeline.
- Enforce an explicit max upload size at the framework/gateway level, not
  just implicitly via whatever the provider or OS allows.

### 5. LLM-call resilience (transport-level, distinct from agent runaway control)

**Problem.** No explicit timeout or retry-with-backoff on transient
provider errors. A rate limit or transient 5xx currently 502s the caller
immediately instead of retrying.

**Resolutions.**
- Add retry-with-exponential-backoff specifically for transient/transport
  errors (429, 5xx, timeouts) -- distinct from the *correctness*-driven
  verification retries in `agent.py`, which are a different concern and
  should stay separate.
- Set explicit request timeouts on the LLM client rather than relying on
  library defaults.
- Add a circuit breaker so a sustained provider outage fails fast for new
  requests instead of every request individually waiting out the same
  timeout.
- Consider automatic provider failover (the existing `LLM_PROVIDER`
  OpenAI/OpenRouter toggle is a manual env-var switch today -- a
  production version could detect sustained failures on one provider and
  fail over automatically, with the switch logged for traceability).
- Cache by content hash: re-investigating the same complaint or
  re-extracting the same document currently pays full LLM cost again with
  no caching layer at all.

### 6. Cost governance

**Problem.** Task 2 already estimates cost (`extraction/cost_estimate.json`),
but nothing enforces a budget in production -- there's no daily/monthly
spend cap and no alert on a cost anomaly.

**Resolutions.**
- Track cumulative spend (tokens x provider pricing) per day/month against
  a configured cap; once the cap is reached, either hard-stop new
  investigations or degrade to a cheaper model, depending on the business
  tolerance.
- Alert on cost anomalies -- e.g. a single request consuming far more
  tokens than the historical average, which often indicates a runaway
  loop or a pathological input rather than legitimate use (ties to Part 1
  gap 3's per-run budget).
- Cache aggressively (see gap 5) -- the cheapest LLM call is the one that
  doesn't happen.
- Track cost per category/complaint type over time; if one category is
  consistently far more expensive (e.g. more verification retries), that's
  a signal worth investigating on its own, not just a cost line item.

### 7. CI/CD and deployment

**Problem.** No CI pipeline -- `test_agent.py`, `tools.py`,
`validate_dataset.py` are all currently run by hand. The Dockerfile has
never been verified with a real `docker build` (no Docker available on
this development machine). No environment separation, no IaC, no
liveness/readiness probes, no graceful shutdown handling.

**Resolutions.**
- Add a GitHub Actions workflow running the full test suite
  (`test_agent.py`, `tools.py`, `validate_dataset.py`) plus a linter on
  every push/PR.
- Add a scheduled (nightly or weekly) run of `eval.py` against the golden
  set as a standing regression check, independent of code changes --
  catches silent provider-side drift (a model update changing behavior
  without any local code change).
- Actually run `docker build` and `docker run` end-to-end at least once
  before claiming the container is deployable -- an unverified Dockerfile
  is a liability, not a deliverable.
- Add `/health`-style liveness and a separate readiness check (e.g. can it
  reach the configured LLM provider and the database) so an orchestrator
  (Kubernetes, ECS, etc.) can make correct restart/routing decisions.
- Separate configuration by environment (dev/staging/prod) rather than one
  `.env` shape assumed everywhere.
- Handle graceful shutdown -- don't kill in-flight LLM calls or
  in-progress investigations on a deploy; drain first.

### 8. Testing depth

**Problem.** Strong unit-test coverage with mocks (`test_agent.py`), but
no integration tests against the actual FastAPI app, no load/concurrency
tests (particularly relevant given the read-modify-write race in gap 1),
and no automated contract tests for API response shapes.

**Resolutions.**
- Add integration tests using FastAPI's `TestClient` against the real app
  with the LLM client mocked at the boundary -- verifies routing, auth,
  and the review-queue read/write logic without spending real API calls.
- Add a concurrency test that fires overlapping `POST /investigate` calls
  and asserts no lost updates in the review queue -- this would currently
  fail given gap 1, and is the most direct way to prove the fix works
  once it's made.
- Add contract tests (e.g. via Pydantic response models validated in
  tests) so a response-shape change is caught in CI rather than by a
  client breaking in production.
- Add basic load testing (Locust, k6) to understand real throughput limits
  before they're discovered by production traffic.

### 9. Documentation and compliance readiness

**Problem.** No deployment runbook, and for a CAPA/quality-system context
specifically, likely no documentation of the traceability guarantees an
ISO 9001 or similar internal quality process would expect.

**Resolutions.**
- Write an actual README covering: how to run locally, how to deploy, what
  environment variables are required, and what to do when something breaks
  (a runbook, not just a features list).
- Document explicitly what is and isn't guaranteed about traceability --
  once Part 1 gaps 6 and 2 (reproducibility + audit trail) are addressed,
  write down what question they can and cannot answer, so a compliance
  reviewer isn't left guessing.
- Record a demo walkthrough (video or scripted transcript) covering the
  golden path and at least one flagged-for-review path, since "we tested
  it in a browser/via curl" isn't discoverable by someone reading the repo
  later.

---

## Priority if shipping incrementally

Not every gap is equally urgent. Roughly in the order I'd actually close
them:

| Priority | Gap | Why it's first |
|---|---|---|
| 1 | Part 1 #1 -- uniform grounding verification | Silent correctness bug the moment any non-Material category hallucinates a citation |
| 2 | Part 1 #2 -- prompt injection defense | Safety issue that only requires real (not synthetic) text to trigger |
| 3 | Part 2 #1 -- state layer / race condition | Silent data corruption under any real concurrent load |
| 4 | Part 1 #7 -- native human-in-the-loop | Changes bad-conclusion prevention from after-the-fact to before-the-fact |
| 5 | Part 1 #5/#6 -- tracing + reproducibility | What you'll wish you had the first time someone asks "why did the agent say that" about a real incident |

Everything else compounds in value once these five are in place, but
won't by itself corrupt data or ship an unnoticed regression the way these
five will.
