# Limitations

Honest account of what's incomplete in this submission and what I'd do
differently with more time, rather than presenting it as more finished
than it is.

**Non-determinism is handled offline, not in production.** `rca/eval.py`
proved this agent isn't fully deterministic even at `temperature=0` -- the
same complaint produced different categories across runs. The eval
harness accounts for that by sampling 3 times; the live API doesn't --
`POST /investigate` returns whatever one sample happens to produce. With
more time, I'd add self-consistency voting (run N samples, take the
majority, treat disagreement itself as a review-trigger) at least for
borderline-confidence cases, rather than only measuring the problem
offline and not mitigating it live.

**Grounding verification is uniform in *format*, not in *rigor*.**
`_verify_material_citation()` in `rca/agent.py` is a genuine mechanical
check -- it looks up the cited complaint IDs and confirms they're real,
cross-machine, matching-symptom evidence. The `citation_signal` used for
every *other* category only checks that *something shaped like* a record
ID appears in the hypothesis text -- it doesn't confirm that ID was
actually returned by a tool call this run. A fabricated-but-plausible ID
would currently score the same as a real one outside Material.
Generalizing the Material verifier's actual grounding check to every
category is the single highest-value fix I'd make next.

**The held-out Material asymmetry is open, not resolved.** 88.9% on
genuinely unseen data (`rca/answer_key_unseen.json`), with one specific,
reproducible failure I could not fully explain -- the same cross-machine
storyline resolved correctly on one machine pair and incorrectly on
another, with both a data-generation bug and a retrieval failure checked
and ruled out before accepting it as a genuine reasoning gap. I'd want
more held-out storylines specifically targeting that asymmetry before
trusting Material's real-world reliability at the same level as the other
three categories.

**No memory across investigations, by design, but worth being explicit
about the tradeoff.** Cross-complaint pattern-finding (recurrence,
cross-machine defects) is handled by deterministic tool queries over the
full dataset, not by the agent recalling past investigations -- this is
more reliable at the current scale (36 complaints) than an LLM-memory
layer would be, since it's exhaustive and doesn't depend on an LLM's
imperfect recall of what it "remembers." It stops being the right call
once complaint volume outgrows what structured filtering handles well; at
that point a semantic/vector memory over resolved CAPAs becomes worth its
cost. Not needed yet, but not free forever either.

**The review queues are read, not resolved.** Both
`GET /investigate/review-queue` and `GET /review-queue` tell you what's
flagged; there's no endpoint yet for a human to record a decision
(approve/override + why). For a CAPA process specifically, that decision
trail -- who reviewed what, and why -- is the actual artifact an audit
would ask for, not just the AI's own output. Designing that properly means
pausing the LangGraph run itself before finalizing a flagged conclusion
(via LangGraph's checkpointer / `interrupt_before`) rather than flagging
after the fact, which I didn't have time to build.

**No structured logging or persisted reasoning trace.** Right now,
diagnosing what happened during a specific past investigation means
either reading whatever `print()` output was on screen when it ran, or
re-running it -- and a re-run isn't guaranteed to reproduce the same
result, given the non-determinism above. `evidence_chain` and
`verification_log` are returned in each response, but nothing persists
them centrally or makes them queryable afterward (e.g. "show me every
investigation last week that called `query_complaints_by_machine` more
than twice," or "what did the agent actually see before concluding X on
this complaint two days ago"). With more time I'd replace the `print()`
calls with structured logging carrying a correlation ID per request, and
either wire in LangSmith (close to free here, since this is already
LangGraph -- one env var) or at minimum persist each run's full message
history to a queryable store, rather than treating the response body as
the only record of what happened.

**Docker is unverified.** No Docker was available in the development
environment, so the `Dockerfile`/`docker-compose.yml` were written
carefully but never actually run. I'd verify this before calling it a
real deployment option rather than a best-effort one.

**`main` is stale relative to the working branch.** All of the work
described in the README -- confidence scoring, both review queues, ad-hoc
complaint input, the demo console -- happened after `main` was last
updated and lives only on `experiment/per-category-investigator`. I'd
normally merge before calling something done; I kept them separate so the
original, simpler baseline stays easy to diff against everything that
came after it, but a reviewer who only looks at `main` will see a
materially earlier version of this project.

**No production hardening beyond what the brief asked for.** No rate
limiting, no TLS termination (this app shouldn't hold certs itself
anyway), no per-user accounts (Basic Auth is one shared credential), no
CI pipeline running the test suite or `eval.py` automatically. None of
this was in scope for a 3-day prototype, but it's the honest gap between
"working prototype" and "something I'd put in front of real plant data."
