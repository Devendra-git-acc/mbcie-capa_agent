"""
Day 3: hit-rate evaluation of the RCA agent across all 18 answer_key.json
complaints. Like run_single.py, this is one of only two files allowed to
read answer_key.json -- the agent itself never sees it.

Why multiple runs per complaint, not one: confirmed directly during Day 2
testing that gpt-4o-mini at temperature=0 is NOT fully deterministic here
-- the identical C012 complaint produced a different category on two
consecutive runs. A single pass/fail per complaint would misrepresent
reliability either way (a lucky pass, or an unlucky fail). Default 3 runs/
complaint balances statistical signal against real API cost and time;
override with `python eval.py N`.

Two metrics per run, not one:
  - category match: does the final root_cause_category equal the ground
    truth? (conclusion_failed always counts as a non-match, per the Day 2
    fix -- a broken run that happens to land on the right label by
    coincidence must never be counted as a hit.)
  - evidence surfaced: did the tool calls made during this run's
    investigation ever return the actual supporting record IDs from the
    answer key, regardless of whether the agent's final conclusion used
    them correctly? This separates two different failure modes that a
    single hit-rate number can't distinguish: "the right evidence was
    right there and the agent still got it wrong" (a reasoning problem)
    vs. "the agent never even saw the right evidence" (a tool/investigation
    problem). Only meaningful for complaints with concrete supporting IDs
    (storylines 1-3); storyline 4 (mold wear) and the red herring have none
    by design -- see rca/answer_key.json's supporting_*_ids being empty for
    those.

Run with:
    cd rca && python eval.py        # 3 runs/complaint (default)
    cd rca && python eval.py 5      # 5 runs/complaint
"""
import json
import sys
from pathlib import Path

from agent import investigate

DEFAULT_RUNS = 3
BASE = Path(__file__).parent


def load_answer_key() -> dict:
    return json.loads((BASE / "answer_key.json").read_text())


def evidence_was_surfaced(evidence_chain: list[dict], truth: dict) -> bool | None:
    supporting_ids = set(truth.get("supporting_downtime_ids", [])) | set(truth.get("supporting_shift_ids", []))
    if not supporting_ids:
        return None  # no concrete IDs to check for this storyline (mold wear / red herring)
    blob = json.dumps([step.get("result") for step in evidence_chain])
    return all(sid in blob for sid in supporting_ids)


def evaluate_complaint(complaint_id: str, truth: dict, n_runs: int) -> list[dict]:
    runs = []
    for i in range(n_runs):
        result = investigate(complaint_id)
        conclusion = result["conclusion"]
        failed = result.get("conclusion_failed", False)
        verification_failed = result.get("material_verification_failed", False)
        category = conclusion.get("root_cause_category")
        # Same discipline as conclusion_failed: a Material conclusion that never
        # passed the mechanical citation check (i.e. survived MAX_VERIFICATION_RETRIES
        # corrections and STILL doesn't hold up) must never count as a hit, even if
        # the category string happens to equal ground truth.
        match = ((not failed) and (not verification_failed) and bool(category)
                  and category.lower() == truth["root_cause_category"].lower())
        runs.append({
            "run": i + 1,
            "conclusion_failed": failed,
            "material_verification_failed": verification_failed,
            "verification_retries_used": len(result.get("verification_log", [])),
            "predicted_category": category,
            "category_match": match,
            "evidence_surfaced": evidence_was_surfaced(result["evidence_chain"], truth),
            "steps_used": result.get("steps_used"),
        })
    return runs


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUNS
    answer_key = load_answer_key()

    per_complaint = {}
    per_category_matches = {}
    per_category_totals = {}
    total_runs = 0
    total_matches = 0
    total_failed = 0
    total_verification_failed = 0
    total_verification_retries = 0
    evidence_checks = []

    for cid, truth in answer_key.items():
        print(f"Evaluating {cid} ({n_runs} runs, truth={truth['root_cause_category']!r})...")
        runs = evaluate_complaint(cid, truth, n_runs)
        matches = sum(1 for r in runs if r["category_match"])
        failed = sum(1 for r in runs if r["conclusion_failed"])
        per_complaint[cid] = {
            "truth_category": truth["root_cause_category"],
            "hit_rate": round(matches / n_runs, 3),
            "conclusion_failed_count": failed,
            "runs": runs,
        }
        cat = truth["root_cause_category"]
        per_category_matches[cat] = per_category_matches.get(cat, 0) + matches
        per_category_totals[cat] = per_category_totals.get(cat, 0) + n_runs
        total_runs += n_runs
        total_matches += matches
        total_failed += failed
        total_verification_failed += sum(1 for r in runs if r["material_verification_failed"])
        total_verification_retries += sum(r["verification_retries_used"] for r in runs)
        evidence_checks += [r["evidence_surfaced"] for r in runs if r["evidence_surfaced"] is not None]
        vf = sum(1 for r in runs if r["material_verification_failed"])
        vr = sum(r["verification_retries_used"] for r in runs)
        extra = f", {vr} verification retr{'y' if vr == 1 else 'ies'}" if vr else ""
        print(f"  {cid}: {matches}/{n_runs} correct, {failed}/{n_runs} conclusion_failed{extra}"
              + (f", {vf}/{n_runs} still-invalid Material after retries" if vf else ""))

    summary = {
        "n_runs_per_complaint": n_runs,
        "total_complaints": len(answer_key),
        "total_investigations": total_runs,
        "overall_hit_rate": round(total_matches / total_runs, 3),
        "conclusion_failed_rate": round(total_failed / total_runs, 3),
        "material_verification_failed_rate": round(total_verification_failed / total_runs, 3),
        "total_verification_retries_used": total_verification_retries,
        "evidence_surfaced_rate": round(sum(evidence_checks) / len(evidence_checks), 3) if evidence_checks else None,
        "per_category_hit_rate": {cat: round(per_category_matches[cat] / per_category_totals[cat], 3)
                                   for cat in per_category_totals},
        "per_complaint": per_complaint,
    }
    (BASE / "eval_results.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print(f"Overall hit rate: {summary['overall_hit_rate']:.1%}  "
          f"({total_matches}/{total_runs} investigations)")
    print(f"conclusion_failed rate: {summary['conclusion_failed_rate']:.1%}")
    print(f"material_verification_failed rate: {summary['material_verification_failed_rate']:.1%}  "
          f"({summary['total_verification_retries_used']} correction attempts made in total)")
    if summary["evidence_surfaced_rate"] is not None:
        print(f"Evidence surfaced rate (storylines with concrete record IDs): "
              f"{summary['evidence_surfaced_rate']:.1%}")
    print("Per-category hit rate:")
    for cat, rate in summary["per_category_hit_rate"].items():
        print(f"  {cat}: {rate:.1%}")
    print(f"\nFull results written to {BASE / 'eval_results.json'}")


if __name__ == "__main__":
    main()