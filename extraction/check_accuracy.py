"""
Compares pipeline.py's output against extraction/data/ground_truth.json --
the Task 2 equivalent of Task 1's run_single.py match check against
answer_key.json. Without this, the only evidence Task 2 extraction is
actually correct is manual eyeballing, which doesn't scale past a couple of
documents and isn't something a reviewer can independently re-run.

ground_truth.json is never read by pipeline.py itself -- same separation as
Task 1's answer_key.json. This script is a validation step, not part of the
pipeline.

Run with:
    cd extraction && python check_accuracy.py
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

STRING_FIELDS = ["doc_type", "doc_number", "doc_date", "vendor_name", "buyer_name", "currency"]
NUMERIC_FIELDS = ["subtotal", "tax_amount", "total_amount"]


def norm_str(v):
    return v.strip().lower() if isinstance(v, str) else v


def numeric_match(a, b, tol_frac=0.01, tol_abs=0.5):
    if a is None or b is None:
        return a is None and b is None
    tol = max(tol_frac * abs(b), tol_abs)
    return abs(a - b) <= tol


def check_document(doc_id: str, truth: dict, output: dict) -> dict:
    if output.get("extraction_failed"):
        return {"doc_id": doc_id, "status": "extraction_failed",
                "reason": output.get("failure_reason", "unknown")}

    extracted = output.get("extracted", {})
    field_results = {}
    for f in STRING_FIELDS:
        field_results[f] = norm_str(extracted.get(f)) == norm_str(truth.get(f))
    for f in NUMERIC_FIELDS:
        field_results[f] = numeric_match(extracted.get(f), truth.get(f))
    field_results["n_line_items"] = len(extracted.get("line_items", [])) == truth.get("n_line_items")

    mismatches = {f: {"extracted": extracted.get(f) if f != "n_line_items" else len(extracted.get("line_items", [])),
                       "truth": truth.get(f)}
                  for f, ok in field_results.items() if not ok}

    return {"doc_id": doc_id, "status": "compared", "fields_checked": len(field_results),
            "fields_correct": sum(field_results.values()), "field_accuracy": field_results,
            "mismatches": mismatches,
            "all_correct": all(field_results.values())}


def main():
    ground_truth = json.loads((DATA_DIR / "ground_truth.json").read_text())
    results = []
    for doc_id, truth in ground_truth.items():
        out_path = OUTPUT_DIR / f"{doc_id}.json"
        if not out_path.exists():
            results.append({"doc_id": doc_id, "status": "not_yet_run"})
            continue
        output = json.loads(out_path.read_text())
        results.append(check_document(doc_id, truth, output))

    compared = [r for r in results if r["status"] == "compared"]
    field_totals = {}
    for r in compared:
        for f, ok in r["field_accuracy"].items():
            field_totals.setdefault(f, [0, 0])
            field_totals[f][0] += int(ok)
            field_totals[f][1] += 1

    report = {
        "total_ground_truth_docs": len(ground_truth),
        "compared": len(compared),
        "extraction_failed": sum(1 for r in results if r["status"] == "extraction_failed"),
        "not_yet_run": sum(1 for r in results if r["status"] == "not_yet_run"),
        "fully_correct_docs": sum(1 for r in compared if r["all_correct"]),
        "per_field_accuracy": {f: round(c / t, 3) for f, (c, t) in field_totals.items()},
        "per_document": results,
    }
    (OUTPUT_DIR / "accuracy_report.json").write_text(json.dumps(report, indent=2))

    print(f"Ground truth docs: {report['total_ground_truth_docs']}  |  "
          f"Compared: {report['compared']}  |  Failed: {report['extraction_failed']}  |  "
          f"Not yet run: {report['not_yet_run']}")
    print(f"Fully correct: {report['fully_correct_docs']}/{report['compared']}")
    print("Per-field accuracy:", json.dumps(report["per_field_accuracy"], indent=2))
    for r in results:
        if r["status"] == "compared" and not r["all_correct"]:
            print(f"  {r['doc_id']}: mismatches -> {json.dumps(r['mismatches'])}")
        elif r["status"] == "extraction_failed":
            print(f"  {r['doc_id']}: EXTRACTION FAILED -> {r['reason']}")


if __name__ == "__main__":
    main()
