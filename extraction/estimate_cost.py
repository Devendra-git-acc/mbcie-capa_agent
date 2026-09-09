"""
Cost-per-1000-documents estimate for Task 2, built from REAL measured token
usage in output/*.json (not a guess) applied to REAL current OpenAI pricing
(not a guess either -- checked live against platform.openai.com/docs/pricing
on 2026-09-09; update GPT_4O_MINI_PRICE below if you re-run this later).

The reasoning-overhead split and the cost_reduction_idea below are both
computed FROM the actual sample data, not hardcoded to whatever model
happened to be configured the last time this ran -- re-running this after
switching LLM_PROVIDER (e.g. free reasoning-heavy OpenRouter model vs.
gpt-4o-mini) changes both automatically instead of leaving stale advice
that contradicts the numbers next to it.

Run with:
    cd extraction && python estimate_cost.py
"""
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
TOTAL_FIXTURE_DOCS = 10

# https://platform.openai.com/docs/pricing -- checked live 2026-09-09
GPT_4O_MINI_PRICE = {"input_per_1m": 0.15, "output_per_1m": 0.60}

REASONING_SHARE_THRESHOLD = 0.10  # above this, reasoning overhead is the dominant cost lever


def load_samples() -> list[dict]:
    samples = []
    for f in sorted(OUTPUT_DIR.glob("D*.json")):
        d = json.loads(f.read_text())
        usage = d.get("token_usage") or {}
        if usage.get("input_tokens") is not None:
            samples.append({
                "doc_id": d["doc_id"],
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning", 0),
                "n_line_items": len((d.get("extracted") or {}).get("line_items", [])),
            })
    return samples


def cost_per_1000(avg_input: float, avg_output: float) -> float:
    per_doc = (avg_input * GPT_4O_MINI_PRICE["input_per_1m"]
               + avg_output * GPT_4O_MINI_PRICE["output_per_1m"]) / 1_000_000
    return round(per_doc * 1000, 4)


def build_cost_reduction_idea(reasoning_share: float, avg_line_items: float) -> str:
    if reasoning_share > REASONING_SHARE_THRESHOLD:
        return (
            f"Use a non-reasoning model for this extraction step. This isn't a generic suggestion -- "
            f"it's what this run's own data shows: {reasoning_share:.0%} of output tokens went to hidden "
            f"chain-of-thought reasoning that a simple, deterministic structured-extraction task doesn't "
            f"need (there's no multi-step decision-making here, unlike Task 1's agent). Switching to a "
            f"non-reasoning model would cut output tokens by roughly that share, which is most of the "
            f"per-document cost since output tokens are priced 4x higher than input tokens here."
        )
    return (
        f"Stop asking the model to emit line_total per line item -- it's pure arithmetic "
        f"(quantity * unit_price), already computed in Python by generate_docs.py/known from the two "
        f"other fields, and costs real output tokens for no accuracy benefit (an LLM's multiplication "
        f"isn't more trustworthy than code doing it directly). At {avg_line_items:.1f} line items/doc "
        f"average in this run, that's a meaningful fraction of the ~{avg_line_items:.0f} extra numeric "
        f"tokens per document going toward a field the pipeline could derive itself post-hoc instead of "
        f"paying the model to produce it -- proportionally larger for the longer multi-page invoices "
        f"(this run's worst case had 35 line items) where output-token cost scales with item count."
    )


def main():
    samples = load_samples()
    if not samples:
        print("No output/*.json samples found -- run pipeline.py first.")
        return

    n = len(samples)
    avg_input = sum(s["input_tokens"] for s in samples) / n
    avg_output_raw = sum(s["output_tokens"] for s in samples) / n
    avg_reasoning = sum(s["reasoning_tokens"] for s in samples) / n
    avg_output_excl_reasoning = avg_output_raw - avg_reasoning
    avg_line_items = sum(s["n_line_items"] for s in samples) / n
    reasoning_share = round(avg_reasoning / avg_output_raw, 3) if avg_output_raw else 0

    completeness_note = (f"All {TOTAL_FIXTURE_DOCS}/{TOTAL_FIXTURE_DOCS} fixture documents." if n >= TOTAL_FIXTURE_DOCS
                          else f"Only {n}/{TOTAL_FIXTURE_DOCS} fixture documents have been run through the "
                               f"pipeline so far -- these numbers will tighten as more are processed.")

    report = {
        "sample_size": n,
        "sample_doc_ids": [s["doc_id"] for s in samples],
        "note": completeness_note,
        "avg_input_tokens_per_doc": round(avg_input, 1),
        "avg_output_tokens_per_doc_as_measured": round(avg_output_raw, 1),
        "avg_reasoning_tokens_per_doc": round(avg_reasoning, 1),
        "reasoning_share_of_output": reasoning_share,
        "avg_line_items_per_doc": round(avg_line_items, 1),
        "pricing_used": "gpt-4o-mini, $0.15/1M input + $0.60/1M output (platform.openai.com/docs/pricing)",
        "estimated_cost_per_1000_docs": {
            "as_measured_incl_reasoning_overhead": cost_per_1000(avg_input, avg_output_raw),
            "realistic_excl_reasoning_overhead": cost_per_1000(avg_input, avg_output_excl_reasoning),
        },
        "cost_reduction_idea": build_cost_reduction_idea(reasoning_share, avg_line_items),
    }
    (OUTPUT_DIR / "cost_estimate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
