"""
Day 1 entry point: run the agent on ONE complaint end-to-end and print the
evidence chain + conclusion, next to the hidden answer key for that
complaint. This is the "prove the mechanism works" step before Day 2
scales it to all 18 complaints via eval.py.

Usage:
    Set LLM_PROVIDER + the matching API key in .env (see .env.example), then:
    python run_single.py            # defaults to C001 (calibration-drift case)
    python run_single.py C012       # try the mold-wear case
    python run_single.py C017       # try the red-herring case
"""
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # must happen before the env-var check below, not just inside agent.py's import

_provider = os.environ.get("LLM_PROVIDER", "openai").lower()
_key_var = "OPENROUTER_API_KEY" if _provider == "openrouter" else "OPENAI_API_KEY"
if not os.environ.get(_key_var):
    print(f"{_key_var} is not set (LLM_PROVIDER={_provider!r}).\n"
          "This repo can't call the model from Claude's sandbox (the API isn't reachable there), "
          "so this step is meant to run on your machine. In .env, set:\n\n"
          f"    LLM_PROVIDER={_provider}\n"
          f"    {_key_var}=...\n\n"
          "then:\n"
          "    python run_single.py\n")
    sys.exit(1)

from agent import investigate  # noqa: E402  (import after the key check, on purpose)

answer_key = json.loads(open("answer_key.json").read())


def main():
    complaint_id = sys.argv[1] if len(sys.argv) > 1 else "C001"
    print(f"Investigating {complaint_id} ...\n")

    result = investigate(complaint_id)

    print("=" * 70)
    print("COMPLAINT")
    print("=" * 70)
    print(json.dumps(result["complaint"], indent=2))

    print("\n" + "=" * 70)
    print(f"EVIDENCE CHAIN ({len(result['evidence_chain'])} steps, {result['steps_used']} agent turns)")
    print("=" * 70)
    for step in result["evidence_chain"]:
        print(f"\n  [{step['step']}] {step['tool']}({step['args']})")
        print(f"      -> {str(step['result'])[:200]}")

    print("\n" + "=" * 70)
    print("AGENT'S CONCLUSION")
    print("=" * 70)
    print(json.dumps(result["conclusion"], indent=2))

    print("\n" + "=" * 70)
    print("GROUND TRUTH (hidden answer key -- for this manual check only)")
    print("=" * 70)
    truth = answer_key.get(complaint_id, {})
    print(json.dumps(truth, indent=2))

    if result.get("conclusion_failed"):
        print("\nCategory match: N/A -- agent never produced a real submit_conclusion call "
              "(forced tool-call failed). Whatever category is shown above is a fallback, not a "
              "genuine answer, even if it happens to equal the ground truth -- do not count this as a match.")
    elif truth:
        match = result["conclusion"].get("root_cause_category", "").lower() == \
            truth.get("root_cause_category", "").lower()
        print(f"\nCategory match: {'YES' if match else 'NO'} "
              f"(agent: {result['conclusion'].get('root_cause_category')!r}, "
              f"truth: {truth.get('root_cause_category')!r})")


if __name__ == "__main__":
    main()
