"""
Shared configuration, loaded once. rca/ and extraction/ both read from
here once main.py (Day 2) puts the repo root on the path -- until then,
rca/agent.py loads its own env vars directly (see the comment there) to
keep the Day 1 scripts runnable standalone with a plain `cd rca && python
run_single.py`, with zero import-path setup.
"""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
RCA_MODEL = os.environ.get("RCA_MODEL", "gpt-4o-mini")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

BASIC_AUTH_USERNAME = os.environ.get("BASIC_AUTH_USERNAME", "reviewer")
BASIC_AUTH_PASSWORD = os.environ.get("BASIC_AUTH_PASSWORD", "")
