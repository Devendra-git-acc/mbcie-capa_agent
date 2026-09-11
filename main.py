"""
FastAPI entry point -- exposes both tasks as one deployable service with
basic authentication. Task 1 (brief-required) wraps the RCA agent. Task 2
(not brief-required, added for demo/deployment convenience) wraps the
document-extraction pipeline, including a genuine file-upload endpoint --
not just replay of the 10 fixture documents.

Run with:
    uvicorn main:app --reload
Then hit http://127.0.0.1:8000/docs (basic-auth prompt covers everything
except /health).
"""
import json
import secrets
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).parent

# rca/agent.py, rca/tools.py, and extraction/pipeline.py all use bare
# imports (e.g. "from tools import ALL_TOOLS") so they stay runnable
# standalone (`cd rca && python run_single.py`, `cd extraction && python
# pipeline.py`) -- see the NOTE comments in those files. Putting each
# package directory itself on sys.path (rather than converting them to
# package-relative imports) lets this file reuse that exact,
# already-verified code unchanged.
sys.path.insert(0, str(REPO_ROOT / "rca"))
sys.path.insert(0, str(REPO_ROOT / "extraction"))

# noinspection PyUnresolvedReferences
# PyCharm's static analyzer can't follow the sys.path.insert() calls above --
# these imports genuinely resolve at runtime (verified: server started, real
# authenticated calls succeeded end-to-end against both). Not a bug.
from agent import investigate  # noqa: E402
import pipeline  # noqa: E402
from shared.config import BASIC_AUTH_PASSWORD, BASIC_AUTH_USERNAME  # noqa: E402

ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}

_effective_password = BASIC_AUTH_PASSWORD
if not _effective_password:
    # An empty configured password would mean secrets.compare_digest("", "")
    # -- i.e. anyone sending no password at all gets in. Generate one at
    # startup instead of shipping a demo service with auth that's trivially
    # bypassable by default.
    _effective_password = secrets.token_urlsafe(12)
    print(f"[main.py] BASIC_AUTH_PASSWORD not set in .env -- generated a random "
          f"password for this run only: {_effective_password}")

_complaints = json.loads((REPO_ROOT / "rca" / "data" / "complaints.json").read_text())
_complaint_ids = [c["complaint_id"] for c in _complaints]

# Task 1's mirror of extraction/output/review_queue.json. Task 2 builds its
# queue once per batch run (run_batch() over all 10 fixtures at once); Task 1
# has no batch runner -- investigations happen one complaint at a time via
# this API -- so the queue is instead built incrementally, updated after
# every POST /investigate/{complaint_id} call rather than written in one
# shot. A complaint that needed review on one run and gets re-investigated
# with a clean result is removed from the queue -- the queue reflects the
# most recent investigation of each complaint, not a history of every flag
# ever raised.
RCA_OUTPUT_DIR = REPO_ROOT / "rca" / "output"
RCA_OUTPUT_DIR.mkdir(exist_ok=True)
RCA_REVIEW_QUEUE_PATH = RCA_OUTPUT_DIR / "review_queue.json"


def _load_rca_review_queue() -> dict:
    if not RCA_REVIEW_QUEUE_PATH.exists():
        return {}
    return json.loads(RCA_REVIEW_QUEUE_PATH.read_text())


def _update_rca_review_queue(complaint_id: str, result: dict) -> None:
    queue = _load_rca_review_queue()
    if result.get("needs_human_review"):
        queue[complaint_id] = {
            "root_cause_category": result["conclusion"].get("root_cause_category"),
            "confidence_score": result.get("confidence_score"),
            "review_reasons": result.get("review_reasons"),
            "investigated_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        queue.pop(complaint_id, None)
    RCA_REVIEW_QUEUE_PATH.write_text(json.dumps(queue, indent=2))


def _update_extraction_review_queue(doc_id: str, result: dict) -> None:
    """Same incremental-update pattern as _update_rca_review_queue above,
    for Task 2's extraction/output/review_queue.json. Only wired into
    POST /extract/sample/{doc_id} (the persisted fixture-corpus path) --
    POST /extract (real uploads) is deliberately ephemeral/one-off and
    doesn't write extraction/output/ at all, so it doesn't touch this
    queue either. Bug fixed here: this endpoint previously wrote the
    per-document JSON but never updated review_queue.json, so a document
    flagged needs_review=True via this endpoint silently never appeared
    in GET /review-queue -- only run_batch() actually populated it,
    despite that endpoint's own docstring claiming otherwise."""
    path = pipeline.OUTPUT_DIR / "review_queue.json"
    queue = json.loads(path.read_text()) if path.exists() else {}
    if pipeline.needs_review(result):
        queue[doc_id] = pipeline.review_reasons_for(result)
    else:
        queue.pop(doc_id, None)
    path.write_text(json.dumps(queue, indent=2))


class AdHocComplaint(BaseModel):
    """A complaint that doesn't have to already exist in complaints.json --
    for POST /investigate, distinct from POST /investigate/{complaint_id}
    which only accepts a known fixture ID. batch_code must still decode to
    a real machine/week (see tools.py's BATCH_RE, format 'M02-2026W13') for
    the investigation to find anything -- an unrecognized machine or week
    isn't an error, it just means query_downtime/query_shifts legitimately
    return nothing and the agent should honestly conclude "Insufficient
    evidence", the same as it would for a real complaint about an
    untracked machine."""
    batch_code: str = Field(..., description="e.g. 'M02-2026W13' -- must match ^M\\d{2}-\\d{4}W\\d{2}$")
    defect_description: str = Field(..., min_length=1)
    complaint_id: str | None = Field(None, description="Optional -- auto-generated (ADHOC-xxxxxxxx) if omitted")
    date_reported: str | None = None
    product_line: str | None = None
    customer_or_dealer: str | None = None
    severity: str | None = None


app = FastAPI(title="MBCIE CAPA RCA Agent", version="0.1.0")
_security = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(_security)) -> str:
    valid_user = secrets.compare_digest(credentials.username, BASIC_AUTH_USERNAME)
    valid_pass = secrets.compare_digest(credentials.password, _effective_password)
    if not (valid_user and valid_pass):
        raise HTTPException(status_code=401, detail="Invalid credentials",
                             headers={"WWW-Authenticate": "Basic"})
    return credentials.username


@app.get("/health")
def health():
    """Unauthenticated -- just proves the service is up."""
    return {"status": "ok"}


@app.get("/complaints", dependencies=[Depends(require_auth)])
def list_complaints():
    return {"complaint_ids": _complaint_ids}


@app.post("/investigate", dependencies=[Depends(require_auth)])
def run_adhoc_investigation(payload: AdHocComplaint):
    """Investigate a complaint typed in directly, instead of picking one of
    the 36 fixture complaints from GET /complaints -- same agent, same
    verification, same confidence scoring, no restriction to what's
    already in complaints.json."""
    complaint = {k: v for k, v in payload.model_dump().items() if v is not None}
    try:
        result = investigate(complaint)
    except ValueError as e:
        # Bad input (missing/unparseable batch_code) -- caught before the
        # graph ever runs, so this is a 400, not a 502.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502,
                             detail=f"Investigation failed -- upstream LLM provider error: "
                                    f"{type(e).__name__}: {e}")
    _update_rca_review_queue(result["complaint_id"], result)
    return result


@app.post("/investigate/{complaint_id}", dependencies=[Depends(require_auth)])
def run_investigation(complaint_id: str):
    if complaint_id not in _complaint_ids:
        raise HTTPException(status_code=404, detail=f"Unknown complaint_id: {complaint_id!r}. "
                                                      f"See GET /complaints for valid IDs.")
    try:
        result = investigate(complaint_id)
        _update_rca_review_queue(complaint_id, result)
        return result
    except Exception as e:
        # An upstream LLM-provider failure (rate limit, bad/missing API key,
        # network error, timeout) must not leak a raw 500 + stack trace to
        # the caller -- 502 signals "the service you depend on failed",
        # distinct from a bug in this API itself.
        raise HTTPException(status_code=502,
                             detail=f"Investigation failed -- upstream LLM provider error: "
                                    f"{type(e).__name__}: {e}")


@app.get("/investigate/review-queue", dependencies=[Depends(require_auth)])
def get_rca_review_queue():
    """Complaints whose most recent investigation was flagged by the
    deterministic review gate (rca/agent.py's _human_review_reasons()) --
    conclusion_failed, unresolved material_verification_failed, low
    composite confidence_score, or genuine "Insufficient evidence". Built
    incrementally from POST /investigate/{complaint_id} calls -- see
    _update_rca_review_queue() above. Task 1's equivalent of Task 2's
    GET /review-queue."""
    return {"review_queue": _load_rca_review_queue()}


@app.get("/documents", dependencies=[Depends(require_auth)])
def list_sample_documents():
    """The 10 fixture invoices/POs under extraction/data/, for repeatable
    demo runs -- distinct from /extract, which takes any uploaded file."""
    docs = pipeline.discover_documents()
    return {"doc_ids": sorted(docs.keys())}


@app.post("/extract/sample/{doc_id}", dependencies=[Depends(require_auth)])
def extract_sample_document(doc_id: str):
    """Run extraction on one of the 10 known fixture documents and persist
    the result to extraction/output/<doc_id>.json -- the same file
    run_batch() would produce, so hitting these one at a time via the API
    (e.g. from /docs) builds up the exact same output/ that the batch
    script does, and extraction/check_accuracy.py can validate either."""
    docs = pipeline.discover_documents()
    if doc_id not in docs:
        raise HTTPException(status_code=404, detail=f"Unknown sample doc_id: {doc_id!r}. See GET /documents.")
    try:
        result = pipeline.extract_document(doc_id, docs[doc_id])
    except Exception as e:
        raise HTTPException(status_code=502,
                             detail=f"Extraction failed -- upstream LLM provider error: {type(e).__name__}: {e}")
    (pipeline.OUTPUT_DIR / f"{doc_id}.json").write_text(json.dumps(result, indent=2))
    _update_extraction_review_queue(doc_id, result)
    return result


@app.post("/extract", dependencies=[Depends(require_auth)])
async def extract_uploaded_document(files: list[UploadFile] = File(...)):
    """Run extraction on a real uploaded document -- a single PDF (digital
    or scanned, multi-page is fine, handled internally), or several image
    files together for a hand-photographed multi-page scan. Not persisted
    to extraction/output/ -- these are one-off, not part of the fixture
    corpus."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    for f in files:
        if Path(f.filename).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(status_code=400,
                                 detail=f"Unsupported file type for {f.filename!r}. "
                                        f"Allowed: {sorted(ALLOWED_UPLOAD_SUFFIXES)}")
    if len(files) > 1 and any(Path(f.filename).suffix.lower() == ".pdf" for f in files):
        raise HTTPException(status_code=400,
                             detail="Multiple files together must all be page images (a multi-page "
                                    "PDF should be uploaded as a single file, not split).")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_paths = []
        for i, f in enumerate(files, 1):
            dest = Path(tmp_dir) / f"upload_p{i}{Path(f.filename).suffix.lower()}"
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            tmp_paths.append(dest)
        try:
            return pipeline.extract_document("uploaded", tmp_paths)
        except Exception as e:
            # Could be a malformed upload OR an upstream provider failure --
            # 422 ("this request could not be processed") is the honest
            # middle ground when we can't cheaply tell which from here.
            raise HTTPException(status_code=422,
                                 detail=f"Extraction failed: {type(e).__name__}: {e}")


@app.get("/review-queue", dependencies=[Depends(require_auth)])
def get_review_queue():
    """Documents flagged for human review -- built either by one batch run
    (extraction/pipeline.py's run_batch(), overwriting the whole queue) or
    incrementally by repeated POST /extract/sample/{doc_id} calls (each one
    adds/removes just that doc_id, via _update_extraction_review_queue()
    above), same file either way. Real uploads (POST /extract) are
    deliberately excluded -- see that endpoint's docstring."""
    path = pipeline.OUTPUT_DIR / "review_queue.json"
    if not path.exists():
        return {"review_queue": {}, "note": "No document has been extracted yet."}
    return {"review_queue": json.loads(path.read_text())}


# Demo console -- a single static page covering both tasks (complaint
# picker + investigation results + Task 1 review queue; sample-doc picker +
# real file upload + extraction results + Task 2 review queue), talking
# directly to the endpoints above with Basic Auth entered in the page
# itself. Mounted under /ui rather than "/" so it can never shadow an API
# route; "/" redirects there for convenience.
app.mount("/ui", StaticFiles(directory=str(REPO_ROOT / "static"), html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
