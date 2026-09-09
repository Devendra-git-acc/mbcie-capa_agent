"""
Task 2 extraction pipeline: turns a folder of invoice/PO documents (mixed
digital PDFs and scanned images, some multi-page) into clean structured
JSON with a per-document confidence score, and routes low-confidence
documents to a human-review queue instead of guessing.

Pipeline stages per document:
  1. Format detection -- PDF with a real text layer is "digital"; anything
     else (a bare image, or a PDF that turns out to have no extractable
     text) is "scanned" and goes through OCR.
  2. Text extraction -- PyMuPDF for digital PDFs (no OCR needed, free,
     exact); pytesseract for scanned images, which also gives per-word OCR
     confidence -- that confidence becomes part of the final score, not
     just a side effect.
  3. LLM structured extraction -- one single-shot call (not an agent loop
     like Task 1; there's nothing to investigate here, just parse) turns
     the raw text into the InvoiceExtraction schema below.
  4. Confidence scoring -- a composite of three independently-checkable
     signals (source reliability, field completeness, arithmetic
     consistency), NOT a self-reported LLM confidence number. LLMs are
     poorly calibrated at rating their own certainty; these three signals
     are cheap, deterministic, and each individually explainable to a
     human reviewing why a document got flagged.
  5. Routing -- below CONFIDENCE_THRESHOLD goes to the review queue with a
     reason string, instead of being accepted as-is.

Run with:
    cd extraction && python pipeline.py
"""
import json
import re
import sys
from pathlib import Path

import pymupdf as fitz
import pytesseract
from PIL import Image
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.config import (LLM_PROVIDER, OPENAI_API_KEY, RCA_MODEL,  # noqa: E402
                            OPENROUTER_API_KEY, OPENROUTER_MODEL)

from langchain_openai import ChatOpenAI  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = 0.75
REQUIRED_FIELDS = ["doc_number", "doc_date", "vendor_name", "buyer_name", "currency", "total_amount"]

# --- Tesseract binary discovery ---------------------------------------------
# pytesseract is just a wrapper around the tesseract.exe binary -- it is NOT
# pip-installable and won't be on PATH on a fresh Windows box by default.
_DEFAULT_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import os  # noqa: E402
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
elif os.name == "nt" and Path(_DEFAULT_WIN_TESSERACT).exists():
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_WIN_TESSERACT


# --- Structured output schema -----------------------------------------------
class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    line_total: float | None = None


class InvoiceExtraction(BaseModel):
    doc_type: str = Field(description="Exactly one of the two literal strings 'invoice' or "
                                       "'purchase_order', lowercase, regardless of how the document "
                                       "titles itself (e.g. 'Tax Invoice' and 'Bill' both -> 'invoice')")
    doc_number: str | None = Field(description="Invoice/PO/Bill number as printed, or null if absent")
    doc_date: str | None = Field(description="Document date as printed (don't reformat), or null")
    vendor_name: str | None = None
    buyer_name: str | None = None
    currency: str | None = Field(description="The 3-letter ISO 4217 code (INR, USD, EUR, ...). Convert "
                                              "from whatever symbol the document uses -- 'Rs.' or a "
                                              "rupee sign means INR, '$' means USD, etc. Never return "
                                              "the raw symbol itself as the value.")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None


EXTRACTION_PROMPT = """You are extracting structured data from a business document (invoice or \
purchase order). The text below was produced by either direct PDF text extraction or OCR -- it may \
contain OCR errors (character confusions, mangled spacing), multiple pages concatenated in order, and \
non-tabular layouts (plain numbered lists instead of grid tables). Extract exactly what is printed. \
Use null for any field you cannot find -- do not invent or estimate a value, and do not silently correct \
numbers that don't add up; report them as printed.

--- DOCUMENT TEXT START ---
{text}
--- DOCUMENT TEXT END ---"""


def _build_model() -> ChatOpenAI:
    """Same provider toggle as rca/agent.py (LLM_PROVIDER=openai|openrouter),
    but reading through shared/config.py directly -- Task 2 has no Day-1
    standalone-script legacy to preserve, so it uses the shared config path
    main.py was always meant to wire everything through."""
    if LLM_PROVIDER == "openrouter":
        return ChatOpenAI(model=OPENROUTER_MODEL, temperature=0, api_key=OPENROUTER_API_KEY,
                           base_url="https://openrouter.ai/api/v1")
    elif LLM_PROVIDER == "openai":
        return ChatOpenAI(model=RCA_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER {LLM_PROVIDER!r}: expected 'openai' or 'openrouter'")


_model = _build_model()
_structured_model = _model.with_structured_output(InvoiceExtraction, include_raw=True)


# --- Document discovery ------------------------------------------------------
def discover_documents() -> dict[str, list[Path]]:
    """Group files in data/ by doc_id (prefix before an optional _pN page
    suffix), sorted into page order. One PDF file = one (possibly
    multi-page) document; scanned docs may be several image files."""
    groups: dict[str, list[Path]] = {}
    for f in sorted(DATA_DIR.iterdir()):
        if f.name.startswith("_") or f.suffix.lower() not in (".pdf", ".jpg", ".jpeg", ".png"):
            continue
        m = re.match(r"^([A-Za-z0-9]+?)(?:_p(\d+))?\.\w+$", f.name)
        doc_id = m.group(1)
        groups.setdefault(doc_id, []).append(f)
    for doc_id, files in groups.items():
        files.sort(key=lambda p: int(m.group(1)) if (m := re.search(r"_p(\d+)", p.name)) else 0)
    return groups


# --- Text extraction ---------------------------------------------------------
def extract_digital_pdf(path: Path) -> tuple[str, float]:
    """Direct text-layer extraction. source_confidence is fixed high --
    this is exact text, not a recognition guess, so the main risk is a
    field being genuinely absent from the document, not misread."""
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text, 0.98


def extract_scanned_images(paths: list[Path]) -> tuple[str, float]:
    """OCR each page image and average pytesseract's own per-word
    confidence across all recognized words on all pages -- this is the
    single biggest signal for whether the extracted text can be trusted."""
    texts = []
    confidences = []
    for p in paths:
        img = Image.open(p)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words = [w for w in data["text"] if w.strip()]
        confs = [int(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and int(c) >= 0]
        texts.append(" ".join(words))
        confidences.extend(confs)
    text = "\n".join(texts)
    mean_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    return text, mean_conf


def is_pdf_actually_digital(path: Path) -> bool:
    doc = fitz.open(path)
    total_chars = sum(len(page.get_text().strip()) for page in doc)
    doc.close()
    return total_chars > 30  # a real text layer will clear this trivially; a scanned PDF with no OCR won't


# --- Confidence scoring -------------------------------------------------------
def score_completeness(extracted: InvoiceExtraction) -> float:
    present = sum(1 for f in REQUIRED_FIELDS if getattr(extracted, f) not in (None, ""))
    has_items = 1 if extracted.line_items else 0
    return (present + has_items) / (len(REQUIRED_FIELDS) + 1)


def check_arithmetic(extracted: InvoiceExtraction) -> bool:
    if extracted.subtotal is None or extracted.tax_amount is None or extracted.total_amount is None:
        return False
    tol = max(0.01 * abs(extracted.total_amount), 0.5)
    if abs((extracted.subtotal + extracted.tax_amount) - extracted.total_amount) > tol:
        return False
    if extracted.line_items:
        item_sum = sum(it.line_total for it in extracted.line_items if it.line_total is not None)
        if item_sum and abs(item_sum - extracted.subtotal) > tol:
            return False
    return True


def score_document(source_confidence: float, extracted: InvoiceExtraction) -> dict:
    completeness = score_completeness(extracted)
    arithmetic_ok = check_arithmetic(extracted)
    overall = round(0.45 * source_confidence + 0.35 * completeness + 0.20 * (1.0 if arithmetic_ok else 0.0), 3)
    reasons = []
    if source_confidence < 0.6:
        reasons.append(f"low OCR confidence ({source_confidence:.2f})")
    if completeness < 0.8:
        reasons.append(f"missing required fields ({completeness:.0%} complete)")
    if not arithmetic_ok:
        reasons.append("totals don't reconcile (subtotal + tax != total, or items don't sum to subtotal)")
    return {"source_confidence": round(source_confidence, 3), "completeness": round(completeness, 3),
            "arithmetic_consistent": arithmetic_ok, "overall_confidence": overall,
            "needs_review": overall < CONFIDENCE_THRESHOLD, "review_reasons": reasons}


# --- Per-document pipeline -----------------------------------------------------
MIN_TEXT_LENGTH = 20  # below this, OCR/extraction produced essentially nothing --
                       # not worth spending an LLM call on, and the model can't
                       # do anything useful with it anyway


def extract_document(doc_id: str, files: list[Path]) -> dict:
    tmp_imgs = []
    try:
        if len(files) == 1 and files[0].suffix.lower() == ".pdf" and is_pdf_actually_digital(files[0]):
            fmt = "digital"
            text, source_confidence = extract_digital_pdf(files[0])
        elif files[0].suffix.lower() == ".pdf":
            # a PDF with no real text layer -- render pages to images and OCR them
            fmt = "scanned"
            doc = fitz.open(files[0])
            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                tmp_path = OUTPUT_DIR / f"_tmp_{doc_id}_p{i + 1}.png"
                pix.save(tmp_path)
                tmp_imgs.append(tmp_path)
            doc.close()
            text, source_confidence = extract_scanned_images(tmp_imgs)
        else:
            fmt = "scanned"
            text, source_confidence = extract_scanned_images(files)
    finally:
        for t in tmp_imgs:
            t.unlink(missing_ok=True)

    if len(text.strip()) < MIN_TEXT_LENGTH:
        # cheap guard, saves an LLM call entirely -- an OCR/extraction failure
        # this severe can't be rescued by the model anyway, and this IS
        # itself a hard review-queue signal
        return {"doc_id": doc_id, "format": fmt, "files": [f.name for f in files],
                "extraction_failed": True,
                "failure_reason": f"extracted text too short ({len(text.strip())} chars) -- "
                                   f"skipped LLM call, source extraction/OCR effectively failed",
                "raw_text_excerpt": text[:500], "token_usage": {}}

    try:
        result = _structured_model.invoke(EXTRACTION_PROMPT.format(text=text))
    except Exception as e:
        # a single document's API failure (rate limit, network, timeout) must
        # not take down the whole batch -- record it and let run_batch() move
        # on to the next document. This is not hypothetical: this is exactly
        # what happened testing D10 against OpenRouter's daily rate limit.
        return {"doc_id": doc_id, "format": fmt, "files": [f.name for f in files],
                "extraction_failed": True, "failure_reason": f"{type(e).__name__}: {e}",
                "raw_text_excerpt": text[:500], "token_usage": {}}

    extracted: InvoiceExtraction = result["parsed"]
    usage = getattr(result["raw"], "usage_metadata", None) or {}

    if extracted is None:
        # model failed to produce valid structured output at all -- this is
        # itself a hard signal, not something to paper over with a fallback
        # confidence number
        return {"doc_id": doc_id, "format": fmt, "files": [f.name for f in files],
                "extraction_failed": True, "failure_reason": "model did not return valid structured output",
                "raw_text_excerpt": text[:500], "token_usage": usage}

    scoring = score_document(source_confidence, extracted)
    return {"doc_id": doc_id, "format": fmt, "files": [f.name for f in files],
            "extraction_failed": False, "extracted": extracted.model_dump(),
            "confidence": scoring, "token_usage": usage, "raw_text_excerpt": text[:500]}


def run_batch() -> dict:
    documents = discover_documents()
    results = {}
    for doc_id, files in documents.items():
        print(f"Extracting {doc_id} ({len(files)} file(s))...")
        try:
            results[doc_id] = extract_document(doc_id, files)
        except Exception as e:
            # belt-and-suspenders: extract_document already catches the LLM
            # call specifically, but format detection/OCR themselves could
            # still raise (e.g. a corrupt file) -- one bad document must
            # never stop the rest of the batch from running
            print(f"  {doc_id} FAILED outside the LLM call: {type(e).__name__}: {e}")
            results[doc_id] = {"doc_id": doc_id, "extraction_failed": True,
                                "failure_reason": f"{type(e).__name__}: {e}", "token_usage": {}}
        (OUTPUT_DIR / f"{doc_id}.json").write_text(json.dumps(results[doc_id], indent=2))

    def _review_reasons(r):
        if r.get("extraction_failed"):
            return [f"extraction failed: {r.get('failure_reason', 'unknown')}"]
        return r.get("confidence", {}).get("review_reasons", [])

    review_queue = {doc_id: _review_reasons(r) for doc_id, r in results.items()
                     if r.get("extraction_failed") or r.get("confidence", {}).get("needs_review")}
    total_prompt_tokens = sum(r.get("token_usage", {}).get("input_tokens", 0) for r in results.values())
    total_completion_tokens = sum(r.get("token_usage", {}).get("output_tokens", 0) for r in results.values())

    summary = {
        "total_documents": len(results),
        "flagged_for_review": len(review_queue),
        "review_queue": review_queue,
        "avg_confidence": round(sum(r.get("confidence", {}).get("overall_confidence", 0)
                                     for r in results.values() if not r.get("extraction_failed"))
                                 / max(1, len(results) - sum(1 for r in results.values()
                                                              if r.get("extraction_failed"))), 3),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_prompt_tokens_per_doc": round(total_prompt_tokens / len(results), 1),
        "avg_completion_tokens_per_doc": round(total_completion_tokens / len(results), 1),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "review_queue.json").write_text(json.dumps(review_queue, indent=2))
    return summary


if __name__ == "__main__":
    summary = run_batch()
    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=2))
