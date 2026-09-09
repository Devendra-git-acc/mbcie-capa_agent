"""
Self-test for pipeline.py's non-LLM logic: document discovery, format
detection, digital text extraction, OCR, and confidence scoring. No API
key or network call needed -- same split as Task 1's tools.py (LLM-free,
instant) vs run_single.py (needs a real model call).

This exists specifically because pipeline.py's LLM-dependent path is
blocked today by OpenRouter's free-tier daily cap -- everything that
DOESN'T need the LLM should still be provably correct in the meantime,
not just assumed correct because it "looks right."

Run with:
    cd extraction && python test_pipeline.py
"""
import json
from unittest.mock import patch

import pymupdf as fitz

import pipeline
from pipeline import InvoiceExtraction, LineItem

DATA_DIR = pipeline.DATA_DIR


def test_discover_documents():
    docs = pipeline.discover_documents()
    assert len(docs) == 10, f"expected 10 documents, found {len(docs)}: {sorted(docs)}"
    assert docs["D01"] == [DATA_DIR / "D01.pdf"]
    assert [f.name for f in docs["D09"]] == ["D09_p1.jpg", "D09_p2.jpg"], \
        "D09's two scanned pages must be grouped together, in page order"
    print(f"discover_documents: {len(docs)} documents, D09 correctly grouped as {[f.name for f in docs['D09']]}")


def test_digital_vs_scanned_detection():
    assert pipeline.is_pdf_actually_digital(DATA_DIR / "D01.pdf") is True

    blank = pipeline.OUTPUT_DIR / "_test_blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(blank)
    doc.close()
    assert pipeline.is_pdf_actually_digital(blank) is False, \
        "a PDF page with no text layer must be classified as needing OCR, not read as empty digital text"
    blank.unlink()
    print("is_pdf_actually_digital: correctly True for a real text-layer PDF, False for a blank page")


def test_digital_extraction():
    text, conf = pipeline.extract_digital_pdf(DATA_DIR / "D01.pdf")
    assert conf == 0.98
    assert "IN-2026-1000" in text and "Shree Balaji Rubber Industries" in text
    print(f"extract_digital_pdf: {len(text)} chars extracted, source_confidence={conf}")


def test_ocr_light_vs_heavy_noise():
    text_light, conf_light = pipeline.extract_scanned_images([DATA_DIR / "D04_p1.jpg"])
    text_heavy, conf_heavy = pipeline.extract_scanned_images([DATA_DIR / "D06_p1.jpg"])
    assert conf_light > conf_heavy, \
        f"light-noise scan ({conf_light}) should OCR with higher confidence than the heavy-noise scan ({conf_heavy})"
    assert "Ahmedabad" in text_light, "OCR on a light-noise scan should recover the vendor name cleanly"
    print(f"OCR confidence: light-noise={conf_light:.3f}, heavy-noise={conf_heavy:.3f} (light > heavy, as expected)")


def test_multipage_scanned_grouping_reads_both_pages():
    docs = pipeline.discover_documents()
    text, conf = pipeline.extract_scanned_images(docs["D09"])
    # a 35-item invoice split across 2 pages -- page 2 must contribute real
    # OCR'd content too, not just page 1's
    assert len(text) > 800, f"expected substantial OCR text from both D09 pages combined, got {len(text)} chars"
    print(f"D09 (scanned, 2 pages): {len(text)} chars OCR'd across both pages, confidence={conf:.3f}")


def test_score_completeness():
    complete = InvoiceExtraction(doc_type="invoice", doc_number="X", doc_date="Y", vendor_name="V",
                                  buyer_name="B", currency="INR", line_items=[LineItem(description="a")],
                                  subtotal=1, tax_amount=1, total_amount=2)
    assert pipeline.score_completeness(complete) == 1.0

    half = InvoiceExtraction(doc_type="invoice", doc_number=None, doc_date="Y", vendor_name="V",
                              buyer_name=None, currency=None, line_items=[],
                              subtotal=1, tax_amount=1, total_amount=2)
    # present: doc_date, vendor_name, total_amount = 3/6 required + 0/1 items -> 3/7
    got = pipeline.score_completeness(half)
    assert abs(got - 3 / 7) < 1e-6, f"expected 3/7={3/7:.4f}, got {got}"
    print(f"score_completeness: complete doc -> 1.0, partial doc -> {got:.4f} (matches hand-calc 3/7)")


def test_check_arithmetic():
    good = InvoiceExtraction(doc_type="invoice", doc_number="X", doc_date="Y", vendor_name="V",
                              buyer_name="B", currency="INR",
                              line_items=[LineItem(description="a", line_total=50), LineItem(description="b", line_total=50)],
                              subtotal=100, tax_amount=18, total_amount=118)
    assert pipeline.check_arithmetic(good) is True

    bad_total = InvoiceExtraction(doc_type="invoice", doc_number="X", doc_date="Y", vendor_name="V",
                                   buyer_name="B", currency="INR", line_items=[],
                                   subtotal=100, tax_amount=18, total_amount=500)
    assert pipeline.check_arithmetic(bad_total) is False

    missing = InvoiceExtraction(doc_type="invoice", doc_number="X", doc_date="Y", vendor_name="V",
                                 buyer_name="B", currency="INR", line_items=[],
                                 subtotal=None, tax_amount=None, total_amount=None)
    assert pipeline.check_arithmetic(missing) is False
    print("check_arithmetic: consistent totals -> True, inconsistent -> False, missing totals -> False")


def test_score_document_review_routing():
    strong = InvoiceExtraction(doc_type="invoice", doc_number="X", doc_date="Y", vendor_name="V",
                                buyer_name="B", currency="INR", line_items=[LineItem(description="a", line_total=100)],
                                subtotal=100, tax_amount=18, total_amount=118)
    scoring = pipeline.score_document(0.98, strong)
    assert scoring["needs_review"] is False, scoring

    weak = InvoiceExtraction(doc_type="invoice", doc_number=None, doc_date=None, vendor_name="V",
                              buyer_name=None, currency=None, line_items=[],
                              subtotal=None, tax_amount=None, total_amount=None)
    scoring2 = pipeline.score_document(0.5, weak)
    assert scoring2["needs_review"] is True, scoring2
    assert len(scoring2["review_reasons"]) == 3, "should flag all three signals: OCR, completeness, arithmetic"
    print(f"score_document: strong extraction -> needs_review=False (conf={scoring['overall_confidence']}); "
          f"weak extraction -> needs_review=True (conf={scoring2['overall_confidence']}, "
          f"reasons={scoring2['review_reasons']})")


def test_empty_text_short_circuits_before_llm_call():
    """A blank PDF page has no text layer -> gets OCR'd -> produces
    ~nothing. This must be caught BEFORE the LLM call, not after -- so the
    mock is set to blow up if invoke() is ever called at all, proving the
    guard actually prevents the call rather than just handling its result."""
    blank = pipeline.OUTPUT_DIR / "_test_blank2.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(blank)
    doc.close()
    with patch.object(pipeline, "_structured_model") as mock_model:
        mock_model.invoke.side_effect = AssertionError("LLM should NOT be called for near-empty text")
        result = pipeline.extract_document("TESTBLANK", [blank])
    blank.unlink()
    assert result["extraction_failed"] is True
    assert "too short" in result["failure_reason"], result
    print(f"empty-text short-circuit: correctly skipped the LLM call entirely ({result['failure_reason']})")


def test_llm_call_failure_is_caught_not_fatal():
    """Simulates exactly what happened testing D10 against OpenRouter's
    rate limit today: the LLM call raises. extract_document() must catch
    it and return a normal failed-result dict, not propagate the exception."""
    with patch.object(pipeline, "_structured_model") as mock_model:
        mock_model.invoke.side_effect = RuntimeError("simulated rate limit (429)")
        result = pipeline.extract_document("D01", [pipeline.DATA_DIR / "D01.pdf"])
    assert result["extraction_failed"] is True
    assert "simulated rate limit" in result["failure_reason"], result
    print(f"LLM call failure: correctly caught and recorded, not raised -> {result['failure_reason']}")


def test_run_batch_survives_a_broken_document():
    """One document raising outside the LLM call (e.g. a missing/corrupt
    file) must not take down the rest of the batch."""
    fake_path = pipeline.DATA_DIR / "DOES_NOT_EXIST.pdf"
    with patch.object(pipeline, "discover_documents", return_value={"BROKEN": [fake_path]}):
        summary = pipeline.run_batch()
    result = json.loads((pipeline.OUTPUT_DIR / "BROKEN.json").read_text())
    assert result["extraction_failed"] is True
    assert summary["total_documents"] == 1
    (pipeline.OUTPUT_DIR / "BROKEN.json").unlink()
    (pipeline.OUTPUT_DIR / "summary.json").unlink(missing_ok=True)
    (pipeline.OUTPUT_DIR / "review_queue.json").unlink(missing_ok=True)
    print(f"run_batch survives a broken document: {result['failure_reason']}")


if __name__ == "__main__":
    test_discover_documents()
    test_digital_vs_scanned_detection()
    test_digital_extraction()
    test_ocr_light_vs_heavy_noise()
    test_multipage_scanned_grouping_reads_both_pages()
    test_score_completeness()
    test_check_arithmetic()
    test_score_document_review_routing()
    test_empty_text_short_circuits_before_llm_call()
    test_llm_call_failure_is_caught_not_fatal()
    test_run_batch_survives_a_broken_document()
    print("\nAll pipeline self-tests passed (no LLM/API calls made).")
