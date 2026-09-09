"""
Synthetic invoice/PO generator for Task 2 (document extraction with cost
control). Deterministic (seeded), like rca/generate_data.py.

Produces 10 documents across 5 genuinely different visual layouts (not the
same template with swapped numbers -- different field labels, different
table column sets, one non-tabular list-style PO, one with an extra
irrelevant column), mixing:
  - digital PDFs (real text layer, extracted directly -- no OCR needed)
  - "scanned" JPEGs (rendered from a PDF then degraded with rotation/noise/
    blur/JPEG re-compression so any embedded text layer is gone and OCR is
    the only way in)
  - one PDF whose item table is long enough to force a real page break
    (tables split across pages, as the brief asks for)
  - one scanned document that is ALSO multi-page (the hardest combined case)
  - one heavily-degraded scan, deliberately, to produce genuinely low OCR
    confidence -- this is the doc that should land in the human-review queue
  - one digital doc with its own document number omitted from the source,
    to test the completeness signal independent of OCR quality

ground_truth.json (extraction/data/ground_truth.json) holds the true field
values for validation only -- pipeline.py never reads it, same separation
as rca/answer_key.json on Task 1.
"""
import json
import random
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT, TA_CENTER

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

STYLES = getSampleStyleSheet()
RIGHT = ParagraphStyle("right", parent=STYLES["Normal"], alignment=TA_RIGHT)
CENTER = ParagraphStyle("center", parent=STYLES["Normal"], alignment=TA_CENTER)

CURRENCY_SYMBOL = {"INR": "Rs.", "USD": "$", "EUR": "EUR "}


def money(v, currency):
    return f"{CURRENCY_SYMBOL[currency]}{v:,.2f}"


def compute_totals(items, tax_rate):
    for it in items:
        it["line_total"] = round(it["quantity"] * it["unit_price"], 2)
    subtotal = round(sum(it["line_total"] for it in items), 2)
    tax_amount = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax_amount, 2)
    return subtotal, tax_amount, total


ITEM_POOL = [
    "Steel Radial Tyre 175/65 R14", "Bicycle Alloy Rim 26in", "Curing Press Gasket Kit",
    "Tyre Valve Stem (pack of 50)", "Bead Wire Coil 2.5mm", "Rubber Compound Batch (25kg)",
    "Bicycle Chain 1/2x1/8in", "Mold Release Agent (5L)", "Tread Rubber Sheet",
    "Bicycle Brake Pad Set", "Carbon Black Filler (bag)", "Tyre Sidewall Label Set",
    "Wheel Bearing Assembly", "Vulcanizing Solution (1L)", "Bicycle Spoke Set (36pc)",
    "Nylon Cord Fabric Roll", "Tyre Repair Patch Kit", "Bicycle Pedal Pair",
]


def random_items(n):
    names = random.sample(ITEM_POOL, min(n, len(ITEM_POOL)))
    if n > len(ITEM_POOL):
        names += random.choices(ITEM_POOL, k=n - len(ITEM_POOL))
    return [{"description": nm, "quantity": random.randint(2, 60),
              "unit_price": round(random.uniform(15, 850), 2)} for nm in names]


# --- Layout renderers ------------------------------------------------------
# Each takes (path, doc) and writes a PDF. `doc` fields intentionally vary in
# which labels/columns are present, matching how real vendors differ.

def layout_a_invoice(path, doc):
    """Classic invoice: title top-left, fields top-right, standard table."""
    d = SimpleDocTemplate(str(path), pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    els = [Paragraph(f"<b>{doc['vendor_name']}</b>", STYLES["Title"]),
           Paragraph("INVOICE", STYLES["Heading2"])]
    if doc.get("doc_number") is not None:
        els.append(Paragraph(f"Invoice No: {doc['doc_number']}", STYLES["Normal"]))
    els.append(Paragraph(f"Date: {doc['doc_date']}", STYLES["Normal"]))
    els.append(Paragraph(f"Bill To: {doc['buyer_name']}", STYLES["Normal"]))
    els.append(Spacer(1, 8 * mm))

    rows = [["Description", "Qty", "Unit Price", "Amount"]]
    for it in doc["items"]:
        rows.append([it["description"], str(it["quantity"]),
                     money(it["unit_price"], doc["currency"]), money(it["line_total"], doc["currency"])])
    t = Table(rows, colWidths=[85 * mm, 20 * mm, 35 * mm, 35 * mm], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    els.append(t)
    els.append(Spacer(1, 6 * mm))
    els.append(Paragraph(f"Subtotal: {money(doc['subtotal'], doc['currency'])}", RIGHT))
    els.append(Paragraph(f"Tax: {money(doc['tax_amount'], doc['currency'])}", RIGHT))
    els.append(Paragraph(f"<b>Total: {money(doc['total_amount'], doc['currency'])}</b>", RIGHT))
    d.build(els)


def layout_b_po(path, doc):
    """PO style: different labels (PO Number/Vendor/Ship To), different column names."""
    d = SimpleDocTemplate(str(path), pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    els = [Paragraph("PURCHASE ORDER", STYLES["Title"])]
    els.append(Paragraph(f"PO Number: {doc['doc_number']}", STYLES["Normal"]))
    els.append(Paragraph(f"PO Date: {doc['doc_date']}", STYLES["Normal"]))
    els.append(Paragraph(f"Vendor: {doc['vendor_name']}", STYLES["Normal"]))
    els.append(Paragraph(f"Ship To: {doc['buyer_name']}", STYLES["Normal"]))
    els.append(Spacer(1, 8 * mm))

    rows = [["Item", "Quantity", "Rate", "Line Total"]]
    for it in doc["items"]:
        rows.append([it["description"], str(it["quantity"]),
                     money(it["unit_price"], doc["currency"]), money(it["line_total"], doc["currency"])])
    t = Table(rows, colWidths=[85 * mm, 25 * mm, 30 * mm, 35 * mm], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.85, 0.9, 1)),
                            ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    els.append(t)
    els.append(Spacer(1, 6 * mm))
    els.append(Paragraph(f"Sub Total: {money(doc['subtotal'], doc['currency'])}", RIGHT))
    els.append(Paragraph(f"GST: {money(doc['tax_amount'], doc['currency'])}", RIGHT))
    els.append(Paragraph(f"<b>Grand Total: {money(doc['total_amount'], doc['currency'])}</b>", RIGHT))
    d.build(els)


def layout_c_compact(path, doc):
    """Compact/dense letterhead: fields inline in one line, minimal table borders."""
    d = SimpleDocTemplate(str(path), pagesize=A4, topMargin=15 * mm, bottomMargin=15 * mm)
    els = [Paragraph(doc["vendor_name"], CENTER), Paragraph("Tax Invoice", CENTER)]
    els.append(Spacer(1, 4 * mm))
    els.append(Paragraph(f"Bill No: {doc['doc_number']}&nbsp;&nbsp;|&nbsp;&nbsp;"
                          f"Bill Date: {doc['doc_date']}&nbsp;&nbsp;|&nbsp;&nbsp;"
                          f"Buyer: {doc['buyer_name']}", STYLES["Normal"]))
    els.append(Spacer(1, 6 * mm))

    rows = [["Particulars", "Qty", "Price", "Value"]]
    for it in doc["items"]:
        rows.append([it["description"], str(it["quantity"]),
                     money(it["unit_price"], doc["currency"]), money(it["line_total"], doc["currency"])])
    t = Table(rows, colWidths=[85 * mm, 20 * mm, 35 * mm, 35 * mm], repeatRows=1)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                            ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    els.append(t)
    els.append(Spacer(1, 6 * mm))
    els.append(Paragraph(f"Total Payable: {money(doc['total_amount'], doc['currency'])} "
                          f"(incl. tax {money(doc['tax_amount'], doc['currency'])})", RIGHT))
    d.build(els)


def layout_d_extracol(path, doc):
    """Right-aligned key-value header block, item table with an extra HSN column."""
    d = SimpleDocTemplate(str(path), pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    els = [Paragraph(f"<b>{doc['vendor_name']}</b>", STYLES["Title"]),
           Paragraph("TAX INVOICE", STYLES["Heading2"])]
    meta = Table([["Invoice #", doc["doc_number"]], ["Date", doc["doc_date"]],
                  ["Customer", doc["buyer_name"]]], colWidths=[30 * mm, 60 * mm])
    meta.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9),
                               ("ALIGN", (0, 0), (0, -1), "LEFT")]))
    els.append(meta)
    els.append(Spacer(1, 8 * mm))

    rows = [["S.No", "Description", "HSN", "Qty", "Rate", "Amount"]]
    for i, it in enumerate(doc["items"], 1):
        rows.append([str(i), it["description"], "4011", str(it["quantity"]),
                     money(it["unit_price"], doc["currency"]), money(it["line_total"], doc["currency"])])
    t = Table(rows, colWidths=[12 * mm, 60 * mm, 18 * mm, 18 * mm, 30 * mm, 32 * mm], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    els.append(t)
    els.append(Spacer(1, 6 * mm))
    els.append(Paragraph(f"<b>Total: {money(doc['total_amount'], doc['currency'])}</b> "
                          f"(Subtotal {money(doc['subtotal'], doc['currency'])} + "
                          f"Tax {money(doc['tax_amount'], doc['currency'])})", RIGHT))
    d.build(els)


def layout_e_nontabular(path, doc):
    """PO as a plain numbered list, no table/gridlines at all."""
    d = SimpleDocTemplate(str(path), pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    els = [Paragraph("Purchase Order", STYLES["Title"])]
    els.append(Paragraph(f"Order Ref: {doc['doc_number']}  --  Dated {doc['doc_date']}", STYLES["Normal"]))
    els.append(Paragraph(f"Supplier: {doc['vendor_name']}  --  For: {doc['buyer_name']}", STYLES["Normal"]))
    els.append(Spacer(1, 8 * mm))
    els.append(Paragraph("Items ordered:", STYLES["Normal"]))
    for i, it in enumerate(doc["items"], 1):
        els.append(Paragraph(f"{i}. {it['description']} &times; {it['quantity']} @ "
                              f"{money(it['unit_price'], doc['currency'])} = "
                              f"{money(it['line_total'], doc['currency'])}", STYLES["Normal"]))
    els.append(Spacer(1, 6 * mm))
    els.append(Paragraph(f"Subtotal {money(doc['subtotal'], doc['currency'])}, "
                          f"tax {money(doc['tax_amount'], doc['currency'])}, "
                          f"total due {money(doc['total_amount'], doc['currency'])}.", STYLES["Normal"]))
    d.build(els)


LAYOUTS = {"A": layout_a_invoice, "B": layout_b_po, "C": layout_c_compact,
           "D": layout_d_extracol, "E": layout_e_nontabular}


# --- Scan simulation --------------------------------------------------------

def make_scanned_images(pdf_path: Path, doc_id: str, noise_level: str) -> list[str]:
    """Render each PDF page to an image, then degrade it (rotation, noise,
    blur, JPEG re-compression) so it has no text layer and must be OCR'd.
    Returns the list of image filenames written."""
    written = []
    pdf = fitz.open(pdf_path)
    for page_index in range(len(pdf)):
        page = pdf[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # ~144dpi
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")

        angle = random.uniform(-2.5, 2.5) if noise_level == "light" else random.uniform(-4, 4)
        img = img.rotate(angle, fillcolor=255, expand=False)

        if noise_level == "heavy":
            img = img.filter(ImageFilter.GaussianBlur(radius=1.3))
        import numpy as np
        arr = np.array(img).astype("int16")
        sigma = 8 if noise_level == "light" else 22
        arr = arr + np.random.normal(0, sigma, arr.shape)
        arr = arr.clip(0, 255).astype("uint8")
        img = Image.fromarray(arr)

        out_name = f"{doc_id}_p{page_index + 1}.jpg"
        quality = 65 if noise_level == "light" else 35
        img.save(DATA_DIR / out_name, "JPEG", quality=quality)
        written.append(out_name)
    pdf.close()
    return written


# --- Document plan -----------------------------------------------------------

VENDORS = ["Shree Balaji Rubber Industries", "Continental Traders Pvt Ltd", "Pune Component Works",
           "Ahmedabad Wheel & Tyre Supplies", "Nashik Precision Parts", "Bharat Tyre Chemicals",
           "MG Bicycle Components", "Sunrise Industrial Supplies"]
BUYERS = ["MBCIE Tyre & Cycle Plant", "MBCIE Procurement Dept."]

plan = [
    dict(doc_id="D01", doc_type="invoice", layout="A", fmt="digital", currency="INR",
         n_items=3, tax_rate=0.18, note="baseline digital"),
    dict(doc_id="D02", doc_type="purchase_order", layout="B", fmt="digital", currency="USD",
         n_items=4, tax_rate=0.08, note="baseline digital, different labels/currency"),
    dict(doc_id="D03", doc_type="invoice", layout="A", fmt="digital", currency="INR",
         n_items=40, tax_rate=0.18, note="long item table -> forces a real page break"),
    dict(doc_id="D04", doc_type="invoice", layout="C", fmt="scanned", currency="INR",
         n_items=3, tax_rate=0.18, note="light scan noise", noise="light"),
    dict(doc_id="D05", doc_type="purchase_order", layout="B", fmt="scanned", currency="USD",
         n_items=4, tax_rate=0.08, note="light scan noise", noise="light"),
    dict(doc_id="D06", doc_type="invoice", layout="C", fmt="scanned", currency="INR",
         n_items=3, tax_rate=0.18, note="HEAVY scan degradation -- meant to land in review queue", noise="heavy"),
    dict(doc_id="D07", doc_type="invoice", layout="D", fmt="digital", currency="EUR",
         n_items=5, tax_rate=0.21, note="extra HSN column, digital"),
    dict(doc_id="D08", doc_type="purchase_order", layout="E", fmt="digital", currency="INR",
         n_items=4, tax_rate=0.18, note="non-tabular plain-list format, digital"),
    dict(doc_id="D09", doc_type="invoice", layout="A", fmt="scanned", currency="INR",
         n_items=35, tax_rate=0.18, note="scanned AND multi-page -- hardest combined case", noise="light"),
    dict(doc_id="D10", doc_type="invoice", layout="A", fmt="digital", currency="INR",
         n_items=3, tax_rate=0.18, note="doc_number omitted from source -- tests completeness signal",
         omit_doc_number=True),
]

ground_truth = {}

for i, spec in enumerate(plan):
    vendor = VENDORS[i % len(VENDORS)]
    buyer = BUYERS[i % len(BUYERS)]
    items = random_items(spec["n_items"])
    subtotal, tax_amount, total = compute_totals(items, spec["tax_rate"])
    doc_number = None if spec.get("omit_doc_number") else f"{spec['doc_type'][:2].upper()}-2026-{1000 + i}"
    doc = {
        "doc_id": spec["doc_id"], "doc_type": spec["doc_type"], "vendor_name": vendor,
        "buyer_name": buyer, "doc_number": doc_number,
        "doc_date": f"2026-0{(i % 9) + 1}-{10 + i:02d}",
        "currency": spec["currency"], "items": items,
        "subtotal": subtotal, "tax_amount": tax_amount, "total_amount": total,
    }

    tmp_pdf = DATA_DIR / f"_tmp_{spec['doc_id']}.pdf"
    LAYOUTS[spec["layout"]](tmp_pdf, doc)

    if spec["fmt"] == "digital":
        final_path = DATA_DIR / f"{spec['doc_id']}.pdf"
        tmp_pdf.replace(final_path)
        files = [final_path.name]
    else:
        files = make_scanned_images(tmp_pdf, spec["doc_id"], spec.get("noise", "light"))
        tmp_pdf.unlink()

    ground_truth[spec["doc_id"]] = {
        "format": spec["fmt"], "layout": spec["layout"], "files": files, "note": spec["note"],
        "doc_type": doc["doc_type"], "doc_number": doc["doc_number"], "doc_date": doc["doc_date"],
        "vendor_name": vendor, "buyer_name": buyer, "currency": doc["currency"],
        "n_line_items": len(items), "subtotal": subtotal, "tax_amount": tax_amount,
        "total_amount": total,
    }
    print(f"{spec['doc_id']}: {spec['fmt']:8s} layout {spec['layout']}  {spec['note']}  -> {files}")

(DATA_DIR / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))
print(f"\nWrote {len(plan)} documents + ground_truth.json to {DATA_DIR}")
