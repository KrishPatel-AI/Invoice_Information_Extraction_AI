"""
Positional OCR-based auto-labeler for the fixed-template invoice dataset.

WHY THIS EXISTS:
The 489-image dataset (Kaggle: "High-Quality Invoice Images for OCR") ships
with NO ground-truth JSON. Every invoice is rendered from the same fixed
template (same label positions: "Seller:" always top-left, "Client:" always
top-right, same table column x-positions), so instead of hand-labeling 489
images we reconstruct ground truth deterministically from OCR word boxes +
known column geometry, then sanity-check every row with invoice arithmetic
(qty * net_price == net_worth, net_worth + vat == gross_worth) to catch and
flag OCR digit errors (e.g. Tesseract misreading '8' as '5').

Records that fail arithmetic checks are auto-corrected from arithmetic
(trusted over noisy OCR digits) AND logged in `_meta.issues` /
review_needed.csv so you can spot-check those specific images before trusting
them as training targets.
"""
import os, re, json, glob
import pytesseract
from PIL import Image

# --- Locate the Tesseract OCR engine binary (not the pytesseract python wrapper) ---
# On Windows this frequently isn't on PATH even after installing it, so we check
# common install locations automatically before giving up. Override by setting
# the TESSERACT_CMD environment variable if yours lives somewhere else.
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
elif os.name == "nt":
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

ROUND = 2
X_SPLIT = 780  # left/right column boundary (seller vs client), fixed by template


def get_words(data):
    words = []
    n = len(data['text'])
    for i in range(n):
        t = data['text'][i].strip()
        if not t or int(data['conf'][i]) < 0:
            continue
        words.append({
            'text': t, 'left': data['left'][i], 'top': data['top'][i],
            'width': data['width'][i], 'height': data['height'][i],
            'conf': int(data['conf'][i])
        })
    return words


def cluster_by_y(words, y_tol=10):
    words = sorted(words, key=lambda w: w['top'])
    clusters = []
    for w in words:
        placed = False
        for c in clusters:
            if abs(c['top'] - w['top']) <= y_tol:
                c['words'].append(w)
                c['top'] = min(c['top'], w['top'])
                placed = True
                break
        if not placed:
            clusters.append({'top': w['top'], 'words': [w]})
    for c in clusters:
        c['words'].sort(key=lambda w: w['left'])
    clusters.sort(key=lambda c: c['top'])
    return clusters


def line_text(words):
    return ' '.join(w['text'] for w in words)


def to_float(s):
    if s is None:
        return None
    s = s.replace(' ', '').replace(',', '.')
    s = re.sub(r'[^0-9.\-]', '', s)
    if not s or s in ('.', '-'):
        return None
    try:
        return round(float(s), ROUND)
    except ValueError:
        return None


def parse_block(words_in_region, side):
    """side: 'left' (seller) or 'right' (client). Returns name/address/tax_id/iban."""
    side_words = [w for w in words_in_region if (w['left'] < X_SPLIT) == (side == 'left')]
    clusters = cluster_by_y(side_words)
    lines = [line_text(c['words']) for c in clusters]
    lines = [l for l in lines if l.strip().lower() not in ('seller:', 'client:')]

    name, address_parts, tax_id, iban = None, [], None, None
    for l in lines:
        low = l.lower()
        if low.startswith('tax id'):
            m = re.search(r'([\d\-]{6,})', l)
            if m:
                tax_id = m.group(1)
        elif low.startswith('iban'):
            m = re.search(r'IBAN:\s*([A-Z0-9]+)', l, re.I)
            if m:
                iban = m.group(1)
        elif name is None:
            name = l.strip()
        else:
            address_parts.append(l.strip())
    out = {"name": name, "address": ', '.join(address_parts) if address_parts else None, "tax_id": tax_id}
    if side == 'left':
        out["iban"] = iban
    return out


def build_col_edges(header_words):
    """Map column name -> x position from the ITEMS header row. Two-word
    headers ('Net price', 'Net worth', 'VAT [%]') are merged into ONE
    semantic column anchored at the first word's x -- treating 'Net' as
    its own column would create a spurious boundary between 'Net price'
    and 'Net worth' and misclassify values near that seam."""
    ws = sorted(header_words, key=lambda w: w['left'])
    col_x = {}
    i = 0
    while i < len(ws):
        t = ws[i]['text'].lower().strip('.')
        nxt = ws[i + 1]['text'].lower().strip('[]') if i + 1 < len(ws) else ''
        if t == 'no':
            col_x['no'] = ws[i]['left']
        elif t == 'description':
            col_x['description'] = ws[i]['left']
        elif t == 'qty':
            col_x['qty'] = ws[i]['left']
        elif t == 'um':
            col_x['um'] = ws[i]['left']
        elif t == 'net' and nxt == 'price':
            col_x['price'] = ws[i]['left']; i += 1
        elif t == 'net' and nxt == 'worth':
            col_x['worth'] = ws[i]['left']; i += 1
        elif t == 'vat':
            col_x['vat'] = ws[i]['left']
            if nxt == '%':
                i += 1
        elif t == 'gross':
            col_x['gross'] = ws[i]['left']
        i += 1
    return col_x


def col_for_x(sorted_edges, x, margin=45):
    """sorted_edges: list of (name, x) sorted ascending. A word belongs to
    the LAST column whose edge it has reached (edge - margin), i.e. each
    column's bin extends from its own header position up to just before the
    next column's header position. This (rather than a symmetric midpoint)
    matches how these tables actually lay out: 'Description' is a WIDE
    column whose header sits at its far LEFT while numeric columns are
    NARROW, so the true boundary sits near the next column's start, not
    halfway between header label anchors."""
    best = None
    for name, edge in sorted_edges:
        if x >= edge - margin:
            best = name
    return best


def parse_invoice(image_path):
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = get_words(data)
    clusters = cluster_by_y(words)
    flat_conf = [w['conf'] for w in words]
    min_conf = min(flat_conf) if flat_conf else 0
    issues = []

    result = {
        "invoice_no": None, "date_of_issue": None,
        "seller": {"name": None, "address": None, "tax_id": None, "iban": None},
        "client": {"name": None, "address": None, "tax_id": None},
        "items": [], "total": {"net_worth": None, "vat": None, "gross_worth": None},
    }

    seller_top = client_top = items_top = summary_top = None
    header_words = None
    for c in clusters:
        txt = line_text(c['words']).lower()
        if 'invoice no' in txt:
            m = re.search(r'(\d+)', line_text(c['words']))
            if m: result['invoice_no'] = m.group(1)
        if 'date of issue' in txt:
            m = re.search(r'(\d{2}/\d{2}/\d{4})', line_text(c['words']))
            if m:
                result['date_of_issue'] = m.group(1)
        if 'seller' in txt and seller_top is None:
            seller_top = c['top']
        if 'client' in txt and client_top is None:
            client_top = c['top']
        if txt.strip() == 'items':
            items_top = c['top']
        if txt.strip() == 'summary':
            summary_top = c['top']
        if header_words is None and any(w['text'].lower().startswith('no') for w in c['words']) \
                and 'description' in txt:
            header_words = c['words']

    if seller_top is not None and items_top is not None:
        region = [w for w in words if seller_top < w['top'] < items_top]
        result['seller'] = {**result['seller'], **parse_block(region, 'left')}
        result['client'] = {**result['client'], **parse_block(region, 'right')}
    else:
        issues.append('seller_client_block_not_found')

    if header_words is not None and items_top is not None and summary_top is not None:
        col_x = build_col_edges(header_words)
        sorted_edges = sorted(col_x.items(), key=lambda kv: kv[1])
        row_clusters = [c for c in clusters if items_top < c['top'] < summary_top]
        current_item = None
        for c in row_clusters:
            # group this row's words by column, concatenating fragments in
            # each column left-to-right (handles OCR splitting "1 245,00"
            # into two word boxes that both land in the same 'worth' bin)
            by_col = {}
            for w in c['words']:
                col = col_for_x(sorted_edges, w['left'])
                by_col.setdefault(col, []).append(w['text'])
            joined = {k: ' '.join(v) for k, v in by_col.items()}
            has_qty = 'qty' in joined and to_float(joined['qty']) is not None

            if has_qty:
                if current_item:
                    result['items'].append(current_item)
                current_item = {"no": len(result['items']) + 1,
                                 "description": joined.get('description', '').strip(),
                                 "qty": to_float(joined.get('qty')),
                                 "um": joined.get('um'),
                                 "net_price": to_float(joined.get('price')),
                                 "net_worth": None, "vat_pct": None, "gross_worth": None}
                if 'worth' in joined:
                    current_item['net_worth'] = to_float(joined['worth'])
                if 'vat' in joined:
                    m = re.search(r'(\d+)', joined['vat'])
                    if m:
                        current_item['vat_pct'] = int(m.group(1))
                if 'gross' in joined:
                    current_item['gross_worth'] = to_float(joined['gross'])
            else:
                # continuation line: wrapped description text only
                if current_item is not None and joined.get('description'):
                    current_item['description'] = (current_item['description'] + ' ' + joined['description']).strip()
        if current_item:
            result['items'].append(current_item)
    else:
        issues.append('items_table_not_found')

    for it in result['items']:
        if it['qty'] is not None and it['net_price'] is not None:
            expected_net = round(it['qty'] * it['net_price'], ROUND)
            if it['net_worth'] is None or abs(it['net_worth'] - expected_net) > 0.02:
                issues.append(f"item{it['no']}_net_worth_corrected_ocr={it['net_worth']}_to={expected_net}")
                it['net_worth'] = expected_net
        if it['net_worth'] is not None and it['vat_pct'] is not None:
            expected_gross = round(it['net_worth'] * (1 + it['vat_pct'] / 100), ROUND)
            if it['gross_worth'] is None or abs(it['gross_worth'] - expected_gross) > 0.02:
                issues.append(f"item{it['no']}_gross_worth_corrected_ocr={it['gross_worth']}_to={expected_gross}")
                it['gross_worth'] = expected_gross

    if result['items'] and all(i['net_worth'] is not None for i in result['items']):
        sum_net = round(sum(i['net_worth'] for i in result['items']), ROUND)
        sum_gross = round(sum(i['gross_worth'] for i in result['items'] if i['gross_worth'] is not None), ROUND)
        result['total']['net_worth'] = sum_net
        result['total']['gross_worth'] = sum_gross
        result['total']['vat'] = round(sum_gross - sum_net, ROUND)
    else:
        issues.append('total_could_not_be_derived')

    if not result['invoice_no']:
        issues.append('invoice_no_missing')
    if not result['date_of_issue']:
        issues.append('date_missing')

    result['_meta'] = {"min_word_conf": min_conf, "issues": issues, "needs_review": len(issues) > 0}
    return result


def _debug_preview(images_dir):
    for f in sorted(glob.glob(os.path.join(images_dir, '*.jpg'))):
        r = parse_invoice(f)
        print(os.path.basename(f))
        print(json.dumps(r, indent=2))
        print()


def rebuild_review_csv(out_labels_dir, review_csv_path):
    """Recovery helper: regenerate review_needed.csv from labels already on
    disk, without re-running OCR. Use this if labeling finished but the CSV
    write step failed (e.g. missing output folder)."""
    import csv
    os.makedirs(os.path.dirname(review_csv_path) or '.', exist_ok=True)
    review_rows = []
    label_files = sorted(glob.glob(os.path.join(out_labels_dir, "*.json")))
    for lbl_path in label_files:
        with open(lbl_path) as f:
            record = json.load(f)
        meta = record.get('_meta', {})
        if meta.get('needs_review'):
            review_rows.append({"id": os.path.splitext(os.path.basename(lbl_path))[0],
                                 "original_filename": "(see ../data/filename_mapping.csv)",
                                 "issues": '; '.join(meta.get('issues', []))})
    with open(review_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["id", "original_filename", "issues"])
        w.writeheader(); w.writerows(review_rows)
    print(f"Rebuilt {review_csv_path}: {len(review_rows)}/{len(label_files)} flagged for review.")


def label_dataset(images_dir, out_images_dir, out_labels_dir, review_csv_path):
    """Run the labeler over every .jpg in images_dir, renaming sequentially to
    invoice_001.jpg / invoice_001.json (per the assessment's required naming),
    and writing a review_needed.csv of anything the arithmetic check flagged."""
    import shutil, csv

    # Fail fast and loud if Tesseract isn't actually reachable, instead of
    # silently burning a full 489-image run that flags everything for review.
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise SystemExit(
            "\nERROR: the Tesseract OCR engine is not installed / not found.\n"
            "pytesseract (the Python package) is just a wrapper -- it needs the actual\n"
            "Tesseract program installed separately.\n\n"
            "Windows: download & run the installer from\n"
            "  https://github.com/UB-Mannheim/tesseract/wiki\n"
            "then either let it install to the default location, or set the path yourself:\n"
            '  $env:TESSERACT_CMD = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"\n\n'
            "macOS:  brew install tesseract\n"
            "Linux:  sudo apt-get install tesseract-ocr\n\n"
            f"(underlying error: {e})"
        )

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(os.path.dirname(review_csv_path) or '.', exist_ok=True)
    files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) +
                    glob.glob(os.path.join(images_dir, '*.jpeg')) +
                    glob.glob(os.path.join(images_dir, '*.png')))
    if not files:
        raise SystemExit(f"No images found in {images_dir}")
    review_rows = []
    mapping = []
    for i, f in enumerate(files, start=1):
        new_id = f"invoice_{i:03d}"
        try:
            record = parse_invoice(f)
        except Exception as e:
            record = {"_meta": {"issues": [f"EXCEPTION: {e}"], "needs_review": True}}
        ext = os.path.splitext(f)[1]
        shutil.copy(f, os.path.join(out_images_dir, new_id + ext))
        with open(os.path.join(out_labels_dir, new_id + '.json'), 'w') as jf:
            json.dump(record, jf, indent=2)
        mapping.append({"id": new_id, "original_filename": os.path.basename(f)})
        if record.get('_meta', {}).get('needs_review'):
            review_rows.append({"id": new_id, "original_filename": os.path.basename(f),
                                 "issues": '; '.join(record['_meta']['issues'])})
        if i % 50 == 0:
            print(f"  labeled {i}/{len(files)}")

    with open(os.path.join(os.path.dirname(out_images_dir), 'filename_mapping.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["id", "original_filename"])
        w.writeheader(); w.writerows(mapping)

    with open(review_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["id", "original_filename", "issues"])
        w.writeheader(); w.writerows(review_rows)

    print(f"\nDone. {len(files)} images labeled -> {out_images_dir} / {out_labels_dir}")
    print(f"{len(review_rows)}/{len(files)} flagged for manual review -> {review_csv_path}")
    if len(review_rows) / len(files) > 0.15:
        print("WARNING: >15% flagged. Open a few flagged images + JSONs and check the OCR/column")
        print("geometry assumptions still hold before trusting the rest as training labels.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        # quick sanity check on a handful of images, prints JSON, writes nothing
        _debug_preview(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--rebuild-review":
        # recovery: labels already exist on disk, just the CSV write failed
        rebuild_review_csv("../data/processed/labels", "../data/review/review_needed.csv")
    else:
        # usage: python label_invoices.py <raw_images_dir>
        # labels every image, renames to invoice_001.jpg/.json, writes review_needed.csv
        raw_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/raw"
        label_dataset(raw_dir,
                      "../data/processed/images",
                      "../data/processed/labels",
                      "../data/review/review_needed.csv")
