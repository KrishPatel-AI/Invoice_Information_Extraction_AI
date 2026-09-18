"""
Evaluate the fine-tuned Donut checkpoint on the held-out TEST split only
(never used for training or model selection).

Usage:
    python evaluate.py --checkpoint /path/to/invoice_donut_final \
                        --test_dir ../data/splits/test \
                        --out_dir ../data/splits/test/eval_report
"""
import os, sys, json, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import schema
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "05_inference"))
from infer import load_model, run_inference

HEADER_FIELDS = [
    ("invoice_no", None),
    ("date_of_issue", None),
    ("seller.name", None), ("seller.address", None), ("seller.tax_id", None), ("seller.iban", None),
    ("client.name", None), ("client.address", None), ("client.tax_id", None),
    ("total.net_worth", "numeric"), ("total.vat", "numeric"), ("total.gross_worth", "numeric"),
]
ITEM_FIELDS = ["description", "qty", "um", "net_price", "net_worth", "vat_pct", "gross_worth"]
NUMERIC_ITEM_FIELDS = {"qty", "net_price", "net_worth", "vat_pct", "gross_worth"}


def get_path(d, dotted):
    cur = d
    for part in dotted.split("."):
        cur = (cur or {}).get(part) if isinstance(cur, dict) else None
    return cur


def values_match(a, b, kind=None):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if kind == "numeric":
        try:
            return abs(float(a) - float(b)) <= 0.01
        except (ValueError, TypeError):
            return False
    return str(a).strip().lower() == str(b).strip().lower()


def loose_match(a, b):
    """More forgiving match used only for error-categorization, not scoring:
    strips punctuation/whitespace to distinguish real errors from formatting noise."""
    import re
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower()) if s is not None else ""
    return norm(a) == norm(b)


def update_counts(counts, field, gt_val, pred_val, kind=None):
    c = counts.setdefault(field, {"tp": 0, "fp": 0, "fn": 0})
    match = values_match(gt_val, pred_val, kind)
    if gt_val is None and pred_val is None:
        return  # both agree nothing here; not counted in P/R/F1 denominators
    if match:
        c["tp"] += 1
    else:
        if gt_val is not None:
            c["fn"] += 1
        if pred_val is not None:
            c["fp"] += 1


def prf1(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if (p and r and (p + r) > 0) else None
    return p, r, f1


def categorize_error(field, gt_val, pred_val, raw_output_had_end_token):
    if pred_val is None and gt_val is not None:
        return "missed field"
    if gt_val is None and pred_val is not None:
        return "hallucinated field"
    if loose_match(gt_val, pred_val):
        return "formatting/normalization error"
    if not raw_output_had_end_token:
        return "truncation"
    return "incorrect value"


def main(checkpoint, test_dir, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    processor, model, device = load_model(checkpoint)

    label_files = sorted(glob.glob(os.path.join(test_dir, "labels", "*.json")))
    if limit:
        label_files = label_files[:limit]

    counts = {}
    item_count_diffs = []
    errors_log = []
    examples = []

    for lbl_path in label_files:
        invoice_id = os.path.splitext(os.path.basename(lbl_path))[0]
        image_path = os.path.join(test_dir, "images", invoice_id + ".jpg")
        with open(lbl_path) as f:
            gt_raw = json.load(f)
        gt_raw.pop("_meta", None)
        gt = schema.normalize(gt_raw)

        result = run_inference(image_path, processor, model, device)
        pred = result["structured_json"]
        had_end = schema.TASK_END_TOKEN in result["raw_model_output"]

        for field, kind in HEADER_FIELDS:
            gt_v, pred_v = get_path(gt, field), get_path(pred, field)
            update_counts(counts, field, gt_v, pred_v, kind)
            if not values_match(gt_v, pred_v, kind):
                errors_log.append({
                    "invoice_id": invoice_id, "field": field,
                    "expected": gt_v, "predicted": pred_v,
                    "category": categorize_error(field, gt_v, pred_v, had_end),
                })

        gt_items, pred_items = gt.get("items", []), pred.get("items", [])
        item_count_diffs.append(len(pred_items) - len(gt_items))
        for i in range(max(len(gt_items), len(pred_items))):
            gt_item = gt_items[i] if i < len(gt_items) else {}
            pred_item = pred_items[i] if i < len(pred_items) else {}
            if i >= len(gt_items):
                errors_log.append({"invoice_id": invoice_id, "field": f"item[{i}]",
                                    "expected": None, "predicted": pred_item, "category": "line-item error (extra item)"})
                continue
            if i >= len(pred_items):
                errors_log.append({"invoice_id": invoice_id, "field": f"item[{i}]",
                                    "expected": gt_item, "predicted": None, "category": "line-item error (missing item)"})
                continue
            for f in ITEM_FIELDS:
                kind = "numeric" if f in NUMERIC_ITEM_FIELDS else None
                gt_v, pred_v = gt_item.get(f), pred_item.get(f)
                update_counts(counts, f"item.{f}", gt_v, pred_v, kind)
                if not values_match(gt_v, pred_v, kind):
                    errors_log.append({
                        "invoice_id": invoice_id, "field": f"item[{i}].{f}",
                        "expected": gt_v, "predicted": pred_v,
                        "category": categorize_error(f, gt_v, pred_v, had_end),
                    })

        if len(examples) < 8:
            examples.append({"invoice_id": invoice_id, "expected": gt, "predicted": pred,
                              "schema_valid": result["is_valid"]})

    report = {"n_test_examples": len(label_files), "fields": {}}
    for field, c in counts.items():
        p, r, f1 = prf1(c)
        report["fields"][field] = {"precision": p, "recall": r, "f1": f1, **c}

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "errors.json"), "w") as f:
        json.dump(errors_log, f, indent=2)
    with open(os.path.join(out_dir, "examples.json"), "w") as f:
        json.dump(examples, f, indent=2)

    print(f"\nEvaluated on {len(label_files)} unseen test images.\n")
    print(f"{'field':<22}{'precision':>10}{'recall':>10}{'f1':>10}{'n':>6}")
    for field, m in report["fields"].items():
        n = m["tp"] + m["fp"] + m["fn"]
        fmt = lambda v: f"{v:.3f}" if v is not None else "  n/a"
        print(f"{field:<22}{fmt(m['precision']):>10}{fmt(m['recall']):>10}{fmt(m['f1']):>10}{n:>6}")

    from collections import Counter
    cat_counts = Counter(e["category"] for e in errors_log)
    print("\nError categories:")
    for cat, n in cat_counts.most_common():
        print(f"  {cat}: {n}")

    print(f"\nFull report written to {out_dir}/ (metrics.json, errors.json, examples.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", default="../data/splits/test")
    ap.add_argument("--out_dir", default="../data/splits/test/eval_report")
    ap.add_argument("--limit", type=int, default=None, help="evaluate on a subset for a quick check")
    args = ap.parse_args()
    main(args.checkpoint, args.test_dir, args.out_dir, args.limit)
