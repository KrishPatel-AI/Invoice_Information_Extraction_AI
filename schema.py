"""
Canonical invoice JSON schema, shared by every stage of the pipeline:
    auto-labeling -> training target -> model prediction -> evaluation -> final JSON

This is the ONE place field names/order are defined so they can't drift
between stages. Import this module from 03_donut_dataset, 05_inference, and
06_evaluation rather than re-defining tags in each script.
"""
import re
import json

# Order matters: this is also the serialization order in the Donut target string.
TOP_FIELDS = ["invoice_no", "date_of_issue"]
PARTY_FIELDS = ["name", "address", "tax_id"]  # seller adds 'iban' on top of this
ITEM_FIELDS = ["description", "qty", "um", "net_price", "net_worth", "vat_pct", "gross_worth"]
TOTAL_FIELDS = ["net_worth", "vat", "gross_worth"]

TASK_START_TOKEN = "<s_invoice>"
TASK_END_TOKEN = "</s_invoice>"

ALL_SPECIAL_TAGS = (
    [TASK_START_TOKEN, TASK_END_TOKEN]
    + [f"<s_{f}>" for f in TOP_FIELDS] + [f"</s_{f}>" for f in TOP_FIELDS]
    + ["<s_seller>", "</s_seller>", "<s_client>", "</s_client>"]
    + [f"<s_{f}>" for f in PARTY_FIELDS] + [f"</s_{f}>" for f in PARTY_FIELDS]
    + ["<s_iban>", "</s_iban>"]
    + ["<s_items>", "</s_items>", "<s_item>", "</s_item>"]
    + [f"<s_{f}>" for f in ITEM_FIELDS] + [f"</s_{f}>" for f in ITEM_FIELDS]
    + ["<s_total>", "</s_total>", "<s_vat>", "</s_vat>"]
)
ALL_SPECIAL_TAGS = sorted(set(ALL_SPECIAL_TAGS))


def _tag(name, value):
    value = "" if value is None else str(value)
    value = value.replace("<", "&lt;").replace(">", "&gt;")
    return f"<s_{name}>{value}</s_{name}>"


def to_donut_target(record: dict) -> str:
    """canonical dict -> Donut training target string."""
    parts = [TASK_START_TOKEN]
    parts.append(_tag("invoice_no", record.get("invoice_no")))
    parts.append(_tag("date_of_issue", record.get("date_of_issue")))

    seller = record.get("seller", {}) or {}
    parts.append("<s_seller>" + "".join(
        _tag(f, seller.get(f)) for f in PARTY_FIELDS
    ) + _tag("iban", seller.get("iban")) + "</s_seller>")

    client = record.get("client", {}) or {}
    parts.append("<s_client>" + "".join(
        _tag(f, client.get(f)) for f in PARTY_FIELDS
    ) + "</s_client>")

    items_str = "<s_items>"
    for item in record.get("items", []) or []:
        items_str += "<s_item>" + "".join(_tag(f, item.get(f)) for f in ITEM_FIELDS) + "</s_item>"
    items_str += "</s_items>"
    parts.append(items_str)

    total = record.get("total", {}) or {}
    parts.append("<s_total>" + "".join(_tag(f, total.get(f)) for f in TOTAL_FIELDS) + "</s_total>")

    parts.append(TASK_END_TOKEN)
    return "".join(parts)


def _extract(tag, text):
    m = re.search(rf"<s_{tag}>(.*?)</s_{tag}>", text, re.DOTALL)
    if not m:
        return None
    v = m.group(1).strip()
    return v.replace("&lt;", "<").replace("&gt;", ">") if v else None


def _extract_block(tag, text):
    m = re.search(rf"<s_{tag}>(.*?)</s_{tag}>", text, re.DOTALL)
    return m.group(1) if m else ""


def from_donut_output(text: str) -> dict:
    """Donut decoder output string -> canonical dict. Never raises: malformed
    or truncated generations degrade to None fields rather than crashing the
    inference pipeline, and the caller should check validate_schema()."""
    text = text.replace(TASK_START_TOKEN, "").replace(TASK_END_TOKEN, "").replace("</s>", "")

    seller_block = _extract_block("seller", text)
    client_block = _extract_block("client", text)
    items_block = _extract_block("items", text)
    total_block = _extract_block("total", text)

    record = {
        "invoice_no": _extract("invoice_no", text),
        "date_of_issue": _extract("date_of_issue", text),
        "seller": {f: _extract(f, seller_block) for f in PARTY_FIELDS},
        "client": {f: _extract(f, client_block) for f in PARTY_FIELDS},
        "items": [],
        "total": {f: _extract(f, total_block) for f in TOTAL_FIELDS},
    }
    record["seller"]["iban"] = _extract("iban", seller_block)

    for item_match in re.finditer(r"<s_item>(.*?)</s_item>", items_block, re.DOTALL):
        item_text = item_match.group(1)
        record["items"].append({f: _extract(f, item_text) for f in ITEM_FIELDS})

    return record


def normalize(record: dict) -> dict:
    """Type-normalize a raw (string-only, straight-from-Donut) record: numeric
    fields to float, dates kept as-is (already MM/DD/YYYY in this dataset),
    whitespace collapsed. Never fabricates a value that wasn't present."""
    def num(v):
        if v is None or v == "":
            return None
        s = re.sub(r"[^0-9.\-]", "", str(v).replace(",", "."))
        try:
            return round(float(s), 2)
        except ValueError:
            return None

    def clean_str(v):
        if v is None:
            return None
        v = re.sub(r"\s+", " ", str(v)).strip()
        return v or None

    out = json.loads(json.dumps(record))  # deep copy
    out["invoice_no"] = clean_str(out.get("invoice_no"))
    out["date_of_issue"] = clean_str(out.get("date_of_issue"))
    for party in ("seller", "client"):
        p = out.get(party) or {}
        for f in ("name", "address", "tax_id", "iban"):
            if f in p:
                p[f] = clean_str(p[f])
        out[party] = p
    for item in out.get("items", []):
        item["description"] = clean_str(item.get("description"))
        item["um"] = clean_str(item.get("um"))
        for f in ("qty", "net_price", "net_worth", "gross_worth"):
            item[f] = num(item.get(f))
        vp = item.get("vat_pct")
        if vp is not None:
            m = re.search(r"(\d+)", str(vp))
            item["vat_pct"] = int(m.group(1)) if m else None
    t = out.get("total") or {}
    for f in TOTAL_FIELDS:
        t[f] = num(t.get(f))
    out["total"] = t
    return out


def validate_schema(record: dict):
    """Returns (is_valid, problems[]). Structural check only -- does not
    require every value to be non-null, since a genuinely blank field on the
    source invoice should stay null rather than be treated as an error."""
    problems = []
    for f in TOP_FIELDS:
        if f not in record:
            problems.append(f"missing top-level field: {f}")
    for party in ("seller", "client"):
        if party not in record or not isinstance(record[party], dict):
            problems.append(f"missing/invalid block: {party}")
    if "items" not in record or not isinstance(record["items"], list):
        problems.append("missing/invalid: items")
    else:
        for i, item in enumerate(record["items"]):
            for f in ITEM_FIELDS:
                if f not in item:
                    problems.append(f"item[{i}] missing field: {f}")
    if "total" not in record or not isinstance(record["total"], dict):
        problems.append("missing/invalid block: total")
    return (len(problems) == 0, problems)
