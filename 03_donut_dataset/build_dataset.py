"""
Convert data/splits/{train,val,test}/labels/*.json into Donut training targets
and write one manifest.jsonl per split: {"file_name": "...", "target": "<s_invoice>...</s_invoice>"}
This manifest + the split's images/ folder is everything Colab needs.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import schema

def build_split(split_name, splits_root="../data/splits"):
    lbl_dir = os.path.join(splits_root, split_name, "labels")
    out_path = os.path.join(splits_root, split_name, "manifest.jsonl")
    n = 0
    with open(out_path, "w") as out:
        for lbl_file in sorted(glob.glob(os.path.join(lbl_dir, "*.json"))):
            with open(lbl_file) as f:
                record = json.load(f)
            record.pop("_meta", None)
            record = schema.normalize(record)
            target = schema.to_donut_target(record)
            invoice_id = os.path.splitext(os.path.basename(lbl_file))[0]
            out.write(json.dumps({"file_name": invoice_id + ".jpg", "target": target}) + "\n")
            n += 1
    print(f"{split_name}: {n} examples -> {out_path}")

if __name__ == "__main__":
    for split in ("train", "val", "test"):
        build_split(split)
