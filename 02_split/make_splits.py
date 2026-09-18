"""
Create reproducible 70/15/15 train/val/test splits over the labeled dataset
and physically copy invoice_XXX.jpg/.json into split subfolders (per the
assessment's required naming convention, kept inside each split folder).
"""
import os, json, random, shutil, glob

SEED = 42
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

def main(images_dir="../data/processed/images", labels_dir="../data/processed/labels",
         out_dir="../data/splits"):
    ids = sorted(os.path.splitext(os.path.basename(f))[0]
                 for f in glob.glob(os.path.join(images_dir, "*.jpg")))
    random.Random(SEED).shuffle(ids)

    n = len(ids)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])
    splits = {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }

    for split_name, split_ids in splits.items():
        img_out = os.path.join(out_dir, split_name, "images")
        lbl_out = os.path.join(out_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for _id in split_ids:
            shutil.copy(os.path.join(images_dir, _id + ".jpg"), os.path.join(img_out, _id + ".jpg"))
            shutil.copy(os.path.join(labels_dir, _id + ".json"), os.path.join(lbl_out, _id + ".json"))

    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump({"seed": SEED, "ratios": SPLIT, **splits}, f, indent=2)

    print(f"Total: {n} | train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"Split manifest saved to {out_dir}/splits.json (seed={SEED}, reproducible)")

if __name__ == "__main__":
    main()
