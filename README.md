# Invoice Information Extraction — Full Guide

This implements the assessment end-to-end: 489 invoice images -> fine-tuned
Donut model -> structured JSON, with evaluation and an optional Streamlit UI.

**Read this whole file once before running anything** — it explains *why*
each step exists, not just the command to type.

---

## 0. The one decision this whole project hinges on

The dataset you were given (Kaggle: *"High-Quality Invoice Images for OCR"*)
ships as **images only — no ground-truth JSON**. To fine-tune anything you
need labeled targets, and hand-labeling 489 images in a 3-day window isn't
realistic.

The fix: **every invoice in this dataset is rendered from the same fixed
template** (Seller/Client always in the same two columns, same table
columns, same header wording). That means we can reconstruct ground truth
deterministically — OCR the words, place them into columns using the
template's known geometry, and cross-check every number with invoice
arithmetic (`qty × net_price == net_worth`, `net_worth + vat == gross_worth`)
to catch and auto-correct OCR digit errors.

This is implemented in `01_data_labeling/label_invoices.py` and was
**tested against your 3 uploaded sample images — all 3 parse with zero
flagged issues and every field matches the source image exactly**,
including a tricky case where a number ("1 245,00") was OCR'd as two
separate word fragments. That said, it was only validated on 3 of 489
images. **Step 1 below tells you exactly how to spot-check it on the full
set before trusting it as training data** — do not skip that check.

Document this decision in your submission's README/report — it's a
legitimate, defensible engineering call, but it's a call, and an assessor
may ask about it.

---

## 1. Local setup (your machine — no GPU needed for this part)

```bash
# system dependency (the OCR engine itself, not just the python wrapper)
sudo apt-get install -y tesseract-ocr        # macOS: brew install tesseract

pip install -r requirements.txt
```

Put your 489 raw images in `data/raw/` (any filenames — they'll get renamed).

### 1a. Run the labeler on the full dataset

```bash
cd 01_data_labeling
python label_invoices.py ../data/raw
```

This writes:
- `data/processed/images/invoice_001.jpg` ... `invoice_489.jpg` (renamed, per the assessment's required naming)
- `data/processed/labels/invoice_001.json` ... (matching ground truth)
- `data/filename_mapping.csv` — traceability back to your original filenames
- `data/review/review_needed.csv` — every invoice the arithmetic check flagged

> If this crashes right at the very end with `FileNotFoundError` on the
> review CSV, your images/labels already finished writing (that failure was
> in the last, separate write step). Don't re-run the whole thing — just run
> `python label_invoices.py --rebuild-review` to regenerate the CSV from the
> labels already on disk in seconds. (This specific bug is already fixed in
> this version, but the recovery command is kept in case any other write
> step ever fails partway through a 489-image run.)

**Manual action required:** open `review_needed.csv`. If it's a small
fraction of the dataset (a rough guideline: under ~15%), spot-check 15-20 of
those flagged images against their JSON by eye, and fix any wrong by hand
editing the `.json` file directly. If it's a large fraction, the labeler's
geometry assumptions may not hold on some sub-template — open a few flagged
images and check what's different before trusting the rest.

Also spot check ~10 **non-flagged** images at random — a clean arithmetic
check doesn't catch every possible error type (e.g. two digits both wrong in
a way that still multiplies out).

### 1b. Split the dataset

```bash
cd ../02_split
python make_splits.py
```

70/15/15 train/val/test, seeded (42) for reproducibility. Writes
`data/splits/{train,val,test}/{images,labels}/` and `data/splits/splits.json`
(the exact file list per split, so anyone can reproduce this split).

### 1c. Build the Donut training manifests

```bash
cd ../03_donut_dataset
python build_dataset.py
```

Converts each split's labels into Donut's target string format (defined
once in `schema.py` so training/inference/evaluation can never drift out of
sync on field names). Writes `manifest.jsonl` into each split folder.

### 1d. Zip everything for Colab

```bash
cd ..
zip -r invoice_project.zip schema.py data/splits
```

Upload `invoice_project.zip` to your Google Drive (e.g. `MyDrive/invoice_project.zip`).

---

## 2. Training (Google Colab — needs a GPU)

Open `04_train_colab/train_donut.ipynb` in Colab
(**Runtime > Change runtime type > T4 GPU**), and run every cell top to
bottom. It:

1. Mounts Drive, unzips your project data.
2. Loads `naver-clova-ix/donut-base` and adds this task's schema tags as
   special tokens.
3. Runs a **smoke test** first (one real forward + backward + optimizer
   step on a tiny batch) — this must print `SMOKE TEST PASSED` before you
   move on. If it doesn't, something in the data pipeline is broken and
   full training will just waste your GPU hours.
4. Trains for real. Hyperparameters (also required by the brief):

   | | |
   |---|---|
   | Learning rate | 3e-5 |
   | Batch size | 2 per device × 4 grad-accum = effective 8 |
   | Epochs | 30 |
   | Optimizer | AdamW (`adamw_torch`) |
   | Loss | token-level cross-entropy (built into the decoder) |
   | Image preprocessing | resize to 960×1280, ImageNet normalize (Donut's own image processor) |

   Model selection uses **validation loss only** (`load_best_model_at_end`,
   `metric_for_best_model="eval_loss"`) — the test split is never touched
   during training, so there's no leakage into model selection.
5. Saves the best checkpoint + processor to
   `MyDrive/invoice_donut_final/`.

**Expected runtime:** roughly 1.5–3 hours on a T4 for ~340 training images
× 30 epochs at this batch size — budget for this; it's the longest single
step in your 3 days. If Colab disconnects, `output_dir` checkpoints let you
resume with `trainer.train(resume_from_checkpoint=True)`.

**Why Donut:** it's an OCR-free image→JSON document transformer, validated
on CORD (receipt parsing — structurally the same problem as this invoice
task), and known to fine-tune well from a few hundred examples. The
alternative, LayoutLMv3, needs a separate OCR+bounding-box pipeline and
handles repeating line-item structures less naturally. This reasoning is
also in the notebook itself — copy it into your report.

---

## 3. Inference

Download (or keep on Drive and mount) `invoice_donut_final/`, then locally:

```bash
cd 05_inference
python infer.py --checkpoint /path/to/invoice_donut_final --image /path/to/some_invoice.jpg
```

Prints the structured JSON. The raw (untrimmed) model output is also kept
in the result dict for debugging, per the pipeline spec. Malformed or
truncated generations degrade to `null` fields rather than crashing or
inventing values — check `is_valid` / `validation_problems` in the result.

---

## 4. Evaluation (on the held-out TEST split — never seen during training)

```bash
cd 06_evaluation
python evaluate.py --checkpoint /path/to/invoice_donut_final
```

Writes to `data/splits/test/eval_report/`:
- `metrics.json` — precision/recall/F1 per field (invoice_no, date_of_issue,
  seller.*, client.*, total.*, and every line-item field), computed as:
  TP = predicted value matches ground truth exactly (numeric fields within
  0.01), FP = predicted something that doesn't match (including predicting
  a value where ground truth is blank), FN = ground truth has a value the
  model missed or got wrong.
- `errors.json` — every individual field mismatch, categorized as one of:
  missed field / hallucinated field / incorrect value /
  formatting-normalization error / truncation / line-item error (extra or
  missing item).
- `examples.json` — expected vs. predicted JSON side by side for 8 test
  invoices, for your report's "Expected output vs Predicted output" section.

For the write-up, pull a few concrete failures from `errors.json` into a
**Problem → Evidence → Likely Cause → Improvement** narrative — this
script gives you the raw material, but the "likely cause" reasoning needs
you to actually look at the flagged image.

---

## 5. Bonus: Streamlit interface

```bash
cd 07_streamlit_app
INVOICE_CHECKPOINT=/path/to/invoice_donut_final streamlit run app.py
```

Upload an image, it runs the real trained checkpoint (not mock data) and
displays the extracted fields + generated JSON.

---

## Project structure

```
schema.py                    canonical JSON schema + Donut encode/decode (single source of truth)
01_data_labeling/             OCR+geometry auto-labeler (ground truth generation)
02_split/                     70/15/15 reproducible split
03_donut_dataset/             label JSON -> Donut training-target manifests
04_train_colab/train_donut.ipynb   fine-tuning notebook (run in Colab)
05_inference/infer.py         image -> JSON pipeline
06_evaluation/evaluate.py     field-level P/R/F1 + error analysis on test split
07_streamlit_app/app.py       bonus upload-and-extract UI
data/                         raw -> processed -> split data (gitignore this; large + regenerable)
```

## Final delivery checklist (map assessment requirements -> files)

| Assessment requirement | Where it's satisfied |
|---|---|
| Dataset inspection & preprocessing | `01_data_labeling/label_invoices.py` + README §0-1 (explain the no-ground-truth problem and the OCR/geometry labeling decision) |
| invoice_001.jpg / invoice_001.json naming | `data/processed/images/`, `data/processed/labels/` after step 1a |
| 70/15/15 (or justified) split | `02_split/make_splits.py`, `data/splits/splits.json` |
| Model selection + justification | README §2 "Why Donut" + notebook markdown cell 1 (copy both into your report) |
| Hyperparameters (LR, batch size, epochs, optimizer, loss, image preprocessing) | Table in README §2 + `training_config.json` saved next to the checkpoint after training |
| Fine-tuning actually executed | `04_train_colab/train_donut.ipynb` — run it; screenshot/keep the training loss curve output for your report |
| Evaluation: Precision/Recall/F1/field-level accuracy | `06_evaluation/evaluate.py` -> `metrics.json` |
| Per-field analysis (invoice no, date, seller, client, tax ID, total, line items) | `metrics.json` breaks out every one of these individually |
| Expected vs Predicted examples | `examples.json` |
| Error analysis (Problem -> Evidence -> Likely Cause -> Improvement) | `errors.json` gives you the raw mismatches + category; you write the narrative from a handful of concrete cases |
| Bonus: Streamlit interface using the real trained model | `07_streamlit_app/app.py` (already wired to the actual checkpoint, not mock data) |

**What to actually zip up and submit:** the whole project folder (this
README, `schema.py`, all numbered stage folders, the trained checkpoint or
a note on where it's hosted since it's too large for a repo, and your
written report pulling from the table above) — plus the executed notebook
(`File > Download > .ipynb` from Colab after training, so it retains its
output cells as proof it actually ran).



- Ground truth is **derived**, not human-annotated — it inherits any blind
  spot in the labeler's template assumptions. The review-CSV process in
  step 1a exists specifically to catch this; don't skip it.
- The dataset is synthetic and single-template — the model will not
  generalize to real-world invoice layouts without further data.
- Line-item matching in evaluation is by position (top-to-bottom order),
  which is reasonable for this fixed template but wouldn't hold for
  invoices where item order isn't stable.
