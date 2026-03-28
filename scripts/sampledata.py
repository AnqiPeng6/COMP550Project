from datasets import load_from_disk, DatasetDict
import os

LANGS = ["en", "de", "fr", "es", "zh", "ja"]
RAW_DIR = "data/raw"
OUT_DIR = "data/subsets"

TRAIN_N = 10000
VAL_N = 2000
SEED = 42

for lang in LANGS:
    ds = load_from_disk(os.path.join(RAW_DIR, lang))

    train_subset = ds["train"].shuffle(seed=SEED).select(range(TRAIN_N))
    val_subset = ds["validation"].shuffle(seed=SEED).select(range(VAL_N))
    test_subset = ds["test"]  # keep full test

    subset_ds = DatasetDict({
        "train": train_subset,
        "validation": val_subset,
        "test": test_subset
    })

    out_path = os.path.join(OUT_DIR, lang)
    subset_ds.save_to_disk(out_path)
    print(f"Saved subset for {lang} to {out_path}")