# scripts/download_data.py
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
from datasets import load_dataset, DatasetDict

LANGUAGES = ["en", "de", "fr", "es", "zh", "ja"]
SAVE_DIR = "data/raw"

# Language-specific Parquet-backed dataset IDs
DATASET_IDS = {
    "en": "SetFit/amazon_reviews_multi_en",
    "de": "SetFit/amazon_reviews_multi_de",
    "fr": "SetFit/amazon_reviews_multi_fr",
    "es": "SetFit/amazon_reviews_multi_es",
    "zh": "SetFit/amazon_reviews_multi_zh",
    "ja": "SetFit/amazon_reviews_multi_ja",
}

def normalize_columns(ds, lang):
    """
    Standardize to:
    - text
    - label
    - label_text (if present)
    - lang
    """
    cols = ds.column_names

    # Find text column
    text_col = None
    for c in ["text", "review_body", "review_text"]:
        if c in cols:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"Could not find text column in {cols}")

    # Find numeric label column
    label_col = None
    for c in ["label", "stars", "rating"]:
        if c in cols:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"Could not find label column in {cols}")

    rename_map = {}
    if text_col != "text":
        rename_map[text_col] = "text"
    if label_col != "label":
        rename_map[label_col] = "label"

    if rename_map:
        ds = ds.rename_columns(rename_map)

    # Add language tag
    ds = ds.map(lambda x: {"lang": lang})

    # Keep only relevant columns
    keep_cols = [c for c in ["text", "label", "label_text", "lang"] if c in ds.column_names]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    return ds

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    for lang in LANGUAGES:
        dataset_id = DATASET_IDS[lang]
        print(f"\nDownloading {lang}: {dataset_id}")

        ds = load_dataset(dataset_id)

        # Some ports may expose a single split, others train/test
        split_names = list(ds.keys())
        print(f"Available splits: {split_names}")

        normalized = {}
        for split in split_names:
            cur = ds[split]
            print(f"  Before normalization [{split}] columns: {cur.column_names}")
            cur = normalize_columns(cur, lang)
            print(f"  After normalization  [{split}] columns: {cur.column_names}")
            print(f"  [{split}] rows: {len(cur)}")
            normalized[split] = cur

        out = DatasetDict(normalized)
        save_path = os.path.join(SAVE_DIR, lang)
        out.save_to_disk(save_path)
        print(f"Saved {lang} to {save_path}")

if __name__ == "__main__":
    main()