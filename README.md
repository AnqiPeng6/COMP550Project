# Cross-Lingual Sentiment Analysis with Multilingual Transformers

## Overview
This project investigates the zero-shot cross-lingual generalization of multilingual transformer models (e.g., mBERT and XLM-R) on sentiment classification. The model is fine-tuned on a single source language and evaluated on different target languages to analyze transfer performance.

We focus on two key research questions:
1. **Symmetry**: Is cross-lingual transfer performance symmetric across language pairs (A → B vs B → A)?
2. **Per-class Performance**: Do certain sentiment classes degrade more during cross-lingual transfer?



## Data Download

To obtain the dataset, run the provided download script:

```bash
python scripts/downloaddata.py
```bash
python scripts/sampledata.py

## Dataset
We use the **Amazon Multilingual Reviews dataset** (via Hugging Face SetFit), covering multiple languages:
- English (en)
- German (de)
- French (fr)
- Spanish (es)
- Chinese (zh)
- Japanese (ja)

Each language contains:
- Train: 200,000 samples  
- Validation: 5,000 samples  
- Test: 5,000 samples  

To improve efficiency, we create controlled subsets:
- Train: 10,000  
- Validation: 2,000  
- Test: 5,000  

Datasets are stored locally after download for reproducibility.

---

## Methodology

### Data Preparation
- Downloaded per-language datasets using Hugging Face `datasets`
- Standardized format: `text`, `label`, `label_text`, `lang`
- Created equal-sized subsets across languages for fair comparison

### Model
- example
- Model: `xlm-roberta-base`
- Task: 5-class sentiment classification
- Fine-tuning on source language only

### Experiments

#### 1. Monolingual Baseline
- Train and test on the same language (e.g., EN → EN)
- Provides upper-bound performance

#### 2. Cross-Lingual Transfer
- Train on source language, test on target language
- Example:
  - EN → ZH
  - ZH → EN

#### 3. Symmetry Analysis
- Compare A → B vs B → A performance gaps

#### 4. Per-Class Analysis
- Compute confusion matrices
- Analyze class-wise performance differences across languages

---

## Evaluation Metrics
- Accuracy
- Macro-F1 score (important for class balance)
- Confusion matrix (for per-class analysis)

---

## Project Structure
