# EXP-001: Data Sources

This document lists the HuggingFace datasets required for EXP-001.
All datasets are publicly available and freely downloadable via the
`datasets` Python library.

---

## Datasets

### 1. XSum (text_summarization task)

- **HuggingFace dataset ID:** `EdinburghNLP/xsum`
- **HuggingFace URL:** https://huggingface.co/datasets/EdinburghNLP/xsum
- **Paper citation:** Narayan et al. (2018). "Don't Give Me the Details, Just the Summary!"
- **Split used:** `test`
- **Sample size:** 100 items (from ~11,334 test examples)
- **Input field:** `document` (BBC news article body)
- **Reference field:** `summary` (one-sentence gold summary)
- **Native metric:** BERTScore-F1 (against gold summary)
- **CoEval benchmark loader:** `xsum` (registered in `Public/benchmark/loaders/__init__.py`)
- **Loader metric:** `bertscore`

**Download command:**
```python
from datasets import load_dataset
ds = load_dataset("EdinburghNLP/xsum", split="test")
```

---

### 2. CNN/DailyMail (news_summarization task)

- **HuggingFace dataset ID:** `abisee/cnn_dailymail`
- **HuggingFace URL:** https://huggingface.co/datasets/abisee/cnn_dailymail
- **Dataset version:** `3.0.0`
- **Paper citation:** Hermann et al. (2015). "Teaching Machines to Read and Comprehend."
- **Split used:** `test`
- **Sample size:** 100 items (from ~11,490 test examples)
- **Input field:** `article` (full news article text)
- **Reference field:** `highlights` (bullet-point gold summary, joined as single string)
- **Native metric:** BERTScore-F1 (against gold highlights)
- **CoEval benchmark loader:** `cnn_dailymail` (registered in `Public/benchmark/loaders/__init__.py`)
- **Loader metric:** `bertscore`

**Download command:**
```python
from datasets import load_dataset
ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
```

---

### 3. CodeSearchNet (code_explanation task)

- **HuggingFace dataset ID:** `code-search-net/code_search_net`
- **HuggingFace URL:** https://huggingface.co/datasets/code-search-net/code_search_net
- **Language subset:** `python` (Python functions only)
- **Paper citation:** Husain et al. (2019). "CodeSearchNet Challenge."
- **Split used:** `test`
- **Sample size:** 100 items (from ~22,176 Python test examples)
- **Input field:** `whole_func_string` (full Python function source code)
- **Reference field:** `func_documentation_string` (gold docstring)
- **Native metric:** BLEU-4 (against gold docstring)
- **CoEval benchmark loader:** `codesearchnet` (registered in `Public/benchmark/loaders/__init__.py`)
- **Loader metric:** `bleu`

**Download command:**
```python
from datasets import load_dataset
ds = load_dataset("code-search-net/code_search_net", "python", split="test")
```

---

## Required Python packages for native metric computation

```
pip install evaluate          # HuggingFace evaluate library
pip install bert_score        # BERTScore implementation
pip install sacrebleu         # SacreBLEU for BLEU-4 computation
pip install transformers      # Required by BERTScore
```

---

## License summary

| Dataset | License | Commercial use |
|---|---|---|
| EdinburghNLP/xsum | MIT | Yes |
| abisee/cnn_dailymail | Apache 2.0 | Yes |
| code-search-net/code_search_net | CC-BY-4.0 | Yes |

All three datasets are permissively licensed and suitable for academic
research and paper publication.

---

## Notes

- XSum and CNN/DailyMail use BERTScore-F1 with the `roberta-large` model
  as the scoring backbone (standard in the NLP evaluation literature).
- CodeSearchNet uses corpus-level BLEU-4 with SacreBLEU smoothing method 1.
- The CoEval `benchmark.compute_scores` module handles metric computation
  automatically after Phase 5 completes, reading `benchmark_native_score`
  from the phase3 JSONL records.
