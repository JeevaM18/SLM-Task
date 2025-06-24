# 📅 Submission Summary - Multilingual Sentiment Classification

## 🚀 Project Title

**Multilingual Sentiment Classification using Hugging Face Transformers**

## ✅ Objective

To build a robust text classification pipeline capable of handling multilingual sentiment analysis (English, Spanish, German) using the Hugging Face Transformers library. This was completed as part of an internship project focused on developing a full-scale NLP workflow.

---

## 🚀 Approach

### Dataset

* Source: `tyqiangz/multilingual-sentiments`
* Sample: 1000 samples each for English, Spanish, and German
* Strategy: Stratified split into Train/Validation/Test sets

### Model

* Initial: `xlm-roberta-base`
* Final: `bert-base-multilingual-cased` (used for better Windows and PyTorch compatibility)

### Pipeline

* Preprocessing: Tokenization with padding and truncation using `AutoTokenizer`
* Training: Hugging Face `Trainer` API
* Metrics: Accuracy, F1 Score, Precision, Recall
* Saving: Final model and tokenizer saved for downstream inference

---

## ⚙️ Key Decisions

* ✅ Switched to `bert-base-multilingual-cased` for Windows and Python 3.13 compatibility
* ✅ Used Trainer API for its seamless integration and evaluation methods
* ✅ Followed modular design with directories for `src/`, `models/`, `notebooks/`, `reports/`
* ✅ Added confusion matrix and metric visualizations in `reports/`

---

## 🛠️ Challenges Faced

* ❌ `xlm-roberta-base` faced SentencePiece build errors on Python 3.13
* ❌ Older Transformers version didn’t support some TrainingArguments (e.g., `evaluation_strategy`)
* ✅ Resolved by switching models and adapting Trainer arguments manually

---

## 📊 Final Evaluation Metrics (Test Set)

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 60.2% |
| F1 Score  | 60.3% |
| Precision | 61.5% |
| Recall    | 60.2% |

---

## 📘 Learnings

* Hands-on experience with multilingual tokenization and text preprocessing
* Deep understanding of Hugging Face Trainer API
* Learned to resolve compatibility issues across OS and Python versions
* Developed a modular, maintainable ML pipeline that supports scaling

---
