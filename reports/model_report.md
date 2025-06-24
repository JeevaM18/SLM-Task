# 🧠 Model Report - Multilingual Sentiment Classification

## 📌 Overview

This report summarizes the design decisions, training process, evaluation outcomes, and possible improvements for the multilingual sentiment classification model developed using the `bert-base-multilingual-cased` transformer.

---

## 🏗️ Model Architecture

* **Base Model**: `bert-base-multilingual-cased` from Hugging Face
* **Model Type**: Transformer-based Sequence Classification
* **Fine-tuning Head**: Linear classification layer on top of pooled output
* **Number of Labels**: 3 (Negative, Neutral, Positive)

---

## 🛠️ Training Configuration

* **Tokenizer**: AutoTokenizer from pre-trained checkpoint
* **Model**: AutoModelForSequenceClassification
* **Training Arguments**:

  * Epochs: 3
  * Learning Rate: 2e-5
  * Batch Size: 16
  * Weight Decay: 0.01
  * Save Total Limit: 1
  * Logging Strategy: per epoch

---

## 📊 Dataset

* **Source**: `tyqiangz/multilingual-sentiments`
* **Languages**: English, Spanish, German
* **Data Size**:

  * Train: 2,100
  * Validation: 450
  * Test: 450

---

## ✅ Performance Metrics

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.602 |
| Precision | 0.615 |
| Recall    | 0.602 |
| F1 Score  | 0.603 |
| Eval Loss | 0.996 |

---

## 📉 Error Analysis

* **Observation**: Most confusion occurs between neutral and positive samples.
* **Short Texts**: Single-word inputs led to ambiguous predictions.
* **Language Influence**: Slight performance dip on German samples.

---

## 🔧 Improvements

* ✅ Use of XLM-Roberta instead of mBERT for better multilingual alignment
* ✅ Increase training epochs (5–10) for better convergence
* ✅ Perform language-specific fine-tuning
* ✅ Apply advanced data augmentation techniques (e.g., back translation)

---

## 🗂️ Artifacts Saved

* `models/trained_model/` – Final model checkpoint
* `models/tokenizer/` – Corresponding tokenizer files
* `reports/evaluation_metrics.json` – Metrics dump
* `reports/confusion_matrix.png` – Visualization

---

## 📌 Summary

The fine-tuned multilingual BERT model achieves over 60% accuracy and balanced F1 score across three languages on sentiment classification. Further tuning and model enhancements can lead to significantly improved cross-lingual generalization and robustness.

---

