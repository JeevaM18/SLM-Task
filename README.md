# SLM-Task

# Multilingual Sentiment Classification

This project implements a multilingual sentiment classification system using transformer-based models from Hugging Face. It is designed to classify textual reviews into three sentiment categories: **positive**, **neutral**, and **negative**, supporting multiple languages.

---

##  Objective

Develop a **robust NLP pipeline** for multilingual sentiment classification using `transformers`, `datasets`, and PyTorch, while ensuring:

- Clear and maintainable code structure
- Reproducibility
- Visual, metric-based evaluation
- Clean packaging for real-world usage or extension

---

## 🧠 Model Summary

- **Base Model:** `bert-base-multilingual-cased` (or `xlm-roberta-base`)
- **Architecture:** Transformer encoder with classification head
- **Task:** Text classification (`num_labels = 3`)
- **Loss Function:** Cross-entropy
- **Optimizer:** AdamW with weight decay
- **Scheduler:** Linear warmup and decay
  
---


## 📁 Project Structure

├── data/               
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── tokenized_*.csv
│
├── models/
│   ├── tokenizer/        
│   └── trained_model/    
│
├── notebooks/             
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation_analysis.ipynb
│
├── reports/
│   ├── confusion_matrix.png
│   ├── evaluation_metrics.json
│   └── model_report.md
│
├── src/
│   ├── config.py             
│   ├── data_preprocessing.py 
│   ├── model_utils.py         
│   └── train_model.py         
│
├── train.py              # Simple entrypoint script
├── requirements.txt
├── .gitignore
└── README.md
