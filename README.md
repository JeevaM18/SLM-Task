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

##  Model Summary

- **Base Model:** `bert-base-multilingual-cased` (or `xlm-roberta-base`)
- **Architecture:** Transformer encoder with classification head
- **Task:** Text classification (`num_labels = 3`)
- **Loss Function:** Cross-entropy
- **Optimizer:** AdamW with weight decay
- **Scheduler:** Linear warmup and decay
  
---


##  Project Structure

/data/
- **train.csv**
- **val.csv**
- **test.csv**

/notebooks/
- **data_exploration.ipynb** - Dataset analysis, class distribution, sample exploration
- **model_training.ipynb** - Interactive model training and experimentation
- **evaluation_analysis.ipynb** - Results analysis, error analysis, visualizations

/src/
- **train_model.py** - Main training script with Hugging Face Trainer
- **data_preprocessing.py** - Text cleaning, tokenization, dataset preparation
- **model_utils.py** - Model loading, saving, prediction utilities
- **config.py** - Training hyperparameters and model configurations
- **inference.py** - Can do prediction

/models/
- **trained_model/** - Fine-tuned model weights and configuration
- **tokenizer/** - Trained tokenizer files
- **.gitkeep** - Maintains directory structure

/reports/
- **model_report.md** - Model architecture decisions, training insights, improvements
- **evaluation_metrics.json** - Detailed metrics (F1, precision, recall, accuracy)
- **confusion_matrix.png** - Classification results visualization

Root Files
- **requirements.txt** - Python dependencies (transformers, torch, datasets, etc.)
- **README.md** - Project documentation (this file)
- **submission.md** - Your approach, model decisions, and key learnings
-- **train.py** - Simple training script entry point
-- **.gitignore** - Files to exclude from git (models/, pycache, etc.)
-- **app.py** - prediction pipeline using streamlit
---

#  Setup Instructions
1. Environment Setup
<pre> pip install -r requirements.txt </pre>

If running into tokenizer issues:
<pre> pip install sentencepiece </pre>

2. Preprocess & Tokenize
<pre> python src/data_preprocessing.py </pre>

3. Train the Model
<pre> python src/train_model.py </pre>

4. Evaluate
Evaluation results (F1, accuracy, precision, recall) are logged and saved in:
<pre> reports/evaluation_metrics.json
 reports/confusion_matrix.png </pre>

---

#  Metrics 
| Metric    | Score |
| --------- | ----- |
| Accuracy  | 60.2% |
| F1 Score  | 60.3% |
| Precision | 61.5% |
| Recall    | 60.2% |

--- 

#  Design Philosophy
This project demonstrates:

- Sound technical foundations: Hugging Face Transformers, PyTorch, clean modular design.
- Maintainability: Each file has a single responsibility, following industry conventions.
- Resourcefulness: Combines Hugging Face datasets, Trainer, and visual evaluation tools.
- Design trade-offs: Lightweight BERT models were preferred over heavier multilingual models to balance performance and training time.

---

# 🚀 Inference & Deployment
This project supports both command-line inference and web-based UI via Streamlit:

**🔍 Inference via Python script**
Run a quick prediction on a single input:
<pre> python src/inference.py </pre>
- The model loads from /models/trained_model/
- Edit the test_input variable inside inference.py to try different texts
- Output includes predicted label, confidence, and class-wise probabilities

# 🌐 Deploy via Streamlit
An interactive UI is available:
<pre> streamlit run app.py </pre>
- Paste any text input to get sentiment prediction
- Behind the scenes, it uses the same inference.py pipeline

---

#  Future Enhancements
- Add multilingual input support in Streamlit app
- Support batch inference on CSV uploads
- Add explanations via SHAP/LIME in UI

