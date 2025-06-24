import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os

# Paths
MODEL_DIR = os.path.join("models", "trained_model")
TOKENIZER_DIR = os.path.join("models", "tokenizer")
LABELS = ["positive", "neutral", "negative"]

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        label = LABELS[pred_class.item()]
    return label, confidence.item(), probs[0].tolist()

# UI setup
st.set_page_config(page_title="Multilingual Sentiment Classifier", layout="centered")
st.title("🌍 Multilingual Sentiment Classifier")
st.markdown("Enter a sentence in any language to classify its sentiment.")

text_input = st.text_area("📝 Enter your text here:", height=150)

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        tokenizer, model = load_model()
        label, confidence, probs = predict(text_input, tokenizer, model)

        st.markdown(f"### 🧠 Prediction: `{label}`")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

        st.markdown("#### 🔍 Class Probabilities")
        st.bar_chart({LABELS[i]: prob for i, prob in enumerate(probs)})

st.markdown("---")
st.caption("Built with 🤗 Transformers, Streamlit and PyTorch")
