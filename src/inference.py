import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
import json

MODEL_DIR = os.path.join("models", "trained_model")
TOKENIZER_DIR = os.path.join("models", "tokenizer")
CONFIG_FILE = os.path.join("config", "config.json")

LABELS = ["positive", "neutral", "negative"]

def load_model():
    print("🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval() 
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        label = LABELS[pred_class.item()]
    return {
        "text": text,
        "label": label,
        "confidence": round(confidence.item(), 4),
        "probabilities": {LABELS[i]: round(p.item(), 4) for i, p in enumerate(probs[0])}
    }

if __name__ == "__main__":
    tokenizer, model = load_model()
    test_input = "This product is not good"
    result = predict(test_input, tokenizer, model)
    print("Inference Result:", json.dumps(result, indent=2))
