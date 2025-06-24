import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def predict(texts, model, tokenizer):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.tolist()

