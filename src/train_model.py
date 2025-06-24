import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import os

os.makedirs("../models/trained_model", exist_ok=True)
os.makedirs("../models/tokenizer", exist_ok=True)

def load_data(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })
    return dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def main():
    print("Python version:", torch.__version__)
    
    dataset = load_data("../data")

    model_ckpt = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=3)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    print("Tokenization complete.")

    training_args = TrainingArguments(
        output_dir="../models/trained_model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate(dataset["test"])
    print("Test Results:", eval_results)

    model.save_pretrained("../models/trained_model")
    tokenizer.save_pretrained("../models/tokenizer")
    print("Model and tokenizer saved!")

if __name__ == "__main__":
    main()
