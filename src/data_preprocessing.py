import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import os
def load_data(data_dir="data"):

    print("Loading CSV files...")

    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    print("Loaded and converted to Hugging Face DatasetDict.")
    return dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def preprocess_dataset(dataset_dict, model_checkpoint="xlm-roberta-base"):
    print(f"Loading tokenizer from checkpoint: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset_dict.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    print(" Saving tokenized CSVs to:", "data")
    os.makedirs("data", exist_ok=True)

    for split in tokenized_datasets.keys():
        df = tokenized_datasets[split].with_format("numpy").to_pandas()
        df.to_csv(os.path.join("data", f"tokenized_{split}.csv"), index=False)
        print(f"tokenized_{split}.csv saved.")


    print("Tokenization complete. Dataset ready for training.")
    return tokenized_datasets, tokenizer
