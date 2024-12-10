from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os

# Step 1: Load your dataset
data_file = "./iris_train.csv"
data = pd.read_csv(data_file)

# Step 2: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(data)

# Step 3: Preprocess dataset
def convert_to_text(example):
    return {
        "text": f"Features: Sepal length {example['SepalLengthCm']} cm, Sepal width {example['SepalWidthCm']} cm, "
                f"Petal length {example['PetalLengthCm']} cm, Petal width {example['PetalWidthCm']} cm."
    }

dataset = dataset.map(convert_to_text)

# Map labels
label_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
dataset = dataset.map(lambda x: {"label": label_mapping[x["Species"]]})

# Tokenize
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_data(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_data, batched=True)

# Step 4: K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df = pd.DataFrame(tokenized_dataset)

results = []
model_checkpoint = "distilbert-base-uncased"  # Start with the pre-trained model

for fold, (train_idx, val_idx) in enumerate(kf.split(df, df["label"])):
    print(f"\n--- Fold {fold + 1} ---")
    
    # Create result directories
    fold_results_dir = f"./results_fold_{fold + 1}"
    os.makedirs(fold_results_dir, exist_ok=True)
    
    # Split data into training and validation sets
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    
    # Load the model (incrementally from the last checkpoint or best model)
    model = DistilBertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=fold_results_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs_fold_{fold + 1}",
        logging_steps=10,
        save_total_limit=2,  # Keep only the last 2 checkpoints
        load_best_model_at_end=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train and save the model
    trainer.train()
    trainer.save_model(fold_results_dir)  # Save the best model for the fold
    
    # Update checkpoint for the next fold
    model_checkpoint = fold_results_dir  # Use the best model of the current fold as the starting point for the next
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    results.append(eval_results)
    
    print(f"Results for Fold {fold + 1}: {eval_results}")

# Step 5: Calculate average metrics across all folds
average_results = {key: sum(r[key] for r in results) / len(results) for key in results[0]}
print("\n--- Average Results Across All Folds ---")
for key, value in average_results.items():
    print(f"{key}: {value:.4f}")

# Final Output
print("\nTraining and evaluation completed. Models and checkpoints are saved in respective fold directories.")
