from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

# Load the full dataset
dataset = load_dataset("LabHC/bias_in_bios")

# Subset the train and validation splits
train_dataset = dataset["train"].shuffle(seed=42).select(range(500))  # Use the first 750 examples for training
validation_dataset = dataset["dev"].shuffle(seed=42).select(range(150))  # Use the first 150 examples for validation

# Create a DatasetDict with the subsets
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset
})

print("Training set size:", len(dataset["train"]))
print("Validation set size:", len(dataset["validation"]))

# Load the model and tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=28)  # Assuming 28 professions

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples["hard_text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Rename the 'profession' column to 'labels'
tokenized_datasets = tokenized_datasets.rename_column("profession", "labels")

# Remove unused columns
tokenized_datasets = tokenized_datasets.remove_columns(["hard_text", "gender"])

# Convert to PyTorch tensors
tokenized_datasets.set_format("torch")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # Subset of 750 entries
    eval_dataset=tokenized_datasets["validation"]  # Subset of 150 entries
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Training complete. Model saved at ./fine_tuned_model")
