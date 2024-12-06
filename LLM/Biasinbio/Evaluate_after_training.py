from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
fine_tuned_model_path = "./fine_tuned_model"  # Path to your fine-tuned model
data_path = "./data"  # Path to the tokenized dataset
batch_size = 16  # Adjust batch size as needed

# Load the fine-tuned model and tokenizer
print("Loading fine-tuned model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# Load the tokenized dataset
print("Loading tokenized dataset...")
tokenized_test_dataset = load_from_disk(data_path)

# Create a DataLoader for evaluation
dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size)

# Set model to evaluation mode
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize storage for predictions and labels
predictions, true_labels, genders = [], [], []

# Progress tracking variables
total_batches = len(dataloader)
total_examples = len(tokenized_test_dataset)
processed_examples = 0

print("Evaluating the fine-tuned model...")
for batch_idx, batch in enumerate(dataloader):
    # Move inputs and labels to the appropriate device
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    labels = batch["labels"].to(device)
    genders_batch = batch["gender"].numpy()

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=-1).cpu().numpy()

    # Store results
    predictions.extend(preds)
    true_labels.extend(labels.cpu().numpy())
    genders.extend(genders_batch)

    # Progress logging
    processed_examples += len(batch["input_ids"])
    if (batch_idx + 1) % (total_batches // 20) == 0 or (batch_idx + 1) == total_batches:
        percentage = (processed_examples / total_examples) * 100
        print(f"Processed {processed_examples}/{total_examples} examples ({percentage:.2f}%)...")

# Convert results to numpy arrays
predictions = np.array(predictions)
true_labels = np.array(true_labels)
genders = np.array(genders)

# Debugging checks
print("\n--- Debugging Information ---")
print("Unique values in true labels:", np.unique(true_labels))
print("Unique values in predictions:", np.unique(predictions))
print("Unique values in genders:", np.unique(genders))

# Classification report
print("\n--- Classification Report ---")
print(classification_report(true_labels, predictions, labels=np.unique(true_labels)))

# Confusion matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(true_labels, predictions, labels=np.unique(true_labels))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Fairness Metrics
male_indices = genders == 0
female_indices = genders == 1

male_accuracy = (true_labels[male_indices] == predictions[male_indices]).mean() if np.sum(male_indices) > 0 else 0
female_accuracy = (true_labels[female_indices] == predictions[female_indices]).mean() if np.sum(female_indices) > 0 else 0

print("\n--- Fairness Metrics ---")
print(f"Male Accuracy: {male_accuracy:.4f}")
print(f"Female Accuracy: {female_accuracy:.4f}")
print(f"Disparity (Male - Female): {male_accuracy - female_accuracy:.4f}")
