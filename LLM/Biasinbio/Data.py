from datasets import load_dataset
from transformers import AutoTokenizer

# Paths
model_path = "./model"  # Path where the tokenizer is saved
data_save_path = "./data"  # Path to save the tokenized dataset

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Download and load the dataset
print("Downloading Bias in Bios test dataset...")
test_dataset = load_dataset("LabHC/bias_in_bios", split="test[:500]")  # Limit to 500 entries

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["hard_text"], truncation=True, padding="max_length", max_length=128)

print("Tokenizing the dataset...")
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Prepare the dataset for saving
tokenized_test_dataset = tokenized_test_dataset.rename_column("profession", "labels")
tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "gender"])

# Save the tokenized dataset
tokenized_test_dataset.save_to_disk(data_save_path)
print(f"Tokenized dataset saved at {data_save_path}")
