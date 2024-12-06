from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define the model name
model_name = "roberta-base"  # Change this to your desired model from Hugging Face

# Download the model and tokenizer
print(f"Downloading model and tokenizer: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=28)  # Update `num_labels` if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
print("Model and tokenizer saved in ./model.")
