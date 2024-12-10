from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

# Step 1: Load your dataset from a CSV file
train_file = "./iris_train.csv"  # Update with your actual file path
data = pd.read_csv(train_file)

# Step 2: Convert the dataset to a Hugging Face Dataset
dataset = Dataset.from_pandas(data)

# Step 3: Preprocess the dataset: Convert all feature columns into a single textual description
def convert_to_text(example):
    features = f"This flower has a sepal length of {example['SepalLengthCm']} cm, a sepal width of {example['SepalWidthCm']} cm, " \
               f"a petal length of {example['PetalLengthCm']} cm, and a petal width of {example['PetalWidthCm']} cm."
    return {"text": features}

dataset = dataset.map(convert_to_text)

# Step 4: Map the 'Species' column to numerical labels for classification
label_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

# Ensure 'Species' is correctly mapped
dataset = dataset.map(lambda x: {"label": label_mapping[x["Species"]]})

# Drop unnecessary columns
dataset = dataset.remove_columns(["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])

# Step 5: Tokenize the dataset
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_data(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_data, batched=True)

# Step 6: Split the dataset into training and evaluation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Step 7: Load a pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Step 8: Define training arguments with increased epochs and logging
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Perform evaluation at the end of each epoch
    save_strategy="epoch",  # Save checkpoints at the end of each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,  # Increased epochs for better training
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Step 9: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Step 10: Train the model
trainer.train()

# Step 11: Evaluate the model on training data to check overfitting
train_results = trainer.evaluate(train_dataset)
print(f"Training Evaluation Results: {train_results}")

# Step 12: Evaluate the model on evaluation data
eval_results = trainer.evaluate(eval_dataset)
print(f"Evaluation Results: {eval_results}")

# Step 13: Calculate accuracy on evaluation data and print predictions
predictions = trainer.predict(eval_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)
actual_labels = eval_dataset["label"]

# Calculate accuracy
correct_predictions = sum(predicted_labels[i] == actual_labels[i] for i in range(len(actual_labels)))
accuracy = correct_predictions / len(actual_labels)

# Print accuracy
print(f"Accuracy on Evaluation Dataset: {accuracy:.2%}")

# Print predictions for all rows
label_reverse_mapping = {v: k for k, v in label_mapping.items()}
print("\nPredictions:")
for i in range(len(eval_dataset)):
    text = eval_dataset[i]["text"]
    actual = label_reverse_mapping[actual_labels[i]]
    predicted = label_reverse_mapping[predicted_labels[i]]
    print(f"Row {i + 1}:")
    print(f"Text: {text}")
    print(f"Actual Label: {actual}, Predicted Label: {predicted}")
    print("-" * 50)
