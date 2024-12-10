from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer
import pandas as pd

# Step 1: Load the evaluation dataset
eval_file = "./iris_eval.csv"  # Update with your actual file path
eval_data = pd.read_csv(eval_file)

# Step 2: Convert the dataset to Hugging Face Dataset
eval_dataset = Dataset.from_pandas(eval_data)

# Step 3: Preprocess the dataset: Convert features into textual descriptions
def convert_to_text(example):
    features = f"Sepal length: {example['SepalLengthCm']}, Sepal width: {example['SepalWidthCm']}, " \
               f"Petal length: {example['PetalLengthCm']}, Petal width: {example['PetalWidthCm']}."
    return {"text": features}

eval_dataset = eval_dataset.map(convert_to_text)

# Step 4: Tokenize the dataset
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_data(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_eval_dataset = eval_dataset.map(tokenize_data, batched=True)

# Step 5: Load the trained model
model = DistilBertForSequenceClassification.from_pretrained("./results_fold_5/checkpoint-18")  # Path to the saved model

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
)

# Step 7: Make predictions
predictions = trainer.predict(tokenized_eval_dataset)

# Step 8: Process predictions
predicted_labels = predictions.predictions.argmax(axis=1)  # Get the predicted class
actual_labels = eval_dataset["Species"]  # Original labels

# Map label indices back to species names
label_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse mapping

# Calculate accuracy
correct_predictions = sum(
    reverse_label_mapping[predicted_labels[i]] == actual_labels[i]
    for i in range(len(eval_dataset))
)
accuracy = correct_predictions / len(eval_dataset)

# Print accuracy
print(f"Accuracy: {accuracy:.2%}")

# Print all rows with actual and predicted labels
print("Results:")
for i in range(len(eval_dataset)):
    text = eval_dataset[i]["text"]
    actual = actual_labels[i]
    predicted = reverse_label_mapping[predicted_labels[i]]
    print(f"Row {i + 1}:")
    print(f"Text: {text}")
    print(f"Actual Label: {actual}, Predicted Label: {predicted}")
    print("-" * 50)
