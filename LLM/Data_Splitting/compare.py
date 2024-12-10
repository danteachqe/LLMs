from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer
import pandas as pd
import matplotlib.pyplot as plt

# Function to load and preprocess the dataset
def load_and_preprocess_dataset(file_path):
    data = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(data)

    # Convert features to textual descriptions
    def convert_to_text(example):
        features = f"Sepal length: {example['SepalLengthCm']}, Sepal width: {example['SepalWidthCm']}, " \
                   f"Petal length: {example['PetalLengthCm']}, Petal width: {example['PetalWidthCm']}."
        return {"text": features}

    dataset = dataset.map(convert_to_text)

    # Tokenize the dataset
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    def tokenize_data(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_data, batched=True)
    return tokenized_dataset, dataset

# Function to evaluate a dataset and calculate accuracy
def evaluate_dataset(model, tokenizer, file_path):
    tokenized_eval_dataset, eval_dataset = load_and_preprocess_dataset(file_path)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )

    # Make predictions
    predictions = trainer.predict(tokenized_eval_dataset)

    # Process predictions
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

    return accuracy

# Paths to the two datasets
dataset1_path = "./iris_eval.csv"  # Update with your first dataset path
dataset2_path = "./iris_train.csv"  # Update with your second dataset path

# Load the trained model
model = DistilBertForSequenceClassification.from_pretrained("./results/checkpoint-60")  # Path to the saved model

# Evaluate both datasets
accuracy1 = evaluate_dataset(model, DistilBertTokenizer.from_pretrained("distilbert-base-uncased"), dataset1_path)
accuracy2 = evaluate_dataset(model, DistilBertTokenizer.from_pretrained("distilbert-base-uncased"), dataset2_path)

# Print accuracies
print(f"Accuracy on Dataset 1: {accuracy1:.2%}")
print(f"Accuracy on Dataset 2: {accuracy2:.2%}")

# Plot the comparison
datasets = ["Dataset 1", "Dataset 2"]
accuracies = [accuracy1, accuracy2]

plt.bar(datasets, accuracies)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparison of Accuracy Across Two Datasets")
plt.show()
