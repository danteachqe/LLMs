from datasets import load_dataset

# Step 1: Load the Iris dataset from Hugging Face
dataset = load_dataset("scikit-learn/iris")

# Step 2: Split the dataset into training (120 samples) and evaluation (30 samples)
split_dataset = dataset["train"].train_test_split(test_size=30, seed=42)

# Step 3: Save the splits to CSV files
train_file = "iris_train.csv"
eval_file = "iris_eval.csv"

split_dataset['train'].to_pandas().to_csv(train_file, index=False)
split_dataset['test'].to_pandas().to_csv(eval_file, index=False)

print(f"Training data saved to '{train_file}'")
print(f"Evaluation data saved to '{eval_file}'")
