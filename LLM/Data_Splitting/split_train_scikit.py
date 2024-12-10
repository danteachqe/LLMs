from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Step 2: Split the dataset into training (120 samples) and evaluation (30 samples)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=30/150, random_state=42)

# Step 3: Convert splits to DataFrames
train_data = pd.DataFrame(X_train, columns=iris.feature_names)
train_data['target'] = y_train

eval_data = pd.DataFrame(X_eval, columns=iris.feature_names)
eval_data['target'] = y_eval

# Step 4: Save the splits to CSV files
train_data.to_csv("iris_train.csv", index=False)
eval_data.to_csv("iris_eval.csv", index=False)

print("Training data saved to 'iris_train.csv'")
print("Evaluation data saved to 'iris_eval.csv'")
