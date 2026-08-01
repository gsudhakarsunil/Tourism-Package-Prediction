import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop CustomerID as it's not needed for model training
df = df.drop(columns=["CustomerID"])

# Define features (X) and target (y)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the split datasets
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("Data preparation complete. Train and test sets saved.")
