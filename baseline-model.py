import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Load training data
training_dataframe = pd.read_csv("./Data/offenseval-gr-training-cleaned-v1.csv")

# Encode labels from text to int (OFF -> 1, NOT -> 0)
label_encoder = LabelEncoder()
training_dataframe["label"] = label_encoder.fit_transform(
    training_dataframe["subtask_a"]
)

# Split the data into Train and Test
X_train, X_validation, y_train, y_validation = train_test_split(
    training_dataframe["tweet"],
    training_dataframe["label"],
    test_size=0.2,
    stratify=training_dataframe["label"],
    random_state=42,
)

# Create Model pipeline: Tf-IDF + Logistic Regression
model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)

# Train model
model.fit(X_train, y_train)

# Evaluate on validation split
y_predict = model.predict(X_validation)
print(
    "\nClassification Report: \n",
    classification_report(y_validation, y_predict, target_names=label_encoder.classes_),
)
print("\nConfusion Matrix: \n", confusion_matrix(y_validation, y_predict))

# --- Predict on test set ---

# Load test set
X_test_dataframe = pd.read_csv("./Data/offenseval-gr-test-cleaned-v1.csv")
y_test_labels_dataframe = pd.read_csv(
    "./Data/offenseval-gr-labels-v1.csv", header=None, names=["id", "label"]
)

# Sort IDs so that they align in both files to avoid mismatches
X_test_dataframe = X_test_dataframe.sort_values("id").reset_index(drop=True)
y_test_labels_dataframe = y_test_labels_dataframe.sort_values("id").reset_index(
    drop=True
)

# Encode values
y_test_true = label_encoder.transform(y_test_labels_dataframe["label"])
X_test_texts = X_test_dataframe["tweet"]

# Predict
y_test_predict = model.predict(X_test_texts)

# Final evaluation on test set

print("\nFinal Test Set Evaluation:\n")
print(
    "Classification Report:\n",
    classification_report(
        y_test_true, y_test_predict, target_names=label_encoder.classes_
    ),
)
print("Confusion Matrix:\n", confusion_matrix(y_test_true, y_test_predict))
