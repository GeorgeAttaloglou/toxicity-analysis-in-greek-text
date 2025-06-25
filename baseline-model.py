import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# Load training data
train_df = pd.read_csv("offenseval-gr-training-v1.csv")

# Encode labels (OFF -> 1, NOT -> 0)
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["subtask_a"])

# Train/test split from training set
X_train, X_val, y_train, y_val = train_test_split(
    train_df["tweet"], train_df["label"], test_size=0.2, stratify=train_df["label"], random_state=42
)

# Create pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate on validation split
y_pred = model.predict(X_val)
print("Classification Report:\n", classification_report(y_val, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# --- Predict on test set ---
# Load test set
X_test_df = pd.read_csv("offenseval-gr-test-v1.csv")
y_test_labels_df = pd.read_csv("offenseval-gr-labela-v1.csv", header=None, names=["id", "label"])

# Sort to align IDs
X_test_df = X_test_df.sort_values("id").reset_index(drop=True)
y_test_labels_df = y_test_labels_df.sort_values("id").reset_index(drop=True)

# Encode true labels
y_test_true = label_encoder.transform(y_test_labels_df["label"])
X_test_texts = X_test_df["tweet"]

# Predict
y_test_pred = model.predict(X_test_texts)

# Final evaluation on test set
cr = classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_)
cm = confusion_matrix(y_test_true, y_test_pred)
print("\nFinal Test Set Evaluation:")
print("Classification Report:\n", cr)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Plot classification report as bar chart
report = classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:2, :3]  # Only classes, only precision/recall/f1

report_df.plot(kind="bar", figsize=(8, 6))
plt.title("Classification Report Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()
