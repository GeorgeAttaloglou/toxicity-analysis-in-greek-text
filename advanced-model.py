import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Load dataset
train_df = pd.read_csv("./Data/offenseval-gr-training-cleaned-v1.csv")
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["subtask_a"])

# Preprocessing
model_name = "nlpaueb/bert-base-greek-uncased-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)


class ToxicityDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    train_df["tweet"],
    train_df["label"],
    test_size=0.2,
    stratify=train_df["label"],
    random_state=42,
)

train_dataset = ToxicityDataset(X_train.tolist(), y_train.tolist())
val_dataset = ToxicityDataset(X_val.tolist(), y_val.tolist())

# Load BERT model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)


def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=1)
    return {"f1": (preds == torch.tensor(p.label_ids)).float().mean().item()}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Validation Evaluation
preds = trainer.predict(val_dataset)
y_pred = torch.argmax(torch.tensor(preds.predictions), axis=1).numpy()
y_true = y_val.to_numpy()
print(
    "\nValidation Classification Report:\n",
    classification_report(y_true, y_pred, target_names=label_encoder.classes_),
)
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Test Set Evaluation
test_df = pd.read_csv("./Data/offenseval-gr-test-cleaned-v1.csv")
test_labels_df = pd.read_csv(
    "./Data/offenseval-gr-labels-v1.csv", header=None, names=["id", "label"]
)
test_df = test_df.sort_values("id").reset_index(drop=True)
test_labels_df = test_labels_df.sort_values("id").reset_index(drop=True)
test_labels = label_encoder.transform(test_labels_df["label"])
test_dataset = ToxicityDataset(test_df["tweet"].tolist(), test_labels.tolist())

preds = trainer.predict(test_dataset)
y_test_pred = torch.argmax(torch.tensor(preds.predictions), axis=1).numpy()
y_test_true = test_labels

print(
    "\nTest Classification Report:\n",
    classification_report(
        y_test_true, y_test_pred, target_names=label_encoder.classes_
    ),
)
print("Confusion Matrix:\n", confusion_matrix(y_test_true, y_test_pred))
