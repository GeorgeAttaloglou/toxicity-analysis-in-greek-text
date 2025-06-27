import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import fasttext

# Load binary .bin model
fasttext_model = fasttext.load_model("cc.el.300.bin")

# Wrap for compatibility with current code
class FTWrapper:
    def __init__(self, model):
        self.model = model
        self.vector_size = model.get_dimension()

    def __contains__(self, word):
        return True

    def __getitem__(self, word):
        return self.model.get_word_vector(word)

fasttext_model = FTWrapper(fasttext_model)

# Load training data
train_df = pd.read_csv("./Data/offenseval-gr-training-v1.csv")
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["subtask_a"])

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(
    train_df["tweet"], train_df["label"], test_size=0.2, stratify=train_df["label"], random_state=42
)

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, labels, embedding_model, max_len=50):
        self.texts = texts
        self.labels = labels
        self.model = embedding_model
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts.iloc[idx].split()[:self.max_len]
        vecs = [self.model[word] if word in self.model else np.zeros(300) for word in tokens]
        padding = [np.zeros(300)] * (self.max_len - len(vecs))
        vecs.extend(padding)
        vecs_np = np.array(vecs, dtype=np.float32)
        return torch.from_numpy(vecs_np), torch.tensor(self.labels.iloc[idx], dtype=torch.long)

# Create DataLoaders
train_dataset = TweetDataset(X_train, y_train, fasttext_model)
val_dataset = TweetDataset(X_val, y_val, fasttext_model)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# BiLSTM model
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, output_dim=2):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        return self.fc(hidden)

# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluation on validation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(targets.numpy())

print("\nValidation Classification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# --- Evaluate on test set ---
test_df = pd.read_csv("./Data/offenseval-gr-test-v1.csv")
test_labels_df = pd.read_csv("./Data/offenseval-gr-labels-v1.csv", header=None, names=["id", "label"])
test_df = test_df.sort_values("id").reset_index(drop=True)
test_labels_df = test_labels_df.sort_values("id").reset_index(drop=True)
test_labels = label_encoder.transform(test_labels_df["label"])

# Dataset for test
test_dataset = TweetDataset(test_df["tweet"], pd.Series(test_labels), fasttext_model)
test_loader = DataLoader(test_dataset, batch_size=32)

# Predict
y_true_test, y_pred_test = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred_test.extend(preds)
        y_true_test.extend(targets.numpy())

print("\nTest Classification Report:\n", classification_report(y_true_test, y_pred_test, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_true_test, y_pred_test))
