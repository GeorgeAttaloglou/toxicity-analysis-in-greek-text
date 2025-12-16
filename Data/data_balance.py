import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read the training data
try:
    # Try reading as TSV (common for OffenseEval datasets)
    df = pd.read_csv('Data/train.tsv', sep='\t')
    print("Successfully loaded data as TSV")
    file_extension = 'tsv'
    separator = '\t'
except:
    try:
        # Try reading as CSV
        df = pd.read_csv('Data/offenseval-gr-training-cleaned-v1.csv')
        print("Successfully loaded data as CSV")
        file_extension = 'csv'
        separator = ','
    except:
        print("Error: Could not find or read the training file.")
        print("Please ensure the file exists in the Data/ folder")
        exit()

# Display original dataset info
print(f"\n{'='*60}")
print("ORIGINAL DATASET")
print(f"{'='*60}")
print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# Find the label column
label_column = None
possible_labels = ['label', 'class', 'subtask_a', 'category', 'offensive']

for col in df.columns:
    if col.lower() in possible_labels:
        label_column = col
        break

# If not found, use the column that likely contains labels
if label_column is None:
    print(f"\nAvailable columns: {df.columns.tolist()}")
    label_column = input("Enter the name of the label column: ")

print(f"\nUsing label column: {label_column}")

# Get category counts
category_counts = df[label_column].value_counts()
print(f"\nOriginal distribution:")
for category, count in category_counts.items():
    pct = (count / len(df) * 100)
    print(f"  {category}: {count} tweets ({pct:.2f}%)")

# Find minority and majority classes
minority_class = category_counts.idxmin()
majority_class = category_counts.idxmax()
minority_count = category_counts.min()

print(f"\nMinority class: {minority_class} ({minority_count} samples)")
print(f"Majority class: {majority_class} ({category_counts.max()} samples)")

# Separate majority and minority classes
df_minority = df[df[label_column] == minority_class]
df_majority = df[df[label_column] == majority_class]

# Undersample majority class to match minority class
df_majority_downsampled = df_majority.sample(n=minority_count, random_state=42)

# Combine the balanced datasets
df_balanced = pd.concat([df_minority, df_majority_downsampled])

# Shuffle the balanced dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Display balanced dataset info
print(f"\n{'='*60}")
print("BALANCED DATASET")
print(f"{'='*60}")
print(f"Dataset shape: {df_balanced.shape}")

balanced_counts = df_balanced[label_column].value_counts()
print(f"\nBalanced distribution:")
for category, count in balanced_counts.items():
    pct = (count / len(df_balanced) * 100)
    print(f"  {category}: {count} tweets ({pct:.2f}%)")

# Calculate reduction
original_total = len(df)
balanced_total = len(df_balanced)
reduction_pct = ((original_total - balanced_total) / original_total * 100)

print(f"\nTotal samples reduced from {original_total} to {balanced_total}")
print(f"Reduction: {original_total - balanced_total} samples ({reduction_pct:.2f}%)")

# Save the balanced dataset
output_filename = f'Data/train_balanced.{file_extension}'

if file_extension == 'tsv':
    df_balanced.to_csv(output_filename, sep='\t', index=False)
else:
    df_balanced.to_csv(output_filename, index=False)

print(f"\n{'='*60}")
print(f"✓ Balanced dataset saved to: {output_filename}")
print(f"{'='*60}")

# Verify the saved file
print("\nVerifying saved file...")
try:
    if file_extension == 'tsv':
        df_verify = pd.read_csv(output_filename, sep='\t')
    else:
        df_verify = pd.read_csv(output_filename)
    
    verify_counts = df_verify[label_column].value_counts()
    print("Saved file distribution:")
    for category, count in verify_counts.items():
        print(f"  {category}: {count} tweets")
    print("\n✓ File saved and verified successfully!")
except Exception as e:
    print(f"Warning: Could not verify saved file: {e}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Original file: Data/train.{file_extension}")
print(f"Balanced file: {output_filename}")
print(f"Original samples: {original_total}")
print(f"Balanced samples: {balanced_total}")
print(f"Samples per class: {minority_count}")
print(f"{'='*60}")