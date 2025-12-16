import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Read the training data
# Adjust the file path based on your data location
# Common formats: CSV, TSV, or JSON
try:
    # Try reading as TSV (common for OffenseEval datasets)
    df = pd.read_csv('Data/train.tsv', sep='\t')
    print("Successfully loaded data as TSV")
except:
    try:
        # Try reading as CSV
        df = pd.read_csv('Data/train_balanced.csv')
        print("Successfully loaded data as CSV")
    except:
        print("Error: Could not find or read the training file.")
        print("Please ensure the file exists in the Data/ folder")
        exit()

# Display basic information about the dataset
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Find the label column (common names: 'label', 'class', 'subtask_a', etc.)
label_column = None
possible_labels = ['label', 'class', 'subtask_a', 'category', 'offensive']

for col in df.columns:
    if col.lower() in possible_labels:
        label_column = col
        break

# If not found, ask user or use the last column
if label_column is None:
    print(f"\nAvailable columns: {df.columns.tolist()}")
    label_column = input("Enter the name of the label column: ")

print(f"\nUsing label column: {label_column}")

# Count tweets in each category
category_counts = df[label_column].value_counts().sort_index()

print(f"\nCategory distribution:")
print(category_counts)
print(f"\nTotal tweets: {len(df)}")

# Calculate percentages
percentages = (category_counts / len(df) * 100).round(2)

# Create bar graph
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars = ax.bar(range(len(category_counts)), 
               category_counts.values, 
               color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'][:len(category_counts)],
               edgecolor='black',
               linewidth=1.5)

# Customize the plot
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Tweets', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Tweets by Category in Training Set', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(category_counts)))
ax.set_xticklabels(category_counts.index, fontsize=11)

# Add value labels on top of bars
for i, (bar, count, pct) in enumerate(zip(bars, category_counts.values, percentages.values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add grid for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()

# Additional statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
for category, count in category_counts.items():
    pct = (count / len(df) * 100)
    print(f"{category}: {count} tweets ({pct:.2f}%)")
print("="*50)