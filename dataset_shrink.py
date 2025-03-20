import pandas as pd
import os

# File paths
input_file = '/home/georgeattaloglou/Downloads/CCC.csv'
output_file = '/home/georgeattaloglou/Downloads/CCC_balanced.csv'

# Parameters
samples_per_category = 600  # Set this to your desired number per category

# Step 1: Load the dataset
data = pd.read_csv(input_file)

# Step 2: Inspect the data structure
# Assuming there is a column named 'Category' that contains the labels
if 'tox_codes_oc' not in data.columns:
    raise ValueError("The dataset does not contain a 'Category' column. Please check the column names.")

# Step 3: Group by category and sample
balanced_data = pd.DataFrame()

for category, group in data.groupby('tox_codes_oc'):
    sampled_group = group.sample(n=min(samples_per_category, len(group)), random_state=42)
    balanced_data = pd.concat([balanced_data, sampled_group], ignore_index=True)

# Step 4: Save the balanced dataset
balanced_data.to_csv(output_file, index=False)

print(f"Balanced dataset saved to {output_file}")
