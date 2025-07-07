import pandas as pd
import unicodedata

def strip_accents_and_lowercase(s):
    """Removes Greek accents and lowercases text."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

files = [
    ("./Data/offenseval-gr-training-v1.csv", "./Data/offenseval-gr-training-cleaned-v1.csv"),
    ("./Data/offenseval-gr-test-v1.csv", "./Data/offenseval-gr-test-cleaned-v1.csv"),
]

for input_path, output_path in files:
    df = pd.read_csv(input_path)
    df['tweet'] = df['tweet'].apply(strip_accents_and_lowercase)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
