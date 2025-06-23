import pandas as pd

def main():
    df = pd.read_csv('./offenseval-gr-training-v1.tsv', sep='\t')
    df.to_csv('./offenseval-gr-training-v1.csv', index=False)

if __name__ == "__main__":
    main()