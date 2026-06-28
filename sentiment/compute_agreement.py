"""Compute human vs model agreement after user fills in validation_sample.csv."""

import pandas as pd


def normalize(label):
    lab = str(label).strip().lower()
    if lab in ('pos', 'positive', '1', 'p'):
        return 'positive'
    if lab in ('neg', 'negative', '-1', 'n'):
        return 'negative'
    if lab in ('neu', 'neutral', '0', 'u'):
        return 'neutral'
    return lab


def main():
    df = pd.read_csv("validation_sample.csv")

    # Drop rows where human_label is still blank
    df = df[df['human_label'].notna() & (df['human_label'].astype(str).str.strip() != '')]
    print(f"Labelled rows: {len(df)}")

    df['human_norm'] = df['human_label'].apply(normalize)
    df['model_norm'] = df['model_label'].apply(normalize)

    agree = (df['human_norm'] == df['model_norm']).sum()
    total = len(df)
    pct = agree / total * 100

    print(f"\nAgreement: {agree} / {total} = {pct:.1f}%")

    print("\nConfusion matrix (rows=human, cols=model):")
    ct = pd.crosstab(df['human_norm'], df['model_norm'])
    print(ct)

    # Disagreements
    disagree = df[df['human_norm'] != df['model_norm']]
    print(f"\nDisagreements ({len(disagree)} rows):")
    print(disagree[['headline', 'human_norm', 'model_norm']].head(20).to_string())

    # Summary notes
    most_confused = disagree.groupby(['human_norm', 'model_norm']).size().sort_values(ascending=False)
    print(f"\nTop disagreement patterns:")
    print(most_confused.head(6))


if __name__ == '__main__':
    main()
