"""Stage 4 — Sample 200 rows for human validation."""

import pandas as pd
import ast


def label_from_score(score):
    if score > 0.1:
        return 'positive'
    if score < -0.1:
        return 'negative'
    return 'neutral'


def main():
    news = pd.read_csv("news_scored.csv")
    print(f"Loaded {len(news)} rows from news_scored.csv")

    # Only sample from articles that were actually scored
    scored = news[news['sentiment_score'].notna()].copy()
    print(f"  {len(scored)} articles have sentiment scores")

    sample = scored.sample(n=min(200, len(scored)), random_state=42).copy()
    sample['model_label'] = sample['sentiment_score'].apply(label_from_score)
    sample['human_label'] = ''

    out = sample[['headline', 'sentiment_score', 'model_label', 'human_label']].reset_index(drop=True)
    out.to_csv("validation_sample.csv", index=False)
    print(f"Saved validation_sample.csv with {len(out)} rows")
    print("\nModel label distribution:")
    print(out['model_label'].value_counts())
    print("\nFirst 5 rows:")
    print(out[['headline', 'model_label']].head(5).to_string())


if __name__ == '__main__':
    main()
