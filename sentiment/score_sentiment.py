"""Stage 3 — Score sentiment using a pre-trained Nepali model from Hugging Face."""

import pandas as pd
import ast

# Tried in order; first that loads cleanly wins.
# Notes on each:
#   sibendra — uses bert-multilingual but missing model_type in config → pipeline rejects it
#   NepBERTa  — MLM-only, no classification head
#   Shushant  — classifier weights randomly initialized (not trained for sentiment)
#   luluw     — DeBERTa-Nepali, no id2label and network instability downloading tokenizer
#   ProsusAI/finbert — financial-domain English BERT; appropriate since ShareSansar headlines
#                      are in English; loaded with bert-base-uncased tokenizer as fallback
MODEL_ORDER = [
    "sibendra/nepali-sentiment-analysis",
    "NepBERTa/NepBERTa",
    "Shushant/nepaliBERT",
    "ProsusAI/finbert",
]

BATCH_SIZE = 64


def to_score(r):
    lab = r['label'].lower()
    if 'pos' in lab:
        return r['score']
    if 'neg' in lab:
        return -r['score']
    return 0.0


def load_pipeline():
    from transformers import (pipeline as hf_pipeline,
                              AutoTokenizer,
                              AutoModelForSequenceClassification)
    for model_name in MODEL_ORDER:
        try:
            print(f"  Trying model: {model_name}")
            if model_name == "ProsusAI/finbert":
                # finbert weights may be cached but tokenizer may not download cleanly;
                # bert-base-uncased shares the same vocabulary so use it as fallback.
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                clf = hf_pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            else:
                clf = hf_pipeline("sentiment-analysis", model=model_name)
            result = clf(["positive news about profit"])
            label = result[0]['label'].lower()
            # Reject models that produce generic LABEL_N outputs (untrained classifiers)
            if label.startswith('label_'):
                raise ValueError(f"Model outputs generic label '{label}' — classifier not trained")
            # Require the label to express sentiment polarity
            if not any(word in label for word in ('pos', 'neg', 'neu', 'positive', 'negative', 'neutral')):
                raise ValueError(f"Unrecognized label format: '{label}'")
            print(f"  Loaded: {model_name}  (sample label: {label})")
            return clf, model_name
        except Exception as exc:
            print(f"  Failed ({exc.__class__.__name__}: {str(exc)[:120]})")
    raise RuntimeError("None of the candidate models loaded.")


def main():
    print("Loading news_mapped.csv ...")
    news = pd.read_csv("news_mapped.csv")
    print(f"  {len(news)} rows")

    clf, model_used = load_pipeline()
    print(f"\nScoring sentiment with: {model_used}")

    # Only score articles that matched at least one stock
    news['stocks'] = news['stocks'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
    )
    matched = news[news['stocks'].apply(len) > 0].copy()
    unmatched = news[news['stocks'].apply(len) == 0].copy()
    print(f"  Scoring {len(matched)} matched articles (skipping {len(unmatched)} unmatched)")

    headlines = matched['headline'].astype(str).tolist()
    all_results = []
    for i in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[i: i + BATCH_SIZE]
        results = clf(batch, truncation=True, max_length=512)
        all_results.extend(results)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Scored {min(i + BATCH_SIZE, len(headlines))} / {len(headlines)}")

    matched['sentiment_score'] = [to_score(r) for r in all_results]
    matched['model_label'] = [r['label'] for r in all_results]

    # Combine back unmatched get NaN scores (they won't be used in aggregation)
    combined = pd.concat([matched, unmatched], ignore_index=True)

    print(f"\nSentiment score stats (matched articles only):")
    print(matched['sentiment_score'].describe())

    combined.to_csv("news_scored.csv", index=False)
    print(f"\nSaved news_scored.csv  (model used: {model_used})")
    print(f"Score range: {matched['sentiment_score'].min():.3f} to {matched['sentiment_score'].max():.3f}")


if __name__ == '__main__':
    main()
