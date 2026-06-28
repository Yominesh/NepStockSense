"""Stage 5 — Aggregate to final (Symbol, Date, avg_sentiment, news_count, sentiment_momentum)."""

import pandas as pd
import ast


def main():
    print("Loading news_scored.csv ...")
    news = pd.read_csv("news_scored.csv")
    print(f"  {len(news)} rows")

    # Parse stocks column
    news['stocks'] = news['stocks'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
    )

    news['date'] = pd.to_datetime(news['date'], errors='coerce')
    before = len(news)
    news = news.dropna(subset=['date'])
    if len(news) < before:
        print(f"  Dropped {before - len(news)} rows with unparseable dates")

    # Explode one row per (article, stock)
    exploded = news.explode('stocks').dropna(subset=['stocks'])
    exploded = exploded[exploded['stocks'].str.strip() != '']
    print(f"  {len(exploded)} stock-article pairs after exploding")

    agg = (
        exploded
        .groupby(['stocks', 'date'])
        .agg(
            avg_sentiment=('sentiment_score', 'mean'),
            news_count=('sentiment_score', 'size'),
        )
        .reset_index()
        .rename(columns={'stocks': 'Symbol', 'date': 'Date'})
    )

    agg = agg.sort_values(['Symbol', 'Date'])
    agg['sentiment_momentum'] = agg.groupby('Symbol')['avg_sentiment'].diff()
    agg['Date'] = agg['Date'].dt.strftime('%Y-%m-%d')

    agg.to_csv("sentiment_features.csv", index=False)

    print(f"\nsentiment_features.csv written — {len(agg)} rows")
    print(f"Columns: {list(agg.columns)}")
    print(f"\nSymbol breakdown:")
    print(agg.groupby('Symbol').agg(rows=('Date', 'count'),
                                     date_min=('Date', 'min'),
                                     date_max=('Date', 'max')).to_string())
    print(f"\nFirst 10 rows:")
    print(agg.head(10).to_string())


if __name__ == '__main__':
    main()
