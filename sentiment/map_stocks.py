"""Stage 2 — Map each news item to one or more of the 7 stock symbols."""

import pandas as pd
import ast


STOCK_KEYWORDS = {
    'BBC': [
        'Bishal Bazar',
        'Bishal Bazar Company',
    ],
    'CIT': [
        'Citizen Investment Trust',
        'CIT Nepal',
    ],
    'HBL': [
        'Himalayan Bank',
        'HBL',
    ],
    'NABIL': [
        'Nabil Bank',
        'Nepal Arab Bank',
        'NABIL'
    ],
    'NRIC': [
        'Nepal Reinsurance',
        'NRIC',
    ],
    'NTC': [
        'Nepal Telecom',
        'Nepal Doorsanchar',
        'Doorsanchar',
        'NTC'
    ],
    'SCB': [
        'Standard Chartered',
        'Standard Chartered Bank Nepal',
        'SCBNL',
    ],
}


def find_stocks(headline):
    h = str(headline).lower()
    matched = []
    for sym, kws in STOCK_KEYWORDS.items():
        if any(str(k).lower() in h for k in kws):
            matched.append(sym)
    return matched


def main():
    print("Loading raw_news.csv ...")
    news = pd.read_csv("raw_news.csv")
    print(f"  {len(news)} rows loaded")

    news['stocks'] = news['headline'].apply(find_stocks)

    matched = news['stocks'].apply(len) > 0
    print(f"\n  Articles matching at least one stock: {matched.sum()} / {len(news)} "
          f"({matched.mean()*100:.1f}%)")

    print("\nSample matches (up to 15 per symbol):\n")
    for sym in STOCK_KEYWORDS:
        sym_rows = news[news['stocks'].apply(lambda s: sym in s)]
        print(f"  {sym}: {len(sym_rows)} matches")
        for _, row in sym_rows.head(10).iterrows():
            print(f"    [{row['date']}] {row['headline'][:90]}")
        print()

    news.to_csv("news_mapped.csv", index=False)
    print("Saved news_mapped.csv")


if __name__ == '__main__':
    main()
