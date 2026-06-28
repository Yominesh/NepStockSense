# NepStockSense

Data Analysis project predicting next-day price direction (Up/Down) for 7 NEPSE stocks by fusing technical price indicators with news sentiment.

**Stocks covered:** BBC, CIT, HBL, NABIL, NRIC, NTC, SCB  
**Date range:** 2020-01-19 to 2025-09-18  
**Train/test split:** Train < 2024-06-01 | Test ≥ 2024-06-01

---

## Project Structure

```
DA_Project/
├── data/                          # Raw stock CSVs (one per symbol)
├── price_model/                   # Phase 1: Scripts + processed data
│   ├── combine_and_baseline.py
│   ├── price_lstm.py
│   └── master_price_table.csv     # Intermediate data (input to fusion)
├── sentiment/                     # Phase 2: Scripts + processed data
│   ├── scrape_news.py
│   ├── map_stocks.py
│   ├── score_sentiment.py
│   ├── aggregate.py
│   ├── prepare_validation.py
│   ├── compute_agreement.py
│   ├── sentiment_features.csv     # Intermediate data (input to fusion)
│   └── validation_sample.csv
├── fusion_model/                  # Phase 3: Fusion model script
│   └── fusion_model.py
├── results/                       # All model outputs
│   ├── price_model/
│   │   ├── feature_importance.png
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_lstm.png
│   │   ├── hyperparameter_results.csv
│   │   ├── model_comparison.csv
│   │   ├── lstm_results.json
│   │   └── lstm_model.keras
│   └── fusion_model/
│       ├── feature_importance_fusion.png
│       ├── confusion_matrix_xgb_fusion.png
│       ├── confusion_matrix_lstm_fusion.png
│       ├── fusion_comparison.csv
│       ├── lstm_fusion_model.keras
│       └── master_fusion_table.csv    # Full merged price + sentiment dataset
└── README.md
```

---

## Phase 1 — Price Baseline

**Scripts:** `price_model/combine_and_baseline.py` → `price_model/price_lstm.py`

Loads 7 stock CSVs from `data/`, cleans and filters to 2020+, builds the binary up/down target, trains an XGBoost baseline, then an LSTM on 20-day rolling windows.

**Run:**
```bash
cd price_model
python combine_and_baseline.py   # produces master_price_table.csv + results/price_model/
python price_lstm.py             # produces results/price_model/lstm_model.keras + metrics
```

**Results:**

| Model | Accuracy | F1 |
|---|---|---|
| XGBoost baseline | 55.30% | 0.2158 |
| LSTM (64 units) | 54.85% | 0.2089 |

Key finding from `feature_importance.png`: `OBV`, `RSI_14`, and `Daily_Return` are the most predictive technical indicators.

---

## Phase 2 — Sentiment Pipeline

Scrapes English-language Nepali financial news from ShareSansar, maps articles to the 7 stock symbols, scores sentiment with **ProsusAI/finbert** (financial-domain BERT), and aggregates to a daily `(Symbol, Date)` table.

**Run the pipeline in order:**
```bash
cd sentiment
python scrape_news.py        # → raw_news.csv
python map_stocks.py         # → news_mapped.csv
python score_sentiment.py    # → news_scored.csv
python aggregate.py          # → sentiment_features.csv
```

**Output schema — `sentiment_features.csv`:**

| Column | Description |
|---|---|
| Symbol | Stock ticker (BBC, CIT, HBL, NABIL, NRIC, NTC, SCB) |
| Date | YYYY-MM-DD |
| avg_sentiment | Mean finbert score for the day (positive=high) |
| news_count | Number of articles mapped to this stock that day |
| sentiment_momentum | 3-day rolling average of avg_sentiment |
| has_news | 1 if real news was scraped that day, 0 if imputed |

---

## Phase 3 — Fusion Model

**Script:** `fusion_model/fusion_model.py`

Left-joins `master_price_table.csv` with `sentiment_features.csv` on `(Symbol, Date)`. Missing sentiment days are forward-filled up to 3 days; a `has_news` binary flag distinguishes real scraped data from imputed values.

Uses a **dual-input architecture** for the LSTM fusion:
- **Price branch:** `LSTM(64) → Dropout(0.2)` — learns temporal patterns from 20-day price sequences
- **Sentiment branch:** `Dense(8, relu)` — takes today's 4 sentiment features as a direct point-in-time input (avoids diluting sparse sentiment across 20 timesteps)
- Both branches merge via `Concatenate → Dense(1, sigmoid)`

The decision threshold is tuned on a held-out validation set using macro F1, rather than a fixed 0.5 cutoff.

**Run:**
```bash
cd fusion_model
python fusion_model.py
```

**Final Results:**

| Model | Accuracy | F1 |
|---|---|---|
| XGBoost — Technical only (baseline) | 55.30% | 0.2158 |
| LSTM — Technical only (baseline) | 54.85% | 0.2089 |
| XGBoost — Fusion (technical + sentiment) | **55.80%** | 0.2079 |
| LSTM — Fusion, dual-input (technical + sentiment) | 51.08% | **0.4830** |

**Key takeaways:**
- XGBoost fusion improved accuracy by +0.50% over its baseline by incorporating sentiment features.
- The LSTM fusion trades raw accuracy for significantly more balanced class prediction — F1 more than doubled (0.21 → 0.48), meaning the model can reliably identify both Up and Down signals rather than defaulting to the majority class. This is a more useful property for real trading decisions.
- The dual-input design was necessary because feeding sparse sentiment (only 11% real coverage) as part of the LSTM sequence caused the model to collapse. Separating price sequences from point-in-time sentiment resolved this.

---

## Installation

```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib transformers torch
```

---

## Data Notes

- Technical indicators (SMA, EMA, RSI, MACD, ATR, Bollinger Bands, OBV) are **pre-computed** in the raw CSVs — do not recompute.
- Train/test split is **always by date** (never random) to prevent look-ahead leakage.
- `Percent Change` column in raw CSVs is text with `%` — ignored; use `Daily_Return` instead.
- Large intermediate news CSVs (`raw_news.csv`, `news_mapped.csv`, `news_scored.csv`) are gitignored. Re-run the sentiment pipeline to regenerate them.
