"""
Phase 3 Fusion Model
Merges price/technical data with sentiment features (left join),
retrains XGBoost and LSTM, compares against Phase 1 baselines.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

os.makedirs("../results/fusion_model", exist_ok=True)

TRAIN_TEST_CUTOFF = "2024-06-01"

PRICE_FEATURES = ['SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI_14', 'MACD',
                  'MACD_Signal', 'ATR_14', 'BB_Upper', 'BB_Lower', 'OBV',
                  'Daily_Return', 'Volume']
SENTIMENT_FEATURES = ['avg_sentiment', 'news_count', 'sentiment_momentum', 'has_news']
ALL_FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES

WINDOW_SIZE = 20

# Load & merge
print("Loading data...")
price = pd.read_csv("../price_model/master_price_table.csv")
sentiment = pd.read_csv("../sentiment/sentiment_features.csv")

price['Date'] = pd.to_datetime(price['Date'])
sentiment['Date'] = pd.to_datetime(sentiment['Date'])

merged = price.merge(sentiment, on=['Symbol', 'Date'], how='left')
merged = merged.sort_values(['Symbol', 'Date']).reset_index(drop=True)
merged['has_news'] = (merged['news_count'] > 0).astype(int)
sentiment_cols = ['avg_sentiment', 'news_count', 'sentiment_momentum']
merged[sentiment_cols] = (
    merged.groupby('Symbol')[sentiment_cols]
    .ffill(limit=3)
    .fillna(0)
)

print(f"Merged rows: {len(merged)}")
print(f"Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")
print(f"Days with sentiment data: {(merged['avg_sentiment'] != 0).sum()} / {len(merged)}")
print(f"Target balance: {merged['target'].value_counts(normalize=True).round(3).to_dict()}\n")

# Time-based split 
train = merged[merged['Date'] < TRAIN_TEST_CUTOFF]
test  = merged[merged['Date'] >= TRAIN_TEST_CUTOFF]
print(f"Train rows: {len(train)} | Test rows: {len(test)}")

X_train, y_train = train[ALL_FEATURES], train['target']
X_test,  y_test  = test[ALL_FEATURES],  test['target']

#  XGBoost fusion
print("\n--- XGBoost Fusion ---")
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                    random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1  = f1_score(y_test, xgb_preds)
print(f"Accuracy: {xgb_acc:.4f}")
print(f"F1:       {xgb_f1:.4f}")
print(classification_report(y_test, xgb_preds))

# Feature importance
importances = pd.Series(xgb.feature_importances_, index=ALL_FEATURES).sort_values()
plt.figure(figsize=(9, 6))
colors = ['#e74c3c' if f in SENTIMENT_FEATURES else '#3498db' for f in importances.index]
importances.plot(kind='barh', color=colors)
plt.title("Feature Importance — XGBoost Fusion (red=sentiment, blue=technical)")
plt.tight_layout()
plt.savefig("../results/fusion_model/feature_importance_fusion.png", dpi=120)
print("Saved results/fusion_model/feature_importance_fusion.png")

# Confusion matrix
cm = confusion_matrix(y_test, xgb_preds)
plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix — XGBoost Fusion")
plt.xlabel("Predicted"); plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xticks([0,1], ['Down','Up']); plt.yticks([0,1], ['Down','Up'])
plt.tight_layout(); plt.savefig("../results/fusion_model/confusion_matrix_xgb_fusion.png", dpi=120)
print("Saved results/fusion_model/confusion_matrix_xgb_fusion.png")

# LSTM fusion (dual-input)
# Price features go into LSTM sequences 
# Sentiment features are passed as a direct Dense branch (no dilution across
# 20 zero-filled timesteps). Both branches merge before the output layer.
print("\n--- LSTM Fusion (dual-input) ---")
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    # Scale price and sentiment separately — fit on train only
    price_scaler = MinMaxScaler()
    sent_scaler  = MinMaxScaler()

    train_p = train.copy(); test_p = test.copy()
    train_p[PRICE_FEATURES]     = price_scaler.fit_transform(train[PRICE_FEATURES])
    train_p[SENTIMENT_FEATURES] = sent_scaler.fit_transform(train[SENTIMENT_FEATURES])
    test_p[PRICE_FEATURES]      = price_scaler.transform(test[PRICE_FEATURES])
    test_p[SENTIMENT_FEATURES]  = sent_scaler.transform(test[SENTIMENT_FEATURES])

    # Build windows: price sequence (20 days) + sentiment snapshot (prediction day only)
    def make_dual_windows(df, window=WINDOW_SIZE):
        Xp, Xs, y = [], [], []
        for sym in df['Symbol'].unique():
            s = df[df['Symbol'] == sym].reset_index(drop=True)
            for i in range(window, len(s)):
                Xp.append(s[PRICE_FEATURES].iloc[i-window:i].values)
                Xs.append(s[SENTIMENT_FEATURES].iloc[i].values)
                y.append(s['target'].iloc[i])
        return np.array(Xp), np.array(Xs), np.array(y)

    print("Building dual windows...")
    Xp_tr, Xs_tr, y_tr = make_dual_windows(train_p)
    Xp_te, Xs_te, y_te = make_dual_windows(test_p)
    print(f"Train windows: {len(Xp_tr)} | Test windows: {len(Xp_te)}")

    # Hold out last 10% of train chronologically for threshold tuning
    cut = int(len(Xp_tr) * 0.9)
    Xp_fit, Xs_fit, y_fit = Xp_tr[:cut], Xs_tr[:cut], y_tr[:cut]
    Xp_val, Xs_val, y_val = Xp_tr[cut:], Xs_tr[cut:], y_tr[cut:]

    # Dual-input model: LSTM for price sequences + Dense for sentiment
    price_in = Input(shape=(WINDOW_SIZE, len(PRICE_FEATURES)), name='price')
    x = LSTM(64)(price_in)
    x = Dropout(0.2)(x)

    sent_in = Input(shape=(len(SENTIMENT_FEATURES),), name='sentiment')
    s = Dense(8, activation='relu')(sent_in)

    combined = Concatenate()([x, s])
    out      = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[price_in, sent_in], outputs=out)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(patience=7, restore_best_weights=True)
    class_weight = {0: 1.0, 1: 1.2}
    model.fit([Xp_fit, Xs_fit], y_fit,
              epochs=30, batch_size=32,
              validation_data=([Xp_val, Xs_val], y_val),
              class_weight=class_weight,
              callbacks=[es], verbose=1)

    # Find optimal threshold on validation set
    val_probs = model.predict([Xp_val, Xs_val]).flatten()
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.65, 0.01):
        f = f1_score(y_val, (val_probs > t).astype(int), average='macro', zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, round(t, 2)
    print(f"Optimal threshold (val F1={best_f1:.4f}): {best_thresh}")

    lstm_probs = model.predict([Xp_te, Xs_te]).flatten()
    lstm_preds = (lstm_probs > best_thresh).astype(int)
    lstm_acc = accuracy_score(y_te, lstm_preds)
    lstm_f1  = f1_score(y_te, lstm_preds)
    print(f"\nLSTM Fusion Accuracy: {lstm_acc:.4f}")
    print(f"LSTM Fusion F1:       {lstm_f1:.4f}")
    print(classification_report(y_te, lstm_preds))

    model.save("../results/fusion_model/lstm_fusion_model.keras")
    print("Saved results/fusion_model/lstm_fusion_model.keras")

    cm_lstm = confusion_matrix(y_te, lstm_preds)
    plt.figure(figsize=(4,4)); plt.imshow(cm_lstm, cmap='Blues')
    plt.title("Confusion Matrix — LSTM Fusion")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_lstm[i,j], ha='center', va='center')
    plt.xticks([0,1],['Down','Up']); plt.yticks([0,1],['Down','Up'])
    plt.tight_layout(); plt.savefig("../results/fusion_model/confusion_matrix_lstm_fusion.png", dpi=120)
    print("Saved results/fusion_model/confusion_matrix_lstm_fusion.png")

    lstm_done = True

except Exception as e:
    print(f"LSTM failed: {e}")
    lstm_acc, lstm_f1 = None, None
    lstm_done = False

# Final comparison
print("\n" + "="*60)
print("FINAL COMPARISON — Baseline vs Fusion")
print("="*60)

comparison = pd.DataFrame([
    {'Model': 'XGBoost — Technical only (baseline)', 'Accuracy': 0.5530, 'F1': 0.2158},
    {'Model': 'LSTM    — Technical only (baseline)', 'Accuracy': 0.5485, 'F1': 0.2089},
    {'Model': 'XGBoost — Fusion (technical + sentiment)', 'Accuracy': round(xgb_acc, 4), 'F1': round(xgb_f1, 4)},
])
if lstm_done:
    comparison = pd.concat([comparison, pd.DataFrame([{
        'Model': 'LSTM    — Fusion (technical + sentiment)',
        'Accuracy': round(lstm_acc, 4), 'F1': round(lstm_f1, 4)
    }])], ignore_index=True)

print(comparison.to_string(index=False))
comparison.to_csv("../results/fusion_model/fusion_comparison.csv", index=False)
print("\nSaved results/fusion_model/fusion_comparison.csv")

xgb_improvement = (xgb_acc - 0.5530) * 100
print(f"\nXGBoost improvement from sentiment: {xgb_improvement:+.2f}%")
if lstm_done:
    lstm_improvement = (lstm_acc - 0.5485) * 100
    print(f"LSTM improvement from sentiment:    {lstm_improvement:+.2f}%")

merged.to_csv("../results/fusion_model/master_fusion_table.csv", index=False)
print("\nSaved results/fusion_model/master_fusion_table.csv (full merged dataset)")
print("\nDONE.")
