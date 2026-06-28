"""
LSTM for price direction prediction
Loads master_price_table.csv, builds rolling 20-day windows per stock,
trains an LSTM classifier, evaluates and saves outputs.

RUN (after combine_and_baseline.py has produced master_price_table.csv):
  python price_lstm.py
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib
os.makedirs("../results/price_model", exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

TRAIN_TEST_CUTOFF = "2024-06-01"
WINDOW_SIZE = 20
FEATURE_COLS = ['SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI_14', 'MACD',
                'MACD_Signal', 'ATR_14', 'BB_Upper', 'BB_Lower', 'OBV',
                'Daily_Return', 'Volume']

# Load 
print("Loading master_price_table.csv...")
data = pd.read_csv("master_price_table.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(['Symbol', 'Date']).reset_index(drop=True)
print(f"Rows: {len(data)} | Symbols: {sorted(data['Symbol'].unique())}")

# Time-based split 
train = data[data['Date'] < TRAIN_TEST_CUTOFF].copy()
test  = data[data['Date'] >= TRAIN_TEST_CUTOFF].copy()
print(f"Train rows: {len(train)} | Test rows: {len(test)}")

# Scale features (fit on train only)
scaler = MinMaxScaler()
train[FEATURE_COLS] = scaler.fit_transform(train[FEATURE_COLS])
test[FEATURE_COLS]  = scaler.transform(test[FEATURE_COLS])

# Build rolling windows per stock
def make_windows(df, window=WINDOW_SIZE):
    X, y = [], []
    for sym in df['Symbol'].unique():
        s = df[df['Symbol'] == sym].reset_index(drop=True)
        for i in range(window, len(s)):
            X.append(s[FEATURE_COLS].iloc[i-window:i].values)
            y.append(s['target'].iloc[i])
    return np.array(X), np.array(y)

print("Building windows...")
X_train, y_train = make_windows(train)
X_test,  y_test  = make_windows(test)
print(f"Train windows: {len(X_train)} | Test windows: {len(X_test)}")

# Train LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
class_weight = {0: 1.0, 1: neg / pos}

model = Sequential([
    LSTM(64, input_shape=(WINDOW_SIZE, len(FEATURE_COLS))),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=32,
          validation_split=0.1, callbacks=[es],
          class_weight=class_weight, verbose=1)

# Evaluate
probs = model.predict(X_test).flatten()
preds = (probs > 0.5).astype(int)
acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds)

print(f"\nLSTM Accuracy: {acc:.4f}")
print(f"LSTM F1:       {f1:.4f}")
print(classification_report(y_test, preds))

# Save outputs
model.save("../results/price_model/lstm_model.keras")
print("Saved results/price_model/lstm_model.keras")

with open("../results/price_model/lstm_results.json", "w") as f:
    json.dump({
        "model": f"LSTM (64 units, dropout 0.2, class-weighted)",
        "window_size": WINDOW_SIZE,
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "accuracy": round(acc, 4),
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }, f, indent=2)
print("Saved results/price_model/lstm_results.json")

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix — LSTM (price only)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xticks([0, 1], ['Down', 'Up']); plt.yticks([0, 1], ['Down', 'Up'])
plt.tight_layout(); plt.savefig("../results/price_model/confusion_matrix_lstm.png", dpi=120)
print("Saved results/price_model/confusion_matrix_lstm.png")

print("\nDONE.")
