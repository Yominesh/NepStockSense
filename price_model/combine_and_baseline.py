"""
Combine, Clean, Build Target, Baseline
Loads all stock CSVs from ./data/, cleans them, filters to 2020+, builds the
up/down target, splits by date, trains an XGBoost baseline, saves outputs.

RUN:
  1. Put all stock CSVs in the data/ folder at the project root (../data)
  2. pip install pandas numpy scikit-learn xgboost matplotlib
  3. python combine_and_baseline.py
"""

import pandas as pd
import glob
import os
import matplotlib
os.makedirs("../results/price_model", exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CONFIG 
DATA_FOLDER = "../data"
START_DATE = "2020-01-01"
TRAIN_TEST_CUTOFF = "2024-06-01"
FEATURE_COLS = ['SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI_14', 'MACD',
                'MACD_Signal', 'ATR_14', 'BB_Upper', 'BB_Lower', 'OBV',
                'Daily_Return', 'Volume']

# STEP 1: Load + combine every CSV
csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
if not csv_files:
    raise SystemExit(f"No CSVs found in {DATA_FOLDER}. Put your stock CSVs there.")
print(f"Found {len(csv_files)} files: {[os.path.basename(f) for f in csv_files]}")
data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"Combined rows: {len(data)} | Stocks: {sorted(data['Symbol'].unique())}")

# STEP 2: Clean
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(['Symbol', 'Date']).reset_index(drop=True)
if 'Percent Change' in data.columns:
    data = data.drop(columns=['Percent Change'])

# STEP 3: Keep 2020 onward
data = data[data['Date'] >= START_DATE].reset_index(drop=True)
print(f"After 2020 filter: {len(data)} rows")

row_num = data.groupby('Symbol').cumcount()
data = data[row_num >= 20].reset_index(drop=True)
data = data.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"After cleaning: {len(data)} rows")

# STEP 4: Build TARGET (up=1 / down=0 next trading day)
data['next_close'] = data.groupby('Symbol')['Close'].shift(-1)
data['target'] = (data['next_close'] > data['Close']).astype(int)
data = data.dropna(subset=['next_close']).reset_index(drop=True)
print("Target balance:", data['target'].value_counts(normalize=True).round(3).to_dict())

#  STEP 5: Time-based split (NEVER shuffle) 
train = data[data['Date'] < TRAIN_TEST_CUTOFF]
test = data[data['Date'] >= TRAIN_TEST_CUTOFF]
print(f"Train rows: {len(train)} | Test rows: {len(test)}")
X_train, y_train = train[FEATURE_COLS], train['target']
X_test, y_test = test[FEATURE_COLS], test['target']

# STEP 6: Train baseline XGBoost 
model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                      random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("\n" + "="*55)
print("BASELINE RESULTS — price + technical indicators only")
print("="*55)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds))

#  STEP 7: Save outputs 
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
plt.figure(figsize=(8, 5)); importances.plot(kind='barh')
plt.title("Feature Importance (XGBoost baseline)"); plt.tight_layout()
plt.savefig("../results/price_model/feature_importance.png", dpi=120)
print("Saved results/price_model/feature_importance.png")

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xticks([0, 1], ['Down', 'Up']); plt.yticks([0, 1], ['Down', 'Up'])
plt.tight_layout(); plt.savefig("../results/price_model/confusion_matrix.png", dpi=120)
print("Saved results/price_model/confusion_matrix.png")

data.to_csv("master_price_table.csv", index=False)
print("Saved master_price_table.csv")
print("\nDONE.")
