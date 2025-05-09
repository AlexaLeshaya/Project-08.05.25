# train_iso.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib, pathlib

DATA   = pathlib.Path("data/btcusd_trim.csv")
MODEL_DIR = pathlib.Path("models"); MODEL_DIR.mkdir(exist_ok=True)

# 1) читаем данные
df = pd.read_csv(DATA, usecols=["Close", "Volume"])

# 2) заполняем пропущенные значения
feat = df.ffill()                       # вместо .fillna(method="ffill")

# 3) обучаем Isolation Forest
iso = IsolationForest(
        contamination=0.002,     # 0.2 % точек считаем аномалиями
        random_state=42
      ).fit(feat)

# 4) сохраняем модель
joblib.dump(iso, MODEL_DIR / "iso_forest.pkl")
print("✓ iso_forest.pkl сохранён")
