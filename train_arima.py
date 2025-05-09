# train_arima.py
import pandas as pd, joblib, pathlib
from pmdarima import auto_arima

series = pd.read_csv("data/btcusd_trim.csv")["Close"]
arima  = auto_arima(series, max_p=3, max_q=3, seasonal=False)
path = pathlib.Path("models/arima.pkl"); path.parent.mkdir(exist_ok=True)
joblib.dump(arima, path); print("✓", path, "size", path.stat().st_size/1024, "KB")
