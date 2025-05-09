# train_lstm.py
import pandas as pd, numpy as np, joblib, pathlib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

DATA = pathlib.Path("data/btcusd_trim.csv")
MODEL_DIR = pathlib.Path("models"); MODEL_DIR.mkdir(exist_ok=True)

# --- данные (Close 15‑мин) ---
series = pd.read_csv(DATA)["Close"].values.reshape(-1,1).astype("float32")
scaler = MinMaxScaler().fit(series)
scaled = scaler.transform(series)

win = 60
X = np.array([scaled[i-win:i] for i in range(win, len(scaled))])
y = scaled[win:]

# --- модель ---
model = Sequential([
    Input(shape=(win,1)),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1)
])
model.compile("adam", "mse")
model.fit(X, y, epochs=5, batch_size=256, verbose=2)

# --- сохраняем ---
model.save(MODEL_DIR/"lstm.h5", include_optimizer=False)
joblib.dump(scaler, MODEL_DIR/"lstm_scaler.pkl")
print("✓ LSTM‑файлы сохранены")
