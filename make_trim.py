import pandas as pd, pathlib

SRC = pathlib.Path("btcusd_1-min_data.csv")
DEST = pathlib.Path("data/btcusd_trim.csv")

cols  = ["Timestamp","Open","High","Low","Close","Volume"]
dtype = dict(Open="float32", High="float32",
             Low="float32", Close="float32", Volume="float32")

df = pd.read_csv(SRC, usecols=cols, dtype=dtype)
df["datetime"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)

start = df["datetime"].max() - pd.Timedelta(days=120)
df120 = df[df["datetime"] >= start]           # фильтр вместо .last

# сохраняем датой как обычную колонку
DEST.parent.mkdir(exist_ok=True)
df120.to_csv(DEST, index=False, float_format="%.5f")
print("✓ файл пересохранён:", DEST)
