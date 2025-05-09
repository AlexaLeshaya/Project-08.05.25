# Prophet → JSON  (стабилен к версиям)
from prophet import Prophet
from prophet.serialize import model_to_json
import json, joblib, pandas as pd

df15 = pd.read_csv("data/btcusd_trim.csv")
df15["ds"] = pd.to_datetime(df15["Timestamp"], unit="s")
prop_df  = df15.rename(columns={"Close":"y"})[["ds","y"]]

m = Prophet(daily_seasonality=True)
m.fit(prop_df)
json.dump(model_to_json(m), open("models/prophet_model.json","w"))