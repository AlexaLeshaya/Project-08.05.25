# app.py  ─────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, pathlib, joblib, plotly.express as px
from prophet.serialize import model_from_json

DATA   = pathlib.Path(__file__).parent/"data/btcusd_trim.csv"
MODELS = pathlib.Path(__file__).parent/"models"
BT_CSV = pathlib.Path(__file__).parent/"backtest/equity_curve.csv"

# ---------- utils ------------------------------------------------
@st.cache_data(show_spinner="Загружаю данные…")
def load_data():
    cols  = ["Timestamp","Open","High","Low","Close","Volume"]
    dtype = dict(Open="float32",High="float32",Low="float32",
                 Close="float32",Volume="float32")
    df = pd.read_csv(DATA, usecols=cols, dtype=dtype)
    df["datetime"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    return df

@st.cache_resource
def _load(name: str):
    p = MODELS / name
    if not p.exists():
        return None, f"Файл {name} не найден"
    try:
        if p.suffix == ".json":               # Prophet
            import json
            from prophet.serialize import model_from_json
            with p.open() as fin:
                return model_from_json(json.load(fin)), None
        elif p.suffix == ".pkl":                     # joblib
            import joblib
            return joblib.load(p), None
        elif p.suffix in (".h5", ".keras"):          # LSTM
            from tensorflow.keras.models import load_model
            return load_model(p, compile=False), None
    except Exception as e:
        return None, str(e)
    return None, "Неизвестный тип файла"

# ---------- pages ------------------------------------------------
def page_eda(df):
    st.subheader("📊 EDA")
    st.write(df.head())
    st.plotly_chart(px.line(df, x="datetime", y="Close",
                            title="BTC ‑ Close price"), use_container_width=True)

def page_forecast(df: pd.DataFrame) -> None:
    st.subheader("🔮 Прогноз")

    model_choice = st.selectbox(
        "Модель",
        ["Prophet", "ARIMA", "LSTM"],
        index=0,
        key="forecast_model"
    )

    # ---------- PROPHET ----------
    if model_choice == "Prophet":
        model, err = _load("prophet_model.json")
        if err:
            st.error(err); return

        future = model.make_future_dataframe(periods=96, freq="15min")
        fc = (model.predict(future)
                    .set_index("ds")["yhat"]
                    .tail(96)
                    .rename("Prophet"))
        st.line_chart(fc, height=300)

    # ---------- ARIMA ----------
    elif model_choice == "ARIMA":
        model, err = _load("arima.pkl")
        if err:
            st.error(err); return

        steps = 96
        y_pred = model.predict(steps)
        if pd.isna(y_pred).all():
            st.error("ARIMA вернула только NaN — пересохраните модель"); return

        idx = pd.date_range(
            df["datetime"].max() + pd.Timedelta(minutes=15),
            periods=steps, freq="15min"
        )
        series = pd.Series(y_pred, index=idx, name="ARIMA")
        st.line_chart(series, height=300)
        st.write(series.tail())

    # ---------- LSTM ----------
    else:  # LSTM
        model, err1 = _load("lstm.h5")            # сперва .h5
        if model is None:
            model, err1 = _load("lstm.keras")     # затем .keras
        scaler, err2 = _load("lstm_scaler.pkl")

        if err1 or err2:
            st.error("Нет LSTM‑файлов или ошибка загрузки"); return

        # один шаг вперёд (15 мин)
        window = scaler.transform(df[["Close"]].values)[-60:].reshape(1, 60, 1)
        next_val = scaler.inverse_transform(model.predict(window))[0, 0]
        st.metric("Прогноз (через 15 мин)", f"{next_val:,.2f} $")

def page_anomaly(df: pd.DataFrame):
    st.subheader("🚨 Аномалии")

    iso, err = _load("iso_forest.pkl")      # ← раскрываем кортеж
    if err:
        st.error(err)
        return

    feats = df[["Close", "Volume"]].ffill()
    df["anom"] = iso.predict(feats)         # теперь iso — это модель

    plot_df = df.tail(4000)
    fig = px.scatter(
        plot_df, x="datetime", y="Close",
        color=plot_df["anom"].map({1: "норма", -1: "аномалия"}),
        color_discrete_map={"норма": "blue", "аномалия": "red"},
        title="Выявленные аномалии Isolation Forest"
    )
    st.plotly_chart(fig, use_container_width=True)

def page_backtest(df=None):        # <— исправление
    st.subheader("💰 Бэктест стратегии")
    if not BT_CSV.exists():
        st.error("Нет equity_curve.csv"); return
    eq = pd.read_csv(BT_CSV, parse_dates=["date"], index_col="date")
    st.line_chart(eq["equity"], height=300)
    st.write(eq.tail())

# ---------- main -------------------------------------------------
def main():
    st.set_page_config(page_title="BTC Forecast", layout="wide")
    st.title("⚡️ Прогнозирование рынка Bitcoin")

    choice = st.sidebar.radio("Навигация",
                              ["EDA", "Прогноз", "Аномалии", "Бэктест"])

    df = load_data()

    pages = dict(EDA=page_eda,
                 Прогноз=page_forecast,
                 Аномалии=page_anomaly,
                 Бэктест=page_backtest)

    pages[choice](df)

if __name__ == "__main__":
    main()