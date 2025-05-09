import backtrader as bt, pandas as pd

class MaCross(bt.Strategy):
    params = dict(pfast=50, pslow=200)
    def __init__(self):
        sma_fast = bt.ind.SMA(period=self.p.pfast)
        sma_slow = bt.ind.SMA(period=self.p.pslow)
        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)
    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()

# --- данные ------------------------------------------------------
df = pd.read_csv("data/btcusd_trim.csv", parse_dates=["datetime"])
data = bt.feeds.PandasData(dataname=df.set_index("datetime"))

# --- движок ------------------------------------------------------
cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(MaCross)
cerebro.broker.set_cash(10_000)

# ➊  аналайзер дневной (или минутный) доходности
cerebro.addanalyzer(bt.analyzers.TimeReturn,
                    timeframe=bt.TimeFrame.Minutes,
                    _name="ret")

results = cerebro.run()
strat = results[0]

# ➋  извлекаем кумулятивную equity
rets   = strat.analyzers.ret.get_analysis()          # dict {datetime:return}
dates  = list(rets.keys())
equity = 10_000 * pd.Series(rets).add(1).cumprod().values

pd.DataFrame({"date": dates, "equity": equity}) \
  .to_csv("backtest/equity_curve.csv", index=False)

print("✓ equity_curve.csv сохранён")
