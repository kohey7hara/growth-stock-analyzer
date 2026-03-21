"""
predictor.py - 株価予測シミュレーション

Prophet優先 → ETS → ARIMA → 線形回帰 → 簡易平均リターン。
80%と95%の信頼区間を算出。全銘柄で予測を保証。
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent

PREDICTOR_ENGINE = "linear"
try:
    from prophet import Prophet
    PREDICTOR_ENGINE = "prophet"
except ImportError:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        PREDICTOR_ENGINE = "ets"
    except ImportError:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            PREDICTOR_ENGINE = "arima"
        except ImportError:
            pass


def _resolve_ticker(ticker):
    """日本株ティッカーの.T補完"""
    if ticker.endswith(".T"):
        return ticker
    # 数字のみ→日本株
    stripped = ticker.replace(".", "")
    if stripped.isdigit():
        return f"{ticker}.T"
    return ticker


def _fetch_history(ticker):
    """yfinanceから過去データを取得。複数period/ticker形式でリトライ。"""
    import yfinance as yf
    resolved = _resolve_ticker(ticker)
    candidates = [resolved] if resolved == ticker else [resolved, ticker]

    for tk in candidates:
        for p in ["2y", "1y", "6mo", "3mo"]:
            try:
                hist = yf.Ticker(tk).history(period=p)
                if not hist.empty and len(hist) >= 5:
                    df = hist[["Close", "Volume"]].dropna(subset=["Close"])
                    if len(df) >= 5:
                        return df
            except Exception:
                continue
    return pd.DataFrame()


def _predict_prophet(df, periods_days):
    from prophet import Prophet
    pdf = df[["Close"]].reset_index()
    pdf.columns = ["ds", "y"]
    pdf["ds"] = pd.to_datetime(pdf["ds"]).dt.tz_localize(None)
    pdf = pdf.dropna()

    max_d = max(periods_days)
    results = []

    m80 = Prophet(daily_seasonality=False, yearly_seasonality=True,
                  weekly_seasonality=True, changepoint_prior_scale=0.05, interval_width=0.80)
    m80.fit(pdf)
    f80 = m80.predict(m80.make_future_dataframe(periods=max_d))

    m95 = Prophet(daily_seasonality=False, yearly_seasonality=True,
                  weekly_seasonality=True, changepoint_prior_scale=0.05, interval_width=0.95)
    m95.fit(pdf)
    f95 = m95.predict(m95.make_future_dataframe(periods=max_d))

    last = pdf["ds"].max()
    for d in periods_days:
        tgt = last + timedelta(days=d)
        i80 = (f80["ds"] - tgt).abs().argsort().iloc[0]
        i95 = (f95["ds"] - tgt).abs().argsort().iloc[0]
        results.append({
            "period_days": d,
            "predicted_price": float(f80.iloc[i80]["yhat"]),
            "lower_80": float(f80.iloc[i80]["yhat_lower"]),
            "upper_80": float(f80.iloc[i80]["yhat_upper"]),
            "lower_95": float(f95.iloc[i95]["yhat_lower"]),
            "upper_95": float(f95.iloc[i95]["yhat_upper"]),
        })
    return results


def _predict_ets(df, periods_days):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    s = df["Close"].dropna().values
    model = ExponentialSmoothing(s, trend="add", seasonal=None, damped_trend=True).fit(optimized=True)
    fc = model.forecast(max(periods_days))
    res_std = np.std(s - model.fittedvalues)
    results = []
    for d in periods_days:
        idx = min(d - 1, len(fc) - 1)
        p = float(fc[idx])
        u = res_std * np.sqrt(d)
        results.append({"period_days": d, "predicted_price": p,
                        "lower_80": p - 1.28 * u, "upper_80": p + 1.28 * u,
                        "lower_95": p - 1.96 * u, "upper_95": p + 1.96 * u})
    return results


def _predict_arima(df, periods_days):
    from statsmodels.tsa.arima.model import ARIMA
    s = df["Close"].dropna().values
    model = ARIMA(s, order=(5, 1, 0)).fit()
    max_d = max(periods_days)
    fc = model.get_forecast(steps=max_d)
    pm = fc.predicted_mean
    c80, c95 = fc.conf_int(alpha=0.20), fc.conf_int(alpha=0.05)
    results = []
    for d in periods_days:
        i = min(d - 1, len(pm) - 1)
        results.append({"period_days": d, "predicted_price": float(pm[i]),
                        "lower_80": float(c80[i, 0]), "upper_80": float(c80[i, 1]),
                        "lower_95": float(c95[i, 0]), "upper_95": float(c95[i, 1])})
    return results


def _predict_linear(df, periods_days):
    s = df["Close"].dropna().values
    n = len(s)
    x = np.arange(n)
    coeffs = np.polyfit(x, s, 1)
    res_std = np.std(s - np.polyval(coeffs, x))
    results = []
    for d in periods_days:
        p = coeffs[0] * (n + d) + coeffs[1]
        u = res_std * np.sqrt(d / 20)
        results.append({"period_days": d, "predicted_price": float(p),
                        "lower_80": float(p - 1.28 * u), "upper_80": float(p + 1.28 * u),
                        "lower_95": float(p - 1.96 * u), "upper_95": float(p + 1.96 * u)})
    return results


def _predict_simple(df, periods_days):
    """最終フォールバック: 過去60日の日次リターン平均ベース"""
    s = df["Close"].dropna().values
    cur = s[-1]
    lookback = min(60, len(s) - 1)
    if lookback < 2:
        lookback = len(s) - 1
    recent = s[-lookback:]
    daily_rets = np.diff(recent) / recent[:-1]
    avg_ret = np.mean(daily_rets) if len(daily_rets) > 0 else 0.0
    std_ret = np.std(daily_rets) if len(daily_rets) > 1 else 0.02

    results = []
    for d in periods_days:
        p = cur * (1 + avg_ret * d)
        u = cur * std_ret * np.sqrt(d)
        results.append({"period_days": d, "predicted_price": float(p),
                        "lower_80": float(p - 1.28 * u), "upper_80": float(p + 1.28 * u),
                        "lower_95": float(p - 1.96 * u), "upper_95": float(p + 1.96 * u)})
    return results


def predict_stock(ticker, periods=None):
    """全銘柄で予測を保証。Noneを返さない（データ完全不在時のみNone）。"""
    if periods is None:
        periods = [1, 7, 30, 90, 180]

    hist_df = _fetch_history(ticker)
    if hist_df.empty:
        return None

    cur = float(hist_df["Close"].iloc[-1])
    predictions = None
    engine = PREDICTOR_ENGINE

    chain = {
        "prophet": ["prophet", "ets", "arima", "linear", "simple"],
        "ets": ["ets", "arima", "linear", "simple"],
        "arima": ["arima", "linear", "simple"],
        "linear": ["linear", "simple"],
    }.get(engine, ["linear", "simple"])

    for eng in chain:
        try:
            n = len(hist_df)
            if eng == "prophet" and n >= 30:
                predictions = _predict_prophet(hist_df, periods)
            elif eng == "ets" and n >= 20:
                predictions = _predict_ets(hist_df, periods)
            elif eng == "arima" and n >= 30:
                predictions = _predict_arima(hist_df, periods)
            elif eng == "linear" and n >= 10:
                predictions = _predict_linear(hist_df, periods)
            elif eng == "simple":
                predictions = _predict_simple(hist_df, periods)
            else:
                continue
            if predictions:
                engine = eng
                break
        except Exception as e:
            logger.warning(f"{eng}予測失敗 ({ticker}): {e}")
            continue

    if not predictions:
        predictions = _predict_simple(hist_df, periods)
        engine = "simple"

    for p in predictions:
        p["current_price"] = cur
        p["return_pct"] = (p["predicted_price"] - cur) / cur * 100
        p["lower_80_pct"] = (p["lower_80"] - cur) / cur * 100
        p["upper_80_pct"] = (p["upper_80"] - cur) / cur * 100
        p["lower_95_pct"] = (p["lower_95"] - cur) / cur * 100
        p["upper_95_pct"] = (p["upper_95"] - cur) / cur * 100
        p["lower_bound"] = p["lower_80"]
        p["upper_bound"] = p["upper_80"]

    return {
        "ticker": ticker,
        "current_price": cur,
        "engine": engine,
        "predictions": predictions,
        "history_df": hist_df,
    }


def predict_all_stocks(tickers):
    """全銘柄の予測を実行してDataFrameで返す"""
    periods = [1, 7, 30, 90, 180]
    rows = []
    for tk in tickers:
        result = predict_stock(tk, periods)
        if result is None:
            continue
        rd = {"ticker": tk, "pred_engine": result["engine"],
              "pred_current_price": result["current_price"]}
        for p in result["predictions"]:
            d = p["period_days"]
            rd[f"pred_{d}d_pct"] = round(p["return_pct"], 2)
            rd[f"pred_{d}d_l80"] = round(p["lower_80_pct"], 2)
            rd[f"pred_{d}d_u80"] = round(p["upper_80_pct"], 2)
            rd[f"pred_{d}d_l95"] = round(p["lower_95_pct"], 2)
            rd[f"pred_{d}d_u95"] = round(p["upper_95_pct"], 2)
        rows.append(rd)

    df = pd.DataFrame(rows)
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "latest_predictions.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"予測データ保存: {path} ({len(df)}銘柄)")
    return df
