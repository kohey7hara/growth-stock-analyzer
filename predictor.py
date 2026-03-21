"""
predictor.py - 株価予測シミュレーション

Prophet優先 → ARIMA → ETS → 線形回帰フォールバック。
80%と95%の信頼区間を算出。
全銘柄で予測が出るよう、最終手段として過去平均騰落率ベースの簡易予測を使用。
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

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


def _fetch_history(ticker, period="2y"):
    """yfinanceから過去データを取得。フォールバックあり。"""
    import yfinance as yf
    for p in [period, "1y", "6mo"]:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=p)
            if not hist.empty and len(hist) >= 10:
                return hist[["Close", "Volume"]].copy()
        except Exception:
            continue
    return pd.DataFrame()


def _predict_prophet(df, periods_days):
    """Prophetによる予測 (80%と95%信頼区間)"""
    from prophet import Prophet

    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

    max_days = max(periods_days)

    # 80%信頼区間
    m80 = Prophet(daily_seasonality=False, yearly_seasonality=True,
                  weekly_seasonality=True, changepoint_prior_scale=0.05,
                  interval_width=0.80)
    m80.fit(prophet_df)
    future = m80.make_future_dataframe(periods=max_days)
    fc80 = m80.predict(future)

    # 95%信頼区間
    m95 = Prophet(daily_seasonality=False, yearly_seasonality=True,
                  weekly_seasonality=True, changepoint_prior_scale=0.05,
                  interval_width=0.95)
    m95.fit(prophet_df)
    fc95 = m95.predict(future)

    last_date = prophet_df["ds"].max()
    results = []
    for days in periods_days:
        target_date = last_date + timedelta(days=days)
        idx80 = (fc80["ds"] - target_date).abs().argsort()[:1]
        idx95 = (fc95["ds"] - target_date).abs().argsort()[:1]
        results.append({
            "period_days": days,
            "predicted_price": float(fc80.iloc[idx80.values[0]]["yhat"]),
            "lower_80": float(fc80.iloc[idx80.values[0]]["yhat_lower"]),
            "upper_80": float(fc80.iloc[idx80.values[0]]["yhat_upper"]),
            "lower_95": float(fc95.iloc[idx95.values[0]]["yhat_lower"]),
            "upper_95": float(fc95.iloc[idx95.values[0]]["yhat_upper"]),
        })
    return results


def _predict_ets(df, periods_days):
    """指数平滑法(ETS)による予測"""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = df["Close"].values
    n = len(series)
    model = ExponentialSmoothing(series, trend="add", seasonal=None,
                                 damped_trend=True).fit(optimized=True)
    max_days = max(periods_days)
    forecast = model.forecast(max_days)

    residuals = series - model.fittedvalues
    res_std = np.std(residuals)

    results = []
    for days in periods_days:
        idx = min(days - 1, len(forecast) - 1)
        pred = float(forecast[idx])
        unc = res_std * np.sqrt(days)
        results.append({
            "period_days": days,
            "predicted_price": pred,
            "lower_80": pred - 1.28 * unc,
            "upper_80": pred + 1.28 * unc,
            "lower_95": pred - 1.96 * unc,
            "upper_95": pred + 1.96 * unc,
        })
    return results


def _predict_arima(df, periods_days):
    """ARIMAによる予測"""
    from statsmodels.tsa.arima.model import ARIMA

    series = df["Close"].values
    model = ARIMA(series, order=(5, 1, 0))
    fitted = model.fit()

    max_days = max(periods_days)
    fc80 = fitted.get_forecast(steps=max_days)
    pred_mean = fc80.predicted_mean
    ci80 = fc80.conf_int(alpha=0.20)
    ci95 = fc80.conf_int(alpha=0.05)

    results = []
    for days in periods_days:
        idx = min(days - 1, len(pred_mean) - 1)
        results.append({
            "period_days": days,
            "predicted_price": float(pred_mean[idx]),
            "lower_80": float(ci80[idx, 0]),
            "upper_80": float(ci80[idx, 1]),
            "lower_95": float(ci95[idx, 0]),
            "upper_95": float(ci95[idx, 1]),
        })
    return results


def _predict_linear(df, periods_days):
    """線形回帰による予測"""
    series = df["Close"].values
    n = len(series)
    x = np.arange(n)
    coeffs = np.polyfit(x, series, 1)
    slope, intercept = coeffs
    fitted = np.polyval(coeffs, x)
    res_std = np.std(series - fitted)

    results = []
    for days in periods_days:
        pred = slope * (n + days) + intercept
        unc = res_std * np.sqrt(days / 20)
        results.append({
            "period_days": days,
            "predicted_price": float(pred),
            "lower_80": float(pred - 1.28 * unc),
            "upper_80": float(pred + 1.28 * unc),
            "lower_95": float(pred - 1.96 * unc),
            "upper_95": float(pred + 1.96 * unc),
        })
    return results


def _predict_simple_avg(df, periods_days):
    """最終手段: 過去の平均月次リターンベースの簡易予測"""
    series = df["Close"].values
    current = series[-1]

    # 月次リターンの平均を計算
    monthly_returns = []
    step = min(21, len(series) // 4)
    if step < 5:
        step = 5
    for i in range(step, len(series), step):
        ret = (series[i] - series[i - step]) / series[i - step]
        monthly_returns.append(ret)

    if not monthly_returns:
        avg_monthly = 0.0
        std_monthly = 0.05
    else:
        avg_monthly = np.mean(monthly_returns)
        std_monthly = max(np.std(monthly_returns), 0.01)

    daily_ret = avg_monthly / 21
    daily_std = std_monthly / np.sqrt(21)

    results = []
    for days in periods_days:
        pred = current * (1 + daily_ret * days)
        unc = current * daily_std * np.sqrt(days)
        results.append({
            "period_days": days,
            "predicted_price": float(pred),
            "lower_80": float(pred - 1.28 * unc),
            "upper_80": float(pred + 1.28 * unc),
            "lower_95": float(pred - 1.96 * unc),
            "upper_95": float(pred + 1.96 * unc),
        })
    return results


def predict_stock(ticker, periods=None):
    """
    株価予測を実行。全銘柄で結果を返す（最終手段: 平均リターンベース）。

    Returns:
        dict with keys: ticker, current_price, engine, predictions, history_df
        predictions[i] keys: period_days, predicted_price,
            lower_80, upper_80, lower_95, upper_95, return_pct,
            lower_80_pct, upper_80_pct, lower_95_pct, upper_95_pct
    """
    if periods is None:
        periods = [1, 7, 30, 90, 180]

    hist_df = _fetch_history(ticker)
    if hist_df.empty:
        return None

    current_price = float(hist_df["Close"].iloc[-1])

    # 予測エンジンのフォールバックチェーン
    engine = PREDICTOR_ENGINE
    predictions = None

    engines = []
    if engine == "prophet":
        engines = ["prophet", "ets", "arima", "linear", "simple"]
    elif engine == "ets":
        engines = ["ets", "arima", "linear", "simple"]
    elif engine == "arima":
        engines = ["arima", "linear", "simple"]
    else:
        engines = ["linear", "simple"]

    for eng in engines:
        try:
            if eng == "prophet" and len(hist_df) >= 30:
                predictions = _predict_prophet(hist_df, periods)
            elif eng == "ets" and len(hist_df) >= 20:
                predictions = _predict_ets(hist_df, periods)
            elif eng == "arima" and len(hist_df) >= 30:
                predictions = _predict_arima(hist_df, periods)
            elif eng == "linear" and len(hist_df) >= 10:
                predictions = _predict_linear(hist_df, periods)
            elif eng == "simple":
                predictions = _predict_simple_avg(hist_df, periods)
            else:
                continue

            if predictions:
                engine = eng
                break
        except Exception as e:
            logger.warning(f"{eng}予測失敗 ({ticker}): {e}")
            continue

    if not predictions:
        predictions = _predict_simple_avg(hist_df, periods)
        engine = "simple"

    # リターン情報を追加
    for pred in predictions:
        pred["current_price"] = current_price
        pred["return_pct"] = (pred["predicted_price"] - current_price) / current_price * 100
        pred["lower_80_pct"] = (pred["lower_80"] - current_price) / current_price * 100
        pred["upper_80_pct"] = (pred["upper_80"] - current_price) / current_price * 100
        pred["lower_95_pct"] = (pred["lower_95"] - current_price) / current_price * 100
        pred["upper_95_pct"] = (pred["upper_95"] - current_price) / current_price * 100
        # 後方互換性
        pred["lower_bound"] = pred["lower_80"]
        pred["upper_bound"] = pred["upper_80"]

    return {
        "ticker": ticker,
        "current_price": current_price,
        "engine": engine,
        "predictions": predictions,
        "history_df": hist_df,
    }
