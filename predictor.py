"""
predictor.py - 株価予測シミュレーション

Prophet優先、インストール不可ならARIMA→線形回帰フォールバック。
過去2年の日足データを学習データに使用。
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# 予測エンジンの判定
PREDICTOR_ENGINE = "linear"  # デフォルト

try:
    from prophet import Prophet
    PREDICTOR_ENGINE = "prophet"
except ImportError:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        PREDICTOR_ENGINE = "arima"
    except ImportError:
        pass

logger.info(f"予測エンジン: {PREDICTOR_ENGINE}")


def _fetch_history(ticker, period="2y"):
    """yfinanceから過去データを取得"""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    hist = tk.history(period=period)
    if hist.empty:
        return pd.DataFrame()
    return hist[["Close"]].copy()


def _predict_prophet(df, periods_days):
    """Prophetによる予測"""
    from prophet import Prophet

    prophet_df = df.reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

    model = Prophet(daily_seasonality=False, yearly_seasonality=True,
                    weekly_seasonality=True, changepoint_prior_scale=0.05)
    model.fit(prophet_df)

    max_days = max(periods_days)
    future = model.make_future_dataframe(periods=max_days)
    forecast = model.predict(future)

    results = []
    last_date = prophet_df["ds"].max()
    for days in periods_days:
        target_date = last_date + timedelta(days=days)
        # 最も近い予測日を取得
        closest = forecast.iloc[(forecast["ds"] - target_date).abs().argsort()[:1]]
        results.append({
            "period_days": days,
            "predicted_price": float(closest["yhat"].values[0]),
            "lower_bound": float(closest["yhat_lower"].values[0]),
            "upper_bound": float(closest["yhat_upper"].values[0]),
        })
    return results


def _predict_arima(df, periods_days):
    """ARIMAによる予測"""
    from statsmodels.tsa.arima.model import ARIMA

    series = df["Close"].values
    model = ARIMA(series, order=(5, 1, 0))
    fitted = model.fit()

    max_days = max(periods_days)
    forecast = fitted.get_forecast(steps=max_days)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.2)  # 80%信頼区間

    results = []
    for days in periods_days:
        idx = days - 1
        if idx >= len(pred_mean):
            idx = len(pred_mean) - 1
        results.append({
            "period_days": days,
            "predicted_price": float(pred_mean[idx]),
            "lower_bound": float(conf_int[idx, 0]),
            "upper_bound": float(conf_int[idx, 1]),
        })
    return results


def _predict_linear(df, periods_days):
    """線形回帰による予測"""
    series = df["Close"].values
    n = len(series)
    x = np.arange(n)

    # 線形回帰
    coeffs = np.polyfit(x, series, 1)
    slope, intercept = coeffs

    # 残差の標準偏差で信頼区間推定
    predicted = np.polyval(coeffs, x)
    residual_std = np.std(series - predicted)

    results = []
    current_price = series[-1]
    for days in periods_days:
        future_x = n + days
        pred = slope * future_x + intercept
        # 信頼区間は期間に応じて広がる
        uncertainty = residual_std * np.sqrt(days / 30) * 1.5
        results.append({
            "period_days": days,
            "predicted_price": float(pred),
            "lower_bound": float(pred - uncertainty),
            "upper_bound": float(pred + uncertainty),
        })
    return results


def predict_stock(ticker, periods=None):
    """
    株価予測を実行

    Args:
        ticker: ティッカーシンボル
        periods: 予測期間リスト (日数)

    Returns:
        dict: {
            "ticker": str,
            "current_price": float,
            "engine": str,
            "predictions": list[dict],
            "history_df": DataFrame,
        }
    """
    if periods is None:
        periods = [1, 7, 30, 90, 180]

    # データ取得
    hist_df = _fetch_history(ticker)
    if hist_df.empty or len(hist_df) < 30:
        return None

    current_price = float(hist_df["Close"].iloc[-1])

    # 予測実行
    engine = PREDICTOR_ENGINE
    try:
        if engine == "prophet":
            predictions = _predict_prophet(hist_df, periods)
        elif engine == "arima":
            predictions = _predict_arima(hist_df, periods)
        else:
            predictions = _predict_linear(hist_df, periods)
    except Exception as e:
        logger.warning(f"{engine}予測失敗、線形回帰にフォールバック: {e}")
        engine = "linear"
        predictions = _predict_linear(hist_df, periods)

    # 予測にリターン情報を追加
    for pred in predictions:
        pred["current_price"] = current_price
        pred["return_pct"] = (pred["predicted_price"] - current_price) / current_price * 100

    return {
        "ticker": ticker,
        "current_price": current_price,
        "engine": engine,
        "predictions": predictions,
        "history_df": hist_df,
    }
