"""
fetch_stock_data.py - 株価・出来高・テクニカル指標の自動取得モジュール

データソース:
  1. yfinance (無料・無制限) — メイン
  2. Alpha Vantage API (バックアップ) — 無料枠25回/日

出力: data/stock_data_{date}.csv
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import yaml
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fetch_stock.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calc_rsi(series, period=14):
    """RSI (Relative Strength Index) を計算"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence) を計算"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger_bands(series, period=20, std_dev=2):
    """ボリンジャーバンドを計算"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def fetch_daily_data_cached(ticker_symbol, full_refresh=False):
    """キャッシュ対応の日足データ取得。初回は5年分、2回目以降は差分のみ。"""
    from db_cache import init_db, get_latest_date, upsert_daily_prices, load_daily_prices

    init_db()
    latest_date = get_latest_date(ticker_symbol)

    if full_refresh or latest_date is None:
        # フル取得: 5年分
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="5y")
        if hist.empty:
            return pd.DataFrame()
        upsert_daily_prices(ticker_symbol, hist)
        logger.debug(f"  {ticker_symbol}: フル取得 {len(hist)}日分")
    else:
        # 差分取得
        start_date = (pd.Timestamp(latest_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        if start_date < today:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(start=start_date)
            if not hist.empty:
                upsert_daily_prices(ticker_symbol, hist)
                logger.debug(f"  {ticker_symbol}: 差分取得 {len(hist)}日分")
            else:
                logger.debug(f"  {ticker_symbol}: 差分なし（最新）")
        else:
            logger.debug(f"  {ticker_symbol}: キャッシュ最新")

    return load_daily_prices(ticker_symbol, days=400)


def calc_price_history_from_cache(ticker_symbol):
    """SQLiteキャッシュから株価推移データを算出"""
    from db_cache import load_full_history

    try:
        hist = load_full_history(ticker_symbol)
        if hist.empty or len(hist) < 30:
            return {}

        current_price = hist["Close"].iloc[-1]

        # N年前の株価を日付ベースで取得
        def price_n_years_ago(years):
            target_date = datetime.now() - timedelta(days=years * 365)
            earlier = hist[hist.index <= target_date]
            if not earlier.empty:
                return earlier["Close"].iloc[-1]
            return None

        price_5y = price_n_years_ago(5)
        price_3y = price_n_years_ago(3)
        price_1y = price_n_years_ago(1)

        def calc_change_rate(old_price):
            if old_price and old_price > 0:
                return round(((current_price - old_price) / old_price) * 100, 1)
            return None

        # 過去1年の最高値・最安値
        one_year_ago = datetime.now() - timedelta(days=365)
        hist_1y = hist[hist.index >= one_year_ago]
        high_1y = None
        high_1y_date = None
        low_1y = None
        low_1y_date = None
        if not hist_1y.empty:
            high_1y = hist_1y["High"].max()
            high_1y_idx = hist_1y["High"].idxmax()
            high_1y_date = high_1y_idx.strftime("%Y-%m-%d") if hasattr(high_1y_idx, 'strftime') else str(high_1y_idx)
            low_1y = hist_1y["Low"].min()
            low_1y_idx = hist_1y["Low"].idxmin()
            low_1y_date = low_1y_idx.strftime("%Y-%m-%d") if hasattr(low_1y_idx, 'strftime') else str(low_1y_idx)

        return {
            "price_5y_ago": round(price_5y, 2) if price_5y else None,
            "price_3y_ago": round(price_3y, 2) if price_3y else None,
            "price_1y_ago": round(price_1y, 2) if price_1y else None,
            "change_5y_pct": calc_change_rate(price_5y),
            "change_3y_pct": calc_change_rate(price_3y),
            "change_1y_pct": calc_change_rate(price_1y),
            "high_1y": round(high_1y, 2) if high_1y else None,
            "high_1y_date": high_1y_date,
            "low_1y": round(low_1y, 2) if low_1y else None,
            "low_1y_date": low_1y_date,
        }
    except Exception as e:
        logger.warning(f"  {ticker_symbol}: 株価推移データ算出エラー - {e}")
        return {}


def fetch_single_stock_yfinance(ticker_symbol, name, sector, market, tech_cfg,
                                 full_refresh=False):
    """yfinanceで1銘柄のデータを取得し、テクニカル指標を計算（キャッシュ対応）"""
    try:
        logger.info(f"取得中: {ticker_symbol} ({name})")

        # キャッシュ対応の日足データ取得
        hist = fetch_daily_data_cached(ticker_symbol, full_refresh)
        if hist.empty:
            logger.warning(f"  {ticker_symbol}: データなし")
            return None

        # 基本情報
        close = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else close
        change_pct = ((close - prev_close) / prev_close) * 100

        # 52週レンジ
        high_52w = hist["Close"].max()
        low_52w = hist["Close"].min()
        pos_52w = ((close - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 50

        # 出来高分析（改善4: 詳細出来高データ）
        vol_current = hist["Volume"].iloc[-1]
        vol_avg_20 = hist["Volume"].tail(20).mean()
        vol_ratio = vol_current / vol_avg_20 if vol_avg_20 > 0 else 1.0
        # 過去5日間の出来高推移
        vol_5d = hist["Volume"].tail(5).tolist()
        vol_5d_str = ",".join([str(int(v)) for v in vol_5d]) if len(vol_5d) > 0 else ""

        # テクニカル指標
        closes = hist["Close"]
        sma_20 = closes.rolling(window=tech_cfg["sma_short"]).mean().iloc[-1]
        sma_50 = closes.rolling(window=tech_cfg["sma_long"]).mean().iloc[-1]
        sma_200 = closes.rolling(window=tech_cfg["sma_trend"]).mean().iloc[-1] if len(closes) >= 200 else np.nan

        rsi = calc_rsi(closes, tech_cfg["rsi_period"]).iloc[-1]

        macd_line, signal_line, macd_hist = calc_macd(
            closes, tech_cfg["macd_fast"], tech_cfg["macd_slow"], tech_cfg["macd_signal"]
        )
        bb_upper, bb_mid, bb_lower = calc_bollinger_bands(
            closes, tech_cfg["bb_period"], tech_cfg["bb_std"]
        )

        # ゴールデンクロス / デッドクロス検出 (SMA20 vs SMA50)
        sma20_series = closes.rolling(window=20).mean()
        sma50_series = closes.rolling(window=50).mean()
        if len(sma20_series.dropna()) >= 2 and len(sma50_series.dropna()) >= 2:
            cross_today = sma20_series.iloc[-1] - sma50_series.iloc[-1]
            cross_yesterday = sma20_series.iloc[-2] - sma50_series.iloc[-2]
            if cross_today > 0 and cross_yesterday <= 0:
                cross_signal = "ゴールデンクロス"
            elif cross_today < 0 and cross_yesterday >= 0:
                cross_signal = "デッドクロス"
            else:
                cross_signal = "なし"
        else:
            cross_signal = "データ不足"

        # ファンダメンタルズ (yfinance info - 常にフレッシュ取得)
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info or {}
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        peg_ratio = info.get("pegRatio")
        market_cap = info.get("marketCap")
        revenue_growth = info.get("revenueGrowth")
        if revenue_growth is not None:
            revenue_growth = revenue_growth * 100  # -> %
        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None:
            dividend_yield = dividend_yield * 100  # -> %
        target_price = info.get("targetMeanPrice")

        # 過去5年間の株価推移データ（キャッシュから算出）
        price_history = calc_price_history_from_cache(ticker_symbol)

        result = {
            "ticker": ticker_symbol.replace(".T", ""),
            "name": name,
            "sector": sector,
            "market": market,
            "price": round(close, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(vol_current),
            "vol_avg_20d": int(vol_avg_20),
            "vol_ratio": round(vol_ratio, 2),
            "vol_5d": vol_5d_str,
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "pos_52w_pct": round(pos_52w, 1),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2) if not np.isnan(sma_200) else None,
            "rsi_14": round(rsi, 1),
            "macd": round(macd_line.iloc[-1], 4),
            "macd_signal": round(signal_line.iloc[-1], 4),
            "macd_histogram": round(macd_hist.iloc[-1], 4),
            "bb_upper": round(bb_upper.iloc[-1], 2),
            "bb_lower": round(bb_lower.iloc[-1], 2),
            "bb_position": round(((close - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])) * 100, 1) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 50,
            "cross_signal": cross_signal,
            "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
            "peg_ratio": round(peg_ratio, 2) if peg_ratio else None,
            "market_cap": market_cap,
            "revenue_growth_pct": round(revenue_growth, 1) if revenue_growth else None,
            "dividend_yield_pct": round(dividend_yield, 2) if dividend_yield else None,
            "target_price": target_price,
            "upside_pct": round(((target_price - close) / close) * 100, 1) if target_price else None,
            "fetch_time": datetime.now().isoformat(),
        }

        # 株価推移データをマージ
        result.update(price_history)

        return result

    except Exception as e:
        logger.error(f"  {ticker_symbol}: エラー - {e}")
        return None


def fetch_via_alpha_vantage(ticker_symbol, api_key):
    """Alpha Vantage APIで株価を取得（バックアップ用）"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": ticker_symbol,
        "apikey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json().get("Global Quote", {})
        if data:
            return {
                "price": float(data.get("05. price", 0)),
                "change_pct": float(data.get("10. change percent", "0").replace("%", "")),
                "volume": int(data.get("06. volume", 0)),
                "high": float(data.get("03. high", 0)),
                "low": float(data.get("04. low", 0)),
            }
    except Exception as e:
        logger.error(f"Alpha Vantage error for {ticker_symbol}: {e}")
    return None


def fetch_all_stocks(config, full_refresh=False):
    """全銘柄のデータを取得"""
    tech_cfg = config["technical"]
    results = []

    category_market_map = {
        "us_stocks": "US",
        "us_etfs": "US",
        "jp_stocks": "JP",
    }
    for category, market in category_market_map.items():
        for stock in config["watchlist"].get(category, []):
            data = fetch_single_stock_yfinance(
                stock["ticker"], stock["name"], stock["sector"], market, tech_cfg,
                full_refresh=full_refresh
            )
            if data:
                results.append(data)

    df = pd.DataFrame(results)

    # CSV保存
    today = datetime.now().strftime("%Y%m%d")
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / f"stock_data_{today}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存完了: {csv_path} ({len(df)}銘柄)")

    # latest_data.csv にも上書き保存
    latest_path = data_dir / "latest_data.csv"
    df.to_csv(latest_path, index=False, encoding="utf-8-sig")

    return df


if __name__ == "__main__":
    config = load_config()
    df = fetch_all_stocks(config)
    print(f"\n{'='*60}")
    print(f"取得完了: {len(df)}銘柄")
    print(f"{'='*60}")
    print(df[["ticker", "name", "price", "change_pct", "rsi_14", "vol_ratio", "pos_52w_pct"]].to_string(index=False))
