"""
portfolio.py - ポートフォリオ分析モジュール

保有銘柄のCSVを読み込み、現在株価・含み損益・年率リターン等を算出する。
"""

import pandas as pd
import yfinance as yf
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

PORTFOLIO_PATH = Path(__file__).parent / "data" / "portfolio.csv"


def load_portfolio(path=None):
    """ポートフォリオCSVを読み込む。ファイルがなければNoneを返す。"""
    p = Path(path) if path else PORTFOLIO_PATH
    if not p.exists():
        logger.info("portfolio.csv が見つかりません")
        return None
    df = pd.read_csv(p, parse_dates=["buy_date"])
    return df


def fetch_usdjpy_rate():
    """現在のUSD/JPYレートを取得。失敗時は150.0を返す。"""
    try:
        t = yf.Ticker("USDJPY=X")
        hist = t.history(period="1d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1])
            logger.info(f"USDJPY: {rate:.2f}")
            return rate
    except Exception as e:
        logger.warning(f"USDJPY取得失敗: {e}")
    return 150.0


def calculate_portfolio(portfolio_df, usdjpy_rate):
    """
    各銘柄の現在株価を取得し、含み損益・年率リターン等を算出する。

    Returns:
        (result_df, totals_dict)
    """
    today = datetime.now().date()
    rows = []

    for _, r in portfolio_df.iterrows():
        ticker = r["ticker"]
        buy_date = r["buy_date"].date() if hasattr(r["buy_date"], "date") else r["buy_date"]
        buy_price = float(r["buy_price"])
        quantity = int(r["quantity"])
        currency = r.get("buy_currency", "JPY")
        memo = r.get("memo", "")
        is_usd = currency == "USD"
        fx = usdjpy_rate if is_usd else 1.0

        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            name = info.get("shortName", ticker)

            # 現在株価
            hist_1d = tk.history(period="5d")
            if hist_1d.empty:
                logger.warning(f"{ticker}: 株価取得失敗、スキップ")
                continue
            current_price = float(hist_1d["Close"].iloc[-1])

            # 保有日数
            holding_days = (today - buy_date).days

            # 含み損益
            unrealized_pnl_pct = (current_price - buy_price) / buy_price * 100
            unrealized_pnl = (current_price - buy_price) * quantity * fx

            # 年率リターン
            if holding_days > 0:
                annualized_return = ((current_price / buy_price) ** (365 / holding_days) - 1) * 100
            else:
                annualized_return = 0.0

            # 円換算評価額
            current_value_jpy = current_price * quantity * fx

            # 1年後予想株価 (3年前価格から年率成長率算出)
            projected_price_1y = None
            try:
                hist_3y = tk.history(period="3y")
                if len(hist_3y) >= 250:
                    price_3y_ago = float(hist_3y["Close"].iloc[0])
                    years_span = (hist_3y.index[-1] - hist_3y.index[0]).days / 365
                    if years_span > 0 and price_3y_ago > 0:
                        annual_growth = (current_price / price_3y_ago) ** (1 / years_span) - 1
                        projected_price_1y = round(current_price * (1 + annual_growth), 2)
            except Exception:
                pass

            rows.append({
                "ticker": ticker,
                "name": name,
                "buy_date": str(buy_date),
                "buy_price": buy_price,
                "quantity": quantity,
                "current_price": round(current_price, 2),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                "unrealized_pnl": round(unrealized_pnl, 0),
                "holding_days": holding_days,
                "annualized_return": round(annualized_return, 2),
                "current_value_jpy": round(current_value_jpy, 0),
                "projected_price_1y": projected_price_1y,
                "memo": memo,
            })
            logger.info(f"  {ticker}: {current_price:.2f} (損益 {unrealized_pnl_pct:+.1f}%)")

        except Exception as e:
            logger.warning(f"{ticker}: 処理エラー: {e}")
            continue

    if not rows:
        return None, {}

    result_df = pd.DataFrame(rows)

    # 合計
    total_current_jpy = result_df["current_value_jpy"].sum()
    total_cost_jpy = sum(
        r["buy_price"] * r["quantity"] * (usdjpy_rate if portfolio_df.iloc[i].get("buy_currency", "JPY") == "USD" else 1.0)
        for i, r in result_df.iterrows()
    )
    total_pnl_jpy = result_df["unrealized_pnl"].sum()
    total_pnl_pct = (total_pnl_jpy / total_cost_jpy * 100) if total_cost_jpy > 0 else 0

    totals = {
        "total_pnl_jpy": round(total_pnl_jpy, 0),
        "total_current_jpy": round(total_current_jpy, 0),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "usdjpy_rate": usdjpy_rate,
    }

    return result_df, totals


def run_portfolio_analysis(path=None):
    """ポートフォリオ分析の統合実行。run.pyから呼ばれる。"""
    portfolio_df = load_portfolio(path)
    if portfolio_df is None:
        return None, {}

    usdjpy_rate = fetch_usdjpy_rate()
    result_df, totals = calculate_portfolio(portfolio_df, usdjpy_rate)
    return result_df, totals
