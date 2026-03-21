"""
fetch_macro.py - マクロ経済指標の自動取得モジュール

データソース: yfinance (無料)
出力: data/macro_data_{date}.csv, data/latest_macro.csv
"""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fetch_macro.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MACRO_TICKERS = {
    "^VIX":     {"name": "VIX恐怖指数",        "category": "リスク指標"},
    "^TNX":     {"name": "米10年国債利回り",     "category": "金利"},
    "UUP":      {"name": "ドルインデックスETF",  "category": "為替"},
    "^GSPC":    {"name": "S&P 500",             "category": "株価指数"},
    "^IXIC":    {"name": "NASDAQ総合",           "category": "株価指数"},
    "^N225":    {"name": "日経225",              "category": "株価指数"},
    "GC=F":     {"name": "金先物",               "category": "コモディティ"},
    "CL=F":     {"name": "原油WTI先物",          "category": "コモディティ"},
    "BTC-USD":  {"name": "ビットコイン",          "category": "暗号資産"},
    "USDJPY=X": {"name": "ドル/円",              "category": "為替"},
}


def fetch_single_macro(ticker_symbol, name, category):
    """1つのマクロ指標を取得"""
    try:
        logger.info(f"  マクロ取得中: {ticker_symbol} ({name})")
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="3mo")

        if hist.empty:
            logger.warning(f"  {ticker_symbol}: データなし")
            return None

        current = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current

        # 1ヶ月前 (~21営業日)
        price_1m = hist["Close"].iloc[-22] if len(hist) >= 22 else hist["Close"].iloc[0]
        # 3ヶ月前 (最初のデータ)
        price_3m = hist["Close"].iloc[0]

        change_1d = ((current - prev_close) / prev_close) * 100
        change_1m = ((current - price_1m) / price_1m) * 100
        change_3m = ((current - price_3m) / price_3m) * 100

        return {
            "ticker": ticker_symbol,
            "name": name,
            "category": category,
            "current_value": round(float(current), 2),
            "prev_close": round(float(prev_close), 2),
            "change_1d_pct": round(float(change_1d), 2),
            "change_1m_pct": round(float(change_1m), 2),
            "change_3m_pct": round(float(change_3m), 2),
            "fetch_time": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"  {ticker_symbol}: エラー - {e}")
        return None


def fetch_all_macro():
    """全マクロ指標を取得してCSV保存"""
    logger.info("=" * 50)
    logger.info("マクロ経済指標取得開始")
    logger.info("=" * 50)

    results = []
    for ticker_symbol, info in MACRO_TICKERS.items():
        data = fetch_single_macro(ticker_symbol, info["name"], info["category"])
        if data:
            results.append(data)

    df = pd.DataFrame(results)

    # CSV保存
    today = datetime.now().strftime("%Y%m%d")
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / f"macro_data_{today}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存完了: {csv_path} ({len(df)}指標)")

    latest_path = data_dir / "latest_macro.csv"
    df.to_csv(latest_path, index=False, encoding="utf-8-sig")

    return df


def interpret_macro(macro_df):
    """マクロデータを解釈してコメント生成"""
    interpretations = []

    for _, row in macro_df.iterrows():
        ticker = row["ticker"]
        value = row["current_value"]
        change_1m = row["change_1m_pct"]

        if ticker == "^VIX":
            if value >= 25:
                interpretations.append({
                    "indicator": "VIX恐怖指数",
                    "value": f"{value:.1f}",
                    "interpretation": "市場不安定",
                    "impact": "ボラティリティ上昇。グロース株に逆風の可能性。リスク管理を強化。",
                })
            elif value <= 20:
                interpretations.append({
                    "indicator": "VIX恐怖指数",
                    "value": f"{value:.1f}",
                    "interpretation": "市場安定",
                    "impact": "リスクオン環境。グロース株に追い風。",
                })
            else:
                interpretations.append({
                    "indicator": "VIX恐怖指数",
                    "value": f"{value:.1f}",
                    "interpretation": "やや警戒",
                    "impact": "市場はやや不安定。慎重な投資判断を。",
                })

        elif ticker == "^TNX":
            if value >= 4.5:
                interpretations.append({
                    "indicator": "米10年国債利回り",
                    "value": f"{value:.2f}%",
                    "interpretation": "金利高水準",
                    "impact": "金利高→グロース株に逆風。バリュー株・高配当株に資金流入しやすい。",
                })
            elif value <= 3.5:
                interpretations.append({
                    "indicator": "米10年国債利回り",
                    "value": f"{value:.2f}%",
                    "interpretation": "金利低下傾向",
                    "impact": "金利低下→グロース株に追い風。成長株の相対的魅力が向上。",
                })
            else:
                interpretations.append({
                    "indicator": "米10年国債利回り",
                    "value": f"{value:.2f}%",
                    "interpretation": "金利中立",
                    "impact": "金利は中立的な水準。個別銘柄のファンダメンタルズ重視。",
                })

        elif ticker == "USDJPY=X":
            if change_1m > 2:
                interpretations.append({
                    "indicator": "ドル/円",
                    "value": f"{value:.1f}円 (前月比+{change_1m:.1f}%)",
                    "interpretation": "円安進行中",
                    "impact": "円安→日本株輸出関連に追い風。米国株の円建て評価額は上昇。",
                })
            elif change_1m < -2:
                interpretations.append({
                    "indicator": "ドル/円",
                    "value": f"{value:.1f}円 (前月比{change_1m:.1f}%)",
                    "interpretation": "円高進行中",
                    "impact": "円高→日本株輸出関連に逆風。米国株の円建て評価額は下落。",
                })
            else:
                interpretations.append({
                    "indicator": "ドル/円",
                    "value": f"{value:.1f}円",
                    "interpretation": "為替安定",
                    "impact": "為替は安定的。為替要因は中立。",
                })

    return interpretations


if __name__ == "__main__":
    df = fetch_all_macro()
    print(f"\n取得完了: {len(df)}指標")
    print(df[["ticker", "name", "current_value", "change_1d_pct", "change_1m_pct"]].to_string(index=False))
    print("\n--- 市場環境の解釈 ---")
    for interp in interpret_macro(df):
        print(f"  {interp['indicator']}: {interp['interpretation']} ({interp['value']})")
        print(f"    → {interp['impact']}")
