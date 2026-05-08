#!/usr/bin/env python3
"""
surge_scanner.py - 急騰・出来高急増銘柄の自動検知スキャナー

ウォッチリスト外の銘柄でも、以下の条件を満たす銘柄を自動検知:
  1. 出来高急増: 直近5日平均出来高が20日平均の2倍以上
  2. 価格急騰: 直近5日で+10%以上の上昇
  3. 新高値: 52週高値を更新

検知された銘柄は surge_alerts.csv に保存され、
run.py から呼ばれて予測対象に自動追加される。

対象ユニバース:
  - 米国: S&P 500 構成銘柄
  - 日本: 日経225 構成銘柄 + TOPIX主要銘柄
"""

import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent

# =====================================================================
# スキャン対象ユニバース
# =====================================================================
# 主要な米国株インデックス構成銘柄（S&P 500のうち主要セクター）
US_UNIVERSE_SECTORS = {
    "半導体": [
        "NVDA", "AMD", "AVGO", "INTC", "QCOM", "MU", "MRVL", "LRCX",
        "KLAC", "AMAT", "CDNS", "SNPS", "TSM", "ARM", "ON", "NXPI",
        "TXN", "ADI", "MCHP", "SWKS", "MPWR",
    ],
    "AI/クラウド": [
        "MSFT", "GOOGL", "AMZN", "META", "AAPL", "CRM", "NOW", "SNOW",
        "DDOG", "NET", "PLTR", "AI", "PATH", "MDB", "ESTC", "ORCL",
        "IBM", "ADBE",
    ],
    "サーバー/インフラ": [
        "SMCI", "DELL", "HPE",
    ],
    "フィンテック": [
        "V", "MA", "PYPL", "SQ", "COIN", "SOFI", "AFRM", "XYZ",
    ],
    "ヘルスケア": [
        "LLY", "NVO", "ISRG", "DXCM", "MRNA", "ABBV", "TMO", "UNH",
        "JNJ", "PFE", "AMGN", "GILD", "REGN", "VRTX", "BMY",
    ],
    "EV/エネルギー": [
        "TSLA", "RIVN", "LCID", "ENPH", "FSLR", "NEE",
    ],
    "消費/エンタメ": [
        "NFLX", "DIS", "SHOP", "UBER", "ABNB", "RBLX", "TTWO", "EA",
    ],
    "防衛/宇宙": [
        "LMT", "RTX", "NOC", "GD", "BA", "RKLB",
    ],
    "量子": [
        "IONQ", "RGTI",
    ],
}

# 日本株ユニバース（日経225主要銘柄 + TOPIX注目銘柄）
JP_UNIVERSE = [
    # 半導体関連
    "8035.T", "6920.T", "6146.T", "6857.T", "6526.T", "6723.T",
    "4063.T", "6963.T", "285A.T",
    # テック
    "9984.T", "6758.T", "6501.T", "6861.T", "6098.T",
    "3993.T", "4443.T", "4478.T",
    # 自動車
    "7203.T", "7267.T", "7269.T",
    # 商社
    "8058.T", "8031.T", "8001.T",
    # 金融
    "8306.T", "8316.T", "8411.T", "8766.T",
    # 防衛/重工
    "7011.T", "7012.T", "7013.T",
    # その他主要
    "9983.T", "7974.T", "6367.T", "4568.T", "9432.T", "9433.T",
    "4661.T", "2914.T", "2801.T", "6594.T", "4816.T", "8830.T",
    "9005.T", "2413.T", "4385.T", "3659.T",
    # === 追加: ウォッチリスト外の注目候補 ===
    "6981.T",  # 村田製作所
    "6762.T",  # TDK
    "6971.T",  # 京セラ
    "6902.T",  # デンソー
    "6273.T",  # SMC
    "6954.T",  # ファナック
    "9101.T",  # 日本郵船
    "9104.T",  # 商船三井
    "4911.T",  # 資生堂
    "6752.T",  # パナソニック
    "6988.T",  # 日東電工
    "7735.T",  # SCREENホールディングス
    "6645.T",  # オムロン
    "7751.T",  # キヤノン
    "6753.T",  # シャープ
    "3382.T",  # セブン&アイ
    "8002.T",  # 丸紅
    "2802.T",  # 味の素
    "6503.T",  # 三菱電機
    "7752.T",  # リコー
]

# =====================================================================
# スキャン閾値
# =====================================================================
THRESHOLDS = {
    # 出来高急増: 直近5日平均 / 20日平均
    "volume_surge_ratio": 2.0,
    # 価格急騰: 直近5日リターン (%)
    "price_surge_5d_pct": 10.0,
    # 価格急騰: 直近1日リターン (%) — 単日の大幅変動
    "price_surge_1d_pct": 7.0,
    # 新高値: 現在値が52週高値の何%以内なら「新高値圏」
    "near_high_pct": 2.0,
    # 最低出来高（流動性フィルタ）: 20日平均出来高がこれ以下は除外
    "min_avg_volume_us": 500_000,
    "min_avg_volume_jp": 100_000,
}


def _get_all_universe_tickers():
    """全ユニバースのティッカーリストを返す"""
    us_tickers = []
    for sector_tickers in US_UNIVERSE_SECTORS.values():
        us_tickers.extend(sector_tickers)
    us_tickers = list(set(us_tickers))
    return us_tickers, JP_UNIVERSE.copy()


def _load_watchlist_tickers():
    """config.yamlから現在のウォッチリストティッカーを読み込む"""
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        return set()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tickers = set()
    for category in ["us_stocks", "us_etfs", "jp_stocks"]:
        for stock in config.get("watchlist", {}).get(category, []):
            tickers.add(stock["ticker"])
    return tickers


def scan_single_ticker(ticker, market="US"):
    """1銘柄をスキャンして急騰/出来高急増を検知"""
    import yfinance as yf

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo")
        if hist.empty or len(hist) < 25:
            return None

        close = hist["Close"]
        volume = hist["Volume"]

        # --- 出来高分析 ---
        vol_avg_5d = volume.tail(5).mean()
        vol_avg_20d = volume.tail(20).mean()
        vol_ratio = vol_avg_5d / vol_avg_20d if vol_avg_20d > 0 else 0

        # 流動性フィルタ
        min_vol = THRESHOLDS["min_avg_volume_us"] if market == "US" else THRESHOLDS["min_avg_volume_jp"]
        if vol_avg_20d < min_vol:
            return None

        # --- 価格分析 ---
        current_price = float(close.iloc[-1])
        price_5d_ago = float(close.iloc[-6]) if len(close) >= 6 else float(close.iloc[0])
        price_1d_ago = float(close.iloc[-2]) if len(close) >= 2 else current_price
        change_5d_pct = (current_price / price_5d_ago - 1) * 100
        change_1d_pct = (current_price / price_1d_ago - 1) * 100

        # --- 52週高値分析 ---
        high_52w = float(close.max())  # 3ヶ月分しかないので近似
        near_high = (current_price / high_52w - 1) * 100 if high_52w > 0 else -999

        # --- アラート判定 ---
        alerts = []

        if vol_ratio >= THRESHOLDS["volume_surge_ratio"]:
            alerts.append(f"出来高急増 x{vol_ratio:.1f}")

        if change_5d_pct >= THRESHOLDS["price_surge_5d_pct"]:
            alerts.append(f"5日急騰 +{change_5d_pct:.1f}%")

        if change_1d_pct >= THRESHOLDS["price_surge_1d_pct"]:
            alerts.append(f"単日急騰 +{change_1d_pct:.1f}%")

        if near_high >= -THRESHOLDS["near_high_pct"]:
            alerts.append("新高値圏")

        if not alerts:
            return None

        # 銘柄名を取得
        try:
            info = tk.info
            name = info.get("shortName", info.get("longName", ticker))
        except Exception:
            name = ticker

        return {
            "ticker": ticker,
            "name": name,
            "market": market,
            "price": current_price,
            "change_1d_pct": round(change_1d_pct, 2),
            "change_5d_pct": round(change_5d_pct, 2),
            "vol_ratio": round(vol_ratio, 2),
            "vol_avg_5d": int(vol_avg_5d),
            "vol_avg_20d": int(vol_avg_20d),
            "near_high_pct": round(near_high, 2),
            "alerts": " / ".join(alerts),
            "alert_count": len(alerts),
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    except Exception as e:
        logger.debug(f"  スキャンエラー {ticker}: {e}")
        return None


def run_surge_scan(include_watchlist=True):
    """
    全ユニバースをスキャンして急騰・出来高急増銘柄を検出

    Args:
        include_watchlist: True=ウォッチリスト内銘柄もスキャン
                          False=ウォッチリスト外のみ

    Returns:
        DataFrame: 検出された銘柄のリスト
    """
    logger.info("=" * 60)
    logger.info("急騰・出来高急増スキャナー開始")
    logger.info("=" * 60)

    us_tickers, jp_tickers = _get_all_universe_tickers()
    watchlist = _load_watchlist_tickers()

    logger.info(f"  ユニバース: 米国 {len(us_tickers)}銘柄, 日本 {len(jp_tickers)}銘柄")
    logger.info(f"  ウォッチリスト: {len(watchlist)}銘柄")

    results = []

    # 米国株スキャン
    logger.info("\n  [1/2] 米国株スキャン中...")
    for i, ticker in enumerate(us_tickers):
        if not include_watchlist and ticker in watchlist:
            continue
        if (i + 1) % 20 == 0:
            logger.info(f"    ... {i + 1}/{len(us_tickers)} 完了")
        result = scan_single_ticker(ticker, market="US")
        if result:
            result["in_watchlist"] = ticker in watchlist
            results.append(result)

    # 日本株スキャン
    logger.info("\n  [2/2] 日本株スキャン中...")
    for i, ticker in enumerate(jp_tickers):
        plain_ticker = ticker.replace(".T", "")
        if not include_watchlist and (ticker in watchlist or plain_ticker in watchlist):
            continue
        if (i + 1) % 20 == 0:
            logger.info(f"    ... {i + 1}/{len(jp_tickers)} 完了")
        result = scan_single_ticker(ticker, market="JP")
        if result:
            result["in_watchlist"] = ticker in watchlist
            results.append(result)

    if not results:
        logger.info("\n  検出なし: 急騰・出来高急増銘柄はありませんでした")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(["alert_count", "change_5d_pct"], ascending=[False, False])

    # CSV保存
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    csv_path = data_dir / f"surge_alerts_{today}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # latest版も保存
    latest_path = data_dir / "latest_surge_alerts.csv"
    df.to_csv(latest_path, index=False, encoding="utf-8-sig")

    logger.info(f"\n  検出完了: {len(df)}銘柄がアラート条件に合致")
    logger.info(f"  保存先: {csv_path}")

    # サマリー表示
    new_finds = df[~df["in_watchlist"]]
    existing = df[df["in_watchlist"]]
    logger.info(f"\n  --- サマリー ---")
    logger.info(f"  ウォッチリスト外の新規検出: {len(new_finds)}銘柄")
    logger.info(f"  ウォッチリスト内のアラート: {len(existing)}銘柄")

    if not new_finds.empty:
        logger.info(f"\n  ★ 新規検出銘柄 (ウォッチリスト外):")
        for _, row in new_finds.iterrows():
            logger.info(
                f"    {row['ticker']:8s} {row['name'][:20]:20s} "
                f"${row['price']:>10,.2f}  5d:{row['change_5d_pct']:+.1f}%  "
                f"Vol:{row['vol_ratio']:.1f}x  [{row['alerts']}]"
            )

    return df


def get_surge_tickers_for_prediction():
    """
    予測対象に追加すべき急騰銘柄のティッカーリストを返す
    (ウォッチリスト外で検出された銘柄のみ)
    """
    latest_path = BASE_DIR / "data" / "latest_surge_alerts.csv"
    if not latest_path.exists():
        return []

    df = pd.read_csv(latest_path)
    if df.empty:
        return []

    # ウォッチリスト外の銘柄のみ
    new_tickers = df[~df["in_watchlist"]]["ticker"].tolist()
    return new_tickers


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    df = run_surge_scan(include_watchlist=True)
    if not df.empty:
        print(f"\n{'='*80}")
        print("  急騰・出来高急増 アラート一覧")
        print(f"{'='*80}")
        for _, row in df.iterrows():
            wl = "★" if not row["in_watchlist"] else " "
            print(
                f"  {wl} {row['ticker']:8s} | {row['name'][:25]:25s} | "
                f"5d:{row['change_5d_pct']:+6.1f}% | "
                f"1d:{row['change_1d_pct']:+6.1f}% | "
                f"Vol:{row['vol_ratio']:4.1f}x | "
                f"{row['alerts']}"
            )
        print(f"{'='*80}")
        print(f"  ★ = ウォッチリスト外の新規検出銘柄")
