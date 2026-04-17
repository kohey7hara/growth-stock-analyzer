"""
backtester.py - TOP10戦略 walk-forward バックテスト

設計:
- 過去N年のOHLCVをyfinanceから取得（parquetでキャッシュ）
- 各リバランス日時点でテクニカル指標のみを計算 → TOP N選定 → 等金額保有
- 次のリバランス日まで保有、到達したら全売却 → 再選定
- リーク無し walk-forward（未来データを使わない）
- ベンチマーク比較（SPY, ^N225）
- KPI: 総/年率リターン、ボラ、Sharpe、最大DD、勝率

使い方:
    from backtester import run_backtest, get_default_tickers
    tickers = get_default_tickers()
    result = run_backtest(tickers, period="5y", rebalance_freq="monthly")
    print(result["metrics"])

CLI:
    python backtester.py --period 5y --freq both
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "backtest_cache"
RESULT_DIR = DATA_DIR / "backtest_results"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
RESULT_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger(__name__)


# ==========================================
# ティッカー取得
# ==========================================

def get_default_tickers():
    """latest_analysis.csv から全銘柄を取得（ETF除外）"""
    ETF = {"VOO", "QQQ", "SOXX", "ARKK", "XLF", "XLE", "XLV"}
    path = DATA_DIR / "latest_analysis.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    tickers = [t for t in df["ticker"].tolist() if t not in ETF]
    return tickers


# =====================================
# ユニバース定義（Step C）
# =====================================

# 大型安定株（S&P500/日経225の中核銘柄）
LARGE_CAP_US = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "BRK-B", "TSM", "V", "JPM", "UNH", "JNJ",
    "WMT", "XOM", "MA", "PG", "HD", "CVX",
    "ABBV", "KO", "PEP", "AVGO", "LLY", "COST",
    "MCD", "TMO", "ABT", "CSCO",
]

LARGE_CAP_JP = [
    "7203", "6758", "8306", "9432", "6098",  # トヨタ、ソニー、MUFG、NTT、リクルート
    "9984", "8316", "8058", "4063", "7974",  # SBG、SMFG、三菱商事、信越化、任天堂
    "8001", "6861", "6367", "4452", "4502",  # 伊藤忠、キーエンス、ダイキン、花王、武田
    "8031", "6594", "7267", "9433", "4063",  # 三井物産、ニデック、ホンダ、KDDI
]


def get_large_cap_universe():
    """大型株のみのユニバース"""
    return LARGE_CAP_US + list(dict.fromkeys(LARGE_CAP_JP))  # 重複除去


def get_expanded_universe():
    """現ユニバース + 大型安定株（重複除去）"""
    base = get_default_tickers()
    large = LARGE_CAP_US + list(dict.fromkeys(LARGE_CAP_JP))
    merged = list(dict.fromkeys(base + large))  # 順序保持で重複除去
    return merged


UNIVERSE_OPTIONS = {
    "current": {
        "name": "現ユニバース（118銘柄、成長株中心）",
        "desc": "latest_analysis.csv のSaaS/AI/バイオ中心の銘柄群。セクター偏り有り",
        "fn": get_default_tickers,
    },
    "large_cap": {
        "name": "大型安定株のみ（約45銘柄）",
        "desc": "S&P500中核+日経225中核。ディフェンシブで市場連動性が高い",
        "fn": get_large_cap_universe,
    },
    "expanded": {
        "name": "拡張ユニバース（現+大型株、約160銘柄）",
        "desc": "成長株に大型安定株を追加。分散効果で安定感UP",
        "fn": get_expanded_universe,
    },
}


def to_yf_symbol(ticker):
    """数字のみの日本株は .T を付与"""
    s = str(ticker).strip()
    if s.replace(".", "").isdigit() and not s.endswith(".T"):
        return s + ".T"
    return s


def from_yf_symbol(yf_symbol):
    return str(yf_symbol).replace(".T", "")


# ==========================================
# OHLCV取得（キャッシュ付き）
# ==========================================

def fetch_historical_ohlcv(tickers, period="5y", use_cache=True, cache_days=1):
    """全銘柄のOHLCVを取得し、long format DataFrame を返す

    - キャッシュ: data/backtest_cache/ohlcv_{period}_{ticker_hash}_{YYYYMMDD}.parquet
    - cache_days=1: 1日以内のキャッシュは再利用
    - v2修正(2026-04-17): ticker_hash をキーに追加（ユニバース変更時のキャッシュ誤ヒット防止）
    """
    import yfinance as yf
    import hashlib as _hl

    # キャッシュキー: tickers の内容もハッシュに含めてユニバース別にキャッシュ分離
    ticker_hash = _hl.md5(",".join(sorted(tickers)).encode()).hexdigest()[:10]
    today = datetime.now().strftime("%Y%m%d")
    cache_file = CACHE_DIR / f"ohlcv_{period}_{ticker_hash}_{today}.parquet"

    if use_cache and cache_file.exists():
        logger.info(f"Cache hit: {cache_file.name}")
        return pd.read_parquet(cache_file)

    # 古いキャッシュも許容（前日まで。ticker_hash 一致のみ）
    if use_cache:
        for f in sorted(CACHE_DIR.glob(f"ohlcv_{period}_{ticker_hash}_*.parquet"), reverse=True):
            age = datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
            if age.days < cache_days:
                logger.info(f"Cache hit (aged): {f.name}")
                return pd.read_parquet(f)

    logger.info(f"Fetching {period} OHLCV for {len(tickers)} tickers...")
    yf_tickers = [to_yf_symbol(t) for t in tickers]

    # yfinance bulk (threads=True で高速化)
    data = yf.download(
        yf_tickers, period=period, interval="1d",
        group_by="ticker", threads=True, progress=False,
        auto_adjust=True,
    )

    rows = []
    # 単一銘柄の場合は MultiIndex にならない
    if len(yf_tickers) == 1:
        df = data.copy()
        df = df.reset_index()
        df["ticker"] = from_yf_symbol(yf_tickers[0])
        rows.append(df[["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]])
    else:
        for yf_t in yf_tickers:
            try:
                if yf_t not in data.columns.get_level_values(0):
                    continue
                df = data[yf_t].copy()
                if df.empty or df["Close"].dropna().empty:
                    continue
                df = df.reset_index()
                df["ticker"] = from_yf_symbol(yf_t)
                rows.append(df[["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]])
            except Exception as e:
                logger.warning(f"Skip {yf_t}: {e}")
                continue

    if not rows:
        raise RuntimeError("OHLCVデータを1銘柄も取得できませんでした")

    result = pd.concat(rows, ignore_index=True)
    result = result.rename(columns={"Date": "date"})
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    result = result.dropna(subset=["Close"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    if use_cache:
        try:
            result.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    logger.info(f"Fetched {len(result)} rows × {result['ticker'].nunique()} tickers")
    return result


# ==========================================
# テクニカル指標 & スコアリング
# ==========================================

def compute_indicators_at(ohlcv, asof_date, min_history=252):
    """asof_date 時点の各銘柄テクニカル指標を計算（未来データは一切見ない）"""
    df = ohlcv[ohlcv["date"] <= asof_date]
    if df.empty:
        return pd.DataFrame()

    results = []
    for ticker, tdf in df.groupby("ticker", sort=False):
        if len(tdf) < min_history:
            continue
        tdf = tdf.sort_values("date")
        close = tdf["Close"].values
        high = tdf["High"].values
        low = tdf["Low"].values
        volume = tdf["Volume"].values

        # RSI14
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_gain = pd.Series(gain).rolling(14).mean().iloc[-1]
        roll_loss = pd.Series(loss).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + roll_gain / roll_loss)) if roll_loss > 0 else 50.0

        # MACD (EMA12, EMA26, signal9)
        s = pd.Series(close)
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - macd_signal).iloc[-1]

        # 52週位置
        last_price = close[-1]
        high_52w = np.max(high[-252:])
        low_52w = np.min(low[-252:])
        if high_52w > low_52w:
            pos_52w = (last_price - low_52w) / (high_52w - low_52w) * 100
        else:
            pos_52w = 50.0

        # SMA
        sma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        sma50 = pd.Series(close).rolling(50).mean().iloc[-1]
        sma200 = pd.Series(close).rolling(200).mean().iloc[-1]

        # 出来高比
        vol20 = pd.Series(volume).rolling(20).mean().iloc[-1]
        vol_ratio = volume[-1] / vol20 if vol20 > 0 else 1.0

        # 5営業日リターン（モメンタム）
        mom_5d = (close[-1] / close[-6] - 1) * 100 if len(close) > 6 else 0

        results.append({
            "ticker": ticker,
            "price": float(last_price),
            "rsi_14": float(rsi),
            "macd_histogram": float(macd_hist),
            "pos_52w_pct": float(pos_52w),
            "sma_20": float(sma20),
            "sma_50": float(sma50),
            "sma_200": float(sma200),
            "vol_ratio": float(vol_ratio),
            "mom_5d_pct": float(mom_5d),
        })

    return pd.DataFrame(results)


def compute_score_mean_reversion(row):
    """逆張り型（現状）: 売られすぎ銘柄を拾う

    - RSI低・52週位置低で大きく加点
    - 下落からのリバウンド期待型
    """
    score = 50.0
    rsi = row["rsi_14"]
    if rsi <= 30: score += 15
    elif rsi <= 40: score += 10
    elif rsi <= 50: score += 5
    elif rsi >= 70: score -= 10
    elif rsi >= 60: score -= 5
    pos = row["pos_52w_pct"]
    if pos <= 15: score += 15
    elif pos <= 30: score += 10
    elif pos <= 50: score += 5
    elif pos >= 90: score -= 10
    if row["macd_histogram"] > 0: score += 10
    else: score -= 5
    p = row["price"]
    sma20, sma50, sma200 = row["sma_20"], row["sma_50"], row["sma_200"]
    if p > sma20 and sma20 > sma50: score += 5
    if p > sma200: score += 5
    if sma50 > sma200: score += 3
    if row["vol_ratio"] >= 1.5: score += 3
    if row["mom_5d_pct"] > 0: score += 2
    return max(0.0, min(100.0, score))


def compute_score_momentum(row):
    """モメンタム型: 強いトレンドに乗る

    - RSI 50-70 (sweet spot) と 52週高値近辺で加点
    - MACD陽線・価格>SMA・上昇配列 で加点
    """
    score = 50.0
    rsi = row["rsi_14"]
    # RSI sweet spot (50-70)
    if 50 <= rsi <= 70: score += 15
    elif 40 <= rsi < 50: score += 5
    elif rsi > 80: score -= 15  # 過熱はペナルティ
    elif rsi < 30: score -= 10  # 下落トレンドは回避

    # 52週位置: 高値近辺がベスト
    pos = row["pos_52w_pct"]
    if pos >= 80: score += 15
    elif pos >= 60: score += 10
    elif pos >= 40: score += 3
    elif pos <= 20: score -= 10  # 底値は買わない

    # MACD: 強い上昇モメンタム
    macd_hist = row["macd_histogram"]
    if macd_hist > 0: score += 15
    else: score -= 10

    # MA 配列: 完全な上昇配列 (Price > SMA20 > SMA50 > SMA200) が理想
    p = row["price"]
    sma20, sma50, sma200 = row["sma_20"], row["sma_50"], row["sma_200"]
    if p > sma20 > sma50 > sma200: score += 12  # Golden alignment
    elif p > sma50 > sma200: score += 8
    elif p > sma200: score += 3
    else: score -= 5  # 長期下降トレンド

    # 出来高: モメンタムを伴う急増
    vol = row["vol_ratio"]
    if vol >= 2.0: score += 5
    elif vol >= 1.3: score += 3

    # 5日モメンタム: 直近上昇
    mom = row["mom_5d_pct"]
    if mom > 2: score += 5
    elif mom > 0: score += 2
    elif mom < -3: score -= 5

    return max(0.0, min(100.0, score))


def compute_score_quality_momentum(row):
    """質の高い順張り: モメンタム＋長期トレンド裏付け

    - モメンタムの中でも「長期上昇が続いているもの」に絞る
    - 52週高値更新を軸に、RSIは過熱しすぎず、出来高も裏付けあり
    """
    score = 50.0
    rsi = row["rsi_14"]
    # 過熱していないモメンタム: 50-65
    if 50 <= rsi <= 65: score += 18
    elif 45 <= rsi < 50: score += 8
    elif rsi > 75: score -= 20  # 明確な過熱はむしろ危険
    elif rsi < 40: score -= 10

    # 52週位置: 最高値付近 (長期上昇のシグナル)
    pos = row["pos_52w_pct"]
    if pos >= 85: score += 18
    elif pos >= 70: score += 12
    elif pos >= 50: score += 3
    elif pos <= 30: score -= 15

    # MACD
    if row["macd_histogram"] > 0: score += 10
    else: score -= 8

    # MA alignment (厳格版): 完全な上昇配列必須
    p = row["price"]
    sma20, sma50, sma200 = row["sma_20"], row["sma_50"], row["sma_200"]
    if p > sma20 > sma50 > sma200: score += 15
    elif sma50 > sma200 and p > sma50: score += 5
    else: score -= 10  # 長期下降はペナルティ大

    # 出来高の裏付け (モメンタム+出来高は強いシグナル)
    if row["vol_ratio"] >= 1.3: score += 5

    # 直近モメンタム (軽め)
    mom = row["mom_5d_pct"]
    if mom > 0: score += 3
    elif mom < -5: score -= 5

    return max(0.0, min(100.0, score))


# 旧名の後方互換 alias
compute_tech_score = compute_score_mean_reversion


# 戦略カタログ
SCORING_STRATEGIES = {
    "mean_reversion": {
        "name": "逆張り型（売られすぎ買い）",
        "short": "逆張り",
        "desc": "RSI低・52週位置低・MACD反転で加点。下落からのリバウンド狙い。現状の買い推奨TOP10と同じロジック",
        "fn": compute_score_mean_reversion,
    },
    "momentum": {
        "name": "モメンタム型（順張り）",
        "short": "モメンタム",
        "desc": "RSI 50-70・52週位置高・上昇配列で加点。強いトレンドに乗る",
        "fn": compute_score_momentum,
    },
    "quality_momentum": {
        "name": "質の高い順張り",
        "short": "質モメンタム",
        "desc": "モメンタム＋長期上昇配列＋過熱回避＋出来高裏付け。より厳しい条件で絞る",
        "fn": compute_score_quality_momentum,
    },
}


# ==========================================
# バックテスト本体
# ==========================================

def _rebalance_dates(start, end, freq, warmup_days=260):
    """ウォームアップを除いたリバランス日リスト"""
    start_eff = start + pd.Timedelta(days=warmup_days)
    if freq == "weekly":
        dates = pd.date_range(start_eff, end, freq="W-FRI")
    elif freq == "monthly":
        dates = pd.date_range(start_eff, end, freq="ME")  # Month End
    else:
        raise ValueError(f"Unknown freq: {freq}")
    return list(dates)


def run_backtest(
    tickers=None,
    period="5y",
    rebalance_freq="monthly",
    top_n=10,
    initial_capital=1_000_000,
    ohlcv=None,
    cost_bps=10,
    strategy="mean_reversion",
    take_profit_pct=None,   # +X% で利確（None なら発動なし）
    stop_loss_pct=None,     # -Y% で損切り
    trailing_stop_pct=None, # 最高値から -Z% で利確
):
    """メインのバックテスト関数

    Args:
        tickers: List[str]. None なら default ticker リスト
        period: "1y", "3y", "5y"
        rebalance_freq: "weekly" or "monthly"
        top_n: 保有銘柄数 (equal weight)
        initial_capital: 初期資金（円）
        ohlcv: 既に取得済みの OHLCV DataFrame（渡せば再取得しない）
        cost_bps: 1往復あたりの取引コスト (bp = 1/10000)
        strategy: SCORING_STRATEGIES キー（"mean_reversion"/"momentum"/"quality_momentum"）
        take_profit_pct: +X% で早期利確（例: 20 → +20%で売却）
        stop_loss_pct: -Y% で損切り（例: 10 → -10%で売却）
        trailing_stop_pct: 最高値から -Z% 下落で売却（例: 5 → トレーリングストップ5%）

    Returns:
        dict: equity_curve, trades, metrics, rebalance_dates, config
    """
    if tickers is None:
        tickers = get_default_tickers()
    if ohlcv is None:
        ohlcv = fetch_historical_ohlcv(tickers, period=period)

    # v2修正(2026-04-17): 指定された tickers で OHLCV をフィルタ
    # （キャッシュに他のユニバースのデータが混在していても安全）
    ticker_set = set(tickers)
    ohlcv = ohlcv[ohlcv["ticker"].isin(ticker_set)].copy()
    if ohlcv.empty:
        raise RuntimeError(f"指定ユニバースに合致するOHLCVデータがありません。tickers={len(tickers)}, ohlcv銘柄数=0")

    # 戦略確定
    if strategy not in SCORING_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(SCORING_STRATEGIES.keys())}")
    score_fn = SCORING_STRATEGIES[strategy]["fn"]

    dates_list = _rebalance_dates(ohlcv["date"].min(), ohlcv["date"].max(), rebalance_freq)
    if not dates_list:
        raise RuntimeError(f"リバランス日が0件。期間={period}")

    # 高速アクセス用: (ticker, date) → Close の辞書
    price_lookup = {
        (r.ticker, r.date): r.Close for r in ohlcv[["ticker", "date", "Close"]].itertuples(index=False)
    }
    # ticker → 時系列DataFrame
    ticker_series = {tk: sub.sort_values("date").reset_index(drop=True)
                     for tk, sub in ohlcv.groupby("ticker", sort=False)}

    cost_factor = 1 - cost_bps / 10000

    cash = float(initial_capital)
    holdings = {}  # ticker -> dict(shares, entry_price, entry_date, high_since_entry)
    equity_curve = []
    trade_log = []

    def _sell(tk, sell_date, sell_price, reason):
        """売却処理"""
        nonlocal cash
        h = holdings[tk]
        shares = h["shares"]
        proceeds = shares * sell_price * cost_factor
        cash += proceeds
        trade_log.append({
            "date": sell_date, "ticker": tk, "action": "sell",
            "shares": shares, "price": sell_price, "value": proceeds,
            "reason": reason,
            "entry_price": h["entry_price"],
            "holding_days": (pd.Timestamp(sell_date) - pd.Timestamp(h["entry_date"])).days,
            "return_pct": (sell_price / h["entry_price"] - 1) * 100,
        })
        del holdings[tk]

    for rb_idx, rb_date in enumerate(dates_list):
        # リバランス日の最寄り取引日
        valid = ohlcv[ohlcv["date"] <= rb_date]
        if valid.empty:
            continue
        asof = valid["date"].max()

        # --- STEP 1: リバランス間の日次チェック (売り時ルールがある場合のみ) ---
        if take_profit_pct is not None or stop_loss_pct is not None or trailing_stop_pct is not None:
            # 前回 asof 〜 今回 asof の間の日次で exit 条件をチェック
            prev_asof = equity_curve[-1]["date"] if equity_curve else ohlcv["date"].min()
            trading_days = sorted({d for d in ohlcv["date"].unique() if prev_asof < d < asof})
            for d in trading_days:
                for tk in list(holdings.keys()):
                    px = price_lookup.get((tk, d))
                    if px is None:
                        continue
                    h = holdings[tk]
                    h["high_since_entry"] = max(h["high_since_entry"], px)
                    ret = (px / h["entry_price"] - 1) * 100
                    trailing_ret = (px / h["high_since_entry"] - 1) * 100
                    if take_profit_pct is not None and ret >= take_profit_pct:
                        _sell(tk, d, px, "take_profit")
                    elif stop_loss_pct is not None and ret <= -stop_loss_pct:
                        _sell(tk, d, px, "stop_loss")
                    elif trailing_stop_pct is not None and trailing_ret <= -trailing_stop_pct:
                        _sell(tk, d, px, "trailing_stop")

        # --- STEP 2: 現在価値を確定（売却直前の time-based リバランス） ---
        portfolio_value = cash
        for tk, h in holdings.items():
            px = price_lookup.get((tk, asof))
            if px is not None:
                portfolio_value += h["shares"] * px
        equity_curve.append({"date": asof, "equity": portfolio_value})

        # --- STEP 3: 既存保有を全売却 (time-based) ---
        for tk in list(holdings.keys()):
            px = price_lookup.get((tk, asof))
            if px is not None:
                _sell(tk, asof, float(px), "rebalance")
            else:
                # データが無い場合: 抹消（事実上0円処理）
                del holdings[tk]

        # --- STEP 4: テクニカル計算 & TOP_N 選定 ---
        inds = compute_indicators_at(ohlcv, asof)
        if inds.empty:
            continue
        inds["score"] = inds.apply(score_fn, axis=1)
        top = inds.nlargest(top_n, "score")

        if len(top) > 0:
            per_cash = cash / len(top) * cost_factor
            for _, r in top.iterrows():
                shares = int(per_cash / r["price"])
                if shares > 0:
                    cost = shares * r["price"]
                    cash -= cost / cost_factor
                    holdings[r["ticker"]] = {
                        "shares": shares,
                        "entry_price": float(r["price"]),
                        "entry_date": asof,
                        "high_since_entry": float(r["price"]),
                    }
                    trade_log.append({
                        "date": asof, "ticker": r["ticker"], "action": "buy",
                        "shares": shares, "price": float(r["price"]), "value": cost,
                        "score": float(r["score"]),
                    })

    # 最終日時点の評価
    last_date = ohlcv["date"].max()
    final_equity = cash
    for tk, h in holdings.items():
        px = price_lookup.get((tk, last_date))
        if px is not None:
            final_equity += h["shares"] * px
    equity_curve.append({"date": last_date, "equity": final_equity})

    eq_df = pd.DataFrame(equity_curve).drop_duplicates(subset="date", keep="last")
    trade_df = pd.DataFrame(trade_log)
    metrics = _compute_metrics(eq_df)

    return {
        "equity_curve": eq_df,
        "trades": trade_df,
        "metrics": metrics,
        "rebalance_dates": dates_list,
        "strategy": strategy,
        "config": {
            "period": period, "rebalance_freq": rebalance_freq,
            "top_n": top_n, "initial_capital": initial_capital,
            "cost_bps": cost_bps, "n_tickers": len(tickers),
            "strategy": strategy,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct,
            "trailing_stop_pct": trailing_stop_pct,
        },
    }


def _compute_metrics(eq_df):
    """資産推移からKPIを算出"""
    if eq_df.empty or len(eq_df) < 2:
        return {}

    eq_df = eq_df.sort_values("date").reset_index(drop=True)
    initial = float(eq_df["equity"].iloc[0])
    final = float(eq_df["equity"].iloc[-1])
    total_return = (final - initial) / initial * 100

    days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days
    years = max(days / 365.25, 0.01)
    annual_return = ((final / initial) ** (1 / years) - 1) * 100

    returns = eq_df["equity"].pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        periods_per_year = len(returns) / years
        annual_vol = returns.std() * np.sqrt(periods_per_year) * 100
        sharpe = (annual_return - 2) / annual_vol if annual_vol > 0 else 0.0
    else:
        annual_vol, sharpe = 0.0, 0.0

    cum_max = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - cum_max) / cum_max * 100
    max_dd = float(drawdown.min())

    win_rate = float((returns > 0).sum() / len(returns) * 100) if len(returns) > 0 else 0.0

    return {
        "initial": round(initial, 2),
        "final": round(final, 2),
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "annual_volatility_pct": round(annual_vol, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "n_rebalances": int(len(eq_df) - 1),
        "period_days": int(days),
        "period_years": round(years, 2),
    }


# ==========================================
# ベンチマーク
# ==========================================

def fetch_benchmark(symbol, period="5y"):
    """ベンチマーク銘柄の推移を取得"""
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})


def benchmark_equity_curve(symbol, period, initial_capital):
    """ベンチマーク価格を初期資金で買い持ちした場合の資産推移"""
    bench = fetch_benchmark(symbol, period=period)
    if bench.empty:
        return pd.DataFrame()
    first = bench["close"].iloc[0]
    bench["equity"] = bench["close"] / first * initial_capital
    return bench[["date", "equity"]]


# ==========================================
# 保存
# ==========================================

def save_results(result, tag):
    """結果を parquet と JSON に保存"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = RESULT_DIR / f"{tag}_{ts}"

    result["equity_curve"].to_parquet(prefix.with_suffix(".equity.parquet"))
    if not result["trades"].empty:
        result["trades"].to_parquet(prefix.with_suffix(".trades.parquet"))

    meta = {
        "tag": tag, "generated_at": ts,
        "config": result["config"], "metrics": result["metrics"],
    }
    with open(prefix.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    # 最新へのシンボリックリンク代替: *_latest.* を上書き
    latest = RESULT_DIR / f"{tag}_latest"
    result["equity_curve"].to_parquet(latest.with_suffix(".equity.parquet"))
    if not result["trades"].empty:
        result["trades"].to_parquet(latest.with_suffix(".trades.parquet"))
    with open(latest.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    return prefix


def load_latest(tag):
    """tag の最新結果を読込。無ければ None"""
    latest = RESULT_DIR / f"{tag}_latest"
    if not latest.with_suffix(".json").exists():
        return None
    with open(latest.with_suffix(".json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    eq = pd.read_parquet(latest.with_suffix(".equity.parquet"))
    trades_path = latest.with_suffix(".trades.parquet")
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    return {
        "equity_curve": eq, "trades": trades,
        "metrics": meta["metrics"], "config": meta["config"],
        "generated_at": meta["generated_at"],
    }


# ==========================================
# CLI
# ==========================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="5y", choices=["1y", "3y", "5y"])
    parser.add_argument("--freq", default="both", choices=["weekly", "monthly", "both"])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--capital", type=int, default=1_000_000)
    args = parser.parse_args()

    tickers = get_default_tickers()
    if not tickers:
        print("❌ data/latest_analysis.csv が見つかりません。先に run.py を実行してください。")
        return

    print(f"🚀 バックテスト開始: {len(tickers)}銘柄, {args.period}")
    ohlcv = fetch_historical_ohlcv(tickers, period=args.period)
    print(f"   OHLCV取得完了: {len(ohlcv)}行")

    freqs = ["weekly", "monthly"] if args.freq == "both" else [args.freq]

    for freq in freqs:
        tag = f"tech_top{args.top_n}_{args.period}_{freq}"
        print(f"\n━━━ {tag} ━━━")
        result = run_backtest(
            tickers=tickers, period=args.period, rebalance_freq=freq,
            top_n=args.top_n, initial_capital=args.capital, ohlcv=ohlcv,
        )
        m = result["metrics"]
        print(f"  期間: {m['period_years']}年 / リバランス: {m['n_rebalances']}回")
        print(f"  総リターン: {m['total_return_pct']:+.2f}%  (年率 {m['annual_return_pct']:+.2f}%)")
        print(f"  Sharpe:    {m['sharpe_ratio']}  / 最大DD: {m['max_drawdown_pct']}%")
        print(f"  勝率:      {m['win_rate_pct']}%  / ボラ: {m['annual_volatility_pct']}%")
        save_results(result, tag)
        print(f"  ✅ 保存: data/backtest_results/{tag}_latest.*")


if __name__ == "__main__":
    main()
