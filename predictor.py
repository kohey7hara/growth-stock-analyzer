"""
predictor.py - レジーム適応型予測エンジン v5

v4→v5の改善ポイント:
  1. ファンダメンタル分析ファクター追加 (F6)
     - アナリスト目標株価との乖離 (upside_pct)
     - PEG比率（成長対比割安度）
     - 売上成長率
  2. マクロ環境ファクター追加 (F7)
     - VIX（恐怖指数）レベル
     - 市場全体のトレンド (S&P500/日経225)
     - 金利動向 (10年国債)
  3. 7ファクター体制でより多面的な分析
  4. レジーム判定にマクロ指標を統合

v4からの継続:
  - レジーム判定（SMAベース）
  - トレンド相場でオシレーター逆張り抑制
  - 動的ウェイト配分
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent


def _resolve_ticker(ticker):
    """日本株ティッカーの.T補完"""
    if ticker.endswith(".T"):
        return ticker
    stripped = ticker.replace(".", "")
    if stripped.isdigit():
        return f"{ticker}.T"
    return ticker


def _fetch_history(ticker, period="2y"):
    """yfinanceから過去データを取得"""
    import yfinance as yf
    resolved = _resolve_ticker(ticker)
    candidates = [resolved] if resolved == ticker else [resolved, ticker]
    for tk in candidates:
        for p in [period, "1y", "6mo"]:
            try:
                hist = yf.Ticker(tk).history(period=p)
                if not hist.empty and len(hist) >= 30:
                    df = hist[["Close", "High", "Low", "Volume"]].dropna(subset=["Close"])
                    if len(df) >= 30:
                        return df
            except Exception:
                continue
    return pd.DataFrame()


# ==========================================
# レジーム判定 — SMAベースで堅牢に
# ==========================================

def _detect_regime(close, high, low, volume):
    """
    レジーム（相場局面）を判定する。
    SMAの配置と傾きで判定（ADXより安定）。

    Returns:
        regime: "STRONG_TREND", "WEAK_TREND", "RANGE"
        trend_direction: +1 (上昇), -1 (下落), 0 (不明)
        strength: 0.0-1.0
    """
    n = len(close)
    if n < 50:
        return "RANGE", 0, 0.3

    sma5 = np.mean(close[-5:])
    sma20 = np.mean(close[-20:])
    sma50 = np.mean(close[-50:])

    # SMA20の傾き（10日前と比較）
    sma20_prev = np.mean(close[-30:-10])
    sma20_slope = (sma20 - sma20_prev) / sma20_prev * 100  # %変化

    # SMA50の傾き（20日前と比較）
    if n >= 70:
        sma50_prev = np.mean(close[-70:-20])
        sma50_slope = (sma50 - sma50_prev) / sma50_prev * 100
    else:
        sma50_slope = sma20_slope * 0.5

    # 10日リターンの方向
    ret_10d = (close[-1] / close[-10] - 1) * 100
    # 20日リターン
    ret_20d = (close[-1] / close[-20] - 1) * 100

    # パーフェクトオーダー判定
    perfect_up = (close[-1] > sma5 > sma20 > sma50)
    perfect_down = (close[-1] < sma5 < sma20 < sma50)

    # トレンド強度スコア (0-8, より厳密に)
    trend_score = 0

    # 価格 vs SMA
    if close[-1] > sma20:
        trend_score += 1
    elif close[-1] < sma20:
        trend_score -= 1

    if sma5 > sma20:
        trend_score += 1
    elif sma5 < sma20:
        trend_score -= 1

    if sma20 > sma50:
        trend_score += 1
    elif sma20 < sma50:
        trend_score -= 1

    # SMA傾き（閾値を厳しく）
    if abs(sma20_slope) > 3:
        trend_score += np.sign(sma20_slope) * 1.5
    elif abs(sma20_slope) > 1.5:
        trend_score += np.sign(sma20_slope) * 0.5

    if abs(sma50_slope) > 2:
        trend_score += np.sign(sma50_slope)

    # 直近リターン（閾値を厳しく）
    if abs(ret_10d) > 5:
        trend_score += np.sign(ret_10d)
    if abs(ret_20d) > 8:
        trend_score += np.sign(ret_20d)

    # 20日ボラティリティ対比リターン（実質的なシャープレシオ）
    daily_rets = np.diff(close[-21:]) / close[-21:-1]
    vol_20 = np.std(daily_rets) * 100
    if vol_20 > 0 and abs(ret_20d) / vol_20 < 0.3:
        # ボラに対してリターンが小さい → レンジ寄り
        trend_score *= 0.6

    # 判定（閾値を引き上げ）
    abs_score = abs(trend_score)
    direction = 1 if trend_score > 0 else (-1 if trend_score < 0 else 0)

    if (abs_score >= 5 or perfect_up or perfect_down) and abs(ret_20d) > 3:
        regime = "STRONG_TREND"
        strength = min(abs_score / 8, 1.0)
    elif abs_score >= 3:
        regime = "WEAK_TREND"
        strength = abs_score / 8
    else:
        regime = "RANGE"
        strength = 0.2
        direction = 0

    return regime, direction, strength


# ==========================================
# ファクター関数 — 各々 -1.0 〜 +1.0
# ==========================================

def _factor_trend(close):
    """F1: トレンドファクター — SMA5/20/50の配置と傾き"""
    n = len(close)
    score = 0.0

    # 短期: 終値 vs SMA5
    if n >= 5:
        sma5 = np.mean(close[-5:])
        score += 0.2 if close[-1] > sma5 else -0.2

    # 中期: SMA5 vs SMA20 + SMA5の傾き
    if n >= 20:
        sma5 = np.mean(close[-5:])
        sma20 = np.mean(close[-20:])
        score += 0.3 if sma5 > sma20 else -0.3

        # SMA5の傾き（方向の加速度）
        if n >= 10:
            sma5_prev = np.mean(close[-10:-5])
            slope = (sma5 - sma5_prev) / sma5_prev * 100
            score += np.clip(slope / 3, -0.2, 0.2)

    # 長期: SMA20 vs SMA50
    if n >= 50:
        sma20 = np.mean(close[-20:])
        sma50 = np.mean(close[-50:])
        score += 0.3 if sma20 > sma50 else -0.3

    return np.clip(score, -1.0, 1.0)


def _factor_oscillator(close, regime, trend_dir):
    """
    F2: オシレーターファクター — RSI + ストキャスティクス
    ★ v4改善: トレンド方向にフィルタリング
    強い下落トレンドでRSI低→「買い」を出さない
    """
    n = len(close)
    if n < 14:
        return 0.0

    # RSI(14)
    deltas = np.diff(close[-15:])
    gain = np.mean(np.maximum(deltas, 0))
    loss = np.mean(np.maximum(-deltas, 0))
    rsi = 100 - (100 / (1 + gain / max(loss, 1e-10)))

    # 基本RSIスコア
    if rsi < 25:
        rsi_score = 0.8
    elif rsi < 35:
        rsi_score = 0.4
    elif rsi < 45:
        rsi_score = 0.1
    elif rsi > 75:
        rsi_score = -0.8
    elif rsi > 65:
        rsi_score = -0.4
    elif rsi > 55:
        rsi_score = -0.1
    else:
        rsi_score = 0.0

    # ★ トレンドフィルター: 強トレンド中の逆張りシグナルを抑制
    if regime == "STRONG_TREND":
        if trend_dir < 0 and rsi_score > 0:
            # 下落トレンド中のRSI買いシグナル → 大幅抑制
            rsi_score *= 0.1  # ほぼ無効化
        elif trend_dir > 0 and rsi_score < 0:
            # 上昇トレンド中のRSI売りシグナル → 大幅抑制
            rsi_score *= 0.1
    elif regime == "WEAK_TREND":
        if trend_dir < 0 and rsi_score > 0:
            rsi_score *= 0.4  # 弱い抑制
        elif trend_dir > 0 and rsi_score < 0:
            rsi_score *= 0.4

    # ストキャスティクス %K
    if n >= 14:
        h14 = np.max(close[-14:])
        l14 = np.min(close[-14:])
        k = (close[-1] - l14) / max(h14 - l14, 1e-10) * 100
        if k < 20:
            stoch_score = 0.5
        elif k > 80:
            stoch_score = -0.5
        else:
            stoch_score = 0.0

        # 同じトレンドフィルター
        if regime == "STRONG_TREND":
            if trend_dir < 0 and stoch_score > 0:
                stoch_score *= 0.1
            elif trend_dir > 0 and stoch_score < 0:
                stoch_score *= 0.1
    else:
        stoch_score = 0.0

    return np.clip(rsi_score * 0.6 + stoch_score * 0.4, -1.0, 1.0)


def _factor_momentum(close):
    """F3: モメンタムファクター — ROCベース + 連続性"""
    n = len(close)
    if n < 20:
        return 0.0

    scores = []

    # 5日ROC
    if n > 5:
        r5 = (close[-1] / close[-6] - 1) * 100
        scores.append(np.clip(r5 / 5, -1, 1) * 0.35)

    # 10日ROC
    if n > 10:
        r10 = (close[-1] / close[-11] - 1) * 100
        scores.append(np.clip(r10 / 8, -1, 1) * 0.35)

    # 20日ROC
    if n > 20:
        r20 = (close[-1] / close[-21] - 1) * 100
        scores.append(np.clip(r20 / 12, -1, 1) * 0.30)

    return np.clip(sum(scores), -1.0, 1.0) if scores else 0.0


def _factor_volume(close, volume):
    """F4: ボリュームファクター — 出来高と価格の関係"""
    n = len(close)
    if n < 20:
        return 0.0

    # 価格方向
    price_dir = 1 if close[-1] > close[-5] else -1

    # 出来高比率
    vol_5 = np.mean(volume[-5:])
    vol_20 = np.mean(volume[-20:])
    vol_ratio = vol_5 / max(vol_20, 1)

    # 高出来高 + 上昇 = 買い確認、高出来高 + 下落 = 売り確認
    if vol_ratio > 1.5:
        return np.clip(price_dir * 0.6, -1, 1)
    elif vol_ratio > 1.2:
        return np.clip(price_dir * 0.3, -1, 1)
    elif vol_ratio < 0.7:
        return 0.0  # 出来高減少 = シグナルなし
    else:
        return np.clip(price_dir * 0.1, -1, 1)


def _factor_reversion(close, regime, trend_dir):
    """
    F5: 回帰ファクター — SMA乖離（逆張り）
    ★ v4改善: 強トレンド中は回帰シグナルを大幅抑制
    """
    n = len(close)
    if n < 20:
        return 0.0

    sma20 = np.mean(close[-20:])
    dev = (close[-1] - sma20) / sma20 * 100

    # 基本の回帰スコア
    if dev < -10:
        rev_score = 0.7
    elif dev < -5:
        rev_score = 0.3
    elif dev > 10:
        rev_score = -0.7
    elif dev > 5:
        rev_score = -0.3
    else:
        rev_score = 0.0

    # ★ トレンドフィルター
    if regime == "STRONG_TREND":
        # 強トレンド中: 回帰シグナルをほぼ無効化
        # 下落トレンドで「売られすぎ→反発」と期待するのは危険
        if (trend_dir < 0 and rev_score > 0) or (trend_dir > 0 and rev_score < 0):
            rev_score *= 0.1  # 逆方向の回帰シグナルは90%カット
        else:
            rev_score *= 0.3  # 同方向でも抑制（トレンドフォローに任せる）
    elif regime == "WEAK_TREND":
        if (trend_dir < 0 and rev_score > 0) or (trend_dir > 0 and rev_score < 0):
            rev_score *= 0.3

    return np.clip(rev_score, -1.0, 1.0)


def _calc_volatility(close):
    """ボラティリティ計算"""
    if len(close) < 20:
        return 0.02
    daily_ret = np.diff(close) / close[:-1]
    vol = np.std(daily_ret[-20:])
    return max(vol, 0.005)


# ==========================================
# F6: ファンダメンタルファクター（CSVデータ活用）
# ==========================================

def _load_stock_fundamentals(ticker):
    """latest_data.csv からファンダメンタルデータを読み込む"""
    try:
        path = BASE_DIR / "data" / "latest_data.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        row = df[df["ticker"] == ticker]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    except Exception:
        return None


def _factor_fundamental(ticker):
    """
    F6: ファンダメンタルファクター
    - アナリスト目標株価 vs 現在株価（upside_pct）
    - PEGレシオ（低い方が割安で成長性あり）
    - 売上成長率
    - 52週位置（極端な高値圏/安値圏を検知）
    Returns: -1.0 ~ +1.0
    """
    data = _load_stock_fundamentals(ticker)
    if data is None:
        return 0.0

    scores = []

    # アナリスト目標株価アップサイド
    upside = data.get("upside_pct")
    if upside is not None and pd.notna(upside):
        upside = float(upside)
        if upside > 30:
            scores.append(0.7)
        elif upside > 15:
            scores.append(0.4)
        elif upside > 5:
            scores.append(0.2)
        elif upside < -20:
            scores.append(-0.6)
        elif upside < -10:
            scores.append(-0.3)
        elif upside < 0:
            scores.append(-0.1)
        else:
            scores.append(0.0)

    # PEGレシオ（低いほど割安成長）
    peg = data.get("peg_ratio")
    if peg is not None and pd.notna(peg):
        peg = float(peg)
        if 0 < peg < 0.8:
            scores.append(0.5)  # 割安成長
        elif 0 < peg < 1.2:
            scores.append(0.2)  # 適正
        elif peg > 3:
            scores.append(-0.4)  # 過大評価
        elif peg > 2:
            scores.append(-0.2)
        elif peg < 0:
            scores.append(-0.3)  # 赤字

    # 売上成長率
    growth = data.get("revenue_growth_pct")
    if growth is not None and pd.notna(growth):
        growth = float(growth)
        if growth > 40:
            scores.append(0.5)
        elif growth > 20:
            scores.append(0.3)
        elif growth > 10:
            scores.append(0.1)
        elif growth < -10:
            scores.append(-0.4)
        elif growth < 0:
            scores.append(-0.2)
        else:
            scores.append(0.0)

    # 52週位置（極端な位置のみ）
    pos_52w = data.get("pos_52w_pct")
    if pos_52w is not None and pd.notna(pos_52w):
        pos_52w = float(pos_52w)
        if pos_52w > 95:
            scores.append(-0.3)  # 52週高値圏 → 過熱
        elif pos_52w < 10:
            scores.append(0.3)  # 52週安値圏 → 反発期待

    if not scores:
        return 0.0
    return np.clip(np.mean(scores), -1.0, 1.0)


# ==========================================
# F7: マクロ環境ファクター
# ==========================================

def _load_macro_data():
    """latest_macro.csv からマクロデータを読み込む"""
    try:
        path = BASE_DIR / "data" / "latest_macro.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        result = {}
        for _, row in df.iterrows():
            tk = row.get("ticker", "")
            result[tk] = {
                "value": row.get("current_value"),
                "change_1d": row.get("change_1d_pct"),
                "change_1m": row.get("change_1m_pct"),
                "change_3m": row.get("change_3m_pct"),
            }
        return result
    except Exception:
        return {}


def _factor_macro(ticker, market="US"):
    """
    F7: マクロ環境ファクター
    - VIXレベル（高い→リスクオフ、低い→リスクオン）
    - 市場インデックスのトレンド（S&P500 or 日経225）
    - 金利動向（上昇→株にネガティブ）
    Returns: -1.0 ~ +1.0
    """
    macro = _load_macro_data()
    if not macro:
        return 0.0

    scores = []

    # VIX
    vix = macro.get("^VIX", {})
    vix_val = vix.get("value")
    vix_1m = vix.get("change_1m")
    if vix_val is not None and pd.notna(vix_val):
        vix_val = float(vix_val)
        if vix_val > 35:
            scores.append(-0.6)  # 恐怖レベル高 → 下落バイアス
        elif vix_val > 25:
            scores.append(-0.3)
        elif vix_val < 15:
            scores.append(0.3)  # 低恐怖 → 上昇バイアス
        elif vix_val < 20:
            scores.append(0.1)
        else:
            scores.append(0.0)

        # VIXの方向（下落中→安心感増加）
        if vix_1m is not None and pd.notna(vix_1m):
            vix_1m = float(vix_1m)
            if vix_1m < -20:
                scores.append(0.3)  # VIX急低下 → 楽観
            elif vix_1m > 30:
                scores.append(-0.3)  # VIX急上昇 → 恐怖

    # 市場インデックストレンド
    if market == "JP":
        idx = macro.get("^N225", {})
    else:
        idx = macro.get("^GSPC", {})

    idx_1m = idx.get("change_1m")
    idx_3m = idx.get("change_3m")
    if idx_1m is not None and pd.notna(idx_1m):
        idx_1m = float(idx_1m)
        if idx_1m > 5:
            scores.append(0.4)
        elif idx_1m > 2:
            scores.append(0.2)
        elif idx_1m < -5:
            scores.append(-0.4)
        elif idx_1m < -2:
            scores.append(-0.2)
        else:
            scores.append(0.0)

    # 金利動向
    tnx = macro.get("^TNX", {})
    tnx_1m = tnx.get("change_1m")
    if tnx_1m is not None and pd.notna(tnx_1m):
        tnx_1m = float(tnx_1m)
        if tnx_1m > 10:
            scores.append(-0.3)  # 金利急上昇 → 株にネガティブ
        elif tnx_1m > 5:
            scores.append(-0.1)
        elif tnx_1m < -10:
            scores.append(0.2)  # 金利低下 → 株にポジティブ

    if not scores:
        return 0.0
    return np.clip(np.mean(scores), -1.0, 1.0)


# ==========================================
# 予測コア — レジーム適応型 v5
# ==========================================

def _predict_direction(close, volume, horizon_days):
    """
    v3互換のインターフェース（high/lowなしで呼べる）。
    内部でclose からhigh/lowを推定。
    """
    # high/lowがない場合はcloseから推定
    high = close * 1.005  # 簡易推定
    low = close * 0.995
    return _predict_direction_full(close, high, low, volume, horizon_days)


def _predict_direction_full(close, high, low, volume, horizon_days,
                            ticker="", market="US"):
    """
    レジーム適応型予測コア v5（7ファクター）。

    v4→v5:
    - ファンダメンタル(F6) + マクロ(F7)ファクター追加
    - 長期予測でファンダメンタルの重みを大きくする
    - マクロ環境を全ホライズンで考慮
    """
    # Step 1: レジーム判定
    regime, trend_dir, strength = _detect_regime(close, high, low, volume)

    # Step 2: 7ファクターを計算
    f_trend = _factor_trend(close)
    f_osc = _factor_oscillator(close, regime, trend_dir)
    f_mom = _factor_momentum(close)
    f_vol = _factor_volume(close, volume)
    f_rev = _factor_reversion(close, regime, trend_dir)
    f_fund = _factor_fundamental(ticker)  # ★ NEW: ファンダメンタル
    f_macro = _factor_macro(ticker, market)  # ★ NEW: マクロ環境

    factors = {
        "トレンド": f_trend,
        "オシレーター": f_osc,
        "モメンタム": f_mom,
        "ボリューム": f_vol,
        "回帰": f_rev,
        "ファンダメンタル": f_fund,
        "マクロ": f_macro,
    }

    # Step 3: レジーム×ホライズンに応じた7ファクター重み付け
    if regime == "STRONG_TREND":
        if horizon_days <= 1:
            weights = {"トレンド": 0.28, "オシレーター": 0.04, "モメンタム": 0.28,
                       "ボリューム": 0.15, "回帰": 0.03,
                       "ファンダメンタル": 0.07, "マクロ": 0.15}
        elif horizon_days <= 7:
            weights = {"トレンド": 0.30, "オシレーター": 0.04, "モメンタム": 0.18,
                       "ボリューム": 0.10, "回帰": 0.08,
                       "ファンダメンタル": 0.15, "マクロ": 0.15}
        else:
            weights = {"トレンド": 0.15, "オシレーター": 0.05, "モメンタム": 0.08,
                       "ボリューム": 0.05, "回帰": 0.22,
                       "ファンダメンタル": 0.30, "マクロ": 0.15}
    elif regime == "WEAK_TREND":
        if horizon_days <= 1:
            weights = {"トレンド": 0.22, "オシレーター": 0.10, "モメンタム": 0.18,
                       "ボリューム": 0.10, "回帰": 0.10,
                       "ファンダメンタル": 0.12, "マクロ": 0.18}
        elif horizon_days <= 7:
            weights = {"トレンド": 0.20, "オシレーター": 0.10, "モメンタム": 0.10,
                       "ボリューム": 0.08, "回帰": 0.17,
                       "ファンダメンタル": 0.18, "マクロ": 0.17}
        else:
            weights = {"トレンド": 0.12, "オシレーター": 0.05, "モメンタム": 0.05,
                       "ボリューム": 0.05, "回帰": 0.28,
                       "ファンダメンタル": 0.30, "マクロ": 0.15}
    else:  # RANGE
        if horizon_days <= 1:
            weights = {"トレンド": 0.10, "オシレーター": 0.22, "モメンタム": 0.05,
                       "ボリューム": 0.10, "回帰": 0.22,
                       "ファンダメンタル": 0.13, "マクロ": 0.18}
        elif horizon_days <= 7:
            weights = {"トレンド": 0.08, "オシレーター": 0.15, "モメンタム": 0.03,
                       "ボリューム": 0.05, "回帰": 0.29,
                       "ファンダメンタル": 0.22, "マクロ": 0.18}
        else:
            weights = {"トレンド": 0.05, "オシレーター": 0.08, "モメンタム": 0.02,
                       "ボリューム": 0.03, "回帰": 0.32,
                       "ファンダメンタル": 0.35, "マクロ": 0.15}

    # 加重平均方向スコア
    direction_score = sum(factors[k] * weights[k] for k in factors)

    # Step 4: 確信度
    signs = [np.sign(v) for v in factors.values() if abs(v) > 0.05]
    if len(signs) >= 3:
        agreement = abs(sum(signs)) / len(signs)
    else:
        agreement = 0.3

    # レジームが明確なほど確信度UP
    confidence = min(agreement * (0.6 + strength * 0.4), 1.0)

    # Step 5: 変動幅の推定
    vol_daily = _calc_volatility(close)
    vol_period = vol_daily * np.sqrt(horizon_days)
    expected_pct = direction_score * vol_period * 100 * 2

    # 信頼区間
    unc = vol_period * 100
    lower_80 = expected_pct - 1.28 * unc
    upper_80 = expected_pct + 1.28 * unc
    lower_95 = expected_pct - 1.96 * unc
    upper_95 = expected_pct + 1.96 * unc

    return {
        "direction_score": round(direction_score, 3),
        "confidence": round(confidence, 2),
        "expected_pct": round(expected_pct, 2),
        "lower_80": round(lower_80, 2),
        "upper_80": round(upper_80, 2),
        "lower_95": round(lower_95, 2),
        "upper_95": round(upper_95, 2),
        "regime": regime,
        "trend_direction": trend_dir,
        "regime_strength": round(strength, 2),
        "factors": {k: round(v, 3) for k, v in factors.items()},
        "volatility": round(vol_daily * 100, 2),
    }


# ==========================================
# 公開API
# ==========================================

def _detect_market(ticker):
    """銘柄の市場を判定"""
    stripped = ticker.replace(".T", "").replace(".", "")
    if stripped.isdigit():
        return "JP"
    # latest_data.csv のmarket列も確認
    try:
        path = BASE_DIR / "data" / "latest_data.csv"
        if path.exists():
            df = pd.read_csv(path)
            row = df[df["ticker"] == ticker]
            if not row.empty:
                return row.iloc[0].get("market", "US")
    except Exception:
        pass
    return "US"


def predict_stock(ticker, periods=None):
    """レジーム適応型確率予測 v5"""
    if periods is None:
        periods = [1, 7, 30]

    hist_df = _fetch_history(ticker)
    if hist_df.empty:
        return None

    close = hist_df["Close"].values
    high = hist_df["High"].values
    low = hist_df["Low"].values
    volume = hist_df["Volume"].values
    cur = float(close[-1])
    market = _detect_market(ticker)

    predictions = []
    all_factors = {}
    regime_info = ""
    volatility = 2.0

    for d in periods:
        result = _predict_direction_full(close, high, low, volume, d,
                                         ticker=ticker, market=market)
        pred_price = cur * (1 + result["expected_pct"] / 100)

        predictions.append({
            "period_days": d,
            "return_pct": result["expected_pct"],
            "predicted_price": round(pred_price, 2),
            "direction_score": result["direction_score"],
            "confidence": result["confidence"],
            "lower_80_pct": result["lower_80"],
            "upper_80_pct": result["upper_80"],
            "lower_95_pct": result["lower_95"],
            "upper_95_pct": result["upper_95"],
            "current_price": cur,
        })

        if d == 1:
            all_factors = result["factors"]
            volatility = result["volatility"]
            regime_info = result["regime"]

    return {
        "ticker": ticker,
        "current_price": cur,
        "engine": "regime_adaptive_v5",
        "factors": all_factors,
        "volatility": volatility,
        "regime": regime_info,
        "predictions": predictions,
        "history_df": hist_df,
    }


def predict_all_stocks(tickers):
    """全銘柄の予測を実行してDataFrameで返す"""
    periods = [1, 7, 30]
    rows = []

    for tk in tickers:
        result = predict_stock(tk, periods)
        if result is None:
            continue

        rd = {
            "ticker": tk,
            "pred_engine": result["engine"],
            "pred_current_price": result["current_price"],
            "volatility": result.get("volatility", 0),
            "regime": result.get("regime", ""),
        }

        # ファクタースコア
        for fname, fval in result.get("factors", {}).items():
            rd[f"factor_{fname}"] = fval

        # 各ホライズンの予測
        for p in result["predictions"]:
            d = p["period_days"]
            rd[f"pred_{d}d_pct"] = p["return_pct"]
            rd[f"pred_{d}d_dir"] = p["direction_score"]
            rd[f"pred_{d}d_conf"] = p["confidence"]
            rd[f"pred_{d}d_l80"] = p["lower_80_pct"]
            rd[f"pred_{d}d_u80"] = p["upper_80_pct"]
            rd[f"pred_{d}d_l95"] = p["lower_95_pct"]
            rd[f"pred_{d}d_u95"] = p["upper_95_pct"]

        rows.append(rd)

    df = pd.DataFrame(rows)
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "latest_predictions.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"予測データ保存: {path} ({len(df)}銘柄)")
    return df
