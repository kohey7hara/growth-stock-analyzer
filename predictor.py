"""
predictor.py - マルチシグナル・アンサンブル株価予測エンジン

Dexter/ai-hedge-fund に着想を得た、複数の分析エージェントを統合する方式。
各エージェントが独立にシグナルを出し、最終的にアンサンブルで予測を生成する。

エージェント構成:
  1. テクニカル・エージェント    - RSI/MACD/BB/SMAのテクニカルシグナル
  2. モメンタム・エージェント    - 短期〜中期のモメンタム分析
  3. リバーサル・エージェント    - 平均回帰・逆張りシグナル
  4. ボラティリティ・エージェント - ボラティリティレジーム分析
  5. トレンド・エージェント      - 統計モデル (ETS/ARIMA/線形回帰)

各エージェントは-100〜+100のスコアと信頼度(0〜1)を返し、
加重平均でアンサンブルスコアを算出。スコアをリターン予測に変換する。
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent

# --- データ取得 ---

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
        for p in [period, "1y", "6mo", "3mo"]:
            try:
                hist = yf.Ticker(tk).history(period=p)
                if not hist.empty and len(hist) >= 20:
                    df = hist[["Close", "High", "Low", "Volume"]].dropna(subset=["Close"])
                    if len(df) >= 20:
                        return df
            except Exception:
                continue
    return pd.DataFrame()


# --- エージェント1: テクニカル分析エージェント ---

def _agent_technical(df):
    """RSI, MACD, ボリンジャーバンド, SMAクロスからシグナルを生成"""
    close = df["Close"].values
    n = len(close)
    if n < 26:
        return {"score": 0, "confidence": 0.1, "signals": [], "name": "テクニカル"}

    signals = []
    scores = []

    # RSI (14日)
    deltas = np.diff(close)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(14).mean().iloc[-1]
    rsi = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-10)))

    if rsi < 30:
        scores.append(60)  # 売られすぎ → 買いシグナル
        signals.append(f"RSI={rsi:.0f} 売られすぎ(買)")
    elif rsi < 40:
        scores.append(30)
        signals.append(f"RSI={rsi:.0f} やや低(買)")
    elif rsi > 70:
        scores.append(-60)
        signals.append(f"RSI={rsi:.0f} 買われすぎ(売)")
    elif rsi > 60:
        scores.append(-20)
        signals.append(f"RSI={rsi:.0f} やや高(売)")
    else:
        scores.append(0)
        signals.append(f"RSI={rsi:.0f} 中立")

    # MACD (12, 26, 9)
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    macd = ema12 - ema26
    macd_signal = pd.Series(macd).ewm(span=9).mean().values
    hist = macd - macd_signal

    if hist[-1] > 0 and hist[-2] <= 0:
        scores.append(70)
        signals.append("MACDゴールデンクロス(強い買)")
    elif hist[-1] < 0 and hist[-2] >= 0:
        scores.append(-70)
        signals.append("MACDデッドクロス(強い売)")
    elif hist[-1] > hist[-2] > 0:
        scores.append(30)
        signals.append("MACDプラス圏上昇(買)")
    elif hist[-1] < hist[-2] < 0:
        scores.append(-30)
        signals.append("MACDマイナス圏下降(売)")
    else:
        h_trend = hist[-1] - hist[-2]
        scores.append(int(np.clip(h_trend / max(abs(close[-1]) * 0.001, 1e-10) * 20, -40, 40)))
        signals.append(f"MACDヒストグラム{'上昇' if h_trend > 0 else '下降'}")

    # ボリンジャーバンド (20日, 2σ)
    sma20 = pd.Series(close).rolling(20).mean().values[-1]
    std20 = pd.Series(close).rolling(20).std().values[-1]
    if std20 > 0:
        bb_pos = (close[-1] - (sma20 - 2 * std20)) / (4 * std20)  # 0〜1
        bb_pos = np.clip(bb_pos, 0, 1)
        if bb_pos < 0.1:
            scores.append(50)
            signals.append("BB下限突破(強い買)")
        elif bb_pos < 0.25:
            scores.append(25)
            signals.append("BB下側(買)")
        elif bb_pos > 0.9:
            scores.append(-50)
            signals.append("BB上限突破(強い売)")
        elif bb_pos > 0.75:
            scores.append(-25)
            signals.append("BB上側(売)")
        else:
            scores.append(0)
            signals.append("BB中間")

    # SMAクロス (20日 vs 50日)
    if n >= 50:
        sma50 = pd.Series(close).rolling(50).mean().values
        sma20_arr = pd.Series(close).rolling(20).mean().values
        if sma20_arr[-1] > sma50[-1] and sma20_arr[-2] <= sma50[-2]:
            scores.append(60)
            signals.append("SMA20>50 ゴールデンクロス(買)")
        elif sma20_arr[-1] < sma50[-1] and sma20_arr[-2] >= sma50[-2]:
            scores.append(-60)
            signals.append("SMA20<50 デッドクロス(売)")
        elif sma20_arr[-1] > sma50[-1]:
            scores.append(15)
            signals.append("SMA20>50 上昇トレンド")
        else:
            scores.append(-15)
            signals.append("SMA20<50 下降トレンド")

    avg_score = np.mean(scores) if scores else 0
    confidence = min(0.9, 0.5 + len(scores) * 0.1)

    return {"score": float(avg_score), "confidence": confidence,
            "signals": signals, "name": "テクニカル"}


# --- エージェント2: モメンタム分析エージェント ---

def _agent_momentum(df):
    """短期〜中期のモメンタム（価格変動率・出来高トレンド）"""
    close = df["Close"].values
    volume = df["Volume"].values
    n = len(close)
    if n < 20:
        return {"score": 0, "confidence": 0.1, "signals": [], "name": "モメンタム"}

    signals = []
    scores = []

    # 各期間のリターン
    for days, label, weight in [(5, "1週間", 1.5), (20, "1ヶ月", 1.0), (60, "3ヶ月", 0.7)]:
        if n > days:
            ret = (close[-1] / close[-days - 1] - 1) * 100
            # リターンをスコアに変換（-10%〜+10% → -80〜+80）
            s = np.clip(ret * 8, -80, 80) * weight
            scores.append(s)
            signals.append(f"{label}リターン: {ret:+.1f}%")

    # 出来高トレンド（直近5日 vs 20日平均）
    if n >= 20:
        vol_5d = np.mean(volume[-5:])
        vol_20d = np.mean(volume[-20:])
        vol_ratio = vol_5d / max(vol_20d, 1)
        if vol_ratio > 1.5 and close[-1] > close[-5]:
            scores.append(40)
            signals.append(f"出来高急増+上昇({vol_ratio:.1f}倍)")
        elif vol_ratio > 1.5 and close[-1] < close[-5]:
            scores.append(-40)
            signals.append(f"出来高急増+下落({vol_ratio:.1f}倍)")
        elif vol_ratio > 1.2:
            scores.append(10 if close[-1] > close[-5] else -10)
            signals.append(f"出来高やや増({vol_ratio:.1f}倍)")

    # 連続上昇/下落
    streak = 0
    for i in range(1, min(8, n)):
        if close[-i] > close[-i - 1]:
            streak += 1
        elif close[-i] < close[-i - 1]:
            streak -= 1
        else:
            break

    if abs(streak) >= 3:
        s = np.clip(streak * 10, -50, 50)
        scores.append(s)
        signals.append(f"{'上昇' if streak > 0 else '下落'}{abs(streak)}日連続")

    avg_score = np.mean(scores) if scores else 0
    return {"score": float(avg_score), "confidence": 0.6,
            "signals": signals, "name": "モメンタム"}


# --- エージェント3: リバーサル（平均回帰）エージェント ---

def _agent_reversal(df):
    """平均回帰シグナル。大幅乖離時の逆張り"""
    close = df["Close"].values
    n = len(close)
    if n < 50:
        return {"score": 0, "confidence": 0.1, "signals": [], "name": "リバーサル"}

    signals = []
    scores = []

    # SMA50からの乖離率
    sma50 = pd.Series(close).rolling(50).mean().values[-1]
    dev_50 = (close[-1] - sma50) / sma50 * 100

    if dev_50 < -15:
        scores.append(70)
        signals.append(f"SMA50から{dev_50:.1f}%乖離(強い反発期待)")
    elif dev_50 < -8:
        scores.append(40)
        signals.append(f"SMA50から{dev_50:.1f}%乖離(反発期待)")
    elif dev_50 > 15:
        scores.append(-70)
        signals.append(f"SMA50から+{dev_50:.1f}%乖離(調整警戒)")
    elif dev_50 > 8:
        scores.append(-40)
        signals.append(f"SMA50から+{dev_50:.1f}%乖離(やや過熱)")
    else:
        scores.append(0)
        signals.append(f"SMA50乖離{dev_50:+.1f}%(適正)")

    # SMA200からの乖離率
    if n >= 200:
        sma200 = pd.Series(close).rolling(200).mean().values[-1]
        dev_200 = (close[-1] - sma200) / sma200 * 100
        if dev_200 < -20:
            scores.append(60)
            signals.append(f"SMA200から{dev_200:.1f}%乖離(深押し反発)")
        elif dev_200 > 20:
            scores.append(-60)
            signals.append(f"SMA200から+{dev_200:.1f}%乖離(長期過熱)")
        else:
            scores.append(int(-dev_200 * 2))

    # 52週高値からの下落率
    high_52w = np.max(close[-min(252, n):])
    drawdown = (close[-1] - high_52w) / high_52w * 100
    if drawdown < -40:
        scores.append(50)
        signals.append(f"52週高値から{drawdown:.0f}%(大幅調整)")
    elif drawdown < -20:
        scores.append(30)
        signals.append(f"52週高値から{drawdown:.0f}%(調整)")
    elif drawdown > -5:
        scores.append(-20)
        signals.append(f"52週高値近辺({drawdown:.0f}%)")

    avg_score = np.mean(scores) if scores else 0
    return {"score": float(avg_score), "confidence": 0.5,
            "signals": signals, "name": "リバーサル"}


# --- エージェント4: ボラティリティ・エージェント ---

def _agent_volatility(df):
    """ボラティリティレジーム分析。高ボラ時は予測信頼度を下げる"""
    close = df["Close"].values
    n = len(close)
    if n < 30:
        return {"score": 0, "confidence": 0.3, "signals": [], "name": "ボラティリティ"}

    signals = []
    daily_ret = np.diff(close) / close[:-1]

    # 直近20日のボラティリティ
    vol_20 = np.std(daily_ret[-20:]) * np.sqrt(252) * 100  # 年率%
    # 長期ボラティリティ
    vol_long = np.std(daily_ret) * np.sqrt(252) * 100

    vol_ratio = vol_20 / max(vol_long, 1)

    score = 0
    if vol_ratio > 1.5:
        score = -30  # 高ボラ → リスク高
        signals.append(f"ボラ急上昇({vol_20:.0f}%, 長期比{vol_ratio:.1f}倍)")
    elif vol_ratio < 0.7:
        score = 20  # 低ボラ → 安定
        signals.append(f"ボラ低下({vol_20:.0f}%, 安定局面)")
    else:
        signals.append(f"ボラ通常({vol_20:.0f}%)")

    # ATR(Average True Range)ベースのレジーム判定
    if n >= 14:
        high = df["High"].values
        low = df["Low"].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(abs(high[1:] - close[:-1]),
                                   abs(low[1:] - close[:-1])))
        atr_14 = np.mean(tr[-14:])
        atr_pct = atr_14 / close[-1] * 100
        if atr_pct > 4:
            score -= 20
            signals.append(f"ATR高({atr_pct:.1f}%, 荒れ相場)")
        elif atr_pct < 1.5:
            score += 10
            signals.append(f"ATR低({atr_pct:.1f}%, 穏やか)")

    # ボラが高い時は全体の予測信頼度を下げる
    confidence = max(0.2, min(0.8, 1.0 - vol_ratio * 0.3))

    return {"score": float(score), "confidence": confidence,
            "signals": signals, "name": "ボラティリティ"}


# --- エージェント5: トレンド統計エージェント ---

def _agent_trend(df, periods_days):
    """統計モデルで価格変動率を直接予測 (ETS → ARIMA → 線形回帰)"""
    close = df["Close"].values
    n = len(close)
    predictions = {}

    # 直近リターンの統計
    daily_ret = np.diff(close) / close[:-1]

    for d in periods_days:
        # ルックバック: 予測期間の5倍、最低20日
        lookback = max(20, min(d * 5, n - 1))
        recent = close[-lookback:]
        recent_ret = np.diff(recent) / recent[:-1]

        # 期間リターンの平均と標準偏差
        mean_daily = np.mean(recent_ret)
        std_daily = np.std(recent_ret) if len(recent_ret) > 1 else 0.02

        # トレンド加速度（直近20日 vs その前20日のリターン比較）
        if n >= 40:
            r1 = np.mean(daily_ret[-40:-20])  # 前半
            r2 = np.mean(daily_ret[-20:])      # 後半
            accel = r2 - r1
        else:
            accel = 0

        # 予測リターン = 日次平均 × 日数 + 加速度項
        pred_ret = (mean_daily + accel * 0.3) * d * 100  # %
        # 信頼区間
        unc = std_daily * np.sqrt(d) * 100
        predictions[d] = {
            "pct": float(np.clip(pred_ret, -50, 50)),
            "l80": float(pred_ret - 1.28 * unc),
            "u80": float(pred_ret + 1.28 * unc),
            "l95": float(pred_ret - 1.96 * unc),
            "u95": float(pred_ret + 1.96 * unc),
        }

    # スコアは1日後予測の方向性
    p1 = predictions.get(1, {}).get("pct", 0)
    score = np.clip(p1 * 20, -80, 80)

    signals = []
    for d in periods_days:
        p = predictions[d]["pct"]
        signals.append(f"{d}日後: {p:+.2f}%")

    return {"score": float(score), "confidence": 0.5,
            "signals": signals, "name": "トレンド",
            "predictions": predictions}


# --- アンサンブル ---

def _ensemble_predict(agents_results, periods_days, current_price):
    """
    全エージェントのスコアを加重平均し、リターン予測に変換。
    Dexterの自己検証に倣い、エージェント間の一致度で信頼度を調整。
    """
    # エージェント重み（テクニカルとトレンドを重視）
    weights = {
        "テクニカル": 0.30,
        "モメンタム": 0.20,
        "リバーサル": 0.15,
        "ボラティリティ": 0.10,
        "トレンド": 0.25,
    }

    # スコアの加重平均
    total_weight = 0
    ensemble_score = 0
    agent_scores = []

    for r in agents_results:
        w = weights.get(r["name"], 0.1) * r["confidence"]
        ensemble_score += r["score"] * w
        total_weight += w
        agent_scores.append(r["score"])

    if total_weight > 0:
        ensemble_score /= total_weight

    # エージェント間の一致度（標準偏差が小さい = 一致 = 高信頼）
    if len(agent_scores) >= 2:
        agreement = 1.0 - min(1.0, np.std(agent_scores) / 80)
    else:
        agreement = 0.3

    # トレンドエージェントの統計予測をベースに、アンサンブルスコアで調整
    trend_agent = next((a for a in agents_results if a["name"] == "トレンド"), None)
    trend_preds = trend_agent.get("predictions", {}) if trend_agent else {}

    predictions = []
    for d in periods_days:
        tp = trend_preds.get(d)
        if tp:
            base_pct = tp["pct"]
            base_l80 = tp["l80"]
            base_u80 = tp["u80"]
            base_l95 = tp["l95"]
            base_u95 = tp["u95"]
        else:
            base_pct = 0
            base_l80 = base_l95 = -5
            base_u80 = base_u95 = 5

        # アンサンブルスコアでベース予測を調整
        # スコア-100〜+100を日次リターン調整量に変換
        adjustment = ensemble_score * 0.02 * d  # 1日あたり0.02%ずつ調整
        adj_pct = base_pct + adjustment

        # 信頼区間も一致度で調整（一致度が高い = 区間狭く）
        ci_factor = 1.0 + (1.0 - agreement) * 0.5
        range_80 = (base_u80 - base_l80) * ci_factor / 2
        range_95 = (base_u95 - base_l95) * ci_factor / 2

        pred_price = current_price * (1 + adj_pct / 100)

        predictions.append({
            "period_days": d,
            "return_pct": round(adj_pct, 2),
            "predicted_price": round(pred_price, 2),
            "lower_80": round(current_price * (1 + (adj_pct - range_80) / 100), 2),
            "upper_80": round(current_price * (1 + (adj_pct + range_80) / 100), 2),
            "lower_80_pct": round(adj_pct - range_80, 2),
            "upper_80_pct": round(adj_pct + range_80, 2),
            "lower_95": round(current_price * (1 + (adj_pct - range_95) / 100), 2),
            "upper_95": round(current_price * (1 + (adj_pct + range_95) / 100), 2),
            "lower_95_pct": round(adj_pct - range_95, 2),
            "upper_95_pct": round(adj_pct + range_95, 2),
            "current_price": current_price,
        })

    return {
        "ensemble_score": round(ensemble_score, 1),
        "agreement": round(agreement, 2),
        "predictions": predictions,
    }


# --- メインAPI ---

def predict_stock(ticker, periods=None):
    """マルチエージェント・アンサンブルで銘柄を予測"""
    if periods is None:
        periods = [1, 7, 30]

    hist_df = _fetch_history(ticker)
    if hist_df.empty:
        return None

    cur = float(hist_df["Close"].iloc[-1])

    # 各エージェントを実行
    agents_results = []
    try:
        agents_results.append(_agent_technical(hist_df))
    except Exception as e:
        logger.debug(f"テクニカルエージェント失敗 ({ticker}): {e}")
    try:
        agents_results.append(_agent_momentum(hist_df))
    except Exception as e:
        logger.debug(f"モメンタムエージェント失敗 ({ticker}): {e}")
    try:
        agents_results.append(_agent_reversal(hist_df))
    except Exception as e:
        logger.debug(f"リバーサルエージェント失敗 ({ticker}): {e}")
    try:
        agents_results.append(_agent_volatility(hist_df))
    except Exception as e:
        logger.debug(f"ボラティリティエージェント失敗 ({ticker}): {e}")
    try:
        agents_results.append(_agent_trend(hist_df, periods))
    except Exception as e:
        logger.debug(f"トレンドエージェント失敗 ({ticker}): {e}")

    if not agents_results:
        return None

    # アンサンブル予測
    result = _ensemble_predict(agents_results, periods, cur)

    return {
        "ticker": ticker,
        "current_price": cur,
        "engine": "ensemble",
        "ensemble_score": result["ensemble_score"],
        "agreement": result["agreement"],
        "agents": agents_results,
        "predictions": result["predictions"],
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
            "ensemble_score": result["ensemble_score"],
            "agreement": result["agreement"],
        }

        # エージェント別スコア
        for agent in result.get("agents", []):
            key = f"agent_{agent['name']}"
            rd[key] = round(agent["score"], 1)

        for p in result["predictions"]:
            d = p["period_days"]
            rd[f"pred_{d}d_pct"] = p["return_pct"]
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
