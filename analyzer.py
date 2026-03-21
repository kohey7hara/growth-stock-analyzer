"""
analyzer.py - 統合スコアリング・買い時判定エンジン

株価データ + SNSセンチメントを統合し、100点満点でスコアリング。
買いシグナル/要注目/様子見を判定する。

スコアリング配分 (config.yamlで調整可能):
  - テクニカル分析: 35%
  - ファンダメンタルズ: 30%
  - SNSセンチメント: 20%
  - モメンタム/出来高: 15%
"""

import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "analyzer.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def score_technical(row, config):
    """
    テクニカル指標に基づくスコア (0-100)

    評価項目:
      1. RSI位置 (売られすぎ=高スコア)
      2. MACD方向 (上向き=高スコア)
      3. ボリンジャーバンド位置 (下限付近=高スコア)
      4. 移動平均線との関係
      5. ゴールデンクロス/デッドクロス
    """
    score = 0
    reasons = []
    tech_cfg = config["technical"]

    # 1. RSI (0-30点)
    rsi = row.get("rsi_14")
    if rsi is not None and not np.isnan(rsi):
        if rsi < tech_cfg["rsi_oversold"]:
            score += 30
            reasons.append(f"RSI売られすぎ({rsi:.0f})")
        elif rsi < 40:
            score += 20
            reasons.append(f"RSI低め({rsi:.0f})")
        elif rsi < 50:
            score += 15
            reasons.append(f"RSI中立({rsi:.0f})")
        elif rsi < tech_cfg["rsi_overbought"]:
            score += 5
            reasons.append(f"RSIやや高({rsi:.0f})")
        else:
            score += 0
            reasons.append(f"RSI買われすぎ({rsi:.0f})")

    # 2. MACD (0-25点)
    macd_hist = row.get("macd_histogram")
    macd_val = row.get("macd")
    if macd_hist is not None and not np.isnan(macd_hist):
        if macd_hist > 0 and macd_val is not None and macd_val > 0:
            score += 25
            reasons.append("MACDプラス圏上昇")
        elif macd_hist > 0:
            score += 20
            reasons.append("MACDヒストグラム上昇")
        elif macd_hist < 0 and abs(macd_hist) < abs(row.get("macd_signal", 1) * 0.1):
            score += 15
            reasons.append("MACD反転間近")
        else:
            score += 0
            reasons.append("MACDマイナス圏")

    # 3. ボリンジャーバンド位置 (0-25点)
    bb_pos = row.get("bb_position")
    if bb_pos is not None and not np.isnan(bb_pos):
        if bb_pos < 10:
            score += 25
            reasons.append(f"BB下限突破({bb_pos:.0f}%)")
        elif bb_pos < 30:
            score += 20
            reasons.append(f"BB下限付近({bb_pos:.0f}%)")
        elif bb_pos < 50:
            score += 10
            reasons.append(f"BB中間以下({bb_pos:.0f}%)")
        else:
            score += 0
            reasons.append(f"BB上方({bb_pos:.0f}%)")

    # 4. クロスシグナル (0-20点)
    cross = row.get("cross_signal", "なし")
    if cross == "ゴールデンクロス":
        score += 20
        reasons.append("ゴールデンクロス発生!")
    elif cross == "デッドクロス":
        score += 0
        reasons.append("デッドクロス警告")
    else:
        # 価格とSMAの関係
        price = row.get("price", 0)
        sma_200 = row.get("sma_200")
        if sma_200 and not np.isnan(sma_200) and price > sma_200:
            score += 10
            reasons.append("200日SMA上")
        elif sma_200 and not np.isnan(sma_200):
            score += 5
            reasons.append("200日SMA下")

    return min(100, score), reasons


def score_fundamental(row, config):
    """
    ファンダメンタルズに基づくスコア (0-100)

    評価項目:
      1. 52週レンジ位置 (安値圏=高スコア)
      2. PER / PEG比率
      3. 売上成長率
      4. アナリスト目標株価との乖離 / 配当利回り
    """
    score = 0
    reasons = []
    market = row.get("market", "US")

    # 1. 52週レンジ位置 (0-30点)
    pos = row.get("pos_52w_pct")
    if pos is not None and not np.isnan(pos):
        if pos < 20:
            score += 30
            reasons.append(f"52週最安値圏({pos:.0f}%)")
        elif pos < 35:
            score += 25
            reasons.append(f"52週安値圏({pos:.0f}%)")
        elif pos < 50:
            score += 15
            reasons.append(f"52週中間以下({pos:.0f}%)")
        elif pos < 70:
            score += 5
            reasons.append(f"52週中間付近({pos:.0f}%)")
        else:
            score += 0
            reasons.append(f"52週高値圏({pos:.0f}%)")

    # 2. バリュエーション (0-25点)
    if market == "US":
        peg = row.get("peg_ratio")
        if peg is not None and not np.isnan(peg):
            if peg < 0.8:
                score += 25
                reasons.append(f"PEG非常に割安({peg:.1f}x)")
            elif peg < 1.2:
                score += 20
                reasons.append(f"PEG割安({peg:.1f}x)")
            elif peg < 2.0:
                score += 10
                reasons.append(f"PEG適正({peg:.1f}x)")
            else:
                score += 0
                reasons.append(f"PEG割高({peg:.1f}x)")
        else:
            pe = row.get("pe_ratio")
            if pe is not None and not np.isnan(pe):
                if pe < 20:
                    score += 25
                elif pe < 30:
                    score += 15
                elif pe < 50:
                    score += 5
                reasons.append(f"PER {pe:.0f}x")
    else:
        pe = row.get("pe_ratio")
        if pe is not None and not np.isnan(pe):
            if pe < 20:
                score += 25
                reasons.append(f"PER割安({pe:.0f}x)")
            elif pe < 30:
                score += 15
                reasons.append(f"PER適正({pe:.0f}x)")
            elif pe < 45:
                score += 5
                reasons.append(f"PERやや高({pe:.0f}x)")
            else:
                score += 0
                reasons.append(f"PER過熱({pe:.0f}x)")

    # 3. 売上成長率 (0-25点)
    rg = row.get("revenue_growth_pct")
    if rg is not None and not np.isnan(rg):
        if rg > 50:
            score += 25
            reasons.append(f"超高成長({rg:.0f}%)")
        elif rg > 25:
            score += 20
            reasons.append(f"高成長({rg:.0f}%)")
        elif rg > 10:
            score += 10
            reasons.append(f"成長({rg:.0f}%)")
        elif rg > 0:
            score += 5
            reasons.append(f"低成長({rg:.0f}%)")
        else:
            score += 0
            reasons.append(f"減収({rg:.0f}%)")

    # 4. 目標株価 or 配当 (0-20点)
    if market == "US":
        upside = row.get("upside_pct")
        if upside is not None and not np.isnan(upside):
            if upside > 30:
                score += 20
                reasons.append(f"上値余地大(+{upside:.0f}%)")
            elif upside > 15:
                score += 15
                reasons.append(f"上値余地あり(+{upside:.0f}%)")
            elif upside > 0:
                score += 5
                reasons.append(f"上値余地小(+{upside:.0f}%)")
            else:
                reasons.append(f"割高({upside:.0f}%)")
    else:
        dy = row.get("dividend_yield_pct")
        if dy is not None and not np.isnan(dy):
            if dy > 3.0:
                score += 20
                reasons.append(f"高配当({dy:.1f}%)")
            elif dy > 1.5:
                score += 10
                reasons.append(f"配当あり({dy:.1f}%)")
            else:
                score += 5
                reasons.append(f"低配当({dy:.1f}%)")

    return min(100, score), reasons


def score_sentiment(row, config):
    """
    SNSセンチメントに基づくスコア (0-100)
    改善2: SNSデータなしの場合のスコアを50→30に変更
    """
    score = 0
    reasons = []

    # 複合センチメント
    cs = row.get("combined_sentiment")
    if cs is not None and not np.isnan(cs):
        if cs > 50:
            score += 40
            reasons.append(f"SNS強気(+{cs:.0f})")
        elif cs > 20:
            score += 30
            reasons.append(f"SNSやや強気(+{cs:.0f})")
        elif cs > -20:
            score += 15
            reasons.append(f"SNS中立({cs:.0f})")
        else:
            score += 0
            reasons.append(f"SNS弱気({cs:.0f})")

    # Google Trends急上昇
    gt_trending = row.get("gtrends_is_trending")
    gt_ratio = row.get("gtrends_trend_ratio")
    if gt_trending:
        score += 30
        reasons.append(f"検索急上昇({gt_ratio:.1f}x)")
    elif gt_ratio and not np.isnan(gt_ratio) and gt_ratio > 1.0:
        score += 15
        reasons.append(f"検索上昇傾向({gt_ratio:.1f}x)")

    # Reddit活発度
    reddit_mentions = row.get("reddit_mentions", 0)
    if reddit_mentions and reddit_mentions > 50:
        score += 30
        reasons.append(f"Reddit話題({reddit_mentions}件)")
    elif reddit_mentions and reddit_mentions > 10:
        score += 15
        reasons.append(f"Redditメンションあり({reddit_mentions}件)")

    if not reasons:
        reasons.append("SNSデータなし")
        score = 30  # 改善2: データなしの場合は30 (50から変更)

    return min(100, score), reasons


def score_momentum(row, config):
    """
    モメンタム/出来高に基づくスコア (0-100)

    評価項目:
      1. 出来高比率 (20日平均比)
      2. 株価の短期トレンド
    """
    score = 0
    reasons = []

    # 出来高比率
    vol_ratio = row.get("vol_ratio")
    if vol_ratio is not None and not np.isnan(vol_ratio):
        if vol_ratio > 3.0:
            score += 50
            reasons.append(f"出来高急増({vol_ratio:.1f}x)")
        elif vol_ratio > 2.0:
            score += 40
            reasons.append(f"出来高増加({vol_ratio:.1f}x)")
        elif vol_ratio > 1.2:
            score += 20
            reasons.append(f"出来高やや増({vol_ratio:.1f}x)")
        elif vol_ratio > 0.8:
            score += 10
            reasons.append(f"出来高平常({vol_ratio:.1f}x)")
        else:
            score += 0
            reasons.append(f"出来高減少({vol_ratio:.1f}x)")

    # SMA20 vs SMA50の関係でトレンド判定
    sma20 = row.get("sma_20")
    sma50 = row.get("sma_50")
    price = row.get("price")
    if sma20 and sma50 and price:
        if not np.isnan(sma20) and not np.isnan(sma50):
            if price > sma20 > sma50:
                score += 50
                reasons.append("強い上昇トレンド")
            elif price > sma20:
                score += 30
                reasons.append("短期上昇")
            elif price > sma50:
                score += 15
                reasons.append("中期上昇")
            else:
                score += 0
                reasons.append("下落トレンド")

    if not reasons:
        score = 50

    return min(100, score), reasons


def generate_analysis_comment(row, tech_score, tech_reasons, fund_score, fund_reasons,
                               mom_score, mom_reasons, total_score, signal):
    """
    改善3: 主要分析コメントを日本語で分かりやすく自動生成

    形式:
    【結論】+ なぜそう判断したかを株初心者でもわかる2-3文
    """
    rsi = row.get("rsi_14")
    bb_pos = row.get("bb_position")
    pos_52w = row.get("pos_52w_pct")
    vol_ratio = row.get("vol_ratio")
    cross = row.get("cross_signal", "なし")
    price = row.get("price", 0)
    sma_200 = row.get("sma_200")
    sma_20 = row.get("sma_20")
    sma_50 = row.get("sma_50")
    macd_hist = row.get("macd_histogram")
    change_1y = row.get("change_1y_pct")

    # 結論ラベルの決定
    if total_score >= 70:
        label = "【強い買いシグナル】"
    elif total_score >= 60:
        # RSI低い + 安値圏 → 押し目買い
        if rsi and rsi < 40 and pos_52w and pos_52w < 35:
            label = "【押し目買いチャンス】"
        else:
            label = "【買い検討】"
    elif total_score >= 55:
        if cross == "ゴールデンクロス":
            label = "【トレンド転換の兆し】"
        elif sma_20 and sma_50 and price and price > sma_20 > sma_50:
            label = "【上昇トレンド継続】"
        else:
            label = "【買い検討】"
    elif total_score >= 40:
        if rsi and rsi > 70:
            label = "【過熱警戒】"
        elif pos_52w and pos_52w > 80:
            label = "【高値圏注意】"
        else:
            label = "【様子見】"
    else:
        if rsi and rsi > 70 and pos_52w and pos_52w > 80:
            label = "【過熱警戒】"
        elif macd_hist and macd_hist < 0 and sma_200 and price and price < sma_200:
            label = "【下落トレンド】"
        else:
            label = "【様子見】"

    # 解説文の生成
    sentences = []

    # 株価位置に関するコメント
    if pos_52w is not None and not np.isnan(pos_52w):
        if pos_52w < 15:
            sentences.append(f"株価が52週安値圏({pos_52w:.0f}%)まで下落")
        elif pos_52w < 30:
            sentences.append(f"株価が52週レンジの下位({pos_52w:.0f}%)に位置")
        elif pos_52w > 85:
            sentences.append(f"株価が52週高値圏({pos_52w:.0f}%)で推移")
        elif pos_52w > 70:
            sentences.append(f"株価が52週レンジの上位({pos_52w:.0f}%)に位置")

    # RSIに関するコメント
    if rsi is not None and not np.isnan(rsi):
        if rsi < 30:
            sentences.append(f"RSIが売られすぎ水準({rsi:.0f})に達しており反発の可能性あり")
        elif rsi < 40:
            sentences.append(f"RSIが売られすぎ水準に接近({rsi:.0f})しており反発の可能性")
        elif rsi > 75:
            sentences.append(f"RSIが買われすぎ水準({rsi:.0f})に達しており調整リスクあり")
        elif rsi > 65:
            sentences.append(f"RSIがやや過熱気味({rsi:.0f})")

    # MACDに関するコメント
    if macd_hist is not None and not np.isnan(macd_hist):
        if macd_hist > 0:
            sentences.append("MACD(売買の勢い指標)が上向きで買いの勢いが継続中")
        elif macd_hist < 0 and abs(macd_hist) < 0.5:
            sentences.append("MACD(売買の勢い指標)が反転しそうな位置")

    # 移動平均線に関するコメント
    if sma_200 and not np.isnan(sma_200) and price:
        if price > sma_200:
            sentences.append("200日移動平均線を上回っており中長期トレンドは強い")
        else:
            sentences.append("200日移動平均線を下回っており中長期トレンドは弱い")

    # クロスシグナル
    if cross == "ゴールデンクロス":
        sentences.append("短期と中期の移動平均線がゴールデンクロスを形成し上昇転換のサイン")
    elif cross == "デッドクロス":
        sentences.append("短期と中期の移動平均線がデッドクロスを形成し下落警戒")

    # 出来高に関するコメント
    if vol_ratio and not np.isnan(vol_ratio):
        if vol_ratio > 3.0:
            sentences.append(f"出来高が平均の{vol_ratio:.1f}倍に急増しており大きな動きの兆候")
        elif vol_ratio > 2.0:
            sentences.append(f"出来高が平均の{vol_ratio:.1f}倍に増加しており注目度が上昇中")

    # 1年間の騰落率
    if change_1y is not None and not np.isnan(change_1y):
        if change_1y > 50:
            sentences.append(f"過去1年で{change_1y:.0f}%上昇と非常に強いパフォーマンス")
        elif change_1y < -30:
            sentences.append(f"過去1年で{change_1y:.0f}%下落しており底値を模索中")

    # 2-3文に制限
    if len(sentences) == 0:
        sentences.append("現時点では明確なシグナルが出ておらず、追加の材料を待ちたい局面")

    comment = label + sentences[0] + "。"
    if len(sentences) > 1:
        comment += sentences[1] + "。"
    if len(sentences) > 2:
        comment += sentences[2] + "。"

    return comment


def calculate_macro_score(macro_df=None):
    """
    マクロ環境スコア (0-100点) を算出

    入力: data/latest_macro.csv (ticker列: ^VIX, ^TNX, ^GSPC, ^N225, USDJPY=X)

    スコアリング:
      - VIX (0-20点): <15→20, 15-20→10, 20-25→5, >25→0
      - S&P500 vs SMA200 (0-20点)
      - 日経225 vs SMA200 (0-20点)
      - 米10年金利 ^TNX (0-20点): <3.5→20, 3.5-4.5→10, >4.5→0
      - ドル円トレンド (0-20点): 1ヶ月変化率で判定

    返却: {"score": int, "label": str, "emoji": str, "detail": str, "bonus": int}
    """
    import yfinance as yf

    if macro_df is None:
        macro_path = BASE_DIR / "data" / "latest_macro.csv"
        if not macro_path.exists():
            return {"score": 50, "label": "データなし", "emoji": "---",
                    "detail": "マクロデータが取得できませんでした", "bonus": 0}
        macro_df = pd.read_csv(macro_path, encoding="utf-8-sig")

    score = 0
    details = []

    # 1. VIX (0-20点)
    vix_row = macro_df[macro_df["ticker"] == "^VIX"]
    if not vix_row.empty:
        vix = vix_row.iloc[0]["current_value"]
        if vix < 15:
            score += 20
            details.append(f"VIX低水準({vix:.1f})→市場安定")
        elif vix < 20:
            score += 10
            details.append(f"VIXやや低({vix:.1f})→やや安定")
        elif vix < 25:
            score += 5
            details.append(f"VIXやや高({vix:.1f})→やや不安定")
        else:
            score += 0
            details.append(f"VIX高水準({vix:.1f})→市場不安定")

    # 2. S&P500 vs SMA200 (0-20点)
    gspc_row = macro_df[macro_df["ticker"] == "^GSPC"]
    if not gspc_row.empty:
        gspc_price = gspc_row.iloc[0]["current_value"]
        try:
            tk = yf.Ticker("^GSPC")
            hist = tk.history(period="1y")
            if not hist.empty and len(hist) >= 200:
                sma200 = hist["Close"].rolling(200).mean().iloc[-1]
                if gspc_price > sma200:
                    ratio = (gspc_price - sma200) / sma200 * 100
                    score += 20
                    details.append(f"S&P500がSMA200上(+{ratio:.1f}%)→強気")
                else:
                    ratio = (gspc_price - sma200) / sma200 * 100
                    score += 5
                    details.append(f"S&P500がSMA200下({ratio:.1f}%)→弱気")
            else:
                score += 10
                details.append("S&P500: SMA200データ不足→中立")
        except Exception:
            score += 10
            details.append("S&P500: SMA200取得失敗→中立")

    # 3. 日経225 vs SMA200 (0-20点)
    nk_row = macro_df[macro_df["ticker"] == "^N225"]
    if not nk_row.empty:
        nk_price = nk_row.iloc[0]["current_value"]
        try:
            tk = yf.Ticker("^N225")
            hist = tk.history(period="1y")
            if not hist.empty and len(hist) >= 200:
                sma200 = hist["Close"].rolling(200).mean().iloc[-1]
                if nk_price > sma200:
                    ratio = (nk_price - sma200) / sma200 * 100
                    score += 20
                    details.append(f"日経225がSMA200上(+{ratio:.1f}%)→強気")
                else:
                    ratio = (nk_price - sma200) / sma200 * 100
                    score += 5
                    details.append(f"日経225がSMA200下({ratio:.1f}%)→弱気")
            else:
                score += 10
                details.append("日経225: SMA200データ不足→中立")
        except Exception:
            score += 10
            details.append("日経225: SMA200取得失敗→中立")

    # 4. 米10年金利 (0-20点)
    tnx_row = macro_df[macro_df["ticker"] == "^TNX"]
    if not tnx_row.empty:
        tnx = tnx_row.iloc[0]["current_value"]
        if tnx < 3.5:
            score += 20
            details.append(f"米金利低水準({tnx:.2f}%)→グロースに追い風")
        elif tnx < 4.5:
            score += 10
            details.append(f"米金利中立({tnx:.2f}%)→影響中立")
        else:
            score += 0
            details.append(f"米金利高水準({tnx:.2f}%)→グロースに逆風")

    # 5. ドル円トレンド (0-20点)
    usdjpy_row = macro_df[macro_df["ticker"] == "USDJPY=X"]
    if not usdjpy_row.empty:
        change_1m = usdjpy_row.iloc[0].get("change_1m_pct", 0)
        if pd.notna(change_1m):
            if -2 <= change_1m <= 2:
                score += 20
                details.append(f"ドル円安定(1M:{change_1m:+.1f}%)→為替リスク低")
            elif change_1m > 2:
                score += 10
                details.append(f"円安進行(1M:{change_1m:+.1f}%)→輸出関連に追い風")
            else:
                score += 10
                details.append(f"円高進行(1M:{change_1m:+.1f}%)→輸入関連に追い風")

    # ラベル判定
    if score >= 80:
        label, emoji, bonus = "買い場", "🟢🟢", 5
    elif score >= 60:
        label, emoji, bonus = "やや強気", "🟢", 0
    elif score >= 40:
        label, emoji, bonus = "中立", "🟡", 0
    else:
        label, emoji, bonus = "弱気", "🔴", -5

    detail_str = " / ".join(details)

    return {
        "score": score,
        "label": label,
        "emoji": emoji,
        "detail": detail_str,
        "bonus": bonus,
    }


def detect_risk_flags(row):
    """
    リスクフラグ検出

    5つの条件をチェックし、該当するアラートをリストで返す。
    各アラート: {"type": str, "label": str, "action": str}
    """
    flags = []
    price = row.get("price", 0)
    change_pct = row.get("change_pct")
    vol_ratio = row.get("vol_ratio")
    pe_ratio = row.get("pe_ratio")
    pos_52w = row.get("pos_52w_pct")
    rsi = row.get("rsi_14")
    macd_val = row.get("macd")

    # 1. 急落アラート: 1日で-5%以上下落 かつ 出来高比率3x以上
    if (change_pct is not None and not np.isnan(change_pct) and change_pct <= -5
            and vol_ratio is not None and not np.isnan(vol_ratio) and vol_ratio >= 3.0):
        flags.append({
            "type": "crash",
            "label": "急落アラート",
            "action": "ニュースを確認してください。粉飾決算・訴訟・規制リスクの可能性",
        })

    # 2. 高ボラティリティ警告: vol_5d データから推定
    #    (vol_5dカラムは "v1,v2,v3,v4,v5" 形式の文字列)
    vol_5d_str = row.get("vol_5d")
    if vol_5d_str and isinstance(vol_5d_str, str):
        try:
            vols = [float(v) for v in vol_5d_str.split(",") if v.strip()]
            if len(vols) >= 3:
                # 出来高の変動率から高ボラを推定
                vol_std = np.std(vols) / (np.mean(vols) + 1e-10)
                if vol_std > 0.8:  # 出来高が非常に不安定
                    flags.append({
                        "type": "high_volatility",
                        "label": "高ボラティリティ警告",
                        "action": "短期トレードには不向き。ポジションサイズを小さく",
                    })
        except (ValueError, TypeError):
            pass

    # change_pctの絶対値が大きい場合も高ボラ判定
    if (change_pct is not None and not np.isnan(change_pct)
            and abs(change_pct) >= 10):
        if not any(f["type"] == "high_volatility" for f in flags):
            flags.append({
                "type": "high_volatility",
                "label": "高ボラティリティ警告",
                "action": "短期トレードには不向き。ポジションサイズを小さく",
            })

    # 3. PER異常値: PER > 200 または PER < 0（赤字）
    if pe_ratio is not None and not np.isnan(pe_ratio):
        if pe_ratio > 200 or pe_ratio < 0:
            flags.append({
                "type": "per_abnormal",
                "label": "PER異常値",
                "action": "投機的銘柄。成長期待で買うなら全体の5%以下に",
            })

    # 4. 52週安値更新中: pos_52w_pct < 3%
    if pos_52w is not None and not np.isnan(pos_52w) and pos_52w < 3:
        flags.append({
            "type": "new_52w_low",
            "label": "52週安値更新中",
            "action": "底値を狙うナンピンは危険。反転確認まで待つ",
        })

    # 5. 逆行シグナル:
    #    - 株価下落中(change_pct<0)なのにRSIが50以上に上昇
    #    - MACDがプラスなのに株価が下がっている
    divergence = False
    if (change_pct is not None and not np.isnan(change_pct) and change_pct < -3):
        if rsi is not None and not np.isnan(rsi) and rsi > 50:
            divergence = True
        if macd_val is not None and not np.isnan(macd_val) and macd_val > 0:
            divergence = True
    if divergence:
        flags.append({
            "type": "divergence",
            "label": "逆行シグナル",
            "action": "テクニカル指標の信頼度が低い。ファンダメンタルズで判断",
        })

    return flags


def analyze(stock_df, sentiment_df=None, config=None):
    """全銘柄を統合スコアリング"""
    if config is None:
        config = load_config()

    weights = config["scoring"]["weights"]

    # マクロスコア取得
    macro_result = calculate_macro_score()
    macro_bonus = macro_result.get("bonus", 0)
    macro_label = macro_result.get("label", "")
    logger.info(f"マクロ環境スコア: {macro_result['score']}点 ({macro_label}) bonus={macro_bonus}")

    # センチメントデータがあればマージ
    if sentiment_df is not None and not sentiment_df.empty:
        df = stock_df.merge(sentiment_df, on="ticker", how="left", suffixes=("", "_sent"))
    else:
        df = stock_df.copy()

    results = []
    for _, row in df.iterrows():
        r = row.to_dict()

        # 各カテゴリのスコア
        tech_score, tech_reasons = score_technical(r, config)
        fund_score, fund_reasons = score_fundamental(r, config)
        sent_score, sent_reasons = score_sentiment(r, config)
        mom_score, mom_reasons = score_momentum(r, config)

        # 加重平均スコア + マクロボーナス
        total = (
            tech_score * weights["technical_score"] +
            fund_score * weights["fundamental_score"] +
            sent_score * weights["sentiment_score"] +
            mom_score * weights["momentum_score"]
        )
        total = round(total + macro_bonus, 1)
        total = max(0, min(100, total))

        # シグナル判定
        if total >= 70:
            signal = "★★★ 強い買い"
        elif total >= 55:
            signal = "★★ 買い"
        elif total >= 40:
            signal = "★ 要注目"
        else:
            signal = "△ 様子見"

        # 改善3: 日本語の分かりやすいコメント生成
        analysis_comment = generate_analysis_comment(
            r, tech_score, tech_reasons, fund_score, fund_reasons,
            mom_score, mom_reasons, total, signal
        )
        # 市況コメントを先頭に付加
        if macro_bonus != 0:
            macro_prefix = f"[市況: {macro_label}({macro_bonus:+d}pt)] "
            analysis_comment = macro_prefix + analysis_comment

        # リスクフラグ検出
        risk_flags = detect_risk_flags(r)
        risk_count = len(risk_flags)
        risk_labels = " / ".join(f["label"] for f in risk_flags) if risk_flags else ""
        risk_actions = " / ".join(f["action"] for f in risk_flags) if risk_flags else ""

        # リスクフラグによるスコア減点
        if risk_count >= 2:
            total = max(0, total - 20)
        elif risk_count == 1:
            total = max(0, total - 10)

        # リスクフラグありの場合シグナルに警告付与
        if risk_flags:
            signal_suffix = " ⚠️要確認"
            # シグナル再判定（減点後の total で）
            if total >= 70:
                signal = "★★★ 強い買い" + signal_suffix
            elif total >= 55:
                signal = "★★ 買い" + signal_suffix
            elif total >= 40:
                signal = "★ 要注目" + signal_suffix
            else:
                signal = "△ 様子見" + signal_suffix

        r.update({
            "tech_score": tech_score,
            "tech_reasons": " / ".join(tech_reasons),
            "fund_score": fund_score,
            "fund_reasons": " / ".join(fund_reasons),
            "sent_score": sent_score,
            "sent_reasons": " / ".join(sent_reasons),
            "mom_score": mom_score,
            "mom_reasons": " / ".join(mom_reasons),
            "total_score": total,
            "signal": signal,
            "risk_count": risk_count,
            "risk_labels": risk_labels,
            "risk_actions": risk_actions,
            "analysis_comment": analysis_comment,
            "analysis_time": datetime.now().isoformat(),
        })
        results.append(r)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("total_score", ascending=False).reset_index(drop=True)

    # 保存
    today = datetime.now().strftime("%Y%m%d")
    data_dir = BASE_DIR / "data"
    csv_path = data_dir / f"analysis_result_{today}.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"分析結果保存: {csv_path}")

    latest_path = data_dir / "latest_analysis.csv"
    result_df.to_csv(latest_path, index=False, encoding="utf-8-sig")

    return result_df


if __name__ == "__main__":
    config = load_config()
    data_dir = BASE_DIR / "data"

    # データ読み込み
    stock_path = data_dir / "latest_data.csv"
    sent_path = data_dir / "latest_sentiment.csv"

    if not stock_path.exists():
        logger.error("株価データがありません。先に fetch_stock_data.py を実行してください")
        exit(1)

    stock_df = pd.read_csv(stock_path)
    sent_df = pd.read_csv(sent_path) if sent_path.exists() else None

    result = analyze(stock_df, sent_df, config)

    print(f"\n{'='*70}")
    print(f"分析完了: {len(result)}銘柄")
    print(f"{'='*70}")
    for _, r in result.iterrows():
        print(f"\n{r['signal']:14s} | {r['ticker']:6s} ({r['name']})")
        print(f"  総合スコア: {r['total_score']:.1f}点")
        print(f"  コメント: {r['analysis_comment']}")
        print(f"  テクニカル: {r['tech_score']}点 - {r['tech_reasons']}")
        print(f"  ファンダ  : {r['fund_score']}点 - {r['fund_reasons']}")
        print(f"  センチメント: {r['sent_score']}点 - {r['sent_reasons']}")
        print(f"  モメンタム: {r['mom_score']}点 - {r['mom_reasons']}")
