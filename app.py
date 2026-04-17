"""
app.py - Growth Stock Analyzer Streamlit ダッシュボード

Usage: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


# ──────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_macro():
    p = DATA_DIR / "latest_macro.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data(ttl=300)
def load_analysis():
    p = DATA_DIR / "latest_analysis.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

@st.cache_data(ttl=300)
def load_ohlc_dynamic(ticker, period="1y", interval="1d"):
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if not hist.empty:
            return hist[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_predictions():
    p = DATA_DIR / "latest_predictions.csv"
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None

def get_prediction_for_ticker(ticker, pred_df):
    """予測CSVから該当銘柄の予測データを取得"""
    if pred_df is None:
        return {}
    row = pred_df[pred_df["ticker"] == ticker]
    if row.empty:
        return {}
    r = row.iloc[0]
    base_price = r.get("pred_current_price", None)
    out = {}
    if base_price is not None and pd.notna(base_price):
        out["base_price"] = float(base_price)
    for d in [1, 7, 30]:
        pct = r.get(f"pred_{d}d_pct")
        if pd.notna(pct):
            out[f"p{d}"] = float(pct)
            out[f"p{d}_l80"] = float(r.get(f"pred_{d}d_l80", 0))
            out[f"p{d}_u80"] = float(r.get(f"pred_{d}d_u80", 0))
            # 予測株価を算出
            if "base_price" in out:
                out[f"p{d}_price"] = round(out["base_price"] * (1 + float(pct) / 100), 2)
    return out

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


# ──────────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────────

st.set_page_config(page_title="Growth Stock Analyzer", layout="wide")
st.sidebar.title("Growth Stock Analyzer")
page = st.sidebar.radio(
    "ページ選択",
    ["ダッシュボード", "買い推奨TOP10", "予測精度検証"],
)


# ──────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────

def short_signal(s):
    if pd.isna(s): return "様子見"
    s = str(s)
    if "強い買い" in s: return "★★★買い"
    if "★★" in s and "買い" in s: return "★★買い"
    if "要注目" in s: return "★注目"
    return "様子見"

def short_risk(labels, count):
    if pd.isna(count) or int(count) == 0: return "-"
    rc = int(count)
    if rc >= 2: return f"🔴要注意({rc}件)"
    m = {"急落アラート":"⚠️急落","高ボラティリティ警告":"⚠️高ボラ",
         "PER異常値":"⚠️PER異常","52週安値更新中":"⚠️安値更新","逆行シグナル":"⚠️逆行"}
    parts = [l.strip() for l in str(labels).split("/") if l.strip()]
    return " ".join(m.get(p, f"⚠️{p[:4]}") for p in parts) or "⚠️"

def fmt_vol(vr, volume):
    if pd.isna(vr): return "-"
    vr = float(vr)
    vs = ""
    if pd.notna(volume):
        vm = float(volume)/10000
        vs = f" ({vm:,.0f}万株)" if vm >= 1 else f" ({float(volume):,.0f}株)"
    icon = "🔥🔥" if vr >= 3 else ("🔥" if vr >= 2 else "")
    return f"{icon} {vr:.1f}x{vs}" if icon else f"{vr:.1f}x{vs}"


# ──────────────────────────────────────────────
# ページ1: ダッシュボード
# ──────────────────────────────────────────────

def page_dashboard():
    st.header("ダッシュボード")
    macro_df = load_macro()
    analysis_df = load_analysis()

    # データ最終更新日時を表示（CSVのanalysis_timeから取得）
    if analysis_df is not None and "analysis_time" in analysis_df.columns:
        try:
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            _latest = pd.to_datetime(analysis_df["analysis_time"]).max()
            _jst = _tz(_td(hours=9))
            _updated = _latest.tz_localize("UTC").astimezone(_jst).strftime("%Y/%m/%d %H:%M:%S")
            st.caption(f"🕐 データ最終更新: {_updated} (JST)")
        except Exception:
            pass

    # --- 市況判定 + リスクサマリー ---
    hcols = st.columns([3, 1])
    with hcols[0]:
        if macro_df is not None and not macro_df.empty:
            try:
                from analyzer import calculate_macro_score
                mr = calculate_macro_score(macro_df)
                st.markdown(
                    f"### {mr['emoji']} 市況判定: **{mr['label']}** ({mr['score']}/100)",
                )
                # サマリーコメント
                summary = mr.get("summary", "")
                if summary:
                    st.markdown(f"<div style='font-size:13px;line-height:1.6;white-space:pre-wrap'>"
                                f"<b>【サマリー】</b>\n{summary}</div>",
                                unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"市況判定失敗: {e}")
    with hcols[1]:
        if analysis_df is not None and "risk_count" in analysis_df.columns:
            total = len(analysis_df)
            risk_n = len(analysis_df[analysis_df["risk_count"] > 0])
            if risk_n > 0:
                st.markdown(f"### ⚠️ リスク検出\n{total}銘柄中**{risk_n}銘柄**")
            else:
                st.markdown(f"### ✅ リスクなし\n{total}銘柄")

    st.markdown("---")

    # --- マクロ指標 ---
    if macro_df is not None and not macro_df.empty:
        targets = {"^VIX":"VIX","^GSPC":"S&P500","^N225":"日経225","USDJPY=X":"ドル円"}
        cols = st.columns(len(targets))
        for col, (tk, lab) in zip(cols, targets.items()):
            row = macro_df[macro_df["ticker"]==tk]
            if not row.empty:
                r = row.iloc[0]
                d = f"{r['change_1d_pct']:+.2f}%" if pd.notna(r.get("change_1d_pct")) else None
                col.metric(lab, f"{r['current_value']:,.2f}", d)
            else:
                col.metric(lab, "N/A")

    if analysis_df is None or analysis_df.empty:
        st.warning("分析データが見つかりません。先に `python run.py` を実行してください。")
        return

    # --- 市場フィルター ---
    st.subheader("全銘柄ランキング")
    market_filter = st.radio(
        "市場",
        ["すべて", "🇺🇸 米国株", "🇯🇵 日本株", "📈 ETF"],
        horizontal=True,
    )
    st.caption("📊 5つのAIエージェント（テクニカル・モメンタム・リバーサル・ボラティリティ・トレンド）のアンサンブル予測")
    st.caption("前日比: 前営業日の終値からの変動率 | 予測: 本日の終値を起点にした予測変動率")
    pred_headers = {
        1: ("1日後予測", "5エージェントの総合予測。短期の方向性は比較的信頼できる"),
        7: ("1週後予測", "1週間後の総合予測。イベントリスクで変動の可能性あり"),
        30: ("1月後予測", "1ヶ月後の総合予測。決算や材料で大きく変動する可能性あり"),
    }

    pred_df = load_predictions()

    # 市場フィルター適用
    ETF_TICKERS = ["VOO","QQQ","SOXX","ARKK","XLF","XLE","XLV"]
    filtered_df = analysis_df.copy()
    if market_filter == "🇺🇸 米国株":
        filtered_df = filtered_df[
            (filtered_df["market"] == "US") &
            (~filtered_df["ticker"].isin(ETF_TICKERS))
        ]
    elif market_filter == "🇯🇵 日本株":
        filtered_df = filtered_df[filtered_df["market"] == "JP"]
    elif market_filter == "📈 ETF":
        filtered_df = filtered_df[filtered_df["ticker"].isin(ETF_TICKERS)]

    st.caption(f"表示: {len(filtered_df)}銘柄 / 全{len(analysis_df)}銘柄")

    rows = []
    for _, r in filtered_df.sort_values("total_score", ascending=False).iterrows():
        tk = r["ticker"]
        name = r.get("name", "")
        market = r.get("market", "US")
        preds = get_prediction_for_ticker(tk, pred_df)

        # Google検索リンク生成
        if market == "JP" or tk.endswith(".T"):
            search_url = f"https://www.google.com/search?q={name}+株価"
        else:
            search_url = f"https://www.google.com/search?q={tk}+stock+price"

        rd = {
            "銘柄": tk,
            "名前": {"text": name, "url": search_url},
            "シグナル": short_signal(r.get("signal","")),
            "スコア": r.get("total_score",0),
            "株価": r.get("price",0),
            "前日比(%)": r.get("change_pct",0),
        }
        for days, (hdr, _) in pred_headers.items():
            pct = preds.get(f"p{days}")
            l80 = preds.get(f"p{days}_l80")
            u80 = preds.get(f"p{days}_u80")
            pred_price = preds.get(f"p{days}_price")
            if pct is not None:
                rd[hdr] = {"pct": pct, "l80": l80, "u80": u80, "price": pred_price}
            else:
                rd[hdr] = None
        rd["出来高比率"] = fmt_vol(r.get("vol_ratio"), r.get("volume"))
        rd["RSI"] = round(r.get("rsi_14",0),1) if pd.notna(r.get("rsi_14")) else "-"
        rd["52週位置"] = f"{r.get('pos_52w_pct',0):.0f}%" if pd.notna(r.get("pos_52w_pct")) else "-"
        rd["リスク"] = short_risk(r.get("risk_labels",""), r.get("risk_count",0))
        rows.append(rd)

    pred_col_names = [h for _, (h, _) in pred_headers.items()]
    header_tips = {h: tip for _, (h, tip) in pred_headers.items()}
    all_cols = ["銘柄","名前","シグナル","スコア","株価","前日比(%)"] + pred_col_names + ["出来高比率","RSI","52週位置","リスク"]

    html = '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px">'
    html += '<thead><tr style="background:#1a1a2e;color:white">'
    for c in all_cols:
        tip = header_tips.get(c, "")
        ta = f' title="{tip}"' if tip else ""
        html += f'<th style="padding:5px 6px;white-space:nowrap;cursor:help"{ta}>{c}</th>'
    html += '</tr></thead><tbody>'

    for rd in rows:
        risk_str = str(rd.get("リスク","✅"))
        has_risk = "⚠️" in risk_str or "🔴" in risk_str
        bg = "background:rgba(255,0,0,0.06);" if has_risk else ""
        html += f'<tr style="{bg}border-bottom:1px solid #eee">'
        for c in all_cols:
            val = rd.get(c, "-")
            style = "padding:4px 6px;white-space:nowrap;"
            if c == "名前":
                if isinstance(val, dict):
                    url = val["url"]
                    text = val["text"]
                    html += f'<td style="{style}"><a href="{url}" target="_blank" style="color:inherit;text-decoration:none">{text} 🔗</a></td>'
                else:
                    html += f'<td style="{style}">{val}</td>'
            elif c in pred_col_names:
                if val is None:
                    html += f'<td style="{style}">-</td>'
                else:
                    pct = val["pct"]
                    l80 = val.get("l80")
                    u80 = val.get("u80")
                    pred_price = val.get("price")
                    if l80 is not None and l80 > 0:
                        color, fw = "#006400", "bold"
                    elif pct >= 0:
                        color, fw = "#228B22", "normal"
                    else:
                        color, fw = "red", "normal"
                    # 予測株価 + 変動率
                    if pred_price is not None:
                        if pred_price > 1000:
                            price_str = f"{pred_price:,.0f}"
                        else:
                            price_str = f"{pred_price:,.2f}"
                        cell = f'<span style="color:{color};font-weight:{fw}">{price_str}<br><small>({pct:+.1f}%)</small></span>'
                    else:
                        cell = f'<span style="color:{color};font-weight:{fw}">{pct:+.1f}%</span>'
                    tip = f"予測: {pct:+.1f}% | 80%信頼区間: {l80:+.1f}%~{u80:+.1f}%" if l80 is not None else ""
                    html += f'<td style="{style}" title="{tip}">{cell}</td>'
            elif c == "前日比(%)":
                try:
                    v = float(val)
                    color = "green" if v >= 0 else "red"
                    html += f'<td style="{style}"><span style="color:{color}">{v:+.2f}%</span></td>'
                except (ValueError, TypeError):
                    html += f'<td style="{style}">{val}</td>'
            elif c == "スコア":
                try:
                    html += f'<td style="{style}"><b>{float(val):.1f}</b></td>'
                except (ValueError, TypeError):
                    html += f'<td style="{style}">{val}</td>'
            elif c == "株価":
                try:
                    html += f'<td style="{style}">{float(val):,.2f}</td>'
                except (ValueError, TypeError):
                    html += f'<td style="{style}">{val}</td>'
            else:
                html += f'<td style="{style}">{val}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'

    st.markdown(html, unsafe_allow_html=True)
    st.caption("予測カラムにマウスを合わせると信頼区間が表示されます。"
               "80%確率=10回中8回はこの範囲に収まる見込み。")


# ──────────────────────────────────────────────
# ページ2: 買い推奨TOP10
# ──────────────────────────────────────────────

def _up_probability(expected_pct, l95, u95):
    """95%信頼区間から正規分布を仮定して上昇確率を算出

    - mean = expected_pct
    - std  = (u95 - l95) / (2 * 1.96)
    - P(ret > 0) = Φ(mean / std)
    """
    try:
        from math import erf, sqrt
        if pd.isna(expected_pct) or pd.isna(l95) or pd.isna(u95):
            return None
        std = (float(u95) - float(l95)) / (2 * 1.96)
        if std <= 0:
            return None
        z = float(expected_pct) / std
        return 0.5 * (1 + erf(z / sqrt(2)))
    except Exception:
        return None


def _historical_hit_rates():
    """horizon別の過去的中率を返す（サイドバーに表示する指標）"""
    try:
        acc_path = DATA_DIR / "prediction_accuracy.csv"
        if not acc_path.exists():
            return {}
        acc = pd.read_csv(acc_path)
        rates = {}
        # horizon_days 列で正規化
        if "horizon_days" in acc.columns:
            for d in [1, 7, 30]:
                sub = acc[acc["horizon_days"] == d]
                if len(sub):
                    rates[d] = sub["direction_correct"].mean() * 100
        return rates
    except Exception:
        return {}


def _signal_action(signal, score, hybrid):
    """シグナル文字列からアクション推奨文言を生成"""
    s = str(signal or "")
    if "強い買い" in s or hybrid >= 70:
        return ("🟢 本格エントリー", "rgba(46,204,113,0.18)")
    if "★★" in s and "買い" in s:
        return ("🟢 分割で打診買い", "rgba(46,204,113,0.12)")
    if "要注目" in s:
        return ("🟡 打診買い（少額）", "rgba(241,196,15,0.14)")
    return ("⚪ 様子見／指値待ち", "rgba(150,150,150,0.10)")


def page_recommend():
    st.header("💎 今買うべき銘柄 TOP10")
    st.caption("スコア × 予測ハイブリッドで並び替えた、中期で最も買い妙味のある10銘柄。1日/1週/1ヶ月後の期待変化率・上昇確率・80%信頼区間つき。")

    df = load_analysis()
    pred_df = load_predictions()

    if df is None or df.empty:
        st.warning("分析データがありません。まず `python run.py` を実行してください。")
        return
    if pred_df is None or pred_df.empty:
        st.warning("予測データ (latest_predictions.csv) がありません。`python run.py` を実行してください。")
        return

    # データ最新時刻
    try:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        _latest = pd.to_datetime(df["analysis_time"]).max()
        _jst = _tz(_td(hours=9))
        _updated = _latest.tz_localize("UTC").astimezone(_jst).strftime("%Y/%m/%d %H:%M:%S")
        st.caption(f"🕐 データ最終更新: {_updated} (JST)")
    except Exception:
        pass

    # 市場フィルタ
    market_filter = st.radio(
        "市場", ["すべて", "🇺🇸 米国株", "🇯🇵 日本株"],
        horizontal=True, key="recommend_market",
    )
    ETF_TICKERS = ["VOO","QQQ","SOXX","ARKK","XLF","XLE","XLV"]
    work = df.copy()
    if market_filter == "🇺🇸 米国株":
        work = work[(work["market"]=="US") & (~work["ticker"].isin(ETF_TICKERS))]
    elif market_filter == "🇯🇵 日本株":
        work = work[work["market"]=="JP"]
    # ETFはそもそも除外
    work = work[~work["ticker"].isin(ETF_TICKERS)]

    # 予測をマージ
    pred_cols = ["ticker","pred_1d_pct","pred_1d_l80","pred_1d_u80","pred_1d_l95","pred_1d_u95",
                 "pred_7d_pct","pred_7d_l80","pred_7d_u80","pred_7d_l95","pred_7d_u95",
                 "pred_30d_pct","pred_30d_l80","pred_30d_u80","pred_30d_l95","pred_30d_u95",
                 "pred_1d_conf","pred_7d_conf","pred_30d_conf"]
    keep = [c for c in pred_cols if c in pred_df.columns]
    merged = work.merge(pred_df[keep], on="ticker", how="left")

    # 上昇確率
    merged["prob_1d"] = merged.apply(lambda r: _up_probability(r.get("pred_1d_pct"), r.get("pred_1d_l95"), r.get("pred_1d_u95")), axis=1)
    merged["prob_7d"] = merged.apply(lambda r: _up_probability(r.get("pred_7d_pct"), r.get("pred_7d_l95"), r.get("pred_7d_u95")), axis=1)
    merged["prob_30d"] = merged.apply(lambda r: _up_probability(r.get("pred_30d_pct"), r.get("pred_30d_l95"), r.get("pred_30d_u95")), axis=1)

    # ハイブリッドスコア: 0.5×スコア正規化 + 0.5×(1週期待リターン × 上昇確率)
    #   期待値正規化: max 15%で打ち切り → 0..1
    def _ev_norm(pct, prob):
        if pd.isna(pct) or pd.isna(prob):
            return 0.0
        ev = float(pct) * float(prob)           # 期待値（%）
        # -5 .. +10 を 0..1 にマップ（上昇側を少し広めに取る）
        return max(0.0, min(1.0, (ev + 5) / 15))

    merged["hybrid_score"] = merged.apply(
        lambda r: 0.5 * (float(r.get("total_score", 0) or 0)/100)
                + 0.5 * _ev_norm(r.get("pred_7d_pct"), r.get("prob_7d")),
        axis=1
    ) * 100

    # 予測データが欠けている銘柄は除外
    merged = merged.dropna(subset=["pred_7d_pct", "prob_7d"])

    top = merged.sort_values("hybrid_score", ascending=False).head(10).reset_index(drop=True)

    if top.empty:
        st.warning("推奨できる銘柄が見つかりませんでした。")
        return

    # サイドバー: 過去的中率
    rates = _historical_hit_rates()
    if rates:
        st.sidebar.markdown("**📊 過去の方向的中率**")
        for d, lab in [(1,"1日"), (7,"1週"), (30,"1ヶ月")]:
            if d in rates:
                st.sidebar.caption(f"{lab}: {rates[d]:.1f}%")
        st.sidebar.caption("※ 50%がランダム、55%↑で良好")

    # 凡例
    st.markdown("""
<div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px 14px;font-size:12px;line-height:1.7;margin:4px 0 16px 0">
<b>📖 読み方:</b> ハイブリッドスコア = 総合スコア(アナライザー) × 0.5 + 7日後期待値(予測×上昇確率) × 0.5 <br>
<b>上昇確率</b> = 95%信頼区間から正規分布を仮定して算出したP(変化率&gt;0) <br>
<b>80%信頼区間</b> = 10回中8回はこの範囲に収まる見込み
</div>
""", unsafe_allow_html=True)

    # 各銘柄カード
    for i, row in top.iterrows():
        rank = i + 1
        ticker = row["ticker"]
        name = row.get("name", "")
        price = row.get("price", 0)
        change = row.get("change_pct", 0)
        market = row.get("market", "")
        sector = row.get("sector", "")
        total_score = row.get("total_score", 0)
        signal = row.get("signal", "")
        hybrid = row.get("hybrid_score", 0)
        analysis_comment = row.get("analysis_comment", "")
        rsi = row.get("rsi_14", None)
        pos52 = row.get("pos_52w_pct", None)
        rev_growth = row.get("revenue_growth_pct", None)
        risk_labels = row.get("risk_labels", "")
        upside = row.get("upside_pct", None)

        action_label, action_bg = _signal_action(signal, total_score, hybrid)
        mkt_flag = "🇺🇸" if market == "US" else ("🇯🇵" if market == "JP" else "🌐")
        change_color = "#27ae60" if change >= 0 else "#e74c3c"

        # カードヘッダ
        st.markdown(f"""
<div style="background:{action_bg};border-radius:10px;padding:14px 18px;margin:8px 0">
<div style="display:flex;justify-content:space-between;align-items:center">
<div style="font-size:18px;font-weight:bold">
  #{rank} {mkt_flag} <code>{ticker}</code> {name}
  <span style="font-size:13px;font-weight:normal;color:#aaa">（{sector}）</span>
</div>
<div style="font-size:16px">
  <span style="font-weight:bold">¥{price:,.2f}</span>
  <span style="color:{change_color};margin-left:8px">{change:+.2f}%</span>
</div>
</div>
<div style="margin-top:8px;font-size:13px;color:#ddd">
  {action_label} ｜ ハイブリッドスコア: <b>{hybrid:.1f}</b> ｜ 総合スコア: {total_score:.1f} ｜ シグナル: {signal}
</div>
</div>
""", unsafe_allow_html=True)

        cA, cB = st.columns([3, 2])

        # 左: 予測テーブル
        with cA:
            st.markdown("**📈 期間別予測**")
            pred_rows = []
            for d, lab, hist_key in [(1,"1日後",1), (7,"1週間後",7), (30,"1ヶ月後",30)]:
                pct = row.get(f"pred_{d}d_pct")
                prob = row.get(f"prob_{d}d")
                l80 = row.get(f"pred_{d}d_l80")
                u80 = row.get(f"pred_{d}d_u80")
                hist = rates.get(hist_key, None)
                if pd.isna(pct):
                    continue
                pred_price = price * (1 + pct/100)
                prob_str = f"{prob*100:.1f}%" if prob is not None and not pd.isna(prob) else "-"
                hist_str = f" (過去的中率 {hist:.0f}%)" if hist is not None else ""
                pred_rows.append({
                    "期間": lab,
                    "期待変化率": f"{pct:+.2f}%",
                    "予想株価": f"{pred_price:,.2f}",
                    "上昇確率": prob_str + hist_str,
                    "80%信頼区間": f"[{l80:+.2f}% 〜 {u80:+.2f}%]",
                })
            if pred_rows:
                st.dataframe(pd.DataFrame(pred_rows), hide_index=True, width="stretch")

        # 右: 選出理由
        with cB:
            st.markdown("**💡 選出理由**")
            reasons = []
            if pd.notna(rsi):
                if rsi <= 30:
                    reasons.append(f"RSI {rsi:.0f}（売られすぎ反発期待）")
                elif rsi <= 45:
                    reasons.append(f"RSI {rsi:.0f}（やや売られすぎ）")
            if pd.notna(pos52):
                if pos52 <= 15:
                    reasons.append(f"52週位置 {pos52:.0f}%（底値圏）")
                elif pos52 <= 35:
                    reasons.append(f"52週位置 {pos52:.0f}%（調整完了圏）")
            if pd.notna(rev_growth) and rev_growth >= 20:
                reasons.append(f"売上成長 +{rev_growth:.0f}%（高成長）")
            if pd.notna(upside) and upside >= 20:
                reasons.append(f"目標株価アップサイド +{upside:.0f}%")
            p7 = row.get("pred_7d_pct")
            pr7 = row.get("prob_7d")
            if pd.notna(p7) and p7 > 0 and pr7 is not None and pr7 >= 0.55:
                reasons.append(f"1週予測 +{p7:.1f}% × 上昇確率 {pr7*100:.0f}%")
            if not reasons:
                reasons.append("総合スコア上位")
            for r in reasons:
                st.markdown(f"- {r}")

            # リスクと詳細
            if pd.notna(risk_labels) and str(risk_labels).strip():
                st.markdown(f"**⚠️ リスク**: {risk_labels}")

            if pd.notna(analysis_comment) and str(analysis_comment).strip():
                with st.expander("📝 アナライザーコメント"):
                    st.caption(str(analysis_comment))

        st.markdown("---")

    # まとめダウンロード用CSV
    export_cols = ["ticker","name","sector","market","price","change_pct",
                   "hybrid_score","total_score","signal",
                   "pred_1d_pct","prob_1d","pred_7d_pct","prob_7d","pred_30d_pct","prob_30d",
                   "rsi_14","pos_52w_pct","revenue_growth_pct","upside_pct","analysis_comment"]
    avail = [c for c in export_cols if c in top.columns]
    csv = top[avail].to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 TOP10をCSVダウンロード", csv, file_name="top10_recommend.csv", mime="text/csv")


# ──────────────────────────────────────────────
# ページ3: 予測精度検証
# ──────────────────────────────────────────────

def page_accuracy():
    st.header("予測精度検証")
    st.caption("過去の予測が実際にどれだけ当たったかを検証します。データは毎日蓄積されます。")

    # 精度データ読み込み
    acc_path = DATA_DIR / "prediction_accuracy.csv"
    if not acc_path.exists():
        st.info("まだ予測精度データがありません。毎日データが更新されると、翌日以降に予測の的中率が表示されます。")
        st.markdown("---")
        st.markdown("**仕組み:** 毎朝の予測をスナップショットとして保存 → 翌日以降、実際の株価と比較して精度を算出")
        return

    acc_df = pd.read_csv(acc_path)
    if acc_df.empty:
        st.info("評価可能な予測データがまだありません。明日以降に結果が表示されます。")
        return

    # === サマリー ===
    st.subheader("全体サマリー")
    total = len(acc_df)
    direction_acc = acc_df["direction_correct"].mean() * 100
    avg_error = acc_df["error_pct"].abs().mean()
    avg_pred = acc_df["predicted_change_pct"].mean()
    avg_actual = acc_df["actual_change_pct"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("総予測件数", f"{total:,}件")
    c2.metric("方向的中率", f"{direction_acc:.1f}%",
              delta="良好" if direction_acc >= 55 else "改善余地あり",
              delta_color="normal" if direction_acc >= 55 else "inverse")
    c3.metric("平均誤差", f"{avg_error:.2f}%")

    # わかりやすい解説
    st.markdown(f"""
<div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:12px 16px;font-size:13px;line-height:1.8;margin:8px 0">
<b>📖 この数字の見方:</b><br>
• <b>方向的中率 {direction_acc:.0f}%</b> →「上がる/下がる」の方向を当てた割合。50%がランダム、<b>55%以上で良好</b>、60%超で優秀<br>
• <b>平均誤差 {avg_error:.2f}%</b> → 予測と実際の株価変動のズレ幅。<b>2%以下なら良好</b>、5%超なら改善が必要<br>
• <b>AIの予測傾向</b> → 平均で「{avg_pred:+.2f}%動く」と予測したが、実際は「{avg_actual:+.2f}%」だった
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # === ホライズン別精度 ===
    st.subheader("予測期間別の精度")
    horizon_order = ["1日後", "1週後", "1ヶ月後"]
    horizon_data = []
    for h in horizon_order:
        h_df = acc_df[acc_df["horizon"] == h]
        if len(h_df) > 0:
            h_acc = h_df['direction_correct'].mean()*100
            h_pred = h_df['predicted_change_pct'].mean()
            h_actual = h_df['actual_change_pct'].mean()
            h_err = h_df['error_pct'].abs().mean()
            horizon_data.append({
                "予測期間": h,
                "件数": len(h_df),
                "方向的中率": f"{h_acc:.1f}%",
                "AI予測(平均)": f"{h_pred:+.2f}%",
                "実際の結果(平均)": f"{h_actual:+.2f}%",
                "平均誤差": f"{h_err:.2f}%",
                "評価": "🎯" if h_acc >= 55 else "⚠️" if h_acc >= 45 else "❌",
            })
    if horizon_data:
        st.dataframe(pd.DataFrame(horizon_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # === フィルター ===
    st.subheader("全銘柄 予測 vs 実績")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        horizon_filter = fc1.selectbox("予測期間", ["1日後"] + [h for h in horizon_order if h != "1日後"])
    with fc2:
        result_filter = fc2.selectbox("結果", ["すべて", "◎的中のみ", "✕外れのみ"])
    with fc3:
        date_options = sorted(acc_df["prediction_date"].unique(), reverse=True)
        date_filter = fc3.selectbox("予測日", ["すべて（最新）"] + list(date_options))

    # フィルター適用
    filtered = acc_df[acc_df["horizon"] == horizon_filter].copy()
    if result_filter == "◎的中のみ":
        filtered = filtered[filtered["direction_correct"] == True]
    elif result_filter == "✕外れのみ":
        filtered = filtered[filtered["direction_correct"] == False]

    if date_filter != "すべて（最新）":
        filtered = filtered[filtered["prediction_date"] == date_filter]
    else:
        # 各銘柄の最新予測日のみ表示
        if not filtered.empty:
            filtered = filtered.sort_values("prediction_date", ascending=False)
            filtered = filtered.drop_duplicates(subset=["ticker"], keep="first")

    if filtered.empty:
        st.info("条件に合う予測がありません")
    else:
        # --- 全銘柄一覧テーブル (HTML) ---
        filtered = filtered.sort_values("error_pct", key=abs)

        html = '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px">'
        html += '<thead><tr style="background:#1a1a2e;color:white">'
        headers = ["銘柄", "名前", "予測日", "基準価格", "現在価格", "予測(%)", "実際(%)", "誤差(%)", "方向", "判定"]
        for h in headers:
            html += f'<th style="padding:6px 8px;white-space:nowrap;text-align:left">{h}</th>'
        html += '</tr></thead><tbody>'

        for _, r in filtered.iterrows():
            dc = r.get("direction_correct", False)
            bg = "background:rgba(0,200,0,0.08);" if dc else "background:rgba(255,0,0,0.08);"
            html += f'<tr style="{bg}border-bottom:1px solid rgba(255,255,255,0.1)">'

            pred_pct = r.get("predicted_change_pct", 0)
            actual_pct = r.get("actual_change_pct", 0)
            error_pct = r.get("error_pct", 0)

            pred_color = "color:#4CAF50" if pred_pct > 0 else "color:#f44336" if pred_pct < 0 else ""
            actual_color = "color:#4CAF50" if actual_pct > 0 else "color:#f44336" if actual_pct < 0 else ""

            direction_icon = "◎" if dc else "✕"
            direction_style = "color:#4CAF50;font-weight:bold" if dc else "color:#f44336;font-weight:bold"

            # 精度判定ラベル
            abs_err = abs(error_pct)
            if dc and abs_err < 2:
                judge = "🎯 精度良好"
            elif dc and abs_err < 5:
                judge = "○ 方向的中"
            elif dc:
                judge = "△ 方向的中(大誤差)"
            else:
                judge = "✕ 外れ"

            base_p = r.get("base_price", 0)
            actual_p = r.get("actual_price", 0)
            # 価格フォーマット（日本株は整数、米国株は小数2桁）
            def fmt_price(p):
                if p > 1000:
                    return f"{p:,.0f}"
                return f"{p:,.2f}"

            cells = [
                (str(r.get("ticker", "")), ""),
                (str(r.get("name", ""))[:10], ""),
                (str(r.get("prediction_date", "")), ""),
                (fmt_price(base_p), "text-align:right"),
                (fmt_price(actual_p), "text-align:right"),
                (f"{pred_pct:+.2f}%", f"text-align:right;{pred_color}"),
                (f"{actual_pct:+.2f}%", f"text-align:right;{actual_color}"),
                (f"{error_pct:+.2f}%", "text-align:right"),
                (direction_icon, f"text-align:center;{direction_style}"),
                (judge, "white-space:nowrap"),
            ]
            for val, style in cells:
                html += f'<td style="padding:4px 8px;{style}">{val}</td>'
            html += '</tr>'

        html += '</tbody></table></div>'
        st.markdown(html, unsafe_allow_html=True)
        st.caption(f"表示: {len(filtered)}件")

    st.markdown("---")

    # === 銘柄別的中率ランキング ===
    st.subheader("銘柄別 的中率ランキング")
    oneday = acc_df[acc_df["horizon"] == "1日後"]
    if not oneday.empty:
        ticker_stats = oneday.groupby("ticker").agg(
            name=("name", "first"),
            count=("direction_correct", "count"),
            accuracy=("direction_correct", "mean"),
            avg_error=("error_pct", lambda x: x.abs().mean()),
            avg_predicted=("predicted_change_pct", "mean"),
            avg_actual=("actual_change_pct", "mean"),
        ).reset_index()
        ticker_stats["accuracy"] = (ticker_stats["accuracy"] * 100).round(1)
        ticker_stats["avg_error"] = ticker_stats["avg_error"].round(2)
        ticker_stats = ticker_stats.sort_values("accuracy", ascending=False)

        rank_html = '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px">'
        rank_html += '<thead><tr style="background:#1a1a2e;color:white">'
        for h in ["#", "銘柄", "名前", "予測回数", "的中率", "平均予測", "平均実際", "平均誤差"]:
            rank_html += f'<th style="padding:6px 8px;text-align:left">{h}</th>'
        rank_html += '</tr></thead><tbody>'

        for i, (_, r) in enumerate(ticker_stats.iterrows()):
            acc_val = r["accuracy"]
            bg = "background:rgba(0,200,0,0.08);" if acc_val >= 55 else "background:rgba(255,0,0,0.08);" if acc_val < 45 else ""
            acc_color = "color:#4CAF50" if acc_val >= 55 else "color:#f44336" if acc_val < 45 else ""
            rank_html += f'<tr style="{bg}border-bottom:1px solid rgba(255,255,255,0.1)">'
            rank_html += f'<td style="padding:4px 8px">{i+1}</td>'
            rank_html += f'<td style="padding:4px 8px">{r["ticker"]}</td>'
            rank_html += f'<td style="padding:4px 8px">{str(r["name"])[:10]}</td>'
            rank_html += f'<td style="padding:4px 8px;text-align:right">{r["count"]}回</td>'
            rank_html += f'<td style="padding:4px 8px;text-align:right;font-weight:bold;{acc_color}">{acc_val:.1f}%</td>'
            rank_html += f'<td style="padding:4px 8px;text-align:right">{r["avg_predicted"]:+.2f}%</td>'
            rank_html += f'<td style="padding:4px 8px;text-align:right">{r["avg_actual"]:+.2f}%</td>'
            rank_html += f'<td style="padding:4px 8px;text-align:right">{r["avg_error"]:.2f}%</td>'
            rank_html += '</tr>'

        rank_html += '</tbody></table></div>'
        st.markdown(rank_html, unsafe_allow_html=True)

    st.markdown("---")

    # === 的中率チャート（日別推移） ===
    if len(acc_df["prediction_date"].unique()) > 1:
        st.subheader("方向的中率の推移（1日後予測）")
        oneday_daily = oneday.groupby("prediction_date").agg(
            accuracy=("direction_correct", "mean"),
            count=("direction_correct", "count")
        ).reset_index()
        oneday_daily["accuracy"] = oneday_daily["accuracy"] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=oneday_daily["prediction_date"], y=oneday_daily["accuracy"],
                             name="的中率", marker_color=[
                                 "#4CAF50" if a >= 55 else "#FF9800" if a >= 45 else "#f44336"
                                 for a in oneday_daily["accuracy"]
                             ]))
        fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%ライン（ランダム）")
        fig.update_layout(yaxis_title="方向的中率(%)", yaxis_range=[0, 100], height=350)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# ルーティング
# ──────────────────────────────────────────────

if page == "ダッシュボード": page_dashboard()
elif page == "買い推奨TOP10": page_recommend()
elif page == "予測精度検証": page_accuracy()
