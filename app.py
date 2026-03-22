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
    out = {}
    for d in [1, 7, 30, 90, 180]:
        pct = r.get(f"pred_{d}d_pct")
        if pd.notna(pct):
            out[f"p{d}"] = float(pct)
            out[f"p{d}_l80"] = float(r.get(f"pred_{d}d_l80", 0))
            out[f"p{d}_u80"] = float(r.get(f"pred_{d}d_u80", 0))
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
    ["ダッシュボード", "銘柄詳細", "ポートフォリオ", "スクリーニング", "予測シミュレーション", "予測精度検証"],
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
    st.caption("📊 予測精度の目安: 1日後は高精度 → 6ヶ月後は参考程度（期間が長いほどブレが大きくなります）")
    st.caption("前日比: 前営業日の終値からの変動率 | 予測: 本日の終値を起点にした予測変動率")
    pred_headers = {
        1: ("1日後(80%)", "予測した騰落の方向(上がるor下がる)が当たる確率が約80%"),
        7: ("1週後(75%)", "1週間後の方向予測。当たる確率は約75%"),
        30: ("1月後(65%)", "1ヶ月後の方向予測。決算や材料で変動の可能性あり。約65%"),
        90: ("3月後(50%)", "3ヶ月後の予測は不確実。方向が当たる確率は約50%"),
        180: ("6月後(40%)", "半年後の予測は不確実。方向が当たる確率は約40%程度"),
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
            if pct is not None:
                rd[hdr] = {"pct": pct, "l80": l80, "u80": u80}
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
                    if l80 is not None and l80 > 0:
                        color, fw = "#006400", "bold"
                    elif pct >= 0:
                        color, fw = "#228B22", "normal"
                    else:
                        color, fw = "red", "normal"
                    cell = f'<span style="color:{color};font-weight:{fw}">{pct:+.1f}%</span>'
                    tip = f"予測: {pct:+.1f}% | この予測が当たる範囲: 80%の確率で{l80:+.1f}%~{u80:+.1f}%" if l80 is not None else ""
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
# ページ2: 銘柄詳細
# ──────────────────────────────────────────────

PERIOD_MAP = {
    "1日":("1d","5m"), "1週":("5d","15m"), "1ヶ月":("1mo","1h"),
    "6ヶ月":("6mo","1d"), "1年":("1y","1d"), "2年":("2y","1wk"), "5年":("5y","1wk"),
}

def page_detail():
    st.header("銘柄詳細")
    analysis_df = load_analysis()
    if analysis_df is None or analysis_df.empty:
        st.warning("分析データがありません。先に `python run.py` を実行してください。")
        return

    options = [f"{r['ticker']} - {r['name']}" for _, r in analysis_df.iterrows()]
    selected_opt = st.sidebar.selectbox("銘柄を選択", options)
    selected = selected_opt.split(" - ")[0]
    row = analysis_df[analysis_df["ticker"]==selected].iloc[0]
    ri = " ⚠️" if row.get("risk_count",0) > 0 else ""
    st.subheader(f"{selected}{ri} - {row.get('name','')}")

    indicators = {"RSI(14)":"rsi_14","MACD":"macd","BB位置":"bb_position",
                  "出来高比率":"vol_ratio","52週位置(%)":"pos_52w_pct","スコア":"total_score","シグナル":"signal"}
    cols = st.columns(len(indicators))
    for col, (lab, key) in zip(cols, indicators.items()):
        val = row.get(key, "N/A")
        col.metric(lab, f"{val:.2f}" if isinstance(val, float) else str(val))

    period_label = st.radio("チャート期間", list(PERIOD_MAP.keys()), index=4, horizontal=True)
    yf_period, yf_interval = PERIOD_MAP[period_label]

    ohlc = load_ohlc_dynamic(selected, period=yf_period, interval=yf_interval)
    if not ohlc.empty:
        ohlc_s = ohlc.sort_index()
        sma_w = [20,50,200] if yf_interval in ["1d","1wk"] else ([20,50] if yf_interval=="1h" else [20])
        for w in sma_w:
            if len(ohlc_s) >= w:
                ohlc_s[f"SMA{w}"] = ohlc_s["Close"].rolling(w).mean()

        rsi_s = compute_rsi(ohlc_s["Close"])
        macd_l, sig_l, hist_l = compute_macd(ohlc_s["Close"])

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.5,0.15,0.15,0.2], vertical_spacing=0.03,
                            subplot_titles=[f"{selected} {period_label}チャート","RSI","MACD","出来高"])

        fig.add_trace(go.Candlestick(x=ohlc_s.index, open=ohlc_s["Open"], high=ohlc_s["High"],
                                      low=ohlc_s["Low"], close=ohlc_s["Close"], name="OHLC"), row=1, col=1)
        for sma, c in {"SMA20":"orange","SMA50":"blue","SMA200":"red"}.items():
            if sma in ohlc_s.columns:
                fig.add_trace(go.Scatter(x=ohlc_s.index, y=ohlc_s[sma], mode="lines", name=sma,
                                          line=dict(color=c, width=1)), row=1, col=1)

        fig.add_trace(go.Scatter(x=ohlc_s.index, y=rsi_s, mode="lines", name="RSI",
                                  line=dict(color="purple", width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.add_trace(go.Scatter(x=ohlc_s.index, y=macd_l, mode="lines", name="MACD",
                                  line=dict(color="blue", width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlc_s.index, y=sig_l, mode="lines", name="Signal",
                                  line=dict(color="orange", width=1)), row=3, col=1)
        fig.add_trace(go.Bar(x=ohlc_s.index, y=hist_l, name="Histogram",
                              marker_color=["green" if v>=0 else "red" for v in hist_l.fillna(0)]), row=3, col=1)

        if "Volume" in ohlc_s.columns:
            va = ohlc_s["Volume"].rolling(20).mean()
            vc = ["red" if (pd.notna(a) and a>0 and v>a*3) else ("orange" if (pd.notna(a) and a>0 and v>a*2) else "steelblue")
                  for v, a in zip(ohlc_s["Volume"], va)]
            fig.add_trace(go.Bar(x=ohlc_s.index, y=ohlc_s["Volume"], name="出来高",
                                  marker_color=vc, opacity=0.7), row=4, col=1)

        fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])], row=1, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, height=900, showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        for i, t in enumerate(["価格","RSI","MACD","出来高"], 1):
            fig.update_yaxes(title_text=t, row=i, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("チャートデータを取得できませんでした。")

    # 出来高分析
    st.subheader("出来高分析")
    vr = row.get("vol_ratio")
    if pd.notna(vr):
        vr = float(vr)
        gc, tc = st.columns([1,2])
        with gc:
            fg = go.Figure(go.Indicator(mode="gauge+number", value=vr,
                title={"text":"出来高比率 (20日平均比)"},
                gauge={"axis":{"range":[0,5]}, "bar":{"color":"royalblue"},
                       "steps":[{"range":[0,1],"color":"lightgray"},{"range":[1,2],"color":"lightyellow"},
                                {"range":[2,3],"color":"orange"},{"range":[3,5],"color":"red"}]},
                number={"suffix":"x"}))
            fg.update_layout(height=250)
            st.plotly_chart(fg, use_container_width=True)
        with tc:
            if vr >= 3: st.error(f"🔥🔥 {vr:.1f}x - 出来高が異常に多い。材料出尽くしの反落に注意。")
            elif vr >= 2: st.warning(f"🔥 {vr:.1f}x - 取引が活発化。短期的に大きく動く可能性あり。")
            elif vr <= 0.5: st.info(f"{vr:.1f}x - 閑散相場。値動きは小さいが流動性リスクに注意。")
            else: st.success(f"{vr:.1f}x - 通常の取引量。")

    # スコア詳細
    st.subheader("スコア詳細")
    cr, ct = st.columns([1,2])
    with cr:
        cats = ["テクニカル","ファンダメンタルズ","センチメント","モメンタム"]
        vals = [row.get("tech_score",0), row.get("fund_score",0), row.get("sent_score",0), row.get("mom_score",0)]
        fr = go.Figure(data=go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
                                             line=dict(color="royalblue")))
        fr.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                         showlegend=False, height=350, title=f"{selected} スコアレーダー")
        st.plotly_chart(fr, use_container_width=True)
    with ct:
        ed = []
        for key, cat, lab, fmt_fn, meaning_fn in [
            ("rsi_14","テクニカル","RSI(14)", lambda v:f"{v:.1f}",
             lambda v:"売られすぎ→反発の可能性" if v<30 else ("買われすぎ→調整リスク" if v>70 else "中立的な水準")),
            ("macd","テクニカル","MACD", lambda v:f"{v:.4f}", lambda v:"買いの勢い" if v>0 else "売りの勢い"),
            ("bb_position","テクニカル","BB位置", lambda v:f"{v:.1f}%",
             lambda v:"バンド下限→反発期待" if v<20 else ("バンド上限→過熱" if v>80 else "中間付近")),
            ("pos_52w_pct","ファンダ","52週位置", lambda v:f"{v:.1f}%",
             lambda v:"安値圏→割安の可能性" if v<30 else ("高値圏" if v>70 else "中間水準")),
            ("pe_ratio","ファンダ","PER", lambda v:f"{v:.1f}x",
             lambda v:"割安" if v<20 else ("割高" if v>40 else "適正範囲")),
            ("vol_ratio","モメンタム","出来高比率", lambda v:f"{v:.2f}x",
             lambda v:"出来高急増" if v>2 else ("やや増" if v>1.2 else "平常")),
        ]:
            val = row.get(key)
            if pd.notna(val):
                ed.append({"カテゴリ":cat,"指標":lab,"値":fmt_fn(val),"解説":meaning_fn(val)})
        if ed: st.table(pd.DataFrame(ed))

    with st.expander("スコア算出理由の詳細"):
        for lab, cn in {"テクニカル":"tech_reasons","ファンダメンタルズ":"fund_reasons",
                        "センチメント":"sent_reasons","モメンタム":"mom_reasons"}.items():
            v = row.get(cn, "")
            if v: st.markdown(f"**{lab}**: {v}")

    comment = row.get("analysis_comment","")
    if comment:
        st.subheader("分析コメント")
        st.info(comment)


# ──────────────────────────────────────────────
# ページ3: ポートフォリオ
# ──────────────────────────────────────────────

def page_portfolio():
    st.header("ポートフォリオ")
    pf = DATA_DIR / "portfolio.csv"
    if not pf.exists():
        st.warning("ポートフォリオファイルが見つかりません。`data/portfolio.csv` を作成してください。")
        return
    with st.spinner("ポートフォリオを分析中..."):
        from portfolio import run_portfolio_analysis
        port_df, totals = run_portfolio_analysis()
    if port_df is None or port_df.empty:
        st.error("ポートフォリオ分析に失敗しました。")
        return
    tc = totals["total_current_jpy"] - totals["total_pnl_jpy"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("投資総額 (円)", f"¥{tc:,.0f}")
    c2.metric("評価総額 (円)", f"¥{totals['total_current_jpy']:,.0f}")
    c3.metric("損益合計 (円)", f"¥{totals['total_pnl_jpy']:,.0f}")
    c4.metric("損益率", f"{totals['total_pnl_pct']:+.2f}%")
    st.caption(f"USD/JPY: {totals['usdjpy_rate']:.2f}")
    st.subheader("保有一覧")
    st.dataframe(port_df, width="stretch", hide_index=True)
    st.subheader("ポートフォリオ構成")
    st.plotly_chart(px.pie(port_df, values="current_value_jpy", names="ticker", title="評価額構成比"),
                    use_container_width=True)


# ──────────────────────────────────────────────
# ページ4: スクリーニング
# ──────────────────────────────────────────────

def page_screening():
    st.header("スクリーニング")
    df = load_analysis()
    if df is None or df.empty:
        st.warning("分析データがありません。先に `python run.py` を実行してください。")
        return
    st.sidebar.subheader("フィルタ条件")
    f = {}
    if "rsi_14" in df.columns: f["rsi"] = st.sidebar.checkbox("RSI < 30", False)
    if "vol_ratio" in df.columns: f["vol"] = st.sidebar.checkbox("出来高比率 >= 2.0", False)
    if "pos_52w_pct" in df.columns: f["pos"] = st.sidebar.checkbox("52週位置 <= 30%", False)
    ms = st.sidebar.slider("最小スコア", 0.0, 100.0, 0.0, 5.0)
    filt = df[df["total_score"] >= ms].copy()
    if f.get("rsi"): filt = filt[filt["rsi_14"] < 30]
    if f.get("vol"): filt = filt[filt["vol_ratio"] >= 2.0]
    if f.get("pos"): filt = filt[filt["pos_52w_pct"] <= 30]
    st.markdown(f"**該当: {len(filt)}件**")
    dc = ["ticker","name","signal","total_score","price","rsi_14","vol_ratio","pos_52w_pct","analysis_comment"]
    av = [c for c in dc if c in filt.columns]
    st.dataframe(filt[av].sort_values("total_score", ascending=False), width="stretch", hide_index=True)


# ──────────────────────────────────────────────
# ページ5: 予測シミュレーション
# ──────────────────────────────────────────────

def page_prediction():
    st.header("予測シミュレーション")
    df = load_analysis()
    if df is None or df.empty:
        st.warning("分析データがありません。")
        return
    options = [f"{r['ticker']} - {r['name']}" for _, r in df.iterrows()]
    selected_opt = st.selectbox("銘柄を選択", options)
    selected = selected_opt.split(" - ")[0]
    investment = st.number_input("投資金額 (円)", min_value=10000, value=1000000, step=100000)
    st.caption(f"投資金額: ¥{investment:,.0f}")

    if st.button("予測を実行"):
        with st.spinner(f"{selected} の予測を計算中..."):
            from predictor import predict_stock
            result = predict_stock(selected)
        if result is None:
            st.error("予測データを取得できませんでした。")
            return
        st.success(f"予測エンジン: {result['engine']}")
        cur = result["current_price"]

        plabels = {1:"1日後",7:"1週間後",30:"1ヶ月後",90:"3ヶ月後",180:"6ヶ月後"}
        st.subheader("予測結果")
        st.markdown(f"現在株価: **{cur:,.2f}** / 投資金額: **¥{investment:,.0f}**")

        for pred in result["predictions"]:
            d = pred["period_days"]
            lab = plabels.get(d, f"{d}日後")
            pp = pred["predicted_price"]
            rp = pred["return_pct"]
            pnl = investment * rp / 100
            c = "green" if rp >= 0 else "red"
            st.markdown(f"**{lab}予測**: {pp:,.2f} (<span style='color:{c}'>{rp:+.2f}%</span>) "
                        f"→ 予想損益: <span style='color:{c}'>¥{pnl:+,.0f}</span>",
                        unsafe_allow_html=True)
            st.caption(f"80%確率: {pred['lower_80']:,.2f}~{pred['upper_80']:,.2f} | "
                       f"95%確率: {pred['lower_95']:,.2f}~{pred['upper_95']:,.2f}")

        st.markdown("---")
        st.caption("80%確率: 過去のパターンから10回中8回はこの範囲に収まる見込み")
        st.caption("95%確率: ほぼ確実にこの範囲に収まるが、範囲が広くなります")

        hist_df = result["history_df"]
        if not hist_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df["Close"], mode="lines", name="実績", line=dict(color="blue")))
            from datetime import timedelta
            ld = hist_df.index[-1]
            pd_list, pp_list, u95, l95 = [ld],[cur],[cur],[cur]
            for p in result["predictions"]:
                td = ld + timedelta(days=p["period_days"])
                pd_list.append(td); pp_list.append(p["predicted_price"])
                u95.append(p["upper_95"]); l95.append(p["lower_95"])
            fig.add_trace(go.Scatter(x=pd_list, y=pp_list, mode="lines+markers", name="予測", line=dict(color="red", dash="dash")))
            fig.add_trace(go.Scatter(x=pd_list, y=u95, mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=pd_list, y=l95, mode="lines", name="95%信頼区間", line=dict(width=0),
                                      fill="tonexty", fillcolor="rgba(255,0,0,0.1)"))
            fig.update_layout(title=f"{selected} 株価予測チャート", yaxis_title="価格", height=500)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# ページ6: 予測精度検証
# ──────────────────────────────────────────────

def page_accuracy():
    st.header("予測精度検証")
    st.caption("過去の予測が実際にどれだけ当たったかを検証します。データは毎日蓄積されます。")

    # 精度データ読み込み
    acc_path = DATA_DIR / "prediction_accuracy.csv"
    if not acc_path.exists():
        st.info("まだ予測精度データがありません。毎日データが更新されると、翌日以降に予測の的中率が表示されます。")
        st.markdown("---")
        st.markdown("**仕組み:**")
        st.markdown("1. 毎朝、その日の予測をスナップショットとして保存")
        st.markdown("2. 翌日以降、実際の株価と比較して精度を算出")
        st.markdown("3. 方向（上昇/下落）の的中率と誤差幅を表示")
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
    evaluated = acc_df[acc_df["status"] == "評価可能"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総予測件数", f"{total}件")
    c2.metric("方向的中率", f"{direction_acc:.1f}%",
              delta="良好" if direction_acc >= 55 else "改善余地あり",
              delta_color="normal" if direction_acc >= 55 else "inverse")
    c3.metric("平均誤差", f"{avg_error:.2f}%")
    c4.metric("評価済み予測", f"{len(evaluated)}件")

    st.markdown("---")

    # === ホライズン別精度 ===
    st.subheader("予測期間別の精度")
    horizon_order = ["1日後", "1週後", "1ヶ月後", "3ヶ月後", "6ヶ月後"]
    horizon_data = []
    for h in horizon_order:
        h_df = acc_df[acc_df["horizon"] == h]
        if len(h_df) > 0:
            horizon_data.append({
                "予測期間": h,
                "件数": len(h_df),
                "方向的中率": f"{h_df['direction_correct'].mean()*100:.1f}%",
                "平均予測": f"{h_df['predicted_change_pct'].mean():+.2f}%",
                "平均実際": f"{h_df['actual_change_pct'].mean():+.2f}%",
                "平均誤差": f"{h_df['error_pct'].abs().mean():.2f}%",
            })
    if horizon_data:
        st.dataframe(pd.DataFrame(horizon_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # === 銘柄別フィルター ===
    st.subheader("銘柄別の予測 vs 実績")

    # フィルター
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        tickers = sorted(acc_df["ticker"].unique())
        ticker_labels = []
        for t in tickers:
            name = acc_df[acc_df["ticker"]==t]["name"].iloc[0] if "name" in acc_df.columns else t
            ticker_labels.append(f"{t} ({name})")
        selected_idx = fc1.selectbox("銘柄", range(len(tickers)), format_func=lambda i: ticker_labels[i])
        selected_ticker = tickers[selected_idx] if tickers else None
    with fc2:
        selected_horizon = fc2.selectbox("予測期間", ["すべて"] + horizon_order)
    with fc3:
        selected_result = fc3.selectbox("結果", ["すべて", "的中のみ", "外れのみ"])

    # フィルター適用
    filtered = acc_df.copy()
    if selected_ticker:
        filtered = filtered[filtered["ticker"] == selected_ticker]
    if selected_horizon != "すべて":
        filtered = filtered[filtered["horizon"] == selected_horizon]
    if selected_result == "的中のみ":
        filtered = filtered[filtered["direction_correct"] == True]
    elif selected_result == "外れのみ":
        filtered = filtered[filtered["direction_correct"] == False]

    if filtered.empty:
        st.info("条件に合う予測がありません")
    else:
        # 結果テーブル
        display_cols = {
            "prediction_date": "予測日",
            "ticker": "銘柄",
            "name": "名前",
            "horizon": "期間",
            "base_price": "基準価格",
            "actual_price": "現在価格",
            "predicted_change_pct": "予測(%)",
            "actual_change_pct": "実際(%)",
            "error_pct": "誤差(%)",
            "direction_correct": "方向",
            "analysis_comment": "分析コメント",
        }
        disp = filtered[list(display_cols.keys())].rename(columns=display_cols)
        disp["方向"] = disp["方向"].map({True: "◎的中", False: "✕外れ"})

        # 色付き表示
        def color_direction(val):
            if val == "◎的中":
                return "background-color: rgba(0,200,0,0.2)"
            return "background-color: rgba(255,0,0,0.2)"

        styled = disp.style.applymap(color_direction, subset=["方向"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    st.markdown("---")

    # === 的中率チャート（日別推移） ===
    if len(acc_df["prediction_date"].unique()) > 1:
        st.subheader("方向的中率の推移")
        daily = acc_df.groupby("prediction_date").agg(
            accuracy=("direction_correct", "mean"),
            count=("direction_correct", "count")
        ).reset_index()
        daily["accuracy"] = daily["accuracy"] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["prediction_date"], y=daily["accuracy"],
                             name="的中率", marker_color=[
                                 "green" if a >= 55 else "red" for a in daily["accuracy"]
                             ]))
        fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%ライン（ランダム）")
        fig.update_layout(yaxis_title="方向的中率(%)", yaxis_range=[0, 100], height=400)
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# ルーティング
# ──────────────────────────────────────────────

if page == "ダッシュボード": page_dashboard()
elif page == "銘柄詳細": page_detail()
elif page == "ポートフォリオ": page_portfolio()
elif page == "スクリーニング": page_screening()
elif page == "予測シミュレーション": page_prediction()
elif page == "予測精度検証": page_accuracy()
