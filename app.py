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
# データ読み込み（キャッシュ付き）
# ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_macro():
    p = DATA_DIR / "latest_macro.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


@st.cache_data(ttl=300)
def load_analysis():
    p = DATA_DIR / "latest_analysis.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


@st.cache_data(ttl=300)
def load_ohlc_dynamic(ticker, period="1y", interval="1d"):
    """yfinanceから動的にOHLCデータを取得（期間・間隔指定可能）"""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval)
        if not hist.empty:
            return hist[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_ohlc(ticker, days=400):
    """db_cache → yfinance フォールバック"""
    try:
        import db_cache
        df = db_cache.load_daily_prices(ticker, days=days)
        if not df.empty:
            return df
    except Exception:
        pass
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        if not hist.empty:
            return hist[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass
    return pd.DataFrame()


def compute_rsi(series, period=14):
    """RSIを計算"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD, シグナル, ヒストグラムを計算"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ──────────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────────

st.set_page_config(page_title="Growth Stock Analyzer", layout="wide")
st.sidebar.title("Growth Stock Analyzer")

page = st.sidebar.radio(
    "ページ選択",
    ["ダッシュボード", "銘柄詳細", "ポートフォリオ", "スクリーニング", "予測シミュレーション"],
)

# ──────────────────────────────────────────────
# サイドバー: スコア重み調整スライダー
# ──────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("スコア重み調整")

w_tech = st.sidebar.slider("テクニカル (%)", 0, 100, 35, key="w_tech")
w_fund = st.sidebar.slider("ファンダメンタルズ (%)", 0, 100, 30, key="w_fund")
w_sent = st.sidebar.slider("センチメント (%)", 0, 100, 20, key="w_sent")
w_mom = st.sidebar.slider("モメンタム (%)", 0, 100, 15, key="w_mom")

# 合計100%自動調整
total_weight = w_tech + w_fund + w_sent + w_mom
if total_weight > 0:
    weights = {
        "technical": w_tech / total_weight,
        "fundamental": w_fund / total_weight,
        "sentiment": w_sent / total_weight,
        "momentum": w_mom / total_weight,
    }
else:
    weights = {"technical": 0.25, "fundamental": 0.25, "sentiment": 0.25, "momentum": 0.25}

st.sidebar.caption(
    f"配分: テク{weights['technical']*100:.0f}% / ファンダ{weights['fundamental']*100:.0f}% "
    f"/ SNS{weights['sentiment']*100:.0f}% / モメ{weights['momentum']*100:.0f}%"
)


def recalculate_scores(df):
    """重み変更時にtotal_scoreを再計算"""
    if df is None or df.empty:
        return df
    result = df.copy()
    for col in ["tech_score", "fund_score", "sent_score", "mom_score"]:
        if col not in result.columns:
            result[col] = 50
    result["total_score"] = (
        result["tech_score"] * weights["technical"] +
        result["fund_score"] * weights["fundamental"] +
        result["sent_score"] * weights["sentiment"] +
        result["mom_score"] * weights["momentum"]
    ).round(1)
    return result.sort_values("total_score", ascending=False)


# ──────────────────────────────────────────────
# ページ1: ダッシュボード
# ──────────────────────────────────────────────

def page_dashboard():
    st.header("ダッシュボード")

    macro_df = load_macro()

    # --- 市況判定表示 ---
    if macro_df is not None and not macro_df.empty:
        try:
            from analyzer import calculate_macro_score
            macro_result = calculate_macro_score(macro_df)
            emoji = macro_result["emoji"]
            label = macro_result["label"]
            mscore = macro_result["score"]
            detail = macro_result["detail"]
            st.markdown(
                f"### {emoji} 市況判定: **{label}** (スコア: {mscore}/100)\n"
                f"<small>{detail}</small>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
        except Exception as e:
            st.warning(f"市況判定の計算に失敗しました: {e}")

    # --- マクロ指標 ---
    if macro_df is not None and not macro_df.empty:
        targets = {"^VIX": "VIX", "^GSPC": "S&P500", "^N225": "日経225", "USDJPY=X": "ドル円"}
        cols = st.columns(len(targets))
        for col, (tk, label) in zip(cols, targets.items()):
            row = macro_df[macro_df["ticker"] == tk]
            if not row.empty:
                r = row.iloc[0]
                val = r["current_value"]
                delta = f"{r['change_1d_pct']:+.2f}%" if pd.notna(r.get("change_1d_pct")) else None
                col.metric(label, f"{val:,.2f}", delta)
            else:
                col.metric(label, "N/A")
    else:
        st.warning("マクロデータ (data/latest_macro.csv) が見つかりません。先に `python run.py` を実行してください。")

    # --- 分析データ読み込み + 重み再計算 ---
    analysis_df = recalculate_scores(load_analysis())

    # --- 買いシグナル TOP10 ---
    if analysis_df is not None and not analysis_df.empty:
        st.subheader("買いシグナル TOP10")
        buy_df = analysis_df[analysis_df["signal"].str.contains("買い", na=False)].copy()
        buy_df = buy_df.sort_values("total_score", ascending=False).head(10)

        if not buy_df.empty:
            rows_list = [buy_df.iloc[i:i+5] for i in range(0, len(buy_df), 5)]
            for chunk in rows_list:
                cols = st.columns(5)
                for col, (_, r) in zip(cols, chunk.iterrows()):
                    with col:
                        st.markdown(f"**{r['ticker']}**")
                        st.caption(r.get("name", ""))
                        score = r.get("total_score", 0)
                        signal = r.get("signal", "")
                        st.markdown(f"スコア: **{score:.1f}** / {signal}")
        else:
            st.info("現在、買いシグナルの銘柄はありません。")

        # --- 全銘柄ランキング ---
        st.subheader("全銘柄ランキング")
        display_cols = ["ticker", "name", "signal", "total_score", "price",
                        "change_pct", "rsi_14", "vol_ratio", "pos_52w_pct"]
        available = [c for c in display_cols if c in analysis_df.columns]
        st.dataframe(
            analysis_df[available].sort_values("total_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("分析データ (data/latest_analysis.csv) が見つかりません。先に `python run.py` を実行してください。")


# ──────────────────────────────────────────────
# ページ2: 銘柄詳細
# ──────────────────────────────────────────────

PERIOD_MAP = {
    "1日": ("1d", "5m"),
    "1週": ("5d", "15m"),
    "1ヶ月": ("1mo", "1h"),
    "6ヶ月": ("6mo", "1d"),
    "1年": ("1y", "1d"),
    "2年": ("2y", "1wk"),
    "5年": ("5y", "1wk"),
}


def page_detail():
    st.header("銘柄詳細")

    analysis_df = recalculate_scores(load_analysis())
    if analysis_df is None or analysis_df.empty:
        st.warning("分析データがありません。先に `python run.py` を実行してください。")
        return

    tickers = analysis_df["ticker"].tolist()
    selected = st.sidebar.selectbox("銘柄を選択", tickers)

    row = analysis_df[analysis_df["ticker"] == selected].iloc[0]
    st.subheader(f"{selected} - {row.get('name', '')}")

    # テクニカル指標
    indicators = {
        "RSI(14)": "rsi_14", "MACD": "macd", "BB位置": "bb_position",
        "出来高比率": "vol_ratio", "52週位置(%)": "pos_52w_pct",
        "スコア": "total_score", "シグナル": "signal",
    }
    cols = st.columns(len(indicators))
    for col, (label, key) in zip(cols, indicators.items()):
        val = row.get(key, "N/A")
        if isinstance(val, float):
            col.metric(label, f"{val:.2f}")
        else:
            col.metric(label, str(val))

    # --- 期間切替ラジオ ---
    period_label = st.radio(
        "チャート期間",
        list(PERIOD_MAP.keys()),
        index=4,  # デフォルト: 1年
        horizontal=True,
    )
    yf_period, yf_interval = PERIOD_MAP[period_label]

    # --- チャートデータ取得 ---
    ohlc = load_ohlc_dynamic(selected, period=yf_period, interval=yf_interval)
    if not ohlc.empty:
        ohlc_sorted = ohlc.sort_index()

        # SMA計算（間隔に応じて調整）
        if yf_interval in ["1d", "1wk"]:
            sma_windows = [20, 50, 200]
        elif yf_interval == "1h":
            sma_windows = [20, 50]
        else:
            sma_windows = [20]

        for window in sma_windows:
            if len(ohlc_sorted) >= window:
                ohlc_sorted[f"SMA{window}"] = ohlc_sorted["Close"].rolling(window).mean()

        # RSI / MACD計算
        rsi_series = compute_rsi(ohlc_sorted["Close"])
        macd_line, signal_line, macd_hist = compute_macd(ohlc_sorted["Close"])

        # 3段構成チャート
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.03,
            subplot_titles=[f"{selected} {period_label}チャート", "RSI", "MACD"],
        )

        # Row1: ローソク足 + SMA
        fig.add_trace(go.Candlestick(
            x=ohlc_sorted.index,
            open=ohlc_sorted["Open"],
            high=ohlc_sorted["High"],
            low=ohlc_sorted["Low"],
            close=ohlc_sorted["Close"],
            name="OHLC",
        ), row=1, col=1)

        colors = {"SMA20": "orange", "SMA50": "blue", "SMA200": "red"}
        for sma, color in colors.items():
            if sma in ohlc_sorted.columns:
                fig.add_trace(go.Scatter(
                    x=ohlc_sorted.index, y=ohlc_sorted[sma],
                    mode="lines", name=sma, line=dict(color=color, width=1),
                ), row=1, col=1)

        # Row2: RSI
        fig.add_trace(go.Scatter(
            x=ohlc_sorted.index, y=rsi_series,
            mode="lines", name="RSI", line=dict(color="purple", width=1),
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Row3: MACD
        fig.add_trace(go.Scatter(
            x=ohlc_sorted.index, y=macd_line,
            mode="lines", name="MACD", line=dict(color="blue", width=1),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=ohlc_sorted.index, y=signal_line,
            mode="lines", name="Signal", line=dict(color="orange", width=1),
        ), row=3, col=1)
        fig.add_trace(go.Bar(
            x=ohlc_sorted.index, y=macd_hist,
            name="Histogram",
            marker_color=["green" if v >= 0 else "red" for v in macd_hist.fillna(0)],
        ), row=3, col=1)

        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=750,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="価格", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("チャートデータを取得できませんでした。")

    # --- スコア詳細セクション ---
    st.subheader("スコア詳細")

    col_radar, col_table = st.columns([1, 2])

    with col_radar:
        # レーダーチャート
        categories = ["テクニカル", "ファンダメンタルズ", "センチメント", "モメンタム"]
        values = [
            row.get("tech_score", 0),
            row.get("fund_score", 0),
            row.get("sent_score", 0),
            row.get("mom_score", 0),
        ]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],  # 閉じるために最初の値を追加
            theta=categories + [categories[0]],
            fill="toself",
            name=selected,
            line=dict(color="royalblue"),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=350,
            title=f"{selected} スコアレーダー",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_table:
        # 指標解説テーブル
        explanation_data = []

        # テクニカル指標
        rsi_val = row.get("rsi_14")
        if pd.notna(rsi_val):
            if rsi_val < 30:
                meaning = "売られすぎ→反発の可能性"
            elif rsi_val > 70:
                meaning = "買われすぎ→調整リスク"
            else:
                meaning = "中立的な水準"
            explanation_data.append({"カテゴリ": "テクニカル", "指標": "RSI(14)", "値": f"{rsi_val:.1f}", "解説": meaning})

        macd_val = row.get("macd")
        if pd.notna(macd_val):
            meaning = "買いの勢い" if macd_val > 0 else "売りの勢い"
            explanation_data.append({"カテゴリ": "テクニカル", "指標": "MACD", "値": f"{macd_val:.4f}", "解説": meaning})

        bb_val = row.get("bb_position")
        if pd.notna(bb_val):
            if bb_val < 20:
                meaning = "バンド下限付近→反発期待"
            elif bb_val > 80:
                meaning = "バンド上限付近→過熱気味"
            else:
                meaning = "バンド中間付近"
            explanation_data.append({"カテゴリ": "テクニカル", "指標": "BB位置", "値": f"{bb_val:.1f}%", "解説": meaning})

        cross_val = row.get("cross_signal", "なし")
        if cross_val != "なし":
            explanation_data.append({"カテゴリ": "テクニカル", "指標": "クロスシグナル", "値": str(cross_val), "解説": "重要な転換サイン"})

        # ファンダメンタルズ
        pos_val = row.get("pos_52w_pct")
        if pd.notna(pos_val):
            meaning = "安値圏→割安の可能性" if pos_val < 30 else ("高値圏" if pos_val > 70 else "中間水準")
            explanation_data.append({"カテゴリ": "ファンダ", "指標": "52週位置", "値": f"{pos_val:.1f}%", "解説": meaning})

        pe_val = row.get("pe_ratio")
        if pd.notna(pe_val):
            meaning = "割安" if pe_val < 20 else ("割高" if pe_val > 40 else "適正範囲")
            explanation_data.append({"カテゴリ": "ファンダ", "指標": "PER", "値": f"{pe_val:.1f}x", "解説": meaning})

        rg_val = row.get("revenue_growth_pct")
        if pd.notna(rg_val):
            meaning = "高成長" if rg_val > 25 else ("成長" if rg_val > 0 else "減収")
            explanation_data.append({"カテゴリ": "ファンダ", "指標": "売上成長率", "値": f"{rg_val:.1f}%", "解説": meaning})

        # センチメント
        cs_val = row.get("combined_sentiment")
        if pd.notna(cs_val):
            meaning = "SNS強気" if cs_val > 20 else ("SNS弱気" if cs_val < -20 else "SNS中立")
            explanation_data.append({"カテゴリ": "センチメント", "指標": "複合センチメント", "値": f"{cs_val:.1f}", "解説": meaning})

        gt_val = row.get("gtrends_trend_ratio")
        if pd.notna(gt_val):
            meaning = "検索急上昇" if gt_val > 1.5 else ("検索上昇傾向" if gt_val > 1.0 else "検索平常")
            explanation_data.append({"カテゴリ": "センチメント", "指標": "Google Trends比率", "値": f"{gt_val:.2f}x", "解説": meaning})

        reddit_val = row.get("reddit_mentions")
        if pd.notna(reddit_val) and reddit_val > 0:
            explanation_data.append({"カテゴリ": "センチメント", "指標": "Redditメンション", "値": f"{int(reddit_val)}件", "解説": "話題度"})

        # モメンタム
        vol_val = row.get("vol_ratio")
        if pd.notna(vol_val):
            meaning = "出来高急増→大きな動きの兆候" if vol_val > 2 else ("出来高やや増" if vol_val > 1.2 else "出来高平常")
            explanation_data.append({"カテゴリ": "モメンタム", "指標": "出来高比率", "値": f"{vol_val:.2f}x", "解説": meaning})

        if explanation_data:
            st.table(pd.DataFrame(explanation_data))

    # スコア理由の詳細表示
    with st.expander("スコア算出理由の詳細"):
        reason_cols = {
            "テクニカル": "tech_reasons",
            "ファンダメンタルズ": "fund_reasons",
            "センチメント": "sent_reasons",
            "モメンタム": "mom_reasons",
        }
        for label, col_name in reason_cols.items():
            val = row.get(col_name, "")
            if val:
                st.markdown(f"**{label}**: {val}")

    # 分析コメント
    comment = row.get("analysis_comment", "")
    if comment:
        st.subheader("分析コメント")
        st.info(comment)


# ──────────────────────────────────────────────
# ページ3: ポートフォリオ
# ──────────────────────────────────────────────

def page_portfolio():
    st.header("ポートフォリオ")

    portfolio_csv = DATA_DIR / "portfolio.csv"
    if not portfolio_csv.exists():
        st.warning(
            "ポートフォリオファイルが見つかりません。\n\n"
            "`data/portfolio.csv` を作成してください。\n\n"
            "フォーマット: `ticker,buy_date,buy_price,quantity,buy_currency,memo`"
        )
        return

    with st.spinner("ポートフォリオを分析中..."):
        from portfolio import run_portfolio_analysis
        port_df, totals = run_portfolio_analysis()

    if port_df is None or port_df.empty:
        st.error("ポートフォリオ分析に失敗しました。銘柄データを確認してください。")
        return

    # サマリー指標
    total_cost = totals["total_current_jpy"] - totals["total_pnl_jpy"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("投資総額 (円)", f"¥{total_cost:,.0f}")
    c2.metric("評価総額 (円)", f"¥{totals['total_current_jpy']:,.0f}")
    c3.metric("損益合計 (円)", f"¥{totals['total_pnl_jpy']:,.0f}")
    c4.metric("損益率", f"{totals['total_pnl_pct']:+.2f}%")

    st.caption(f"USD/JPY: {totals['usdjpy_rate']:.2f}")

    # 保有一覧
    st.subheader("保有一覧")
    st.dataframe(port_df, use_container_width=True, hide_index=True)

    # 円グラフ
    st.subheader("ポートフォリオ構成")
    fig = px.pie(
        port_df,
        values="current_value_jpy",
        names="ticker",
        title="評価額構成比",
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# ページ4: スクリーニング
# ──────────────────────────────────────────────

def page_screening():
    st.header("スクリーニング")

    analysis_df = recalculate_scores(load_analysis())
    if analysis_df is None or analysis_df.empty:
        st.warning("分析データがありません。先に `python run.py` を実行してください。")
        return

    st.sidebar.subheader("フィルタ条件")

    filters = {}
    if "rsi_14" in analysis_df.columns:
        filters["rsi_low"] = st.sidebar.checkbox("RSI < 30 (売られすぎ)", value=False)
    if "vol_ratio" in analysis_df.columns:
        filters["vol_high"] = st.sidebar.checkbox("出来高比率 >= 2.0", value=False)
    if "pos_52w_pct" in analysis_df.columns:
        filters["pos_low"] = st.sidebar.checkbox("52週位置 <= 30%", value=False)

    min_score = st.sidebar.slider(
        "最小トータルスコア",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
    )

    filtered = analysis_df.copy()
    filtered = filtered[filtered["total_score"] >= min_score]

    if filters.get("rsi_low"):
        filtered = filtered[filtered["rsi_14"] < 30]
    if filters.get("vol_high"):
        filtered = filtered[filtered["vol_ratio"] >= 2.0]
    if filters.get("pos_low"):
        filtered = filtered[filtered["pos_52w_pct"] <= 30]

    st.markdown(f"**該当銘柄数: {len(filtered)}件**")

    display_cols = ["ticker", "name", "signal", "total_score", "price",
                    "rsi_14", "vol_ratio", "pos_52w_pct", "analysis_comment"]
    available = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available].sort_values("total_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# ──────────────────────────────────────────────
# ページ5: 予測シミュレーション
# ──────────────────────────────────────────────

def page_prediction():
    st.header("予測シミュレーション")

    analysis_df = load_analysis()
    if analysis_df is None or analysis_df.empty:
        st.warning("分析データがありません。先に `python run.py` を実行してください。")
        return

    tickers = analysis_df["ticker"].tolist()
    selected = st.selectbox("銘柄を選択", tickers)
    investment = st.number_input("投資金額 (円)", min_value=10000, value=1000000, step=100000)

    if st.button("予測を実行"):
        with st.spinner(f"{selected} の予測を計算中..."):
            from predictor import predict_stock
            result = predict_stock(selected)

        if result is None:
            st.error("予測に必要なデータを取得できませんでした。")
            return

        st.success(f"予測エンジン: {result['engine']}")
        current_price = result["current_price"]

        # 予測結果テーブル
        period_labels = {1: "1日後", 7: "1週間後", 30: "1ヶ月後", 90: "3ヶ月後", 180: "6ヶ月後"}
        table_data = []
        for pred in result["predictions"]:
            days = pred["period_days"]
            label = period_labels.get(days, f"{days}日後")
            predicted = pred["predicted_price"]
            return_pct = pred["return_pct"]
            pnl = investment * return_pct / 100
            table_data.append({
                "期間": label,
                "予測株価": f"{predicted:,.2f}",
                "予想リターン": f"{return_pct:+.2f}%",
                "予想損益(円)": f"¥{pnl:+,.0f}",
                "信頼区間下限": f"{pred['lower_bound']:,.2f}",
                "信頼区間上限": f"{pred['upper_bound']:,.2f}",
            })

        st.subheader("予測結果")
        st.markdown(f"現在株価: **{current_price:,.2f}** / 投資金額: **¥{investment:,.0f}**")
        st.table(pd.DataFrame(table_data))

        # チャート: 実績 + 予測 + 信頼区間
        hist_df = result["history_df"]
        if not hist_df.empty:
            fig = go.Figure()

            # 実績線
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["Close"],
                mode="lines", name="実績",
                line=dict(color="blue"),
            ))

            # 予測線
            from datetime import timedelta
            last_date = hist_df.index[-1]
            pred_dates = [last_date]
            pred_prices = [current_price]
            upper_prices = [current_price]
            lower_prices = [current_price]

            for pred in result["predictions"]:
                target_date = last_date + timedelta(days=pred["period_days"])
                pred_dates.append(target_date)
                pred_prices.append(pred["predicted_price"])
                upper_prices.append(pred["upper_bound"])
                lower_prices.append(pred["lower_bound"])

            fig.add_trace(go.Scatter(
                x=pred_dates, y=pred_prices,
                mode="lines+markers", name="予測",
                line=dict(color="red", dash="dash"),
            ))

            # 信頼区間シェード
            fig.add_trace(go.Scatter(
                x=pred_dates, y=upper_prices,
                mode="lines", name="上限",
                line=dict(width=0),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=pred_dates, y=lower_prices,
                mode="lines", name="信頼区間",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 0, 0, 0.1)",
            ))

            fig.update_layout(
                title=f"{selected} 株価予測チャート",
                yaxis_title="価格",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "※ この予測は過去データに基づく統計的推定であり、将来の株価を保証するものではありません。"
            "投資判断は自己責任で行ってください。"
        )


# ──────────────────────────────────────────────
# ルーティング
# ──────────────────────────────────────────────

if page == "ダッシュボード":
    page_dashboard()
elif page == "銘柄詳細":
    page_detail()
elif page == "ポートフォリオ":
    page_portfolio()
elif page == "スクリーニング":
    page_screening()
elif page == "予測シミュレーション":
    page_prediction()
