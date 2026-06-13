#!/usr/bin/env python3
"""
出来高を伴う反転シグナルのスキャン（Fable 5 独自分析・GSAモデルとは独立）

仮説: 売られすぎ＋出来高膨張（機関の仕込み）＋上昇への転換 = 急騰の前兆。
出来高なき反発はダマシ（dead-cat）として除外する。

シグナル（多日OHLCVから計算）:
  A. 底値圏      : 52週高値からの下落率が大きい / 200MA以下
  B. 出来高膨張  : 直近5日平均出来高 / 20日平均出来高（>1.2で加点）
  C. 上昇日集中  : 直近15日で「上昇日の出来高合計 / 下落日の出来高合計」(>1.1=accumulation)
  D. フォロースルー: 直近7日に +1.5%以上 かつ 出来高が前日超＆20日平均の1.2倍超 の日（オニール流FTD）
  E. 転換確認    : 終値が20MAを回復 / RSI(14)が30-55の反転帯で上向き
  F. 初動        : 直近10日リターンがプラス（既に動き始めている）
"""
import csv, sys, io, math
import yfinance as yf
import pandas as pd
import numpy as np

ANALYSIS = sys.argv[1] if len(sys.argv) > 1 else "data/latest_analysis.csv"

def jp_to_yf(t):
    return t + ".T" if t[0].isdigit() else t

# 1) ユニバース読み込み（csvで正しくパース）
rows = []
with open(ANALYSIS, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# 2) 全銘柄対象（売られすぎに限定せず、出来高反転は高値圏以外すべて見る）
#    ただし52週位置85%超の高値追いは除外（反転テーマでないため）
def fnum(x):
    try: return float(x)
    except: return float("nan")

cand = []
for r in rows:
    pos = fnum(r.get("pos_52w_pct"))
    if math.isnan(pos) or pos > 85:
        continue
    cand.append((r["ticker"], r.get("name",""), r.get("sector",""), pos))

tickers = [c[0] for c in cand]
yf_map = {c[0]: jp_to_yf(c[0]) for c in cand}
print(f"対象: {len(tickers)}銘柄をダウンロード中...", file=sys.stderr)

# 3) OHLCV一括取得（90日）
data = yf.download(list(yf_map.values()), period="90d", interval="1d",
                   group_by="ticker", auto_adjust=False, progress=False, threads=True)

results = []
for tk, name, sector, pos in cand:
    yt = yf_map[tk]
    try:
        df = data[yt].dropna()
    except Exception:
        continue
    if len(df) < 30:
        continue
    close = df["Close"]; vol = df["Volume"]; high = df["High"]; low = df["Low"]
    last = close.iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50s = close.rolling(50).mean()
    ma50 = ma50s.iloc[-1]
    ma50_slope = (ma50s.iloc[-1]/ma50s.iloc[-11]-1) if len(ma50s.dropna())>11 else 0  # 50MAの傾き
    high60 = high.tail(60).max()
    dist_from_high = last/high60 - 1  # マイナスほど売られている
    # 下降トレンド判定: 50MA以下 かつ 50MAが下向き = まだ落ちている
    in_downtrend = (last < ma50) and (ma50_slope < 0)
    # 新安値判定: 直近5日に60日安値を更新＝下げ止まっていない
    low60 = low.tail(60).min()
    making_new_lows = low.tail(5).min() <= low60 * 1.01
    reclaim50 = last > ma50  # 50MA奪回＝より強い反転確認

    v20 = vol.tail(20).mean(); v5 = vol.tail(5).mean()
    vol_expand = v5/v20 if v20>0 else 0

    # 上昇/下落日の出来高比（直近15日）= accumulation/distribution
    chg = close.diff()
    win = df.tail(15)
    up_v = win["Volume"][win["Close"].diff()>0].sum()
    dn_v = win["Volume"][win["Close"].diff()<0].sum()
    ad_ratio = up_v/dn_v if dn_v>0 else (3.0 if up_v>0 else 1.0)

    # フォロースルー・デイ（直近7日）
    ftd = False; ftd_day=None
    for i in range(len(df)-7, len(df)):
        if i<1: continue
        ret = close.iloc[i]/close.iloc[i-1]-1
        if ret>=0.015 and vol.iloc[i]>vol.iloc[i-1] and vol.iloc[i]>1.2*v20:
            ftd=True; ftd_day=df.index[i].strftime("%m/%d")
    # RSI14
    d = close.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean()
    rsi = (100-100/(1+ g.iloc[-1]/l.iloc[-1])) if l.iloc[-1]>0 else 100
    rsi_prev = None
    try:
        gp=d.clip(lower=0).rolling(14).mean().iloc[-4]; lp=(-d.clip(upper=0)).rolling(14).mean().iloc[-4]
        rsi_prev = (100-100/(1+gp/lp)) if lp>0 else 100
    except: pass
    rsi_up = (rsi_prev is not None and rsi>rsi_prev)

    reclaim20 = last>ma20
    ret10 = close.iloc[-1]/close.iloc[-11]-1 if len(close)>11 else 0

    # スコアリング（出来高を最重視）
    score = 0
    score += 20 if vol_expand>=1.5 else (12 if vol_expand>=1.2 else (5 if vol_expand>=1.0 else 0))  # B
    score += 20 if ad_ratio>=1.5 else (12 if ad_ratio>=1.1 else (3 if ad_ratio>=0.9 else 0))         # C
    score += 22 if ftd else 0                                                                          # D
    score += 10 if reclaim20 else 0                                                                    # E1
    score += 8 if (30<=rsi<=58 and rsi_up) else (3 if rsi_up else 0)                                  # E2
    score += 8 if 0<ret10<=0.20 else (2 if ret10>0.20 else 0)                                          # F（行き過ぎは減点気味）
    # 底値ボーナス（売られすぎから動くほど妙味）
    score += 8 if dist_from_high<=-0.30 else (4 if dist_from_high<=-0.15 else 0)                       # A
    score += 12 if reclaim50 else 0                                                                    # G: 50MA奪回=強い反転
    # ダマシ除外: 出来高膨張なしの上昇は大幅減点
    if ret10>0.05 and vol_expand<1.0 and ad_ratio<1.0:
        score -= 15
    # ★長期トレンド・フィルター（オリエンタルランド型の「下降中の弱い戻り」を除外）★
    # まだ50MA下＆50MA下向き = 落ちるナイフ。出来高を伴う決定的なブレイク(FTD+大商い+50MA奪回)以外は大幅減点
    decisive_breakout = ftd and vol_expand>=1.3 and reclaim50
    if in_downtrend and not decisive_breakout:
        score -= 25
    # まだ新安値を更新中なら下げ止まっていない=除外
    if making_new_lows:
        score -= 20

    results.append(dict(ticker=tk,name=name[:14],sector=sector[:10],pos52=pos,
        last=last, dist_high=dist_from_high*100, vol_expand=vol_expand, ad=ad_ratio,
        ftd=ftd, ftd_day=ftd_day, rsi=rsi, rsi_up=rsi_up, reclaim20=reclaim20,
        reclaim50=reclaim50, downtrend=in_downtrend, new_lows=making_new_lows,
        ret10=ret10*100, score=score))

results.sort(key=lambda x:-x["score"])
# CSV出力（定期便が読む）
import csv as _csv
with open("data/volume_reversal_signals.csv","w",encoding="utf-8-sig",newline="") as f:
    w=_csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader()
    for r in results: w.writerow(r)

print(f"\n{'='*128}\n出来高反転シグナル・ランキング（上位25）  ※出来高重視＋長期下降トレンド除外\n{'='*128}")
print(f"{'順':>2} {'銘柄':<6}{'名前':<15}{'Score':>5} {'52w':>4} {'高値乖離':>7} {'出来高5/20':>9} {'上昇/下落V':>9} {'FTD':>7} {'RSI':>5}{'↑':>2} {'>50MA':>6} {'下降':>4} {'新安値':>5} {'10日%':>6}")
for i,r in enumerate(results[:25],1):
    print(f"{i:>2} {r['ticker']:<6}{r['name']:<15}{r['score']:>5.0f} {r['pos52']:>3.0f}% {r['dist_high']:>6.1f}% {r['vol_expand']:>8.2f}x {r['ad']:>8.2f} {(r['ftd_day'] or '-'):>7} {r['rsi']:>5.0f}{('Y' if r['rsi_up'] else '·'):>2} {('Y' if r['reclaim50'] else '·'):>6} {('▼' if r['downtrend'] else '·'):>4} {('!' if r['new_lows'] else '·'):>5} {r['ret10']:>+5.1f}")
# 参考: オリエンタルランドが何位に落ちたか
olc=[(i,r) for i,r in enumerate(results,1) if r['ticker']=='4661']
if olc: i,r=olc[0]; print(f"\n[検証] 4661 オリエンタルランド: {i}位 / score={r['score']:.0f}（下降={r['downtrend']}, 新安値={r['new_lows']}, 50MA奪回={r['reclaim50']}）")
