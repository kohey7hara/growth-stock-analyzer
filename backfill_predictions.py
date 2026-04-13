"""
backfill_predictions.py - v4エンジンで過去の予測履歴を再生成する

使い方:
  python backfill_predictions.py

やること:
  1. data/stock_data_*.csv から過去の分析日一覧を取得
  2. 各日の銘柄リストに対してv4予測エンジンを実行
  3. prediction_history/ に予測CSVを上書き保存
  4. prediction_tracker を再実行して精度を再計算

注意: yfinanceが必要（ローカル実行前提）
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from predictor import predict_all_stocks


def get_past_dates():
    """data/stock_data_*.csv から過去の日付一覧を取得"""
    pattern = str(BASE_DIR / "data" / "stock_data_*.csv")
    files = sorted(glob.glob(pattern))
    dates = []
    for f in files:
        fname = os.path.basename(f)
        # stock_data_20260320.csv → 20260320
        date_str = fname.replace("stock_data_", "").replace(".csv", "")
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
            dates.append((date_str, dt, f))
        except ValueError:
            continue
    return dates


def get_tickers_for_date(csv_path):
    """指定日の分析CSVから銘柄リストを取得"""
    try:
        df = pd.read_csv(csv_path)
        if "ticker" in df.columns:
            return df["ticker"].tolist()
    except Exception as e:
        print(f"  [WARN] {csv_path} 読み込みエラー: {e}")
    return []


def main():
    print("=" * 60)
    print("  v4予測エンジンで過去の予測履歴を再生成")
    print("=" * 60)

    dates = get_past_dates()
    if not dates:
        print("過去のstock_dataが見つかりません。")
        return

    print(f"\n対象日数: {len(dates)}日")
    for ds, dt, f in dates:
        print(f"  {ds} ({dt.strftime('%A')})")

    # prediction_history ディレクトリ
    hist_dir = BASE_DIR / "data" / "prediction_history"
    hist_dir.mkdir(exist_ok=True)

    for date_str, dt, csv_path in dates:
        print(f"\n--- {date_str} の予測を再生成中 ---")
        tickers = get_tickers_for_date(csv_path)
        if not tickers:
            print(f"  銘柄リストが空です。スキップ。")
            continue

        print(f"  銘柄数: {len(tickers)}")

        try:
            # v4エンジンで予測実行
            pred_df = predict_all_stocks(tickers)

            if pred_df is not None and not pred_df.empty:
                # prediction_history に保存
                out_path = hist_dir / f"predictions_{date_str}.csv"
                pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"  ✓ {len(pred_df)}銘柄の予測を保存: {out_path.name}")
            else:
                print(f"  [WARN] 予測結果が空です")

        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    # 精度再計算
    print("\n--- 予測精度を再計算 ---")
    try:
        from prediction_tracker import evaluate_predictions
        evaluate_predictions()
        print("✓ prediction_accuracy.csv を更新しました")
    except Exception as e:
        print(f"[WARN] 精度再計算エラー: {e}")
        print("  手動で run.py を実行してください")

    print("\n完了！git add & push してダッシュボードに反映してください:")
    print("  git add data/prediction_history/ data/prediction_accuracy.csv")
    print("  git commit -m 'refactor: v4エンジンで予測履歴を再生成'")
    print("  git push")


if __name__ == "__main__":
    main()
