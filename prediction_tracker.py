"""
prediction_tracker.py - 予測精度トラッキング

毎日の予測を保存し、翌日以降に実際の株価と比較して精度を検証する。
- 予測のスナップショットを日付付きで保存
- 過去の予測と実際の価格を比較（各ホライズン毎に正しい日付の株価を使用）
- 的中率・平均誤差・要因分析を生成

v2 修正: 全ホライズンで「今日の株価」を使うバグを修正。
  1日後予測→翌日の株価、7日後→7日後の株価と比較するように変更。
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRACKER_DIR = DATA_DIR / "prediction_history"


def save_daily_snapshot():
    """
    当日の予測データと株価データをスナップショットとして保存。
    毎日run.pyから呼ばれる。
    """
    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    # 予測データを保存
    pred_path = DATA_DIR / "latest_predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        pred_df["snapshot_date"] = today
        snapshot_path = TRACKER_DIR / f"predictions_{today}.csv"
        pred_df.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
        logger.info(f"  予測スナップショット保存: {snapshot_path}")
    else:
        logger.warning("  予測データなし: スナップショットスキップ")
        return

    # 株価データも保存（比較用の基準価格）
    stock_path = DATA_DIR / "latest_data.csv"
    if stock_path.exists():
        stock_df = pd.read_csv(stock_path)
        stock_df["snapshot_date"] = today
        stock_snapshot = TRACKER_DIR / f"prices_{today}.csv"
        stock_df.to_csv(stock_snapshot, index=False, encoding="utf-8-sig")
        logger.info(f"  株価スナップショット保存: {stock_snapshot}")


def _load_all_price_snapshots():
    """
    全ての株価スナップショットをメモリに読み込む。
    Returns: {date_str: {ticker: price}}
    """
    price_map = {}
    price_files = sorted(TRACKER_DIR.glob("prices_*.csv"))

    for pf in price_files:
        try:
            date_str = pf.stem.split("_")[1]
            df = pd.read_csv(pf)
            price_col = "close" if "close" in df.columns else "price"
            if "ticker" in df.columns and price_col in df.columns:
                price_map[date_str] = dict(zip(df["ticker"], df[price_col]))
        except Exception as e:
            logger.warning(f"株価ファイル読み込みエラー ({pf}): {e}")
            continue

    # 最新データも追加
    current_path = DATA_DIR / "latest_data.csv"
    if current_path.exists():
        try:
            df = pd.read_csv(current_path)
            price_col = "close" if "close" in df.columns else "price"
            if "ticker" in df.columns and price_col in df.columns:
                # fetch_time から日付を取得
                if "fetch_time" in df.columns:
                    first_time = str(df["fetch_time"].iloc[0])
                    try:
                        latest_date = datetime.fromisoformat(first_time).strftime("%Y%m%d")
                        price_map[latest_date] = dict(zip(df["ticker"], df[price_col]))
                    except Exception:
                        pass
        except Exception:
            pass

    return price_map


def _find_closest_price(price_map, ticker, target_date_str, max_offset_days=3):
    """
    target_date_strの株価を検索。見つからなければ前後max_offset_days以内で探す。
    （土日祝で株価データがない日を補完）
    """
    # まず正確な日付を試す
    prices = price_map.get(target_date_str, {})
    if ticker in prices:
        return float(prices[ticker]), target_date_str

    # 前後の日付で探す（土日祝対応）
    try:
        target_dt = datetime.strptime(target_date_str, "%Y%m%d")
    except ValueError:
        return None, None

    for offset in range(1, max_offset_days + 1):
        for delta in [offset, -offset]:
            check_dt = target_dt + timedelta(days=delta)
            check_str = check_dt.strftime("%Y%m%d")
            prices = price_map.get(check_str, {})
            if ticker in prices:
                return float(prices[ticker]), check_str

    return None, None


def evaluate_predictions():
    """
    過去の予測と実際の株価を比較して精度を評価する。

    v2修正: 各ホライズンごとに正しい日付の株価を使用する。
    - 1日後予測 → 翌日の株価と比較
    - 7日後予測 → 7日後の株価と比較
    - 30日後予測 → 30日後の株価と比較
    """
    TRACKER_DIR.mkdir(parents=True, exist_ok=True)

    # 全株価スナップショットを読み込み
    price_map = _load_all_price_snapshots()
    if not price_map:
        logger.warning("株価スナップショットがありません")
        return pd.DataFrame()

    # 銘柄名マッピング（最新データから）
    name_map = {}
    current_path = DATA_DIR / "latest_data.csv"
    if current_path.exists():
        try:
            cdf = pd.read_csv(current_path)
            if "ticker" in cdf.columns and "name" in cdf.columns:
                name_map = dict(zip(cdf["ticker"], cdf["name"]))
        except Exception:
            pass

    # 過去の予測スナップショットを読み込み
    prediction_files = sorted(TRACKER_DIR.glob("predictions_*.csv"))
    if not prediction_files:
        logger.info("過去の予測データがありません")
        return pd.DataFrame()

    results = []
    today = datetime.now()

    for pred_file in prediction_files:
        try:
            pred_df = pd.read_csv(pred_file)
            snapshot_date_str = pred_file.stem.split("_")[1]
            snapshot_date = datetime.strptime(snapshot_date_str, "%Y%m%d")
            days_since = (today - snapshot_date).days

            if days_since < 1:
                continue

            # 予測日の基準株価を取得
            base_prices = price_map.get(snapshot_date_str, {})

            # 各銘柄の予測を評価
            for _, row in pred_df.iterrows():
                ticker = row.get("ticker", "")
                if not ticker:
                    continue

                # 基準価格（予測日の株価）
                base_price = None
                if ticker in base_prices:
                    base_price = float(base_prices[ticker])
                else:
                    # pred_current_price をフォールバック
                    bp = row.get("pred_current_price", row.get("base_price", None))
                    if bp is not None and pd.notna(bp) and bp != 0:
                        base_price = float(bp)

                if base_price is None or base_price == 0:
                    continue

                # 各予測ホライズンを評価
                # v6: 1日予測を廃止し、7日/30日/90日に再設計
                # （過去のCSVに pred_1d_pct があっても評価しない＝旧データは履歴として残すのみ）
                horizons = [
                    ("1週後", 7, "pred_7d_pct"),
                    ("1ヶ月後", 30, "pred_30d_pct"),
                    ("3ヶ月後", 90, "pred_90d_pct"),
                ]

                for horizon_name, horizon_days, col_name in horizons:
                    predicted_pct = row.get(col_name, None)
                    if predicted_pct is None or pd.isna(predicted_pct):
                        continue

                    # ★ v2修正: ホライズンに応じた正しい日付の株価を取得
                    target_date = snapshot_date + timedelta(days=horizon_days)
                    target_date_str = target_date.strftime("%Y%m%d")

                    if target_date > today:
                        # まだ到達していない → スキップ（未到達データは精度計算に含めない）
                        continue

                    actual_price, found_date = _find_closest_price(
                        price_map, ticker, target_date_str, max_offset_days=3
                    )

                    if actual_price is None:
                        continue

                    # 実際の変化率
                    actual_change_pct = ((actual_price - base_price) / base_price) * 100

                    # 方向の正解判定
                    direction_correct = (
                        (predicted_pct > 0 and actual_change_pct > 0) or
                        (predicted_pct < 0 and actual_change_pct < 0) or
                        (abs(predicted_pct) < 0.1 and abs(actual_change_pct) < 1)
                    )

                    error_pct = actual_change_pct - predicted_pct

                    # 要因分析コメント生成
                    comment = _generate_analysis_comment(
                        ticker, predicted_pct, actual_change_pct, error_pct, direction_correct, days_since
                    )

                    results.append({
                        "ticker": ticker,
                        "name": name_map.get(ticker, ticker),
                        "prediction_date": snapshot_date.strftime("%Y-%m-%d"),
                        "horizon": horizon_name,
                        "horizon_days": horizon_days,
                        "days_elapsed": days_since,
                        "status": "評価可能",
                        "base_price": round(base_price, 2),
                        "predicted_change_pct": round(float(predicted_pct), 2),
                        "actual_change_pct": round(actual_change_pct, 2),
                        "error_pct": round(error_pct, 2),
                        "actual_price": round(actual_price, 2),
                        "direction_correct": direction_correct,
                        "analysis_comment": comment,
                    })

        except Exception as e:
            logger.warning(f"予測ファイル処理エラー ({pred_file}): {e}")
            continue

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # 結果を保存
    result_df.to_csv(DATA_DIR / "prediction_accuracy.csv", index=False, encoding="utf-8-sig")
    logger.info(f"  予測精度レポート生成: {len(result_df)}件")

    return result_df


def get_accuracy_summary(result_df=None):
    """予測精度のサマリーを生成する。"""
    if result_df is None:
        acc_path = DATA_DIR / "prediction_accuracy.csv"
        if acc_path.exists():
            result_df = pd.read_csv(acc_path)
        else:
            return {}

    if result_df.empty:
        return {}

    evaluated = result_df[result_df["status"] == "評価可能"]
    if evaluated.empty:
        evaluated = result_df

    direction_accuracy = evaluated["direction_correct"].mean() * 100 if len(evaluated) > 0 else 0
    avg_error = evaluated["error_pct"].abs().mean() if len(evaluated) > 0 else 0

    # ホライズン別の精度 (v6: 1日後を削除、3ヶ月後を追加)
    horizon_stats = {}
    for horizon in ["1週後", "1ヶ月後", "3ヶ月後"]:
        h_df = evaluated[evaluated["horizon"] == horizon]
        if len(h_df) > 0:
            horizon_stats[horizon] = {
                "count": len(h_df),
                "direction_accuracy": round(h_df["direction_correct"].mean() * 100, 1),
                "avg_error": round(h_df["error_pct"].abs().mean(), 2),
                "avg_predicted": round(h_df["predicted_change_pct"].mean(), 2),
                "avg_actual": round(h_df["actual_change_pct"].mean(), 2),
            }

    # 銘柄別の精度
    ticker_stats = evaluated.groupby("ticker").agg(
        direction_accuracy=("direction_correct", "mean"),
        avg_error=("error_pct", lambda x: x.abs().mean()),
        count=("ticker", "count"),
    ).reset_index()
    ticker_stats["direction_accuracy"] = (ticker_stats["direction_accuracy"] * 100).round(1)
    ticker_stats["avg_error"] = ticker_stats["avg_error"].round(2)

    return {
        "total_predictions": len(result_df),
        "evaluated": len(evaluated),
        "direction_accuracy": round(direction_accuracy, 1),
        "avg_error": round(avg_error, 2),
        "horizon_stats": horizon_stats,
        "best_tickers": ticker_stats.nlargest(5, "direction_accuracy").to_dict("records"),
        "worst_tickers": ticker_stats.nsmallest(5, "direction_accuracy").to_dict("records"),
    }


def _generate_analysis_comment(ticker, predicted_pct, actual_pct, error_pct, direction_correct, days_elapsed):
    """予測と実際の差異についてのコメントを生成"""
    abs_error = abs(error_pct)

    if direction_correct and abs_error < 2:
        return f"◎ 精度良好: 予測{predicted_pct:+.1f}% → 実際{actual_pct:+.1f}%（誤差{abs_error:.1f}%）"
    elif direction_correct and abs_error < 5:
        return f"○ 方向的中: 予測{predicted_pct:+.1f}% → 実際{actual_pct:+.1f}%（誤差{abs_error:.1f}%）"
    elif direction_correct:
        if actual_pct > predicted_pct:
            return f"△ 上振れ: 予測{predicted_pct:+.1f}%を超えて{actual_pct:+.1f}%（想定以上の上昇）"
        else:
            return f"△ 下振れ: 予測{predicted_pct:+.1f}%より低く{actual_pct:+.1f}%（上昇幅が予測未満）"
    else:
        if predicted_pct > 0 and actual_pct < 0:
            return f"✕ 外れ: 上昇予測{predicted_pct:+.1f}%に対し{actual_pct:+.1f}%下落"
        elif predicted_pct < 0 and actual_pct > 0:
            return f"✕ 外れ: 下落予測{predicted_pct:+.1f}%に対し{actual_pct:+.1f}%上昇"
        else:
            return f"✕ 外れ: 予測{predicted_pct:+.1f}% → 実際{actual_pct:+.1f}%（誤差{abs_error:.1f}%）"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 予測スナップショット保存 ===")
    save_daily_snapshot()
    print("\n=== 予測精度評価 ===")
    result_df = evaluate_predictions()
    if not result_df.empty:
        summary = get_accuracy_summary(result_df)
        print(f"全体方向的中率: {summary.get('direction_accuracy', 0)}%")
        print(f"平均誤差: {summary.get('avg_error', 0)}%")
        for h, stats in summary.get("horizon_stats", {}).items():
            print(f"  {h}: 的中率{stats['direction_accuracy']}%, "
                  f"AI予測平均{stats['avg_predicted']:+.2f}%, "
                  f"実際平均{stats['avg_actual']:+.2f}%")
    else:
        print("評価可能な過去予測がまだありません")
