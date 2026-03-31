"""
prediction_tracker.py - 予測精度トラッキング

毎日の予測を保存し、翌日以降に実際の株価と比較して精度を検証する。
- 予測のスナップショットを日付付きで保存
- 過去の予測と実際の価格を比較
- 的中率・平均誤差・要因分析を生成
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


def evaluate_predictions():
    """
    過去の予測と現在の実際の株価を比較して精度を評価する。
    戻り値: DataFrame with columns:
        - ticker, name, prediction_date, horizon (1日後, 1週後, etc.)
        - predicted_change_pct, actual_change_pct, error_pct
        - direction_correct (方向が合っていたか)
        - base_price, predicted_price, actual_price
        - analysis_comment (要因分析コメント)
    """
    TRACKER_DIR.mkdir(parents=True, exist_ok=True)

    # 現在の株価を取得
    current_prices_path = DATA_DIR / "latest_data.csv"
    if not current_prices_path.exists():
        logger.warning("最新の株価データがありません")
        return pd.DataFrame()

    current_df = pd.read_csv(current_prices_path)
    # price カラムを使用（close がない場合のフォールバック）
    price_col = "close" if "close" in current_df.columns else "price"
    if "ticker" not in current_df.columns or price_col not in current_df.columns:
        logger.warning(f"株価データにticker/{price_col}カラムがありません")
        return pd.DataFrame()

    current_prices = dict(zip(current_df["ticker"], current_df[price_col]))

    # 銘柄名マッピング
    name_map = {}
    if "name" in current_df.columns:
        name_map = dict(zip(current_df["ticker"], current_df["name"]))

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
            snapshot_date_str = pred_file.stem.split("_")[1]  # predictions_20260322 -> 20260322
            snapshot_date = datetime.strptime(snapshot_date_str, "%Y%m%d")
            days_elapsed = (today - snapshot_date).days

            if days_elapsed < 1:
                continue  # 当日の予測はまだ評価不可

            # 対応する当時の株価を取得
            price_file = TRACKER_DIR / f"prices_{snapshot_date_str}.csv"
            if price_file.exists():
                base_prices_df = pd.read_csv(price_file)
                bp_col = "close" if "close" in base_prices_df.columns else "price"
                base_prices = dict(zip(base_prices_df["ticker"], base_prices_df[bp_col]))
            else:
                base_prices = {}

            # 各銘柄の予測を評価
            for _, row in pred_df.iterrows():
                ticker = row.get("ticker", "")
                if not ticker or ticker not in current_prices:
                    continue

                base_price = base_prices.get(ticker, row.get("pred_current_price", row.get("base_price", None)))
                if base_price is None or pd.isna(base_price) or base_price == 0:
                    continue

                actual_price = current_prices[ticker]
                actual_change_pct = ((actual_price - base_price) / base_price) * 100

                # 各予測ホライズンを評価
                horizons = [
                    ("1日後", 1, "pred_1d_pct"),
                    ("1週後", 7, "pred_7d_pct"),
                    ("1ヶ月後", 30, "pred_30d_pct"),
                ]

                for horizon_name, horizon_days, col_name in horizons:
                    predicted_pct = row.get(col_name, None)
                    if predicted_pct is None or pd.isna(predicted_pct):
                        continue

                    # この予測期間が経過しているかチェック
                    if days_elapsed < horizon_days:
                        status = "未到達"
                        # 途中経過として表示
                        progress_pct = (days_elapsed / horizon_days) * 100
                    else:
                        status = "評価可能"
                        progress_pct = 100

                    # 方向の正解判定
                    direction_correct = (predicted_pct > 0 and actual_change_pct > 0) or \
                                       (predicted_pct < 0 and actual_change_pct < 0) or \
                                       (predicted_pct == 0 and abs(actual_change_pct) < 1)

                    error_pct = actual_change_pct - predicted_pct

                    # 要因分析コメント生成
                    comment = _generate_analysis_comment(
                        ticker, predicted_pct, actual_change_pct, error_pct, direction_correct, days_elapsed
                    )

                    results.append({
                        "ticker": ticker,
                        "name": name_map.get(ticker, ticker),
                        "prediction_date": snapshot_date.strftime("%Y-%m-%d"),
                        "horizon": horizon_name,
                        "horizon_days": horizon_days,
                        "days_elapsed": days_elapsed,
                        "status": status,
                        "base_price": round(base_price, 2),
                        "predicted_change_pct": round(predicted_pct, 2),
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
    """
    予測精度のサマリーを生成する。
    戻り値: dict with summary statistics
    """
    if result_df is None:
        acc_path = DATA_DIR / "prediction_accuracy.csv"
        if acc_path.exists():
            result_df = pd.read_csv(acc_path)
        else:
            return {}

    if result_df.empty:
        return {}

    # 全体の方向的中率
    evaluated = result_df[result_df["status"] == "評価可能"]
    if evaluated.empty:
        # 未到達でも途中経過の方向で計算
        evaluated = result_df

    direction_accuracy = evaluated["direction_correct"].mean() * 100 if len(evaluated) > 0 else 0
    avg_error = evaluated["error_pct"].abs().mean() if len(evaluated) > 0 else 0

    # ホライズン別の精度
    horizon_stats = {}
    for horizon in ["1日後", "1週後", "1ヶ月後", "3ヶ月後", "6ヶ月後"]:
        h_df = evaluated[evaluated["horizon"] == horizon]
        if len(h_df) > 0:
            horizon_stats[horizon] = {
                "count": len(h_df),
                "direction_accuracy": round(h_df["direction_correct"].mean() * 100, 1),
                "avg_error": round(h_df["error_pct"].abs().mean(), 2),
                "avg_predicted": round(h_df["predicted_change_pct"].mean(), 2),
                "avg_actual": round(h_df["actual_change_pct"].mean(), 2),
            }

    # 銘柄別の精度（上位・下位）
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
            return f"✕ 外れ: 上昇予測{predicted_pct:+.1f}%に対し{actual_pct:+.1f}%下落。突発的な悪材料の可能性"
        elif predicted_pct < 0 and actual_pct > 0:
            return f"✕ 外れ: 下落予測{predicted_pct:+.1f}%に対し{actual_pct:+.1f}%上昇。好材料やリバウンドの可能性"
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
    else:
        print("評価可能な過去予測がまだありません（明日以降に結果が出ます）")
