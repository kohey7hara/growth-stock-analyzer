#!/usr/bin/env python3
"""
run.py - Growth Stock Analyzer メイン実行スクリプト

使い方:
  python run.py              # 全ステップ実行 (株価→マクロ→SNS→分析→ポートフォリオ→Excel)
  python run.py --stock-only # 株価データのみ取得
  python run.py --macro-only # マクロ指標のみ取得
  python run.py --analyze    # 既存データで分析のみ実行
  python run.py --report     # 既存分析結果でExcelのみ生成
  python run.py --full-refresh # SQLiteキャッシュを無視して全データ再取得
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Growth Stock Analyzer")
    parser.add_argument("--stock-only", action="store_true", help="株価データのみ取得")
    parser.add_argument("--macro-only", action="store_true", help="マクロ指標のみ取得")
    parser.add_argument("--sentiment-only", action="store_true", help="SNSデータのみ取得")
    parser.add_argument("--analyze", action="store_true", help="既存データで分析のみ")
    parser.add_argument("--report", action="store_true", help="既存分析でExcelのみ")
    parser.add_argument("--full-refresh", action="store_true", help="全データを再取得(キャッシュ無視)")
    args = parser.parse_args()

    start = datetime.now()
    logger.info("=" * 60)
    logger.info(f"Growth Stock Analyzer 開始: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 個別実行フラグのリスト
    individual_flags = [args.stock_only, args.macro_only, args.sentiment_only, args.analyze, args.report]
    any_individual = any(individual_flags)

    # デフォルトは全ステップ
    run_stock = not any_individual or args.stock_only
    run_macro = not any_individual or args.macro_only
    run_sentiment = not any_individual or args.sentiment_only
    run_analyze = not any_individual or args.analyze
    run_report = not any_individual or args.report

    stock_df = None
    sentiment_df = None
    analysis_df = None
    macro_df = None
    portfolio_data = None

    # Step 1: 株価データ取得
    if run_stock:
        logger.info("\n[Step 1/6] 株価データ取得中...")
        try:
            from fetch_stock_data import fetch_all_stocks, load_config
            config = load_config()
            stock_df = fetch_all_stocks(config, full_refresh=args.full_refresh)
            logger.info(f"  完了: {len(stock_df)}銘柄取得")
        except Exception as e:
            logger.error(f"  エラー: {e}")
            if not args.stock_only:
                logger.info("  → 既存データで続行を試みます")

    # Step 2: マクロ経済指標取得
    if run_macro:
        logger.info("\n[Step 2/6] マクロ経済指標取得中...")
        try:
            from fetch_macro import fetch_all_macro
            macro_df = fetch_all_macro()
            logger.info(f"  完了: {len(macro_df)}指標取得")
        except Exception as e:
            logger.warning(f"  マクロ指標取得スキップ: {e}")

    # Step 3: SNSセンチメント取得
    if run_sentiment:
        logger.info("\n[Step 3/6] SNSセンチメントデータ取得中...")
        try:
            from fetch_sentiment import fetch_all_sentiment, load_config as load_config2
            config = load_config2()
            sentiment_df = fetch_all_sentiment(config)
            logger.info(f"  完了: {len(sentiment_df)}銘柄取得")
        except Exception as e:
            logger.warning(f"  SNSデータ取得スキップ: {e}")
            logger.info("  → SNSなしで分析を続行します")

    # Step 4: 統合分析
    if run_analyze:
        logger.info("\n[Step 4/6] 統合スコアリング分析中...")
        try:
            import pandas as pd
            from analyzer import analyze, load_config as load_config3
            config = load_config3()

            data_dir = BASE_DIR / "data"
            if stock_df is None:
                stock_path = data_dir / "latest_data.csv"
                if stock_path.exists():
                    stock_df = pd.read_csv(stock_path)
                else:
                    logger.error("  株価データが見つかりません")
                    sys.exit(1)

            if sentiment_df is None:
                sent_path = data_dir / "latest_sentiment.csv"
                if sent_path.exists():
                    sentiment_df = pd.read_csv(sent_path)

            analysis_df = analyze(stock_df, sentiment_df, config)
            logger.info(f"  完了: {len(analysis_df)}銘柄分析")

            # 結果サマリー出力
            print(f"\n{'='*60}")
            print("  分析結果サマリー")
            print(f"{'='*60}")
            for _, r in analysis_df.head(5).iterrows():
                print(f"  {r['signal']:14s} | {r['ticker']:6s} | スコア: {r['total_score']:.1f}")
            print(f"{'='*60}\n")

        except Exception as e:
            logger.error(f"  分析エラー: {e}")
            import traceback
            traceback.print_exc()

    # Step 4.5: 全銘柄予測
    if run_analyze:
        logger.info("\n[Step 4.5/6] 全銘柄予測データ生成中...")
        try:
            import pandas as pd
            from predictor import predict_all_stocks
            data_dir = BASE_DIR / "data"
            if analysis_df is None:
                ap = data_dir / "latest_analysis.csv"
                if ap.exists():
                    analysis_df = pd.read_csv(ap)
            if analysis_df is not None:
                tickers = analysis_df["ticker"].tolist()
                pred_df = predict_all_stocks(tickers)
                logger.info(f"  完了: {len(pred_df)}銘柄の予測データ生成")
            else:
                logger.warning("  分析データなし: 予測スキップ")
        except Exception as e:
            logger.warning(f"  予測データ生成スキップ: {e}")
            import traceback
            traceback.print_exc()

    # Step 5: ポートフォリオ分析
    if run_report:
        logger.info("\n[Step 5/6] ポートフォリオ分析中...")
        try:
            portfolio_path = BASE_DIR / "data" / "portfolio.csv"
            if portfolio_path.exists():
                from portfolio import run_portfolio_analysis
                portfolio_data = run_portfolio_analysis()
                if portfolio_data[0] is not None:
                    logger.info(f"  完了: {len(portfolio_data[0])}銘柄の保有情報")
                else:
                    logger.info("  ポートフォリオデータなし: スキップ")
            else:
                logger.info("  portfolio.csv未作成: スキップ")
        except Exception as e:
            logger.warning(f"  ポートフォリオ分析スキップ: {e}")

    # Step 6: Excelレポート生成
    if run_report:
        logger.info("\n[Step 6/6] Excelレポート生成中...")
        try:
            import pandas as pd
            from generate_report import generate_excel_report

            if analysis_df is None:
                analysis_path = BASE_DIR / "data" / "latest_analysis.csv"
                if analysis_path.exists():
                    analysis_df = pd.read_csv(analysis_path)
                else:
                    logger.error("  分析結果が見つかりません")
                    sys.exit(1)

            if macro_df is None:
                macro_path = BASE_DIR / "data" / "latest_macro.csv"
                if macro_path.exists():
                    macro_df = pd.read_csv(macro_path)

            output_path = generate_excel_report(
                analysis_df,
                macro_df=macro_df,
                portfolio_data=portfolio_data,
            )
            logger.info(f"  完了: {output_path}")

        except Exception as e:
            logger.error(f"  レポート生成エラー: {e}")
            import traceback
            traceback.print_exc()

    # Step 7: 予測スナップショット保存 & 精度評価
    if run_analyze:
        logger.info("\n[Step 7/7] 予測精度トラッキング...")
        try:
            from prediction_tracker import save_daily_snapshot, evaluate_predictions
            # 今日の予測をスナップショットとして保存
            save_daily_snapshot()
            # 過去の予測と現在の価格を比較して精度を評価
            accuracy_df = evaluate_predictions()
            if accuracy_df is not None and not accuracy_df.empty:
                correct = accuracy_df["direction_correct"].mean() * 100
                logger.info(f"  予測方向的中率: {correct:.1f}% ({len(accuracy_df)}件)")
            else:
                logger.info("  過去の予測データがまだありません（明日以降に精度レポートが生成されます）")
        except Exception as e:
            logger.warning(f"  予測トラッキングスキップ: {e}")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"\n全処理完了: {elapsed:.1f}秒")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
