"""
fetch_sentiment.py - SNS・話題性データの自動取得モジュール

データソース:
  1. Google Trends (pytrends) — 無料
  2. Reddit (公開JSON API / PRAW) — APIキー不要で基本取得可能
  3. X / Twitter API — 有料 ($100/月~)、キー未設定時はスキップ

出力: data/sentiment_data_{date}.csv
"""

import pandas as pd
import numpy as np
import yaml
import logging
import time
import requests as req
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fetch_sentiment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 1. Google Trends (改善2: nan除去 + レート制限対策強化)
# ============================================================
def fetch_google_trends(tickers_with_names, period="today 3-m"):
    """
    Google Trendsで各銘柄の検索ボリューム推移を取得。
    改善: nanバグ修正、sleep 10秒、リトライ3回
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.warning("pytrends未インストール: pip install pytrends")
        return {}

    pytrends = TrendReq(hl="en-US", tz=540)  # JST
    results = {}

    keywords = [t["ticker"].replace(".T", "") for t in tickers_with_names]

    for i in range(0, len(keywords), 5):
        batch = keywords[i:i+5]
        success = False

        for retry in range(3):  # リトライ3回
            try:
                logger.info(f"Google Trends取得中: {batch} (試行 {retry + 1}/3)")
                pytrends.build_payload(batch, cat=0, timeframe=period, geo="")
                df = pytrends.interest_over_time()

                if df.empty:
                    success = True
                    break

                for kw in batch:
                    if kw in df.columns:
                        series = df[kw]
                        # 改善2: nanを除去してから計算
                        series = series.dropna()
                        if series.empty:
                            continue

                        current = series.iloc[-1]
                        avg = series.mean()
                        max_val = series.max()

                        # nan チェック: int変換前にnanを除去
                        if pd.isna(current) or pd.isna(avg) or pd.isna(max_val):
                            continue

                        recent_avg = series.tail(7).mean() if len(series) >= 7 else current
                        if pd.isna(recent_avg):
                            recent_avg = current
                        trend_ratio = recent_avg / avg if avg > 0 else 1.0

                        is_trending = trend_ratio > 1.5

                        results[kw] = {
                            "gtrends_current": int(current),
                            "gtrends_avg": round(float(avg), 1),
                            "gtrends_max": int(max_val),
                            "gtrends_trend_ratio": round(float(trend_ratio), 2),
                            "gtrends_is_trending": is_trending,
                        }

                success = True
                break

            except Exception as e:
                logger.warning(f"Google Trends error for {batch} (試行 {retry + 1}): {e}")
                if retry < 2:
                    logger.info(f"  → {10 * (retry + 1)}秒後にリトライ...")
                    time.sleep(10 * (retry + 1))  # 10秒, 20秒, 30秒

        if not success:
            logger.error(f"Google Trends: {batch} は3回リトライ後も失敗")

        time.sleep(10)  # バッチ間は10秒待機

    return results


# ============================================================
# 2. Reddit (改善2: APIキー不要の公開エンドポイント対応)
# ============================================================
def fetch_reddit_public(tickers_with_names, config):
    """
    Reddit公開JSONエンドポイントで銘柄メンション数を取得。
    APIキー不要。User-Agent付きでリクエスト。
    """
    sent_cfg = config["sentiment"]
    results = {}

    headers = {
        "User-Agent": "GrowthStockAnalyzer/1.0 (stock analysis tool)"
    }

    bullish_keywords = ["buy", "calls", "moon", "rocket", "bullish", "long", "undervalued", "dip"]
    bearish_keywords = ["sell", "puts", "crash", "dump", "bearish", "short", "overvalued", "bubble"]

    for stock in tickers_with_names:
        ticker = stock["ticker"].replace(".T", "")
        total_mentions = 0
        total_score = 0
        total_comments = 0
        bullish_count = 0
        bearish_count = 0

        for sub_name in sent_cfg["reddit_subreddits"]:
            try:
                url = f"https://www.reddit.com/r/{sub_name}/search.json"
                params = {
                    "q": ticker,
                    "sort": "new",
                    "t": "week",
                    "limit": 50,
                    "restrict_sr": "true",
                }
                resp = req.get(url, headers=headers, params=params, timeout=15)

                if resp.status_code == 429:
                    logger.warning(f"Reddit rate limit for r/{sub_name}, waiting 60s...")
                    time.sleep(60)
                    resp = req.get(url, headers=headers, params=params, timeout=15)

                if resp.status_code != 200:
                    logger.warning(f"Reddit r/{sub_name} returned {resp.status_code}")
                    continue

                data = resp.json()
                posts = data.get("data", {}).get("children", [])

                for post in posts:
                    post_data = post.get("data", {})
                    total_mentions += 1
                    total_score += post_data.get("score", 0)
                    total_comments += post_data.get("num_comments", 0)

                    text = (post_data.get("title", "") + " " + post_data.get("selftext", "")).lower()
                    if any(kw in text for kw in bullish_keywords):
                        bullish_count += 1
                    if any(kw in text for kw in bearish_keywords):
                        bearish_count += 1

                time.sleep(2)  # レート制限対策
            except Exception as e:
                logger.error(f"Reddit error for {ticker} in r/{sub_name}: {e}")

        sentiment_score = 0
        if bullish_count + bearish_count > 0:
            sentiment_score = round((bullish_count - bearish_count) / (bullish_count + bearish_count) * 100, 1)

        results[ticker] = {
            "reddit_mentions": total_mentions,
            "reddit_total_score": total_score,
            "reddit_total_comments": total_comments,
            "reddit_bullish": bullish_count,
            "reddit_bearish": bearish_count,
            "reddit_sentiment": sentiment_score,
        }

        logger.info(f"  {ticker}: {total_mentions}件のメンション, センチメント={sentiment_score}")

    return results


def fetch_reddit_mentions(tickers_with_names, config):
    """
    Redditのデータ取得。PRAWがあればPRAW、なければ公開APIを使用。
    """
    api_cfg = config["api_keys"]

    # APIキーが設定されている場合はPRAWを使用
    if api_cfg["reddit_client_id"] != "YOUR_REDDIT_CLIENT_ID":
        try:
            import praw
            return _fetch_reddit_praw(tickers_with_names, config)
        except ImportError:
            logger.info("prawが未インストール。公開APIにフォールバック")

    # APIキーなしの場合は公開JSONエンドポイントを使用
    logger.info("Reddit: 公開JSONエンドポイントで取得")
    return fetch_reddit_public(tickers_with_names, config)


def _fetch_reddit_praw(tickers_with_names, config):
    """PRAWを使用したRedditデータ取得（従来の方法）"""
    import praw

    api_cfg = config["api_keys"]
    sent_cfg = config["sentiment"]

    try:
        reddit = praw.Reddit(
            client_id=api_cfg["reddit_client_id"],
            client_secret=api_cfg["reddit_client_secret"],
            user_agent=api_cfg["reddit_user_agent"],
        )
    except Exception as e:
        logger.error(f"Reddit接続エラー: {e}")
        return {}

    results = {}
    lookback_days = sent_cfg["reddit_lookback_days"]
    cutoff_time = datetime.utcnow() - timedelta(days=lookback_days)

    bullish_keywords = ["buy", "calls", "moon", "rocket", "bullish", "long", "undervalued", "dip"]
    bearish_keywords = ["sell", "puts", "crash", "dump", "bearish", "short", "overvalued", "bubble"]

    for stock in tickers_with_names:
        ticker = stock["ticker"].replace(".T", "")
        total_mentions = 0
        total_score = 0
        total_comments = 0
        bullish_count = 0
        bearish_count = 0

        for sub_name in sent_cfg["reddit_subreddits"]:
            try:
                subreddit = reddit.subreddit(sub_name)
                search_query = f"${ticker} OR {ticker}"
                for submission in subreddit.search(search_query, time_filter="week", limit=100):
                    created = datetime.utcfromtimestamp(submission.created_utc)
                    if created < cutoff_time:
                        continue
                    total_mentions += 1
                    total_score += submission.score
                    total_comments += submission.num_comments
                    text = (submission.title + " " + (submission.selftext or "")).lower()
                    if any(kw in text for kw in bullish_keywords):
                        bullish_count += 1
                    if any(kw in text for kw in bearish_keywords):
                        bearish_count += 1
                time.sleep(1)
            except Exception as e:
                logger.error(f"Reddit error for {ticker} in r/{sub_name}: {e}")

        sentiment_score = 0
        if bullish_count + bearish_count > 0:
            sentiment_score = round((bullish_count - bearish_count) / (bullish_count + bearish_count) * 100, 1)

        results[ticker] = {
            "reddit_mentions": total_mentions,
            "reddit_total_score": total_score,
            "reddit_total_comments": total_comments,
            "reddit_bullish": bullish_count,
            "reddit_bearish": bearish_count,
            "reddit_sentiment": sentiment_score,
        }
        logger.info(f"  {ticker}: {total_mentions}件のメンション, センチメント={sentiment_score}")

    return results


# ============================================================
# 3. X (Twitter) API v2 (改善2: キー未設定時はスキップ、Noneを返す)
# ============================================================
def fetch_twitter_mentions(tickers_with_names, config):
    """
    X (Twitter) APIで銘柄メンション数・センチメントを取得。
    改善: APIキー未設定時はエラーにせずNoneを返す。
    """
    api_cfg = config["api_keys"]

    if api_cfg["twitter_bearer_token"] == "YOUR_TWITTER_BEARER_TOKEN":
        logger.info("Twitter APIキー未設定。スキップします（正常動作）")
        return None  # エラーではなくNone

    try:
        import tweepy
    except ImportError:
        logger.warning("tweepy未インストール: pip install tweepy")
        return None

    try:
        client = tweepy.Client(bearer_token=api_cfg["twitter_bearer_token"])
    except Exception as e:
        logger.error(f"Twitter接続エラー: {e}")
        return None

    results = {}
    lookback_hours = config["sentiment"]["twitter_lookback_hours"]
    start_time = datetime.utcnow() - timedelta(hours=lookback_hours)

    bullish_keywords = ["buy", "bullish", "moon", "long", "undervalued", "dip"]
    bearish_keywords = ["sell", "bearish", "crash", "short", "overvalued", "dump"]

    for stock in tickers_with_names:
        ticker = stock["ticker"].replace(".T", "")
        try:
            query = f"${ticker} OR #{ticker} -is:retweet lang:en"
            tweets = client.search_recent_tweets(
                query=query,
                max_results=100,
                start_time=start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                tweet_fields=["public_metrics", "created_at"],
            )

            if not tweets.data:
                results[ticker] = {
                    "twitter_mentions": 0, "twitter_likes": 0,
                    "twitter_retweets": 0, "twitter_sentiment": 0,
                }
                continue

            total_mentions = len(tweets.data)
            total_likes = sum(t.public_metrics["like_count"] for t in tweets.data)
            total_retweets = sum(t.public_metrics["retweet_count"] for t in tweets.data)

            bullish = sum(1 for t in tweets.data if any(kw in t.text.lower() for kw in bullish_keywords))
            bearish = sum(1 for t in tweets.data if any(kw in t.text.lower() for kw in bearish_keywords))

            sentiment = 0
            if bullish + bearish > 0:
                sentiment = round((bullish - bearish) / (bullish + bearish) * 100, 1)

            results[ticker] = {
                "twitter_mentions": total_mentions,
                "twitter_likes": total_likes,
                "twitter_retweets": total_retweets,
                "twitter_sentiment": sentiment,
            }
            logger.info(f"  {ticker}: {total_mentions}ツイート, センチメント={sentiment}")
            time.sleep(1)

        except Exception as e:
            logger.error(f"Twitter error for {ticker}: {e}")
            results[ticker] = {
                "twitter_mentions": 0, "twitter_likes": 0,
                "twitter_retweets": 0, "twitter_sentiment": 0,
            }

    return results


# ============================================================
# メイン: 全SNSデータを統合
# ============================================================
def fetch_all_sentiment(config):
    """全SNSデータを取得して統合"""
    all_stocks = (config["watchlist"].get("us_stocks", [])
                  + config["watchlist"].get("us_etfs", [])
                  + config["watchlist"].get("jp_stocks", []))

    logger.info("=" * 50)
    logger.info("SNSセンチメントデータ取得開始")
    logger.info("=" * 50)

    # 1. Google Trends
    logger.info("\n--- Google Trends ---")
    raw_period = config["sentiment"]["google_trends_period"]
    period_map = {
        "1m": "today 1-m", "3m": "today 3-m", "6m": "today 6-m",
        "12m": "today 12-m", "1y": "today 12-m",
    }
    gt_period = period_map.get(raw_period, raw_period)
    gtrends = fetch_google_trends(all_stocks, gt_period)

    # 2. Reddit
    logger.info("\n--- Reddit ---")
    reddit = fetch_reddit_mentions(all_stocks, config)

    # 3. Twitter/X
    logger.info("\n--- X (Twitter) ---")
    twitter = fetch_twitter_mentions(all_stocks, config)

    # 統合
    rows = []
    for stock in all_stocks:
        ticker = stock["ticker"].replace(".T", "")
        row = {"ticker": ticker, "name": stock["name"]}

        # Google Trends
        gt = gtrends.get(ticker, {})
        row.update(gt)

        # Reddit
        rd = reddit.get(ticker, {}) if reddit else {}
        row.update(rd)

        # Twitter (Noneの場合はスキップ)
        tw = twitter.get(ticker, {}) if twitter else {}
        row.update(tw)

        # 複合センチメントスコア (-100 ~ +100)
        scores = []
        if gt.get("gtrends_trend_ratio"):
            gt_ratio = gt["gtrends_trend_ratio"]
            if not pd.isna(gt_ratio):
                gt_score = min(100, max(-100, (gt_ratio - 1.0) * 100))
                scores.append(gt_score)
        if rd.get("reddit_sentiment") is not None and rd.get("reddit_mentions", 0) > 0:
            scores.append(rd["reddit_sentiment"])
        if tw.get("twitter_sentiment") is not None and tw.get("twitter_mentions", 0) > 0:
            scores.append(tw["twitter_sentiment"])

        row["combined_sentiment"] = round(sum(scores) / len(scores), 1) if scores else 0
        row["sentiment_data_sources"] = len(scores)
        row["fetch_time"] = datetime.now().isoformat()

        rows.append(row)

    df = pd.DataFrame(rows)

    # 保存
    today = datetime.now().strftime("%Y%m%d")
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / f"sentiment_data_{today}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"\n保存完了: {csv_path}")

    latest_path = data_dir / "latest_sentiment.csv"
    df.to_csv(latest_path, index=False, encoding="utf-8-sig")

    return df


if __name__ == "__main__":
    config = load_config()
    df = fetch_all_sentiment(config)
    print(f"\n取得完了: {len(df)}銘柄")
    if "combined_sentiment" in df.columns:
        print(df[["ticker", "name", "combined_sentiment", "sentiment_data_sources"]].to_string(index=False))
