#!/bin/bash
# ============================================================
# Growth Stock Analyzer - セットアップスクリプト
# ============================================================
# 使い方: chmod +x setup.sh && ./setup.sh

set -e

echo "=================================="
echo " Growth Stock Analyzer セットアップ"
echo "=================================="

# Python バージョン確認
python3 --version || { echo "Python 3が必要です"; exit 1; }

# 仮想環境作成
if [ ! -d "venv" ]; then
    echo "仮想環境を作成中..."
    python3 -m venv venv
fi

# 仮想環境有効化
source venv/bin/activate

# パッケージインストール
echo "パッケージをインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

# ディレクトリ作成
mkdir -p data output logs

echo ""
echo "=================================="
echo " セットアップ完了!"
echo "=================================="
echo ""
echo "次のステップ:"
echo "  1. config.yaml を編集してAPIキーを設定"
echo "     - Alpha Vantage: https://www.alphavantage.co/support/#api-key (無料)"
echo "     - Reddit: https://www.reddit.com/prefs/apps (無料)"
echo "     - X (Twitter): https://developer.x.com/ ($100/月~)"
echo ""
echo "  2. 実行:"
echo "     source venv/bin/activate"
echo "     python run.py"
echo ""
echo "  3. 定期実行 (オプション):"
echo "     crontab -e"
echo "     0 7 * * 1-5 cd $(pwd) && source venv/bin/activate && python run.py"
echo ""
