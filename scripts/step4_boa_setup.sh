#!/bin/bash
# STEP4: BOA (Body-and-Organ-Analysis) Docker環境構築

echo "========================================="
echo "STEP4: BOA Docker環境構築"
echo "========================================="

# 1. toolsディレクトリに移動
cd ~/ctrate_ws/tools || exit 1

# 2. BOAリポジトリのクローン
if [ -d "BOA" ]; then
    echo "BOAディレクトリが既に存在します。スキップします。"
else
    echo "1. BOAリポジトリをクローン中..."
    git clone https://github.com/UMEssen/Body-and-Organ-Analysis.git BOA
fi

cd BOA || exit 1

# 3. .envファイルの作成
echo ""
echo "2. 環境設定ファイル(.env)を作成中..."

if [ -f ".env" ]; then
    echo ".envファイルが既に存在します。バックアップを作成します。"
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
fi

# .env_sampleから.envをコピー
if [ -f ".env_sample" ]; then
    cp .env_sample .env
else
    # .env_sampleがない場合は基本的な.envを作成
    cat > .env << 'EOF'
# BOA環境設定

# データディレクトリ（CT-RATEデータの場所）
DATA_DIR=/home/soya/ctrate_ws/data
OUTPUT_DIR=/home/soya/ctrate_ws/outputs/boa

# GPU設定
CUDA_VISIBLE_DEVICES=0

# TotalSegmentatorライセンスディレクトリ
TOTALSEG_LICENSE_DIR=/home/soya/.totalsegmentator

# メモリ設定
SHARED_MEMORY_SIZE=8g

# ワーカー数
NUM_WORKERS=1

# ログレベル
LOG_LEVEL=INFO
EOF
fi

echo "3. .envファイルの内容:"
echo "----------------------------"
cat .env
echo "----------------------------"

# 4. 必要なディレクトリの作成
echo ""
echo "4. 必要なディレクトリを作成中..."
mkdir -p ~/ctrate_ws/outputs/boa
mkdir -p ~/ctrate_ws/logs/boa

# 5. docker-compose設定ファイルの確認
echo ""
echo "5. Docker Compose設定ファイルの確認..."

if [ -f "docker-compose-win.yml" ]; then
    COMPOSE_FILE="docker-compose-win.yml"
    echo "Windows用のdocker-compose-win.ymlを使用します"
elif [ -f "docker-compose.yml" ]; then
    COMPOSE_FILE="docker-compose.yml"
    echo "標準のdocker-compose.ymlを使用します"
else
    echo "エラー: docker-compose設定ファイルが見つかりません"
    exit 1
fi

# 6. Docker Composeでコンテナをビルド・起動
echo ""
echo "6. BOAコンテナの準備..."
echo "注意: 初回実行時はイメージのダウンロードに時間がかかります"

read -p "BOAコンテナを起動しますか？ (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "コンテナをビルド・起動中..."
    docker compose -f "$COMPOSE_FILE" up -d --build
    
    # コンテナの状態確認
    echo ""
    echo "7. コンテナの状態確認:"
    docker compose -f "$COMPOSE_FILE" ps
    
    # BOAコンテナが起動しているか確認
    if docker compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        echo ""
        echo "✓ BOAコンテナが正常に起動しました"
        
        # TotalSegmentatorライセンスの設定確認
        echo ""
        echo "8. TotalSegmentatorライセンスの確認..."
        if [ -f ~/.totalsegmentator/totalseg_license.txt ]; then
            echo "✓ TotalSegmentatorライセンスが検出されました"
        else
            echo "警告: TotalSegmentatorライセンスが見つかりません"
            echo "STEP3を先に実行してライセンスを設定してください"
        fi
    else
        echo "エラー: BOAコンテナの起動に失敗しました"
        echo "ログを確認してください:"
        docker compose -f "$COMPOSE_FILE" logs
        exit 1
    fi
else
    echo "コンテナの起動をスキップしました"
fi

# 7. 実行例スクリプトの作成
echo ""
echo "9. BOA実行例スクリプトを作成中..."

cat > ~/ctrate_ws/scripts/run_boa_single.sh << 'EOF'
#!/bin/bash
# BOAで単一症例を処理する例

# 入力CTファイル（引数で指定）
INPUT_CT=$1
if [ -z "$INPUT_CT" ]; then
    echo "使用法: $0 <入力CTファイルパス>"
    exit 1
fi

# 出力ディレクトリ名を入力ファイル名から生成
BASENAME=$(basename "$INPUT_CT" .nii.gz)
OUTPUT_DIR="/outputs/boa/${BASENAME}"

# BOAコンテナで実行
cd ~/ctrate_ws/tools/BOA
docker compose exec boa bash -c "
    boa_cli run \
        --input '$INPUT_CT' \
        --output '$OUTPUT_DIR' \
        --tasks total heartchambers body_composition
"

echo "処理完了: $OUTPUT_DIR"
EOF

chmod +x ~/ctrate_ws/scripts/run_boa_single.sh

echo ""
echo "========================================="
echo "STEP4 完了: BOA環境が構築されました"
echo ""
echo "使用方法:"
echo "  単一症例の処理: ~/ctrate_ws/scripts/run_boa_single.sh <CTファイルパス>"
echo "  コンテナ停止: cd ~/ctrate_ws/tools/BOA && docker compose down"
echo "  コンテナ再起動: cd ~/ctrate_ws/tools/BOA && docker compose up -d"
echo ""
echo "次はSTEP5（EAT/PAT抽出）に進んでください"
echo "========================================="